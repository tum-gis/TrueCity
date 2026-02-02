#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys
from tqdm import tqdm

# Ensure upstream KPConv utils/models are importable
sys.path.append('/home/stud/nguyenti/storage/user/tum-di-lab/EARLy_notebooks/kpconv/_kpconv_upstream')

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import KPConv
from utils.metrics import metrics as compute_metrics


def robust_p2p_fitting_regularizer(net):
    """
    More robust version of p2p_fitting_regularizer that handles numerical edge cases
    """
    fitting_loss = 0
    repulsive_loss = 0
    eps = 1e-8  # Small epsilon to prevent division by zero
    
    for m in net.modules():
        if isinstance(m, KPConv) and m.deformable:
            
            ##############
            # Fitting loss
            ##############
            
            # Get the distance to closest input point and normalize
            if m.min_d2 is not None:
                # Add epsilon to prevent division by zero
                KP_min_d2 = m.min_d2 / (m.KP_extent ** 2 + eps)
                # Clamp to reasonable range to prevent explosion
                KP_min_d2 = torch.clamp(KP_min_d2, 0.0, 100.0)
                fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))
            
            ################
            # Repulsive loss
            ################
            
            if m.deformed_KP is not None:
                # Normalized KP locations
                KP_locs = m.deformed_KP / (m.KP_extent + eps)
                
                # Point should not be close to each other
                for i in range(min(net.K, KP_locs.shape[1])):  # Safety check
                    if i < KP_locs.shape[1] - 1:
                        other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                        if other_KP.shape[1] > 0:  # Make sure we have other points
                            distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2) + eps)
                            rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                            repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K
    
    # Clamp final loss to reasonable range
    total_loss = net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)
    total_loss = torch.clamp(total_loss, 0.0, 1000.0)  # Prevent explosion
    
    return total_loss


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0
        # Track last validation mIoU in percent to show during training progress
        self.last_val_miou_pct = None
        # Track last training (per-batch) mIoU in percent
        self.last_train_miou_pct = None

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
    

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file: mirror terminal-like progress (no CSV header)
            open(join(config.saving_path, 'training.txt'), "w").close()

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        best_miou = -1.0
        use_amp = bool(getattr(config, 'mixed_precision', False))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        for epoch in range(config.max_epoch):

            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0
            with tqdm(total=len(training_loader), desc=f"Epoch {epoch+1}/{config.max_epoch}", unit="batch") as pbar:
                for batch in training_loader:

                    # Check kill signal (running_PID.txt deleted)
                    if config.saving and not exists(PID_file):
                        continue

                    ##################
                    # Processing batch
                    ##################

                    # New time
                    t = t[-1:]
                    t += [time.time()]

                    if 'cuda' in self.device.type:
                        batch.to(self.device)

                    # Determine accumulation steps
                    base_bs = getattr(config, 'batch_num', 1)
                    eff_bs = getattr(config, 'effective_batch_size', None)
                    acc_steps = (eff_bs // base_bs) if eff_bs else 1

                    # Only zero grad at the beginning of an accumulation window
                    if (self.step % acc_steps) == 0:
                        self.optimizer.zero_grad()

                    # Forward pass
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = net(batch, config)
                    else:
                        outputs = net(batch, config)
                    labels_all = batch.labels
                    loss = net.loss(outputs, labels_all)
                    acc = net.accuracy(outputs, labels_all)

                    # Compute per-batch training mIoU (ignored labels already masked out)
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        targets = labels_all.cpu().numpy()
                        Cb = fast_confusion(preds, targets, np.arange(config.num_classes))
                        IoUs_b = IoU_from_confusions(Cb)
                        self.last_train_miou_pct = float(100.0 * np.mean(IoUs_b))

                    # Use torch's loss as-is (CrossEntropyLoss); rely on warmup/accumulation to avoid NaNs

                    t += [time.time()]

                    # Backward (scale loss for accumulation)
                    if use_amp:
                        scaler.scale(loss / acc_steps).backward()
                    else:
                        (loss / acc_steps).backward()

                    # Step optimizer only at the end of accumulation window
                    if ((self.step + 1) % acc_steps) == 0:
                        if config.grad_clip_norm > 0:
                            # Check if any parameters have gradients before clipping
                            has_gradients = any(param.grad is not None and param.grad.numel() > 0 
                                              for param in net.parameters() if param.requires_grad)
                            if has_gradients:
                                if use_amp:
                                    scaler.unscale_(self.optimizer)
                                    torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                                else:
                                    torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                            else:
                                print("Warning: No gradients found, skipping gradient clipping")
                        if use_amp:
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()

                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(self.device)

                    t += [time.time()]

                    # Average timing
                    if self.step < 2:
                        mean_dt = np.array(t[1:]) - np.array(t[:-1])
                    else:
                        mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Update progress bar with metrics
                    postfix_dict = {
                        'loss': f'{loss.item():.3f}',
                        'acc': f'{100*acc:.1f}%',
                        'mIoU': f'{self.last_train_miou_pct:.2f}%' if self.last_train_miou_pct is not None else 'N/A'
                    }
                    pbar.set_postfix(postfix_dict)
                    pbar.update(1)

                    # Log file
                    if config.saving:
                        with open(join(config.saving_path, 'training.txt'), "a") as file:
                            message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                            file.write(message.format(self.epoch,
                                                      self.step,
                                                      net.output_loss,
                                                      net.reg_loss,
                                                      acc,
                                                      t[-1] - t0))

                    self.step += 1

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate schedule (linear interpolation if enabled)
            if getattr(config, 'lr_schedule', None) == 'linear':
                # progress in [0,1]
                progress = min(1.0, max(0.0, float(self.epoch) / max(1, (config.max_epoch - 1))))
                lr_now = float(config.lr_start) + progress * (float(config.lr_end) - float(config.lr_start))
                for param_group in self.optimizer.param_groups:
                    base_lr = lr_now
                    if 'lr' in param_group and param_group['lr'] != self.optimizer.defaults.get('lr', config.learning_rate):
                        # keep deform params scaled proportionally to main lr
                        pass
                    param_group['lr'] = base_lr

            # Also apply discrete decay schedule if configured
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            net.eval()
            miou = self.validation(net, val_loader, config)
            net.train()

            # Save best model (miou is returned as fraction 0-1)
            if miou is not None and miou > best_miou:
                best_miou = miou
                # Save next to current checkpoint, named with dataset
                dataset_name = getattr(config, 'dataset_name', getattr(config, 'dataset', 'dataset'))
                best_name = f"best_model_{dataset_name}.pth"
                best_path = join(checkpoint_directory, best_name)
                torch.save(net.state_dict(), best_path)
                print(f'New best mIoU: {best_miou*100:.2f}%, saved to {best_path}')

        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config):

        if config.dataset_task == 'classification':
            return self.object_classification_validation(net, val_loader, config)
        elif config.dataset_task == 'segmentation':
            return self.object_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'cloud_segmentation':
            return self.cloud_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'slam_segmentation':
            return self.slam_segmentation_validation(net, val_loader, config)
        else:
            raise ValueError('No validation method implemented for this network type')

    def object_classification_validation(self, net, val_loader, config):
        """
        Perform a round of validation and show/save results
        :param net: network object
        :param val_loader: data loader for validation set
        :param config: configuration object
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Number of classes predicted by the model
        nc_model = config.num_classes
        softmax = torch.nn.Softmax(1)

        # Initialize global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = np.zeros((val_loader.dataset.num_models, nc_model))

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in val_loader:

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            probs += [softmax(outputs).cpu().detach().numpy()]
            targets += [batch.labels.cpu().numpy()]
            obj_inds += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * len(obj_inds) / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        probs = np.vstack(probs)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)

        ###################
        # Voting validation
        ###################

        self.val_probs[obj_inds] = val_smooth * self.val_probs[obj_inds] + (1-val_smooth) * probs

        ############
        # Confusions
        ############

        validation_labels = np.array(val_loader.dataset.label_values)

        # Compute classification results
        C1 = fast_confusion(targets,
                            np.argmax(probs, axis=1),
                            validation_labels)

        # Compute votes confusion
        C2 = fast_confusion(val_loader.dataset.input_labels,
                            np.argmax(self.val_probs, axis=1),
                            validation_labels)


        # Saving (optionnal)
        if config.saving:
            print("Save confusions")
            conf_list = [C1, C2]
            file_list = ['val_confs.txt', 'vote_confs.txt']
            for conf, conf_file in zip(conf_list, file_list):
                test_file = join(config.saving_path, conf_file)
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')
                else:
                    with open(test_file, "w") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')

        val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        vote_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
        print('Accuracies : val = {:.1f}% / vote = {:.1f}%'.format(val_ACC, vote_ACC))

        return C1

    def cloud_segmentation_validation(self, net, val_loader, config, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else config.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Fallback path if dataset does not expose upstream attributes
        if not (hasattr(val_loader.dataset, 'input_labels') and hasattr(val_loader.dataset, 'label_values') and hasattr(val_loader.dataset, 'validation_labels')):
            # Simple validation: aggregate confusion over batches
            from utils.metrics import fast_confusion, IoU_from_confusions
            C = np.zeros((nc_model, nc_model), dtype=np.int64)
            with torch.no_grad():
                phase = getattr(val_loader.dataset, 'set', 'val')
                # Log start of phase
                if config.saving:
                    from os.path import join
                    with open(join(config.saving_path, 'training.txt'), 'a') as f:
                        f.write(f"=== {phase.title()} Evaluation Start ===\n")
                with tqdm(total=len(val_loader), desc=f"{phase.title()} Eval", unit="batch") as pbar:
                    for batch in val_loader:
                        if 'cuda' in self.device.type:
                            batch.to(self.device)
                        outputs = net(batch, config)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        targets = batch.labels.cpu().numpy()
                        mask = targets >= 0
                        preds = preds[mask]
                        targets = targets[mask]
                        C += fast_confusion(preds, targets, np.arange(nc_model))
                        pbar.update(1)
            IoUs = IoU_from_confusions(C)
            mIoU = 100 * np.mean(IoUs)
            print(f'{phase.title()} mIoU (fallback): {mIoU:.2f}%')
            # Track for progress display/logging
            self.last_val_miou_pct = float(mIoU)

            # Also report accuracy and per-class stats
            PRE, REC, F1, IoU_pc, ACC = compute_metrics(C)
            overall_acc = 100.0 * ACC
            print(f'{phase.title()} accuracy (fallback): {overall_acc:.2f}%')

            names_map = getattr(val_loader.dataset, 'label_to_names', None)
            class_names = [names_map[i] if isinstance(names_map, dict) and i in names_map else str(i)
                           for i in range(nc_model)]
            print('Per-class metrics (fallback):')
            for i in range(nc_model):
                print(f'  [{i:02d}] {class_names[i]:<24} IoU={100*IoU_pc[i]:5.2f}% '
                      f'Prec={100*PRE[i]:5.2f}% Rec={100*REC[i]:5.2f}% F1={100*F1[i]:5.2f}%')

            # Log summary to training.txt
            if config.saving:
                with open(join(config.saving_path, 'training.txt'), 'a') as f:
                    f.write(f"[{phase.title()}] mIoU (fallback): {mIoU:.2f}%\n")
                    f.write(f"[{phase.title()}] accuracy (fallback): {overall_acc:.2f}%\n")
                    for i in range(nc_model):
                        f.write(
                            f"[{phase.title()}] class[{i:02d}] {class_names[i]:<24} IoU={100*IoU_pc[i]:5.2f}% "
                            f"Prec={100*PRE[i]:5.2f}% Rec={100*REC[i]:5.2f}% F1={100*F1[i]:5.2f}%\n"
                        )
                    f.write(f"=== {phase.title()} Evaluation End ===\n")

            return mIoU / 100.0  # Convert percentage to fraction for consistency

        #print(nc_tot)
        #print(nc_model)

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_labels]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in val_loader.dataset.validation_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop with progress bar
        phase = getattr(val_loader.dataset, 'set', 'val')
        # Log start of phase
        if config.saving:
            with open(join(config.saving_path, 'training.txt'), 'a') as f:
                f.write(f"=== {phase.title()} Evaluation Start ===\n")
        with tqdm(total=len(val_loader), desc=f"{phase.title()} Eval", unit="batch") as pbar:
            for i, batch in enumerate(val_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                labels = batch.labels.cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    target = labels[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                       + (1 - val_smooth) * probs

                    # Stack all prediction for this epoch
                    predictions.append(probs)
                    targets.append(target)
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (time.time() - last_display) > 1.0:
                    last_display = time.time()
                    pbar.set_postfix({'t1(ms)': f"{1000*mean_dt[0]:.1f}", 't2(ms)': f"{1000*mean_dt[1]:.1f}"})
                pbar.update(1)

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)


        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)


        t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        t5 = time.time()

        # Also report accuracy and per-class stats
        PRE, REC, F1, IoU_pc, ACC = compute_metrics(C)
        overall_acc = 100.0 * ACC
        print('Validation accuracy: {:.2f}%'.format(overall_acc))
        # Map class indices to names after removing ignored labels if available
        names_map = getattr(val_loader.dataset, 'label_to_names', None)
        label_values = getattr(val_loader.dataset, 'label_values', None)
        ignored = set(getattr(val_loader.dataset, 'ignored_labels', []))
        if isinstance(label_values, np.ndarray) and label_values.size == (IoUs.shape[0] + len(ignored)):
            valid_label_values = [lv for lv in label_values.tolist() if lv not in ignored]
            class_names = [names_map.get(lv, str(lv)) if isinstance(names_map, dict) else str(lv) for lv in valid_label_values]
        else:
            class_names = [names_map[i] if isinstance(names_map, dict) and i in names_map else str(i)
                           for i in range(IoUs.shape[0])]
        print('Per-class metrics:')
        for i in range(IoUs.shape[0]):
            print(f'  [{i:02d}] {class_names[i]:<24} IoU={100*IoU_pc[i]:5.2f}% '
                  f'Prec={100*PRE[i]:5.2f}% Rec={100*REC[i]:5.2f}% F1={100*F1[i]:5.2f}%')

        # Log summary to training.txt
        if config.saving:
            with open(join(config.saving_path, 'training.txt'), 'a') as f:
                f.write(f"[{phase.title()}] accuracy: {overall_acc:.2f}%\n")
                for i in range(IoUs.shape[0]):
                    f.write(
                        f"[{phase.title()}] class[{i:02d}] {class_names[i]:<24} IoU={100*IoU_pc[i]:5.2f}% "
                        f"Prec={100*PRE[i]:5.2f}% Rec={100*REC[i]:5.2f}% F1={100*F1[i]:5.2f}%\n"
                    )

        # Saving (optionnal)
        if config.saving:

            # Name of saving file
            test_file = join(config.saving_path, 'val_IoUs.txt')

            # Line to write:
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line = line + '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

            # Save potentials
            if val_loader.dataset.use_potentials:
                pot_path = join(config.saving_path, 'potentials')
                if not exists(pot_path):
                    makedirs(pot_path)
                files = val_loader.dataset.files
                for i, file_path in enumerate(files):
                    pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
                    cloud_name = file_path.split('/')[-1]
                    pot_name = join(pot_path, cloud_name)
                    pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
                    write_ply(pot_name,
                            [pot_points.astype(np.float32), pots],
                            ['x', 'y', 'z', 'pots'])

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))
        # Track for progress display/logging
        self.last_val_miou_pct = float(mIoU)
        if config.saving:
            with open(join(config.saving_path, 'training.txt'), 'a') as f:
                f.write(f"[{phase.title()}] mean IoU = {mIoU:.1f}%\n")
                f.write(f"=== {phase.title()} Evaluation End ===\n")

        # Log
        if config.saving:
            with open(join(config.saving_path, 'progress_log.txt'), 'a') as lf:
                lf.write(f'Validation mIoU: {mIoU:.3f}\n')

        # Save predicted cloud occasionally
        if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
            val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch + 1))
            if not exists(val_path):
                makedirs(val_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):

                # Get points
                points = val_loader.dataset.load_evaluation_points(file_path)

                # Get probs on our own ply points
                sub_probs = self.validation_probs[i]

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                # Get the predicted labels
                sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                # Reproject preds on the evaluations points
                preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)

                # Path of saved validation file
                cloud_name = file_path.split('/')[-1]
                val_name = join(val_path, cloud_name)

                # Save file
                labels = val_loader.dataset.validation_labels[i].astype(np.int32)
                write_ply(val_name,
                          [points, preds, labels],
                          ['x', 'y', 'z', 'preds', 'class'])

        # Display timings
        t7 = time.time()
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('Confs bis . {:.1f}s'.format(t4 - t3))
            print('IoU ....... {:.1f}s'.format(t5 - t4))
            print('Save1 ..... {:.1f}s'.format(t6 - t5))
            print('Save2 ..... {:.1f}s'.format(t7 - t6))
            print('\n************************\n')

        return mIoU / 100.0

    def slam_segmentation_validation(self, net, val_loader, config, debug=True):
        """
        Validation method for slam segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Do not validate if dataset has no validation cloud
        if val_loader is None:
            return

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Create folder for validation predictions
        if not exists (join(config.saving_path, 'val_preds')):
            makedirs(join(config.saving_path, 'val_preds'))

        # initiate the dataset validation containers
        val_loader.dataset.val_points = []
        val_loader.dataset.val_labels = []

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        inds = []
        val_i = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            stk_probs = softmax(outputs).cpu().detach().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            f_inds = batch.frame_inds.cpu().numpy()
            r_inds_list = batch.reproj_inds
            r_mask_list = batch.reproj_masks
            labels_list = batch.val_labels
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                probs = stk_probs[i0:i0 + length]
                proj_inds = r_inds_list[b_i]
                proj_mask = r_mask_list[b_i]
                frame_labels = labels_list[b_i]
                s_ind = f_inds[b_i, 0]
                f_ind = f_inds[b_i, 1]

                # Project predictions on the frame points
                proj_probs = probs[proj_inds]

                # Safe check if only one point:
                if proj_probs.ndim < 2:
                    proj_probs = np.expand_dims(proj_probs, 0)

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

                # Predicted labels
                preds = val_loader.dataset.label_values[np.argmax(proj_probs, axis=1)]

                # Save predictions in a binary file
                filename = '{:s}_{:07d}.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                filepath = join(config.saving_path, 'val_preds', filename)
                if exists(filepath):
                    frame_preds = np.load(filepath)
                else:
                    frame_preds = np.zeros(frame_labels.shape, dtype=np.uint8)
                frame_preds[proj_mask] = preds.astype(np.uint8)
                np.save(filepath, frame_preds)

                # Save some of the frame pots
                if f_ind % 20 == 0:
                    seq_path = join(val_loader.dataset.path, 'sequences', val_loader.dataset.sequences[s_ind])
                    velo_file = join(seq_path, 'velodyne', val_loader.dataset.frames[s_ind][f_ind] + '.bin')
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    frame_points = frame_points.reshape((-1, 4))
                    write_ply(filepath[:-4] + '_pots.ply',
                              [frame_points[:, :3], frame_labels, frame_preds],
                              ['x', 'y', 'z', 'gt', 'pre'])

                # Update validation confusions
                frame_C = fast_confusion(frame_labels,
                                         frame_preds.astype(np.int32),
                                         val_loader.dataset.label_values)
                val_loader.dataset.val_confs[s_ind][f_ind, :, :] = frame_C

                # Stack all prediction for this epoch
                predictions += [preds]
                targets += [frame_labels[proj_mask]]
                inds += [f_inds[b_i, :]]
                val_i += 1
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        t3 = time.time()

        #######################################
        # Results on this subpart of validation
        #######################################

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        C *= np.expand_dims(val_loader.dataset.class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        #####################################
        # Results on the whole validation set
        #####################################

        t4 = time.time()

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in val_loader.dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)

        if debug:
            s = '\n'
            for cc in C_tot:
                for c in cc:
                    s += '{:8.1f} '.format(c)
                s += '\n'
            print(s)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            IoU_list = [IoUs, val_IoUs]
            file_list = ['subpart_IoUs.txt', 'val_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(config.saving_path, IoU_file)

                # Line to write:
                line = ''
                for IoU in IoUs_to_save:
                    line += '{:.3f} '.format(IoU)
                line = line + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} : subpart mIoU = {:.1f} %'.format(config.dataset, mIoU))
        mIoU = 100 * np.mean(val_IoUs)
        print('{:s} :     val mIoU = {:.1f} %'.format(config.dataset, mIoU))
        if config.saving:
            with open(join(config.saving_path, 'progress_log.txt'), 'a') as lf:
                lf.write(f'Validation mIoU: {mIoU:.3f}\n')

        t6 = time.time()

        # Display timings
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('IoU1 ...... {:.1f}s'.format(t4 - t3))
            print('IoU2 ...... {:.1f}s'.format(t5 - t4))
            print('Save ...... {:.1f}s'.format(t6 - t5))
            print('\n************************\n')

        return mIoU / 100.0



































