#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on DiLab dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from os.path import join, exists
import signal
import torch

# Ensure current repo root is first on sys.path (avoid picking stale copies)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kpconv.dilab import DiLabDataset, DiLabCollate, set_runtime_config
import kpconv.dilab as _dilab_mod
from kpconv.config import DiLabConfig
from kpconv.trainer import ModelTrainer, robust_p2p_fitting_regularizer

# KPFCNN from upstream path
sys.path.append('/home/stud/nguyenti/storage/user/tum-di-lab/EARLy_notebooks/kpconv/_kpconv_upstream')
from models.architectures import KPFCNN
import models.architectures as arch
from models.blocks import KPConv


def parse_args():
    p = argparse.ArgumentParser(description='Train KPConv on DiLab (2048-pt shards)')
    p.add_argument('--data_path', type=str, default='/home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree/data_25_octree', help='Root with train/ val/ test/')
    p.add_argument('--batch_size', type=int, default=None, help='Batch size')
    p.add_argument('--ebatch_size', type=int, default=None, help='Effective batch size for gradient accumulation (must be a multiple of batch size)')
    p.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config.max_epoch)')
    p.add_argument('--lr', type=float, default=None, help='Learning rate')
    p.add_argument('--resume', type=str, default=None, help='Checkpoint file or checkpoints dir')
    p.add_argument('--save_dir', type=str, default=None, help='Output dir for logs/checkpoints')
    p.add_argument('--best_model_dir', type=str, default='/home/stud/nguyenti/storage/user/EARLy', help='Directory to save best model .pth')
    p.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES (e.g., 0)')
    p.add_argument('--workers', type=int, default=None, help='Override DataLoader workers (input_threads)')
    return p.parse_args()


# Eps-hardened deformable fitting regularizer to avoid NaNs/Infs
# This replaces the upstream function at runtime so KPFCNN.loss uses it

def _safe_p2p_fitting_regularizer(net):
    eps = 1e-8
    fitting_loss = 0.0
    repulsive_loss = 0.0

    for m in net.modules():
        if isinstance(m, KPConv) and getattr(m, 'deformable', False):
            # Fitting loss term
            if getattr(m, 'min_d2', None) is not None:
                denom = (m.KP_extent ** 2) + eps
                kp_min_norm = m.min_d2 / denom
                kp_min_norm = torch.nan_to_num(kp_min_norm, nan=0.0, posinf=1e6, neginf=0.0)
                kp_min_norm = torch.clamp(kp_min_norm, 0.0, 1e6)
                fitting_loss = fitting_loss + net.l1(kp_min_norm, torch.zeros_like(kp_min_norm))

            # Repulsive loss term
            if getattr(m, 'deformed_KP', None) is not None:
                kp_locs = m.deformed_KP / (m.KP_extent + eps)
                kp_locs = torch.nan_to_num(kp_locs, nan=0.0, posinf=1e6, neginf=-1e6)
                K = getattr(net, 'K', kp_locs.shape[1])
                for i in range(min(K, kp_locs.shape[1])):
                    if i < kp_locs.shape[1] - 1:
                        other = torch.cat([kp_locs[:, :i, :], kp_locs[:, i + 1:, :]], dim=1).detach()
                        if other.shape[1] > 0:
                            d2 = torch.sum((other - kp_locs[:, i:i + 1, :]) ** 2, dim=2)
                            d2 = torch.clamp(d2, min=0.0)
                            distances = torch.sqrt(d2 + eps)
                            rep = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                            repulsive_loss = repulsive_loss + net.l1(rep, torch.zeros_like(rep)) / max(K, 1)

    total = net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)
    total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0)
    total = torch.clamp(total, 0.0, 1e6)
    return total

# Patch upstream regularizer
arch.p2p_fitting_regularizer = _safe_p2p_fitting_regularizer


def _worker_init_fn(worker_id):
    # Ensure workers use this repo root first
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def patch_kpfcnn_for_stability(net):
    """
    Patch KPFCNN model to use robust regularizer and add numerical stability checks
    """
    import torch  # Import torch here for the patched function
    
    original_loss = net.loss
    
    def robust_loss(self, outputs, labels):
        """
        Enhanced loss function with numerical stability
        """
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        # Original outputs shape: [N_points, C]
        # Ensure logits are [N, C] and targets are [N]
        outputs = outputs  # [N, C]
        target = target.squeeze(0) if target.dim() > 1 else target  # [N]

        # Clamp target to valid range [0, num_classes-1]
        num_classes = outputs.shape[-1]
        target = torch.clamp(target, 0, num_classes - 1)

        # Cross entropy loss with numerical stability
        outputs = torch.clamp(outputs, -100, 100)
        self.output_loss = self.criterion(outputs, target)
        
        # Check for NaN in output loss and use a meaningful fallback
        if torch.isnan(self.output_loss):
            print("NaN detected in output loss, computing fallback loss")
            # Use a simple cross-entropy-like fallback that maintains gradients
            probs = torch.softmax(outputs, dim=1)  # softmax over classes
            probs = torch.clamp(probs, min=1e-8, max=1.0-1e-8)
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).float()  # [N, C]
            self.output_loss = -torch.mean(torch.sum(target_one_hot * torch.log(probs), dim=1))

        # Regularization of deformable offsets with robust function
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = robust_p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)
            
        # Check for NaN in reg loss and provide meaningful fallback
        if torch.isnan(self.reg_loss):
            print("NaN detected in regularization loss, using minimal regularization")
            # Use a small but meaningful regularization term
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            param_norm = sum(torch.norm(p)**2 for p in self.parameters() if p.requires_grad and p.grad is not None)
            self.reg_loss = 1e-6 * param_norm / total_params if total_params > 0 else torch.tensor(1e-6, device=outputs.device, requires_grad=True)

        # Combined loss with reasonable bounds
        combined_loss = self.output_loss + self.reg_loss
        combined_loss = torch.clamp(combined_loss, 1e-6, 100.0)  # Ensure it's not too small or too large
        
        return combined_loss
    
    # Bind the new loss function to the net instance
    net.loss = robust_loss.__get__(net, net.__class__)
    print("Applied numerical stability patches to KPFCNN model")
    return net


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = DiLabConfig()

    if args.batch_size is not None:
        config.batch_num = args.batch_size
    if args.epochs is not None:
        config.max_epoch = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.workers is not None:
        config.input_threads = args.workers

    # Validate and set effective batch size for gradient accumulation
    if args.ebatch_size is not None:
        base_bs = args.batch_size if args.batch_size is not None else getattr(config, 'batch_num', 1)
        assert args.ebatch_size >= base_bs, f"ebatch_size ({args.ebatch_size}) must be >= batch_size ({base_bs})"
        assert args.ebatch_size % base_bs == 0, f"ebatch_size ({args.ebatch_size}) must be a multiple of batch_size ({base_bs})"
        config.effective_batch_size = args.ebatch_size
    else:
        config.effective_batch_size = None

    # Dataset root
    if args.data_path:
        config.data_root = args.data_path

    # Derive dataset name from data path
    dataset_name = os.path.basename(os.path.normpath(getattr(config, 'data_root', 'dataset')))
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Saving path with dataset name - using format requested by user: kp_conv_<dataset>_<time>_log
    if args.save_dir is not None:
        config.saving_path = args.save_dir
    else:
        config.saving_path = os.path.join('results', f'kp_conv_{dataset_name}_{timestamp}_log')

    # Best model path
    model_name = 'kpconv_segmentation'
    best_fname = f'{dataset_name}_{model_name}.pth'
    config.best_model_path = os.path.join(args.best_model_dir, best_fname)
    config.dataset_name = dataset_name

    # Resolve checkpoint to resume from
    chosen_chkp = None
    chosen_weights = None
    if args.resume:
        if os.path.isdir(args.resume):
            cand = os.path.join(args.resume, 'current_chkp.tar')
            if os.path.exists(cand):
                chosen_chkp = cand
            else:
                tars = [f for f in os.listdir(args.resume) if f.endswith('.tar')]
                pths = [f for f in os.listdir(args.resume) if f.endswith('.pth')]
                if tars:
                    chosen_chkp = os.path.join(args.resume, sorted(tars)[-1])
                elif pths:
                    chosen_weights = os.path.join(args.resume, sorted(pths)[-1])
        elif os.path.isfile(args.resume):
            if args.resume.endswith('.pth'):
                chosen_weights = args.resume
            else:
                chosen_chkp = args.resume

    print('\nData Preparation')
    print('****************')
    print(f'Using dilab module from: {_dilab_mod.__file__}')
    print(f'sys.path[0]: {sys.path[0]}')
    # Provide runtime config to dilab collate
    set_runtime_config(config)
    train_ds = DiLabDataset(config, set='training')
    val_ds = DiLabDataset(config, set='validation')
    test_ds = DiLabDataset(config, set='test')

    print(f'Dataset: {dataset_name}')
    print(f'  Train files: {len(train_ds)} | Val files: {len(val_ds)} | Test files: {len(test_ds)}')
    print(f'  Batch size: {config.batch_num} | LR: {config.learning_rate} | Save dir: {config.saving_path}')
    print(f'  Workers: {config.input_threads}')

    train_loader = DataLoader(
        train_ds,
                             batch_size=config.batch_num,
                             collate_fn=DiLabCollate,
                             shuffle=True,
                             num_workers=config.input_threads,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
                             batch_size=config.batch_num,
                             collate_fn=DiLabCollate,
                             shuffle=False,
                             num_workers=config.input_threads,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
                             batch_size=config.batch_num,
                             collate_fn=DiLabCollate,
                             shuffle=False,
                             num_workers=config.input_threads,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )

    print('\nModel Preparation')
    print('*****************')
    t1 = time.time()
    labels = list(range(config.num_classes))
    net = KPFCNN(config, labels, np.array([]))

    # Patch the model for numerical stability
    # Use upstream KPFCNN loss (torch CrossEntropyLoss + p2p regularizer)

    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)

    # If a standalone weights file (.pth) was provided, load it now
    if chosen_weights is not None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        state = torch.load(chosen_weights, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            net.load_state_dict(state['model_state_dict'])
        else:
            net.load_state_dict(state)
        print(f"Loaded model weights from {chosen_weights}")

    # Write an overview log
    os.makedirs(config.saving_path, exist_ok=True)
    with open(os.path.join(config.saving_path, 'run_overview.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Train files: {len(train_ds.files)} | Val files: {len(val_ds.files)}\n")
        f.write(f"Batch size: {config.batch_num} | Epochs: {config.max_epoch} | LR: {config.learning_rate} | Save dir: {config.saving_path}\n")
        f.write(f"Workers: {config.input_threads}\n")
        if getattr(config, 'effective_batch_size', None):
            f.write(f"Effective batch size: {config.effective_batch_size}\n")

    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')
    trainer.train(net, train_loader, val_loader, config)

    # Run test at the end
    print('\nStart testing')
    print('*************')
    net.eval()
    trainer.validation(net, test_loader, config)