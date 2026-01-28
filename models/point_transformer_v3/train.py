import os
import time
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from config import get_config
from data_utils import create_dataloaders
from model_v3 import AverageMeter, intersectionAndUnionGPU
from model_v3 import PointTransformerV3
from test import run_test
import argparse

def get_logger():
    """Set up logger for console output"""
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PTv3WithHead(nn.Module):
    def __init__(self, num_classes, in_channels, enable_flash: bool = False, head_dropout: float = 0.5, drop_path: float = 0.0):
        super().__init__()
        self.backbone = PointTransformerV3(in_channels=in_channels, enable_flash=enable_flash, drop_path=drop_path)
        self.cls = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, data_dict):
        point = self.backbone(data_dict)
        logits = self.cls(point.feat)
        return logits

def train(train_loader, model, criterion, optimizer, classes, current_epoch, epochs, log_txt_path, seg_label_to_cat):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = epochs * len(train_loader)
    
    for i, (grid_coord, coord, feat, target, offset) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        data_time.update(time.time() - end)
    
        grid_coord, coord, feat, target, offset = (
            grid_coord.cuda(non_blocking=True),
            coord.cuda(non_blocking=True), 
            feat.cuda(non_blocking=True), 
            target.cuda(non_blocking=True), 
            offset.cuda(non_blocking=True)
        )
        
        data_dict = {
            'coord': coord,
            'grid_coord': grid_coord,
            'feat': feat.float(),
            'offset': offset 
        }
        
        if target.shape[-1] == 1:
            target = target[:, 0]
            
        # Check for invalid targets and log warnings
        tmin, tmax = target.min().item(), target.max().item()
        if tmin < 0 or tmax >= classes:
            with open(log_txt_path, "a") as f:
                f.write(f"[WARN] Invalid target in TRAIN batch {i}: min={tmin}, max={tmax}, "
                        f"fixed to ignore_index=255\n")
            target[(target < 0) | (target >= classes)] = 255
            
        output = model(data_dict)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        n = grid_coord.size(0)
        intersection, union, target = intersectionAndUnionGPU(output, target, classes, ignore_label=255)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = current_epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    valid_mask = union_meter.sum > 0
    mIoU = np.mean(iou_class[valid_mask]) if valid_mask.any() else 0.0
    mAcc = np.mean(accuracy_class[valid_mask]) if valid_mask.any() else 0.0
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    print('Train result at epoch [{}/{}]: loss/mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
        current_epoch+1, epochs, loss_meter.avg, mIoU, mAcc, allAcc))
    for i in range(classes):
        print('{:<20}Result: iou/accuracy {:.4f}/{:.4f}.'.format(seg_label_to_cat[i], iou_class[i], accuracy_class[i]))
    return loss_meter.avg, mIoU, mAcc, allAcc

def validate(val_loader, model, criterion, classes, ignore_label, log_txt_path, seg_label_to_cat):
    """Validation loop with invalid target checking"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    
    with torch.no_grad():
        for i, (grid_coord, coord, feat, target, offset) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
            data_time.update(time.time() - end)
            
            grid_coord, coord, feat, target, offset = (
                grid_coord.cuda(non_blocking=True),
                coord.cuda(non_blocking=True),
                feat.cuda(non_blocking=True),
                target.cuda(non_blocking=True),
                offset.cuda(non_blocking=True),
            )
            
            data_dict = {
                'coord': coord,
                'grid_coord': grid_coord,
                'feat': feat.float(),
                'offset': offset 
            }
            
            if target.shape[-1] == 1:
                target = target[:, 0]

            tmin, tmax = target.min().item(), target.max().item()
            if tmin < 0 or tmax >= classes:
                with open(log_txt_path, "a") as f:
                    f.write(f"[WARN] Invalid target in VALID batch {i}: min={tmin}, max={tmax}, "
                            f"fixed to ignore_index={ignore_label}\n")
                target[(target < 0) | (target >= classes)] = ignore_label

            output = model(data_dict)
            loss = criterion(output, target)
            output = output.max(1)[1]
            n = grid_coord.size(0)

            intersection, union, target = intersectionAndUnionGPU(output, target, classes, ignore_label)
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )

            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), n)
            batch_time.update(time.time() - end)
            end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    valid_mask = union_meter.sum > 0
    mIoU = np.mean(iou_class[valid_mask]) if valid_mask.any() else 0.0
    mAcc = np.mean(accuracy_class[valid_mask]) if valid_mask.any() else 0.0
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print('Val result: loss/mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(loss_meter.avg, mIoU, mAcc, allAcc))
    for i in range(classes):
        print('{:<20}Result: iou/accuracy {:.4f}/{:.4f}.'.format(seg_label_to_cat[i], iou_class[i], accuracy_class[i]))

    return loss_meter.avg, mIoU, mAcc, allAcc, iou_class, accuracy_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Dataset root. Either the parent datav2_final dir or a specific datav2_XXX_octree_fps dir containing train/ val/ test/")
    parser.add_argument("--workers", type=int, default=None, help="Number of DataLoader workers (overrides config.workers)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config.batch_size)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config.base_lr)")
    parser.add_argument("--enable_flash", action="store_true", help="Enable FlashAttention in backbone (requires flash_attn installed)")
    parser.add_argument("--label_smoothing", "-label_smoothing", "-ls", type=float, default=None, help="CrossEntropy label smoothing (overrides config.label_smoothing)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (overrides config.weight_decay)")
    parser.add_argument("--dropout", type=float, default=None, help="Head dropout (overrides config.head_dropout)")
    parser.add_argument("--voxel_size", type=float, default=None, help="Voxel size for grid (overrides config.voxel_size)")
    parser.add_argument("--drop_path", type=float, default=None, help="Stochastic depth drop_path for PTv3 (overrides config.drop_path)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config.epochs)")
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.data_path:
        cfg_kwargs["base_data_root"] = args.data_path
    if args.workers is not None:
        cfg_kwargs["workers"] = args.workers
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg_kwargs["base_lr"] = args.lr
    if args.label_smoothing is not None:
        cfg_kwargs["label_smoothing"] = args.label_smoothing
    if args.weight_decay is not None:
        cfg_kwargs["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        cfg_kwargs["head_dropout"] = args.dropout
    if args.voxel_size is not None:
        cfg_kwargs["voxel_size"] = args.voxel_size
    if args.drop_path is not None:
        cfg_kwargs["drop_path"] = args.drop_path
    if args.epochs is not None:
        cfg_kwargs["epochs"] = args.epochs

    cfg = get_config(**cfg_kwargs)
    set_random_seed(cfg.manual_seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_start_time = time.time()

    # prepare log file
    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    log_txt_path = os.path.join(cfg.save_path, "model", "training_log.txt")
    with open(log_txt_path, "w") as f:
        f.write("===== Training Configuration =====\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {cfg.epochs}\n")
        f.write(f"Batch size: {cfg.batch_size}\n")
        f.write(f"npoints_batch: {cfg.npoints_batch}\n")
        f.write(f"Learning rate: {cfg.base_lr}\n")
        f.write(f"Momentum: {cfg.momentum}\n")
        f.write(f"Weight decay: {cfg.weight_decay}\n")
        f.write(f"Seed: {cfg.manual_seed}\n")
        f.write(f"Classes: {cfg.class_names}\n")
        f.write("==================================\n\n")

    print("=" * 60)
    print(f"[INFO] Using device: {device}")
    print("=" * 60)
    print("[INFO] Loading data ...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(cfg)
    
    if cfg.fast_debug:
        cfg.epochs = 2
        train_iter = iter(train_loader)
        train_loader = [next(train_iter) for _ in range(2)]
        val_iter = iter(val_loader)
        val_loader = [next(val_iter) for _ in range(1)]
        print("[DEBUG] Fast debug mode enabled: 2 train batches, 1 val batch, 2 epochs")

    total_steps = cfg.epochs * len(train_loader)
    print(f"[INFO] Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    class_names = cfg.class_names
    num_classes = len(class_names)
    # Ensure num_classes consistent after label merging
    if getattr(cfg, 'merge_label_from', None) is not None:
        with open(log_txt_path, "a") as f:
            f.write(f"Merging label {cfg.merge_label_from} -> {cfg.merge_label_to}\n")
    seg_label_to_cat = {i: cat for i, cat in enumerate(class_names)}
    print(f"[INFO] Number of classes: {num_classes}")
    print("[INFO] Building model ...")

    # Use new PTv3WithHead model
    model = PTv3WithHead(
        in_channels=cfg.feature_dim,
        num_classes=num_classes,
        enable_flash=args.enable_flash or getattr(cfg, 'enable_flash', False),
        head_dropout=cfg.head_dropout,
        drop_path=getattr(cfg, 'drop_path', 0.0),
    )
    
    model = torch.nn.DataParallel(model).cuda()
    print("[INFO] Model built successfully")
    print("[INFO] Setting up loss, optimizer, and scheduler ...")
    weight_tensor = None
    if getattr(cfg, 'use_class_weights', False) and getattr(cfg, 'class_weights', None) is not None:
        weight_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label, weight=weight_tensor, label_smoothing=getattr(cfg, 'label_smoothing', 0.0)).cuda()
    
    # AdamW optimizer with param groups (no decay on norm/bias)
    def get_param_groups(module: nn.Module):
        decay_params, no_decay_params = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            is_bias = name.endswith('.bias')
            is_norm = ('norm' in name.lower()) or isinstance(param, torch.nn.Parameter) and (param.dim() == 1)
            if is_bias or is_norm:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(get_param_groups(model), lr=cfg.base_lr, betas=(0.9, 0.999))
    
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(cfg.epochs * 0.6), int(cfg.epochs * 0.8)],
        gamma=cfg.gamma,
    )
    print("[INFO] Training setup complete")

    best_iou = 0
    best_results = None

    print("=" * 60)
    print(f"[INFO] Starting training for {cfg.epochs} epochs ...")
    print("=" * 60)

    for epoch in range(cfg.epochs):
        # Use new train function
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader, model, criterion, optimizer, num_classes, epoch, cfg.epochs, log_txt_path, seg_label_to_cat
        )
        scheduler.step()

        # Validation
        if cfg.evaluate and (epoch + 1) % cfg.eval_freq == 0:
            loss_val, mIoU_val, mAcc_val, allAcc_val, val_iou_class, val_acc_class = validate(
                val_loader, model, criterion, num_classes, cfg.ignore_label, log_txt_path, seg_label_to_cat
            )

            is_best = mIoU_val > best_iou
            if is_best:
                best_iou = mIoU_val
                best_results = (epoch + 1, mIoU_val, mAcc_val, allAcc_val, val_iou_class, val_acc_class)

            if (epoch + 1) % cfg.save_freq == 0:
                filename = os.path.join(cfg.save_path, "model", "model_last.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_iou": best_iou,
                        "is_best": is_best,
                    },
                    filename,
                )
                if is_best:
                    shutil.copyfile(filename, os.path.join(cfg.save_path, "model", "model_best.pth"))
                    print('Best validation mIoU updated to: {:.4f}'.format(best_iou))

            if (epoch + 1) % 1 == 0:
                with open(log_txt_path, "a") as f:
                    f.write(f"Train result at epoch [{epoch+1}/{cfg.epochs}]: "
                            f"mIoU/mAcc/allAcc {mIoU_train:.4f}/{mAcc_train:.4f}/{allAcc_train:.4f}\n")
                    f.write(f"Val result: mIoU/mAcc/allAcc {mIoU_val:.4f}/{mAcc_val:.4f}/{allAcc_val:.4f}\n")
                    for i in range(num_classes):
                        f.write(f"{seg_label_to_cat[i]:<20} Result: iou/accuracy {val_iou_class[i]:.4f}/{val_acc_class[i]:.4f}\n")
                    f.write(f"Best validation mIoU so far: {best_iou:.4f}\n\n")

    print("=" * 60)
    if best_results:
        with open(log_txt_path, "a") as f:
            f.write("===== Best Model Results =====\n")
            f.write(f"Best Epoch: {best_results[0]}\n")
            f.write(f"mIoU/mAcc/allAcc: {best_results[1]:.4f}/{best_results[2]:.4f}/{best_results[3]:.4f}\n")
            for i in range(num_classes):
                f.write(f"{seg_label_to_cat[i]:<20} Result: iou/accuracy {best_results[4][i]:.4f}/{best_results[5][i]:.4f}\n")
            f.write("==================================\n\n")
        print(f"[INFO] Best result at epoch {best_results[0]}: "
              f"mIoU={best_results[1]:.4f}, mAcc={best_results[2]:.4f}, allAcc={best_results[3]:.4f}")
    print("[INFO] Training finished.")
    print("=" * 60)

    train_time = time.time() - train_start_time
    if best_results:
        print("[INFO] Running final test on best model ...")
        test_start_time = time.time()
        run_test(cfg)
        test_time = time.time() - test_start_time
        print("[INFO] Test finished.")
    else:
        test_start_time = time.time()
        run_test(cfg, checkpoint_path=os.path.join(cfg.save_path, "model", "model_last.pth"))
        test_time = time.time() - test_start_time
        print("[WARN] No best_results found, test with last model.")

    total_time = train_time + test_time

    def format_time(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"

    print(f"[INFO] Training time: {format_time(train_time)}")
    print(f"[INFO] Testing time: {format_time(test_time)}")
    print(f"[INFO] Total time: {format_time(total_time)}")

    with open(log_txt_path, "a") as f:
        f.write(f"Training time: {format_time(train_time)}\n")
        f.write(f"Testing time: {format_time(test_time)}\n")
        f.write(f"Total time: {format_time(total_time)}\n")


if __name__ == "__main__":
    main()