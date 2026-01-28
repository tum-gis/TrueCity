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
from model import Model, AverageMeter, intersectionAndUnionGPU
from test import run_test

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

def validate(val_loader, model, criterion, classes, ignore_label, log_txt_path):
    """Validation loop with invalid target checking"""
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (points, feat, target, offset) in enumerate(val_loader):
            points, feat, target, offset = (
                points.cuda(non_blocking=True),
                feat.cuda(non_blocking=True),
                target.cuda(non_blocking=True),
                offset.cuda(non_blocking=True),
            )
            if target.shape[-1] == 1:
                target = target[:, 0]

            tmin, tmax = target.min().item(), target.max().item()
            if tmin < 0 or tmax >= classes:
                # print(f"[WARN] Invalid target detected in VALID batch {i}: min={tmin}, max={tmax}, "
                #       f"expected 0~{classes-1}. Replacing with ignore_index={ignore_label}")
                with open(log_txt_path, "a") as f:
                    f.write(f"[WARN] Invalid target in VALID batch {i}: min={tmin}, max={tmax}, "
                            f"fixed to ignore_index={ignore_label}\n")
                target[(target < 0) | (target >= classes)] = ignore_label

            output = model([points, feat, offset])
            loss = criterion(output, target)
            output = output.max(1)[1]

            intersection, union, target = intersectionAndUnionGPU(output, target, classes, ignore_label)
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )

            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            n = points.size(0)
            loss_meter.update(loss.item(), n)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return loss_meter.avg, mIoU, mAcc, allAcc, iou_class, accuracy_class


def main():
    cfg = get_config()
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
    seg_label_to_cat = {i: cat for i, cat in enumerate(class_names)}
    print(f"[INFO] Number of classes: {num_classes}")
    print("[INFO] Building model ...")

    if cfg.model_name == "point_transformer_v1":
        from model import Model
        model = Model(c=cfg.feature_dim, k=num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {cfg.model_name}")

    model = torch.nn.DataParallel(model).cuda()
    print("[INFO] Model built successfully")
    print("[INFO] Setting up loss, optimizer, and scheduler ...")
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.base_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
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

    step = 0
    with tqdm(total=total_steps, desc="Training Progress", unit="batch") as pbar:
        for epoch in range(cfg.epochs):
            model.train()
            loss_meter = AverageMeter()
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()

            for i, (points, feat, target, offset) in enumerate(train_loader):
                points, feat, target, offset = (
                    points.cuda(non_blocking=True),
                    feat.cuda(non_blocking=True),
                    target.cuda(non_blocking=True),
                    offset.cuda(non_blocking=True),
                )
                if target.shape[-1] == 1:
                    target = target[:, 0]

                tmin, tmax = target.min().item(), target.max().item()
                if tmin < 0 or tmax >= num_classes:
                    # print(f"[WARN] Invalid target detected in TRAIN batch {i}: min={tmin}, max={tmax}, "
                    #       f"expected 0~{num_classes-1}. Replacing with ignore_index={cfg.ignore_label}")
                    with open(log_txt_path, "a") as f:
                        f.write(f"[WARN] Invalid target in TRAIN batch {i}: min={tmin}, max={tmax}, "
                                f"fixed to ignore_index={cfg.ignore_label}\n")
                    target[(target < 0) | (target >= num_classes)] = cfg.ignore_label

                output = model([points, feat, offset])
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = output.max(1)[1]
                n = points.size(0)
                intersection, union, target = intersectionAndUnionGPU(output, target, num_classes, cfg.ignore_label)
                intersection, union, target = (
                    intersection.cpu().numpy(),
                    union.cpu().numpy(),
                    target.cpu().numpy(),
                )
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)
                loss_meter.update(loss.item(), n)

                step += 1
                pbar.update(1)

            scheduler.step()

            # Train results for the epoch
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)

            valid_mask = union_meter.sum > 0
            if valid_mask.any():
                mIoU_train = np.mean(iou_class[valid_mask])
                mAcc_train = np.mean((intersection_meter.sum / (target_meter.sum + 1e-10))[valid_mask])
            else:
                mIoU_train = 0.0
                mAcc_train = 0.0

            allAcc_train = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            # Validation
            if cfg.evaluate and (epoch + 1) % cfg.eval_freq == 0:
                loss_val, mIoU_val, mAcc_val, allAcc_val, val_iou_class, val_acc_class = validate(
                    val_loader, model, criterion, num_classes, cfg.ignore_label, log_txt_path
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

                if (epoch + 1) % 1 == 0:
                    with open(log_txt_path, "a") as f:
                        f.write(f"Note: mIoU and mAcc computed with valid_mask (only classes present in data)\n")
                        f.write(f"Valid classes this epoch: {np.where(valid_mask)[0].tolist()}\n")
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
