import os
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import argparse
from datetime import datetime

from ocnn.octree import Octree
from ocnn.octree.points import Points

from point_transformer.config import get_config  # reuse same config
from point_transformer.data_utils import create_dataloaders
from octreeformer.model import Model, AverageMeter


def get_logger():
    logger_name = "octreeformer-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def normalize_xyz_unit_sphere(xyz: torch.Tensor) -> torch.Tensor:
    center = xyz.mean(dim=0, keepdim=True)
    xyz = xyz - center
    scale = torch.norm(xyz, dim=1).max().clamp(min=1e-6)
    xyz = xyz / scale
    return xyz * 0.99


def build_octree(points: torch.Tensor, features: torch.Tensor, depth: int = 8, full_depth: int = 2,
                 batch_ids: torch.Tensor = None, query_depth: int | None = None):
    # Normalize xyz to [-1,1]
    xyz = normalize_xyz_unit_sphere(points[:, :3])
    batch_sz = int(batch_ids.max().item()) + 1 if batch_ids is not None and batch_ids.numel() > 0 else 1
    pts = Points(points=xyz, normals=None, features=features, labels=None,
                 batch_id=batch_ids, batch_size=batch_sz)
    octree = Octree(depth=depth, full_depth=full_depth, batch_size=pts.batch_size, device=points.device)
    octree.build_octree(pts)
    # Neighbors for backbone convs
    octree.construct_all_neigh()
    # Per-point queries: normalized xyz in [-1,1] with batch id in last column
    bcol = batch_ids.view(-1, 1).to(xyz.dtype) if batch_ids is not None else torch.zeros((xyz.size(0), 1), dtype=xyz.dtype, device=xyz.device)
    query_pts = torch.cat([xyz, bcol], dim=1)
    return octree, depth, query_pts


def run_validate(val_loader, model, criterion, classes, ignore_label, log_txt_path):
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (points, feat, target, offset) in enumerate(val_loader):
            points = points.cuda(non_blocking=True)
            feat = feat.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)

            if target.shape[-1] == 1:
                target = target[:, 0]

            # Build batch ids matching ocnn Points API; offsets are cumulative counts
            batch_ids = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
            prev = 0
            for b, cur in enumerate(offset):
                batch_ids[prev:cur] = b
                prev = cur

            # Compute query depth: depth_in - stem_down + head_up (cap to octree.depth)
            stem_down = getattr(model.backbone, 'stem_down', 2)
            head_up = getattr(model.head, 'num_up', 2)
            query_depth = 8 - stem_down + head_up
            octree, depth, query_xyzb = build_octree(points, feat, depth=8, full_depth=2, batch_ids=batch_ids, query_depth=query_depth)
            data_init = octree.get_input_feature('PL', nempty=True)
            output = model(data_init, octree, depth, query_xyzb)
            loss = criterion(output, target)
            pred = output.max(1)[1]

            intersection, union, tgt = intersectionAndUnionGPU(pred, target, classes, ignore_label)
            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            target_meter.update(tgt.cpu().numpy())

            n = points.size(0)
            loss_meter.update(loss.item(), n)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return loss_meter.avg, mIoU, mAcc, allAcc, iou_class, accuracy_class


def run_test(test_loader, model, criterion, classes, ignore_label, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_txt_path = os.path.join(save_dir, "test_log.txt")

    loss, mIoU, mAcc, allAcc, iou_class, acc_class = run_validate(
        test_loader, model, criterion, classes, ignore_label, log_txt_path
    )

    with open(log_txt_path, "w") as f:
        f.write('Test result: loss/mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.\n'.format(
            loss, mIoU, mAcc, allAcc))
        for ci in range(classes):
            f.write('{:<20}Result: iou/accuracy {:.4f}/{:.4f}.\n'.format(
                str(ci), iou_class[ci], acc_class[ci]))

    print('Test result: loss/mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
        loss, mIoU, mAcc, allAcc))
    return loss, mIoU, mAcc, allAcc


def get_param_groups(module: nn.Module, weight_decay: float):
    decay_params, no_decay_params = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        is_bias = name.endswith('.bias')
        is_norm = ('norm' in name.lower()) or (param.dim() == 1)
        if is_bias or is_norm:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Dataset root. Either the parent datav2_final dir or a specific datav2_XXX_octree_fps dir containing train/ val/ test/")
    parser.add_argument("--workers", type=int, default=None, help="Number of DataLoader workers (compat flag; not used by current loaders)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config.batch_size)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config.base_lr)")
    parser.add_argument("--enable_flash", action="store_true", help="Compat flag; ignored for OctreeFormer")
    parser.add_argument("--label_smoothing", "-label_smoothing", "-ls", type=float, default=None, help="CrossEntropy label smoothing")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (overrides config.weight_decay)")
    parser.add_argument("--dropout", type=float, default=None, help="Head dropout for classifier (maps to head_drop)")
    parser.add_argument("--voxel_size", type=float, default=None, help="Compat flag; ignored here")
    parser.add_argument("--drop_path", type=float, default=None, help="Stochastic depth drop_path for backbone")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config.epochs)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps to simulate larger effective batch size")
    parser.add_argument("--max_points_per_item", type=int, default=None, help="Optional downsampling cap per item (e.g., 250000) for very large files")
    parser.add_argument("--no_dwconv", action="store_true", help="Disable CUDA dwconv extension; use OctreeGroupConv instead")
    parser.add_argument("--use_dwconv", action="store_true", help="Enable CUDA dwconv extension explicitly if available")
    parser.add_argument("--save_root", type=str, default="/home/stud/nguyenti/storage/user/tum-di-lab/results", help="Root directory to store training runs")
    parser.add_argument("--run_name", type=str, default=None, help="Optional custom run directory name; defaults to octformer_<data>_<timestamp>")
    parser.add_argument("--test_only", action="store_true", help="Only run evaluation on the test set using the best checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to evaluate (defaults to <run_dir>/model/model_best.pth)")
    parser.add_argument("--run_dir", type=str, default=None, help="Existing run directory to load checkpoint from (overrides auto timestamp save_path)")
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.data_path:
        cfg_kwargs["base_data_root"] = args.data_path
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg_kwargs["base_lr"] = args.lr
    if args.weight_decay is not None:
        cfg_kwargs["weight_decay"] = args.weight_decay
    if args.epochs is not None:
        cfg_kwargs["epochs"] = args.epochs

    cfg = get_config(**cfg_kwargs)

    # If data_path is a prepared root with train/val/test, override preprocessed roots
    if args.data_path and os.path.isdir(args.data_path):
        train_dir = os.path.join(args.data_path, 'train')
        val_dir = os.path.join(args.data_path, 'val')
        test_dir = os.path.join(args.data_path, 'test')
        if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
            cfg.use_preprocessed = True
            cfg.train_data_root = train_dir
            cfg.val_data_root = val_dir
            cfg.test_data_root = test_dir

    # Override save path to results/<run_name>
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_tag = os.path.basename(args.data_path.rstrip('/')) if args.data_path else 'data'
    auto_name = f"octformer_{base_tag}_{ts}"
    run_name = args.run_name if args.run_name else auto_name
    cfg.save_path = os.path.join(args.save_root, run_name)
    if args.run_dir is not None:
        cfg.save_path = args.run_dir
    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_path, 'model'), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log_txt_path = os.path.join(cfg.save_path, "model", "training_log.txt")
    with open(log_txt_path, "w") as f:
        f.write("===== Training Configuration =====\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {cfg.epochs}\n")
        f.write(f"Batch size: {cfg.batch_size}\n")
        f.write(f"Learning rate: {cfg.base_lr}\n")
        f.write(f"Weight decay: {cfg.weight_decay}\n")
        f.write(f"Seed: {cfg.manual_seed}\n")
        f.write(f"Classes: {cfg.class_names}\n")
        f.write("==================================\n\n")

    print("=" * 60)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Save path: {cfg.save_path}")
    print("=" * 60)
    print("[INFO] Loading data ...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(cfg)
    total_steps = cfg.epochs * len(train_loader)
    print(f"[INFO] Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    class_names = cfg.class_names
    num_classes = len(class_names)
    seg_label_to_cat = {i: cat for i, cat in enumerate(class_names)}
    print(f"[INFO] Number of classes: {num_classes}")
    print("[INFO] Building model ...")

    # Map CLI dropout/drop_path into model kwargs
    model_kwargs = {}
    if args.drop_path is not None:
        model_kwargs["drop_path"] = args.drop_path
    if args.dropout is not None:
        model_kwargs["head_drop"] = [args.dropout, args.dropout]
    # Allow disabling dwconv (CUDA extension) to avoid PTX/toolchain issues
    if getattr(args, "no_dwconv", False):
        model_kwargs["use_dwconv"] = False
    if getattr(args, "use_dwconv", False):
        model_kwargs["use_dwconv"] = True

    # Use PL node features -> 6 input channels
    feature_key = 'PL'
    in_ch = 3 + (3 if 'L' in feature_key else 0) + (1 if 'D' in feature_key else 0)
    model = Model(c=in_ch, k=num_classes, **model_kwargs).cuda()
    print("[INFO] Model built successfully")
    print("[INFO] Setting up loss, optimizer, and scheduler ...")
    # criterion with optional class weights and label smoothing
    weight_tensor = None
    if getattr(cfg, 'use_class_weights', False) and getattr(cfg, 'class_weights', None) is not None:
        weight_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label, weight=weight_tensor,
                                    label_smoothing=(args.label_smoothing if args.label_smoothing is not None else 0.0)).cuda()

    # TEST-ONLY MODE
    if args.test_only:
        ckpt_path = args.checkpoint if args.checkpoint is not None else os.path.join(cfg.save_path, "model", "model_best.pth")
        print(f"[INFO] Loading checkpoint for test: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        run_test(test_loader, model, criterion, num_classes, cfg.ignore_label, os.path.join(cfg.save_path, "model"))
        return

    # AdamW optimizer with param groups
    optimizer = torch.optim.AdamW(get_param_groups(model, cfg.weight_decay), lr=cfg.base_lr, betas=(0.9, 0.999))

    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(cfg.epochs * 0.6), int(cfg.epochs * 0.8)], gamma=cfg.gamma
    )
    print("[INFO] Training setup complete")

    best_iou = 0
    best_results = None

    step = 0
    print("=" * 60)
    print(f"[INFO] Starting training for {cfg.epochs} epochs ...")
    print("=" * 60)
    for epoch in range(cfg.epochs):
        model.train()
        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.epochs}", unit="batch") as pbar:
            for i, (points, feat, target, offset) in enumerate(train_loader):
                points = points.cuda(non_blocking=True)
                feat = feat.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                offset = offset.cuda(non_blocking=True)
                if target.shape[-1] == 1:
                    target = target[:, 0]

                # Prepare batch ids for ocnn Points
                batch_ids = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
                prev = 0
                for b, cur in enumerate(offset):
                    batch_ids[prev:cur] = b
                    prev = cur

                # Optional per-item downsampling for very large files
                if args.max_points_per_item is not None:
                    new_points = []
                    new_feat = []
                    new_target = []
                    new_batch_ids = []
                    prev = 0
                    for b, cur in enumerate(offset):
                        cnt = int((cur - prev).item())
                        if cnt > args.max_points_per_item:
                            idx = torch.randperm(cnt, device=points.device)[:args.max_points_per_item]
                        else:
                            idx = torch.arange(cnt, device=points.device)
                        sel = (prev + idx)
                        new_points.append(points[sel])
                        new_feat.append(feat[sel])
                        new_target.append(target[sel])
                        new_batch_ids.append(torch.full((idx.numel(),), b, dtype=torch.long, device=points.device))
                        prev = cur
                    points = torch.cat(new_points, dim=0)
                    feat = torch.cat(new_feat, dim=0)
                    target = torch.cat(new_target, dim=0)
                    batch_ids = torch.cat(new_batch_ids, dim=0)
                    # rebuild offset
                    counts = torch.bincount(batch_ids, minlength=int(batch_ids.max().item())+1)
                    offset = torch.cumsum(counts, dim=0).to(torch.int32)

                stem_down = getattr(model.backbone, 'stem_down', 2)
                head_up = getattr(model.head, 'num_up', 2)
                query_depth = 8 - stem_down + head_up
                octree, depth, query_xyzb = build_octree(points, feat, depth=8, full_depth=2, batch_ids=batch_ids, query_depth=query_depth)
                output = model(octree.get_input_feature('PL', nempty=True), octree, depth, query_xyzb)
                loss = criterion(output, target)

                # Diagnostics (only for first batch per epoch)
                if i == 0:
                    work_depth = min(depth, query_depth)
                    n_nodes = int(octree.nnum_nempty[work_depth].item()) if work_depth <= octree.depth else -1
                    diag_msg = f"[DIAG] raw_points={points.size(0)}, query_pts={query_xyzb.size(0)}, work_depth={work_depth}, nodes={n_nodes}"
                    print(diag_msg)
                    with open(log_txt_path, "a") as f:
                        f.write(diag_msg + "\n")

                # Gradient accumulation
                loss = loss / max(1, args.accum_steps)
                loss.backward()
                if (i + 1) % max(1, args.accum_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pred = output.max(1)[1]
                n = points.size(0)
                intersection, union, tgt = intersectionAndUnionGPU(pred, target, num_classes, cfg.ignore_label)
                intersection_meter.update(intersection.cpu().numpy())
                union_meter.update(union.cpu().numpy())
                target_meter.update(tgt.cpu().numpy())
                loss_meter.update(loss.item(), n)

                step += 1
                pbar.update(1)

        scheduler.step()

        # epoch metrics
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        valid_mask = union_meter.sum > 0
        mIoU_train = np.mean(iou_class[valid_mask]) if valid_mask.any() else 0.0
        mAcc_train = np.mean((intersection_meter.sum / (target_meter.sum + 1e-10))[valid_mask]) if valid_mask.any() else 0.0
        allAcc_train = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        # Console print for train epoch summary
        print('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            epoch + 1, cfg.epochs, mIoU_train, mAcc_train, allAcc_train))
        for ci in range(num_classes):
            print('{:<20}Result: iou/accuracy {:.4f}/{:.4f}.'.format(seg_label_to_cat[ci], iou_class[ci],
                                                                      (intersection_meter.sum / (target_meter.sum + 1e-10))[ci]))

        if cfg.evaluate and (epoch + 1) % cfg.eval_freq == 0:
            loss_val, mIoU_val, mAcc_val, allAcc_val, val_iou_class, val_acc_class = run_validate(
                val_loader, model, criterion, num_classes, cfg.ignore_label, log_txt_path
            )

            # Diagnostics for first val batch
            if i == 0:
                work_depth = min(depth, query_depth)
                n_nodes = int(octree.nnum_nempty[work_depth].item()) if work_depth <= octree.depth else -1
                diag_msg = f"[DIAG-VAL] raw_points={points.size(0)}, query_pts={query_xyzb.size(0)}, work_depth={work_depth}, nodes={n_nodes}"
                print(diag_msg)
                with open(log_txt_path, "a") as f:
                    f.write(diag_msg + "\n")

            # Console print for validation epoch summary
            print('Val result: loss/mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                loss_val, mIoU_val, mAcc_val, allAcc_val))
            for ci in range(num_classes):
                print('{:<20}Result: iou/accuracy {:.4f}/{:.4f}.'.format(seg_label_to_cat[ci], val_iou_class[ci], val_acc_class[ci]))

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

            # Append per-epoch summary to log
            with open(log_txt_path, "a") as f:
                f.write(f"Train result at epoch [{epoch+1}/{cfg.epochs}]: "
                        f"mIoU/mAcc/allAcc {mIoU_train:.4f}/{mAcc_train:.4f}/{allAcc_train:.4f}\n")
                f.write(f"Val result: mIoU/mAcc/allAcc {mIoU_val:.4f}/{mAcc_val:.4f}/{allAcc_val:.4f}\n")
                for ci in range(num_classes):
                    f.write(f"{seg_label_to_cat[ci]:<20} Result: iou/accuracy {val_iou_class[ci]:.4f}/{val_acc_class[ci]:.4f}\n")
                f.write(f"Best validation mIoU so far: {best_iou:.4f}\n\n")

    print("=" * 60)
    if best_results:
        print(f"[INFO] Best result at epoch {best_results[0]}: mIoU={best_results[1]:.4f}, mAcc={best_results[2]:.4f}, allAcc={best_results[3]:.4f}")
    print("[INFO] Training finished.")
    print("=" * 60)


if __name__ == "__main__":
    main() 