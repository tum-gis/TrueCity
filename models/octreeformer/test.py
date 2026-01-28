import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from importlib.util import find_spec

# Ensure current directory is on sys.path when running via absolute path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from point_transformer.config import get_config
from point_transformer.data_utils import create_dataloaders
from model import Model, AverageMeter
from train import intersectionAndUnionGPU, build_octree

HAS_SCIPY = find_spec('scipy') is not None
if HAS_SCIPY:
    from scipy.spatial import cKDTree


def normalize_center_scale_np(xyz: np.ndarray) -> np.ndarray:
    center = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - center
    norms = np.linalg.norm(xyz, axis=1)
    max_norm = np.max(norms) if norms.size > 0 else 1.0
    max_norm = max(max_norm, 1e-6)
    xyz = xyz / max_norm
    return xyz * 0.99


def validate(loader, model, criterion, num_classes, ignore_label, class_names, log_file=None,
             fast_debug=False, test_blocks=None, export_path=None, start_sample_idx=0):
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    export_chunks = [] if export_path is not None else None

    sample_idx = start_sample_idx
    with torch.no_grad():
        for i, (points, feat, target, offset) in enumerate(tqdm(loader, desc="Testing", unit="batch")):
            points = points.cuda(non_blocking=True)
            feat = feat.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)

            if target.shape[-1] == 1:
                target = target[:, 0]

            # Build batch ids from cumulative offsets
            batch_ids = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
            prev = 0
            for b, cur in enumerate(offset):
                batch_ids[prev:cur] = b
                prev = cur

            # Build octree and query points (normalized xyz with batch id)
            octree, depth, query_xyzb = build_octree(points, feat, depth=8, full_depth=2, batch_ids=batch_ids, query_depth=None)
            data_init = octree.get_input_feature('PL', nempty=True)

            output = model(data_init, octree, depth, query_xyzb)
            loss = criterion(output, target)
            pred = output.max(1)[1]

            # metrics
            n = points.size(0)
            intersection, union, tgt = intersectionAndUnionGPU(pred, target, num_classes, ignore_label)
            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            target_meter.update(tgt.cpu().numpy())
            loss_meter.update(loss.item(), n)

            # Export: use original raw xyz from test_blocks and align with predictions
            if export_path is not None:
                bs = int(offset.shape[0])
                prev = 0
                for b in range(bs):
                    cur = int(offset[b].item())
                    pred_lbl = pred[prev:cur].detach().cpu().numpy().astype(np.int32)

                    if test_blocks is not None:
                        raw_block = test_blocks[sample_idx]
                        raw_xyz = raw_block[:, :3].astype(np.float32)
                        if raw_xyz.shape[0] == pred_lbl.shape[0]:
                            save_arr = np.concatenate([raw_xyz, pred_lbl[:, None].astype(np.float32)], axis=1)
                        else:
                            # Fallback: map raw points to nearest predicted point (normalized space)
                            pred_xyz = query_xyzb[prev:cur, :3].detach().cpu().numpy().astype(np.float32)
                            raw_norm = normalize_center_scale_np(raw_xyz)
                            if HAS_SCIPY:
                                nn_idx = cKDTree(pred_xyz).query(raw_norm, k=1)[1]
                                raw_pred = pred_lbl[nn_idx]
                            else:
                                raw_pred = np.empty((raw_norm.shape[0],), dtype=np.int32)
                                tile = 50000
                                for s in range(0, raw_norm.shape[0], tile):
                                    e = min(s + tile, raw_norm.shape[0])
                                    best_d2 = None
                                    best_idx = None
                                    for ps in range(0, pred_xyz.shape[0], tile):
                                        pe = min(ps + tile, pred_xyz.shape[0])
                                        d2 = ((raw_norm[s:e, None, :] - pred_xyz[None, ps:pe, :]) ** 2).sum(axis=2)
                                        min_idx = np.argmin(d2, axis=1)
                                        min_d2 = d2[np.arange(d2.shape[0]), min_idx]
                                        if best_d2 is None:
                                            best_d2 = min_d2
                                            best_idx = min_idx + ps
                                        else:
                                            mask = min_d2 < best_d2
                                            best_d2[mask] = min_d2[mask]
                                            best_idx[mask] = (min_idx + ps)[mask]
                                    raw_pred[s:e] = pred_lbl[best_idx]
                            save_arr = np.concatenate([raw_xyz, raw_pred[:, None].astype(np.float32)], axis=1)
                    else:
                        # No raw blocks provided; fallback to using current (likely normalized) points
                        src_xyz = points[prev:cur, :3].detach().cpu().numpy().astype(np.float32)
                        save_arr = np.concatenate([src_xyz, pred_lbl[:, None].astype(np.float32)], axis=1)

                    if export_chunks is not None:
                        export_chunks.append(save_arr)
                    prev = cur
                    sample_idx += 1

            if fast_debug and i >= 0:
                break

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    valid_mask = union_meter.sum > 0
    mIoU = np.mean(iou_class[valid_mask]) if valid_mask.any() else 0.0
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(accuracy_class[valid_mask]) if valid_mask.any() else 0.0
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if log_file:
        with open(log_file, "w") as f:
            f.write(f"Test Results: mIoU={mIoU:.4f}, mAcc={mAcc:.4f}, allAcc={allAcc:.4f}\n")
            for i, cls in enumerate(class_names):
                f.write(f"{cls:<20} IoU={iou_class[i]:.4f}, Acc={accuracy_class[i]:.4f}\n")

    print(f"[TEST] mIoU={mIoU:.4f}, mAcc={mAcc:.4f}, allAcc={allAcc:.4f}")
    for i, cls in enumerate(class_names):
        print(f"{cls:<20} IoU={iou_class[i]:.4f}, Acc={accuracy_class[i]:.4f}")

    if export_path is not None and export_chunks is not None and len(export_chunks) > 0:
        big_arr = np.vstack(export_chunks)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        np.save(export_path, big_arr)

    return loss_meter.avg, mIoU, mAcc, allAcc


def run_test(cfg, checkpoint_path=None, export=False, export_filename=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = cfg.class_names
    num_classes = len(class_names)

    # Data
    _, _, test_loader, test_blocks = create_dataloaders(cfg)

    if getattr(cfg, "fast_debug", False):
        print("[DEBUG] Running test in fast_debug mode (1 batch only)")
        from itertools import islice
        test_loader = list(islice(test_loader, 1))

    # Model
    feature_key = 'PL'
    in_ch = 3 + (3 if 'L' in feature_key else 0) + (1 if 'D' in feature_key else 0)
    model = Model(c=in_ch, k=num_classes).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.save_path, "model", "model_best.pth")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")

    log_file = os.path.join(cfg.save_path, "model", "test_log.txt")

    # Determine export path if requested
    export_path = None
    if export and checkpoint_path is not None:
        save_dir = os.path.dirname(checkpoint_path)
        run_folder = os.path.basename(os.path.dirname(save_dir))
        default_name = f"{run_folder}.npy"
        export_path = os.path.join(save_dir, export_filename or default_name)

    results = validate(
        test_loader, model, criterion,
        num_classes, cfg.ignore_label,
        class_names, log_file,
        fast_debug=getattr(cfg, "fast_debug", False),
        test_blocks=test_blocks, export_path=export_path, start_sample_idx=0
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="OctreeFormer Test-Only Runner with Export")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset root. Either the parent datav2_final dir or a specific datav2_XXX_octree_fps dir containing train/ val/ test/")
    parser.add_argument("--workers", type=int, default=None, help="Number of DataLoader workers (compat; handled in config/dataloaders if applicable)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for test loader (overrides config.batch_size)")
    parser.add_argument("--fast_debug", action="store_true", help="Limit to 1 batch for quick debug")
    parser.add_argument("--export", action="store_true", help="Export one big .npy with [X,Y,Z,pred] next to checkpoint")
    parser.add_argument("--export_filename", type=str, default=None, help="Export filename; default is the run folder name (e.g., octformer_...timestamp.npy)")
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.data_path:
        cfg_kwargs["base_data_root"] = args.data_path
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.fast_debug:
        cfg_kwargs["fast_debug"] = True

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

    run_test(cfg, checkpoint_path=args.checkpoint, export=args.export, export_filename=args.export_filename)


if __name__ == "__main__":
    main() 