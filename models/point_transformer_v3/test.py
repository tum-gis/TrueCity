import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import create_dataloaders
from model_v3 import AverageMeter, intersectionAndUnionGPU, PointTransformerV3
import argparse
from config import get_config
from augmentations import normalize_center_scale
from augmentations import default_augmentation
from data_utils import collate_fn as ptv3_collate_fn
from data_utils import load_preprocessed_data, default_transforms, _apply_label_mapping


class PTv3WithHead(nn.Module):
    def __init__(self, num_classes, in_channels, enable_flash: bool = False, head_dropout: float = 0.0, drop_path: float = 0.0):
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


def build_test_only_loader(cfg):
    """Create only the test dataloader and return (test_loader, test_blocks)."""
    if getattr(cfg, 'use_preprocessed', True):
        test_blocks = load_preprocessed_data(cfg.test_data_root)
        _apply_label_mapping(test_blocks, cfg.num_classes, getattr(cfg, 'merge_label_from', None), getattr(cfg, 'merge_label_to', None))
        trans = default_transforms(train=False, config=cfg)
        test_data = [trans(block.astype(np.float32)) for block in test_blocks]
    else:
        # Load raw test split (expects .npy blocks under cfg.test_data_root)
        test_blocks = []
        for f in os.listdir(cfg.test_data_root):
            if f.endswith('.npy'):
                arr = np.load(os.path.join(cfg.test_data_root, f)).astype(np.float32)
                test_blocks.append(arr)
        trans = default_transforms(train=False, config=cfg)
        test_data = [trans(block) for block in test_blocks]

    # DataLoader with same collate and workers
    dl = DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=ptv3_collate_fn,
        num_workers=getattr(cfg, 'workers', 0),
        pin_memory=True,
    )
    # Ensure voxel size override is propagated
    ptv3_collate_fn.voxel_size_override = getattr(cfg, 'voxel_size', None)
    return dl, test_blocks


def validate(loader, model, criterion, num_classes, ignore_label, class_names, log_file=None, fast_debug=False, test_blocks=None, export_path=None, start_sample_idx=0, cfg=None, apply_train_aug: bool = False, voxel_size: float = None):
    """Run evaluation on a dataloader

    If apply_train_aug is True, apply the same train-time normalization/augmentations to coords
    and recompute grid_coord per sample before feeding the model.
    """
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    pred_list = []

    sample_idx = start_sample_idx
    export_chunks = [] if export_path is not None else None
    with torch.no_grad():
        for i, (grid_coord, coord, feat, target, offset) in enumerate(tqdm(loader, desc="Testing", unit="batch")):
            grid_coord, coord, feat, target, offset = (
                grid_coord.cuda(non_blocking=True),
                coord.cuda(non_blocking=True),
                feat.cuda(non_blocking=True),
                target.cuda(non_blocking=True),
                offset.cuda(non_blocking=True),
            )

            if target.shape[-1] == 1:
                target = target[:, 0]

            # Optionally re-apply train-time augmentation/normalization and recompute grid coords per sample
            use_aug = bool(apply_train_aug)
            vx = float(voxel_size) if voxel_size is not None else float(getattr(cfg, 'voxel_size', 0.01) if cfg is not None else 0.01)
            if use_aug:
                # Build augmented per-sample coords/feats/grid_coord using the loader-provided normalized coords
                bs = int(offset.shape[0])
                prev = 0
                new_coords = []
                new_feats = []
                new_grid_coords = []
                # For export alignment when requested, we will also prepare augmented raw coords using identical RNG
                aug_raw_coords_per_sample = [] if (export_path is not None and test_blocks is not None) else None
                for b in range(bs):
                    cur = int(offset[b].item())
                    cur_coords = coord[prev:cur].detach().cpu().numpy().astype(np.float32)
                    # Save RNG state to reproduce the same augmentation for raw points later
                    rng_state = np.random.get_state()
                    aug_coords_np, _, _ = default_augmentation(cur_coords, cfg)
                    # Recompute grid coords using the same logic as collate_fn (per-sample shift)
                    aug_coords = torch.from_numpy(aug_coords_np).to(coord.device)
                    coord_min = aug_coords.min(dim=0).values
                    coord_shifted = aug_coords - coord_min
                    g = torch.floor(coord_shifted / vx).to(torch.int32)
                    new_coords.append(aug_coords)
                    new_feats.append(aug_coords.clone())
                    new_grid_coords.append(g)

                    if aug_raw_coords_per_sample is not None:
                        raw_block = test_blocks[sample_idx]
                        raw_xyz = raw_block[:, :3].astype(np.float32)
                        raw_norm = normalize_center_scale(raw_xyz)
                        # Restore RNG and apply identical augmentation to raw-normalized coords
                        np.random.set_state(rng_state)
                        raw_aug_np, _, _ = default_augmentation(raw_norm, cfg)
                        aug_raw_coords_per_sample.append(raw_aug_np.astype(np.float32))

                    prev = cur
                    sample_idx += 1

                # Concatenate back
                coord = torch.cat(new_coords, dim=0)
                feat = torch.cat(new_feats, dim=0)
                grid_coord = torch.cat(new_grid_coords, dim=0)
            else:
                aug_raw_coords_per_sample = None

            data_dict = {
                'coord': coord,
                'grid_coord': grid_coord,
                'feat': feat.float(),
                'offset': offset,
            }

            output = model(data_dict)
            loss = criterion(output, target)
            output = output.max(1)[1]

            # Optional evaluation-time label remapping only if enabled in cfg
            if bool(getattr(cfg, 'test_eval_merge', False)):
                from_ids = getattr(loader, 'eval_merge_label_from', globals().get('EVAL_MERGE_FROM', None))
                to_id = getattr(loader, 'eval_merge_label_to', globals().get('EVAL_MERGE_TO', None))
                if from_ids is not None and to_id is not None:
                    for src in from_ids:
                        output[output == int(src)] = int(to_id)
                        target[target == int(src)] = int(to_id)

            n = grid_coord.size(0)
            intersection, union, target = intersectionAndUnionGPU(output, target, num_classes, ignore_label)
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )

            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            loss_meter.update(loss.item(), n)

            pred_list.append(output.cpu().numpy())

            # Optional export: accumulate mapped predictions back to original block coordinates
            if export_path is not None and test_blocks is not None:
                bs = int(offset.shape[0])
                prev = 0
                local_sample_idx = sample_idx - bs
                for b in range(bs):
                    cur = int(offset[b].item())
                    pred_xyz = coord[prev:cur].detach().cpu().numpy().astype(np.float32)
                    pred_lbl = output[prev:cur].detach().cpu().numpy().astype(np.int32)

                    raw_block = test_blocks[local_sample_idx]
                    raw_xyz = raw_block[:, :3].astype(np.float32)

                    # Build the comparison coordinates in the same frame as pred_xyz
                    if use_aug and aug_raw_coords_per_sample is not None:
                        raw_cmp = aug_raw_coords_per_sample[b]
                    else:
                        raw_norm = normalize_center_scale(raw_xyz)
                        raw_cmp = raw_norm

                    # NN mapping from augmented/normalized raw to current preds
                    d2 = ((raw_cmp[:, None, :] - pred_xyz[None, :, :]) ** 2).sum(axis=2)
                    nn_idx = np.argmin(d2, axis=1)
                    raw_pred = pred_lbl[nn_idx]

                    save_arr = np.concatenate([raw_xyz, raw_pred[:, None].astype(np.float32)], axis=1)
                    if export_chunks is not None:
                        export_chunks.append(save_arr)

                    prev = cur
                    local_sample_idx += 1

            if fast_debug and i >= 0:
                break

    # metrics
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    valid_mask = union_meter.sum > 0
    mIoU = np.mean(iou_class[valid_mask]) if valid_mask.any() else 0.0
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(accuracy_class[valid_mask]) if valid_mask.any() else 0.0
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # log results
    if log_file:
        with open(log_file, "w") as f:
            f.write(f"Test Results: mIoU={mIoU:.4f}, mAcc={mAcc:.4f}, allAcc={allAcc:.4f}\n")
            for i, cls in enumerate(class_names):
                f.write(f"{cls:<20} IoU={iou_class[i]:.4f}, Acc={accuracy_class[i]:.4f}\n")

    print(f"[TEST] mIoU={mIoU:.4f}, mAcc={mAcc:.4f}, allAcc={allAcc:.4f}")
    # Print per-class to stdout
    for i, cls in enumerate(class_names):
        print(f"{cls:<20} IoU={iou_class[i]:.4f}, Acc={accuracy_class[i]:.4f}")

    # Save one big npy if requested
    if export_path is not None and export_chunks is not None and len(export_chunks) > 0:
        big_arr = np.vstack(export_chunks)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        np.save(export_path, big_arr)

    return loss_meter.avg, mIoU, mAcc, allAcc, iou_class, accuracy_class, np.hstack(pred_list), sample_idx


def run_test(cfg, checkpoint_path=None, export=False, export_filename=None):
    """Load best model and run testing"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = cfg.class_names
    num_classes = len(class_names)

    # Choose loader path: full dataloaders (train/val/test) or test-only loader
    use_full = bool(getattr(cfg, 'use_full_dataloaders_for_test', False))
    if use_full:
        _, _, test_loader, test_blocks = create_dataloaders(cfg)
        print("[INFO] Using create_dataloaders() for test (train/val/test built)")
    else:
        test_loader, test_blocks = build_test_only_loader(cfg)
        print("[INFO] Using test-only loader")

    if getattr(cfg, "fast_debug", False):
        print("[DEBUG] Running test in fast_debug mode (1 batch only)")
        from itertools import islice
        test_loader = list(islice(test_loader, 1))

    # Model
    model = PTv3WithHead(
        num_classes=num_classes,
        in_channels=cfg.feature_dim,
        enable_flash=getattr(cfg, 'enable_flash', False),
        head_dropout=getattr(cfg, 'head_dropout', 0.0),
        drop_path=getattr(cfg, 'drop_path', 0.0),
    )
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.save_path, "model", "model_best.pth")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")

    log_file = os.path.join(cfg.save_path, "model", "test_log.txt")

    # Optionally enable eval-time label merge for parity comparisons
    if bool(getattr(cfg, 'test_eval_merge', False)):
        for _ldr in [test_loader]:
            setattr(_ldr, 'eval_merge_label_from', getattr(cfg, 'eval_merge_label_from', None))
            setattr(_ldr, 'eval_merge_label_to', getattr(cfg, 'eval_merge_label_to', None))
        globals()['EVAL_MERGE_FROM'] = getattr(cfg, 'eval_merge_label_from', None)
        globals()['EVAL_MERGE_TO'] = getattr(cfg, 'eval_merge_label_to', None)

    # Determine export path if requested
    export_path = None
    if export and checkpoint_path is not None:
        save_dir = os.path.dirname(checkpoint_path)
        # Default filename: run folder name (parent of 'model')
        run_folder = os.path.basename(os.path.dirname(save_dir))
        default_name = f"{run_folder}.npy"
        export_path = os.path.join(save_dir, export_filename or default_name)

    # Whether to apply train-time augs during test (default False)
    test_apply_train_aug = bool(getattr(cfg, 'test_apply_train_aug', False))

    results = validate(
        test_loader, model, criterion,
        num_classes, cfg.ignore_label,
        class_names, log_file,
        fast_debug=getattr(cfg, "fast_debug", False),
        test_blocks=test_blocks, export_path=export_path, start_sample_idx=0,
        cfg=cfg, apply_train_aug=test_apply_train_aug, voxel_size=getattr(cfg, 'voxel_size', None)
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="PTv3 Test-Only Runner")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset root. Either the parent datav2_final dir or a specific datav2_XXX_octree_fps dir containing train/ val/ test/")
    parser.add_argument("--voxel_size", type=float, default=None, help="Voxel size for grid (overrides config.voxel_size)")
    parser.add_argument("--workers", type=int, default=None, help="Number of DataLoader workers (overrides config.workers)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for test loader (overrides config.batch_size)")
    parser.add_argument("--enable_flash", action="store_true", help="Enable FlashAttention in backbone (requires flash_attn installed)")
    parser.add_argument("--fast_debug", action="store_true", help="Limit to 1 batch for quick debug")
    parser.add_argument("--export", action="store_true", help="Export one big .npy with [X,Y,Z,pred] next to checkpoint")
    parser.add_argument("--export_filename", type=str, default=None, help="Export filename; default is the run folder name (e.g., ptv3_...timestamp.npy)")
    parser.add_argument("--apply_train_aug", action="store_true", help="Apply train-time augmentation/normalization at test time (off by default)")
    parser.add_argument("--use_full_dataloaders", action="store_true", help="Use create_dataloaders() (train/val/test) for test to match training pipeline")
    parser.add_argument("--eval_merge", action="store_true", help="Enable eval-time label merge (off by default to match validation)")
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.data_path:
        cfg_kwargs["base_data_root"] = args.data_path
    if args.voxel_size is not None:
        cfg_kwargs["voxel_size"] = args.voxel_size
    if args.workers is not None:
        cfg_kwargs["workers"] = args.workers
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.enable_flash:
        cfg_kwargs["enable_flash"] = True
    if args.fast_debug:
        cfg_kwargs["fast_debug"] = True
    if args.apply_train_aug:
        cfg_kwargs["test_apply_train_aug"] = True
    if args.use_full_dataloaders:
        cfg_kwargs["use_full_dataloaders_for_test"] = True
    if args.eval_merge:
        cfg_kwargs["test_eval_merge"] = True

    cfg = get_config(**cfg_kwargs)
    run_test(cfg, checkpoint_path=args.checkpoint, export=args.export, export_filename=args.export_filename)


if __name__ == "__main__":
    main()
