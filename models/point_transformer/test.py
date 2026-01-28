import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from model import Model, AverageMeter, intersectionAndUnionGPU
from data_utils import create_dataloaders


def validate(loader, model, criterion, num_classes, ignore_label, class_names, log_file=None, fast_debug=False):
    """Run evaluation on a dataloader"""
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    pred_list = []

    with torch.no_grad():
        for i, (points, feat, target, offset) in enumerate(tqdm(loader, desc="Testing", unit="batch")):
            points, feat, target, offset = (
                points.cuda(non_blocking=True),
                feat.cuda(non_blocking=True),
                target.cuda(non_blocking=True),
                offset.cuda(non_blocking=True),
            )
            if target.shape[-1] == 1:
                target = target[:, 0]

            output = model([points, feat, offset])
            loss = criterion(output, target)
            output = output.max(1)[1]

            n = points.size(0)
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

            if fast_debug and i >= 0:
                break

    # metrics
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    valid_mask = union_meter.sum > 0
    if valid_mask.any():
        mIoU = np.mean(iou_class[valid_mask])
    else:
        mIoU = 0.0

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

    return loss_meter.avg, mIoU, mAcc, allAcc, iou_class, accuracy_class, np.hstack(pred_list)


def run_test(cfg, checkpoint_path=None):
    """Load best model and run testing"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = cfg.class_names
    num_classes = len(class_names)

    # Data
    _, _, test_loader, _ = create_dataloaders(cfg)

    if getattr(cfg, "fast_debug", False):
        print("[DEBUG] Running test in fast_debug mode (1 batch only)")
        test_iter = iter(test_loader)
        test_loader = [next(test_iter)]

    # Model
    model = Model(c=cfg.feature_dim, k=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.save_path, "model", "model_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    log_file = os.path.join(cfg.save_path, "model", "test_log.txt")
    results = validate(
        test_loader, model, criterion,
        num_classes, cfg.ignore_label,
        class_names, log_file,
        fast_debug=getattr(cfg, "fast_debug", False)
    )

    return results
