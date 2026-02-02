import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# Model loader (folder name has a hyphen; use path-based import without try/except)
def load_randlanet_network():
    from importlib.machinery import SourceFileLoader
    import sys
    randla_dir = Path(__file__).resolve().parent
    if str(randla_dir) not in sys.path:
        sys.path.insert(0, str(randla_dir))
    model_path = randla_dir / "model.py"
    mod = SourceFileLoader("randlanet_model", str(model_path)).load_module()
    return mod.Network


@dataclass
class RandLANetConfig:
    num_classes: int
    num_layers: int = 4
    d_out: List[int] = None
    k_neighbors: int = 16
    pool_k: int = 16
    subsample_ratio: int = 4  # N -> N/ratio per layer

    def __post_init__(self):
        if self.d_out is None:
            self.d_out = [16, 64, 128, 256]


def knn_indices_batch(xyz: torch.Tensor, k: int) -> torch.Tensor:
    # xyz: [B, N, 3] -> [B, N, k]
    B, N, _ = xyz.shape
    idx_all = []
    for b in range(B):
        dist = torch.cdist(xyz[b].unsqueeze(0), xyz[b].unsqueeze(0), p=2).squeeze(0)  # [N, N]
        dist = dist + torch.eye(N, device=xyz.device) * 1e6
        nn_idx = torch.topk(dist, k, dim=-1, largest=False)[1]  # [N, k]
        idx_all.append(nn_idx)
    return torch.stack(idx_all, dim=0)


def random_subsample_indices(N: int, ratio: int, device: torch.device) -> torch.Tensor:
    n_sub = max(1, N // ratio)
    perm = torch.randperm(N, device=device)
    return perm[:n_sub]


def knn_query(src_xyz: torch.Tensor, dst_xyz: torch.Tensor, k: int) -> torch.Tensor:
    # src_xyz: [B, Ns, 3], dst_xyz: [B, Nd, 3] -> [B, Nd, k] (indices into src)
    out = []
    for b in range(src_xyz.shape[0]):
        dist = torch.cdist(dst_xyz[b].unsqueeze(0), src_xyz[b].unsqueeze(0), p=2).squeeze(0)  # [Nd, Ns]
        nn_idx = torch.topk(dist, k, dim=-1, largest=False)[1]
        out.append(nn_idx)
    return torch.stack(out, dim=0)


def build_hierarchy(
    xyz: torch.Tensor,
    num_layers: int,
    k_neighbors: int,
    pool_k: int,
    subsample_ratio: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    xyz: [B, N0, 3]
    returns:
      xyz_list:        list length L, each [B, Ni, 3]
      neigh_idx_list:  list length L, each [B, Ni, k_neighbors]
      sub_idx_list:    list length L, each [B, Ni+1, pool_k]
      interp_idx_list: list length L, each [B, Ni, 1]
    """
    device = xyz.device

    xyz_list = []
    neigh_idx_list = []
    sub_idx_list = []
    interp_idx_list = []

    cur_xyz = xyz
    for i in range(num_layers):
        neigh_idx = knn_indices_batch(cur_xyz, k_neighbors)  # [B, Ni, k]
        xyz_list.append(cur_xyz)
        neigh_idx_list.append(neigh_idx)

        sub_inds_b = []
        for b in range(cur_xyz.shape[0]):
            sub_inds = random_subsample_indices(cur_xyz.shape[1], subsample_ratio, device)
            sub_inds_b.append(sub_inds)
        min_len = min([s.shape[0] for s in sub_inds_b])
        sub_inds_b = [s[:min_len] for s in sub_inds_b]
        sub_inds = torch.stack(sub_inds_b, dim=0)  # [B, N_sub]

        next_xyz = torch.stack([cur_xyz[b, sub_inds[b]] for b in range(cur_xyz.shape[0])], dim=0)  # [B, Nsub, 3]
        pool_idx = knn_query(cur_xyz, next_xyz, pool_k)   # [B, Nsub, pool_k]
        interp_knn = knn_query(next_xyz, cur_xyz, 1)      # [B, Ni, 1]

        sub_idx_list.append(pool_idx)
        interp_idx_list.append(interp_knn)
        cur_xyz = next_xyz

    return xyz_list, neigh_idx_list, sub_idx_list, interp_idx_list


class RandLANetTrainer:
    def __init__(
        self,
        cfg: RandLANetConfig,
        class_weights: Optional[np.ndarray] = None,
        log_dir: Optional[Path] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_decay: float = 0.7,
        step_size: int = 10,
        label_index: int = -1,
        accum_steps: int = 1,
        lr_schedule: str = "step",
        lr_final: float = 1e-5,
    ):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label_index = label_index
        self.accum_steps = max(1, int(accum_steps))

        Network = load_randlanet_network()
        self.model = Network(cfg).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        ) if class_weights is not None else nn.CrossEntropyLoss()

        # Build param groups: no decay for bias and norm-like parameters
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_bias = name.endswith('.bias')
            is_norm_like = ('norm' in name.lower()) or (param.dim() == 1)
            if is_bias or is_norm_like:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.lr = lr
        self.lr_decay = lr_decay
        self.step_size = step_size
        self.lr_schedule = lr_schedule
        self.lr_final = lr_final
        self.max_epochs: Optional[int] = None

        self.log_dir = Path(log_dir) if log_dir is not None else Path("./log/sem_seg_randlanet")
        (self.log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.best_miou = 0.0

    def _make_end_points(self, xyz_bnx3: torch.Tensor) -> dict:
        feats = xyz_bnx3.permute(0, 2, 1).contiguous()  # [B, 3, N]
        xyz_list, neigh_list, sub_list, interp_list = build_hierarchy(
            xyz_bnx3,
            self.cfg.num_layers,
            self.cfg.k_neighbors,
            self.cfg.pool_k,
            self.cfg.subsample_ratio,
        )
        return {
            "features": feats,
            "xyz": xyz_list,
            "neigh_idx": neigh_list,
            "sub_idx": sub_list,
            "interp_idx": interp_list,
        }

    @staticmethod
    def _flatten_logits_targets(logits_bcn: torch.Tensor, target_bn: torch.Tensor):
        B, C, N = logits_bcn.shape
        logits = logits_bcn.permute(0, 2, 1).reshape(B * N, C)
        target = target_bn.reshape(B * N)
        return logits, target

    def train_one_epoch(self, loader, num_classes: int, epoch: int, save_every: int = 5):
        self.model.train()
        total_correct = 0
        total_seen = 0
        total_seen_class = np.zeros(num_classes, dtype=np.int64)
        total_correct_class = np.zeros(num_classes, dtype=np.int64)
        total_iou_deno = np.zeros(num_classes, dtype=np.int64)
        loss_sum = 0.0

        if self.lr_schedule == "linear" and self.max_epochs is not None and self.max_epochs > 1:
            t = epoch / float(self.max_epochs - 1)
            lr = self.lr + (self.lr_final - self.lr) * t
        else:
            lr = max(self.lr * (self.lr_decay ** (epoch // self.step_size)), self.lr_final)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        pbar = tqdm(loader, desc=f"Train {epoch}", unit="batch")
        step_idx = 0
        # Ensure training log exists
        (self.log_dir).mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / 'training.txt', 'a') as lf:
            for batch in pbar:
                # Expect batch [B, N, D] with xyz at [:,:,:3], labels at given label_index
                points = batch[:, :, :3].float().to(self.device)
                target = batch[:, :, self.label_index].long().to(self.device)

                end_points = self._make_end_points(points)
                out = self.model(end_points)
                logits_bcn = out["logits"]  # [B, C, N]

                logits, target_flat = self._flatten_logits_targets(logits_bcn, target)
                loss = self.criterion(logits, target_flat)
                # Gradient accumulation
                if (step_idx % self.accum_steps) == 0:
                    self.optimizer.zero_grad()
                (loss / self.accum_steps).backward()
                if ((step_idx + 1) % self.accum_steps) == 0 or ((step_idx + 1) == len(loader)):
                    self.optimizer.step()

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = target_flat.cpu().numpy()
                    total_correct += (pred == labels).sum()
                    total_seen += labels.size
                    loss_sum += float(loss)
                    for l in range(num_classes):
                        total_seen_class[l] += (labels == l).sum()
                        total_correct_class[l] += ((pred == l) & (labels == l)).sum()
                        total_iou_deno[l] += ((pred == l) | (labels == l)).sum()
                # Running metrics
                running_acc = total_correct / float(max(1, total_seen))
                iou_per_class = total_correct_class / (total_iou_deno + 1e-6)
                running_miou = float(np.mean(iou_per_class))
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.4f}", mIoU=f"{running_miou:.4f}")
                # Mirror to log file
                lf.write(f"Train epoch {epoch} step {step_idx} loss={loss.item():.4f} acc={running_acc:.4f} mIoU={running_miou:.4f}\n")
                lf.flush()
                os.fsync(lf.fileno())
                step_idx += 1

        miou = np.mean(total_correct_class / (total_iou_deno + 1e-6))
        acc = total_correct / float(total_seen)
        mean_loss = loss_sum / max(1, len(loader))

        if (epoch % save_every) == 0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(ckpt, self.log_dir / "checkpoints" / f"model_epoch_{epoch}.pth")

        # Epoch summary
        with open(self.log_dir / 'training.txt', 'a') as lf:
            lf.write(f"[Train] loss: {mean_loss:.4f} acc: {acc:.4f} mIoU: {miou:.4f}\n")

        return dict(train_miou=miou, train_acc=acc, train_loss=mean_loss)

    @torch.no_grad()
    def eval_one_epoch(self, loader, num_classes: int, epoch: int, phase: str = "Val"):
        self.model.eval()
        total_correct = 0
        total_seen = 0
        total_seen_class = np.zeros(num_classes, dtype=np.int64)
        total_correct_class = np.zeros(num_classes, dtype=np.int64)
        total_iou_deno = np.zeros(num_classes, dtype=np.int64)
        total_pred_class = np.zeros(num_classes, dtype=np.int64)
        loss_sum = 0.0

        pbar = tqdm(loader, desc=f"{phase} {epoch}", unit="batch")
        (self.log_dir).mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / 'training.txt', 'a') as lf:
            lf.write(f"=== {phase} Evaluation Start ===\n")
        for batch in pbar:
            points = batch[:, :, :3].float().to(self.device)
            target = batch[:, :, self.label_index].long().to(self.device)

            end_points = self._make_end_points(points)
            out = self.model(end_points)
            logits_bcn = out["logits"]

            logits, target_flat = self._flatten_logits_targets(logits_bcn, target)
            loss = self.criterion(logits, target_flat)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            labels = target_flat.cpu().numpy()
            total_correct += (pred == labels).sum()
            total_seen += labels.size
            loss_sum += float(loss)
            for l in range(num_classes):
                total_seen_class[l] += (labels == l).sum()
                total_correct_class[l] += ((pred == l) & (labels == l)).sum()
                total_iou_deno[l] += ((pred == l) | (labels == l)).sum()
                total_pred_class[l] += (pred == l).sum()
            # Running pbar
            running_acc = total_correct / float(max(1, total_seen))
            iou_per_class = total_correct_class / (total_iou_deno + 1e-6)
            running_miou = float(np.mean(iou_per_class))
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.4f}", mIoU=f"{running_miou:.4f}")

        miou = np.mean(total_correct_class / (total_iou_deno + 1e-6))
        acc = total_correct / float(total_seen)
        mean_loss = loss_sum / max(1, len(loader))

        if miou >= self.best_miou:
            self.best_miou = miou
            ckpt = {
                "epoch": epoch,
                "class_avg_iou": miou,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(ckpt, self.log_dir / "checkpoints" / "best_model_randlanet.pth")

        # Per-class metrics
        prec = np.zeros(num_classes, dtype=np.float64)
        rec = np.zeros(num_classes, dtype=np.float64)
        f1 = np.zeros(num_classes, dtype=np.float64)
        iou_pc = total_correct_class / (total_iou_deno + 1e-6)
        acc_pc = total_correct_class / (total_seen_class + 1e-6)
        for l in range(num_classes):
            tp = float(total_correct_class[l])
            pp = float(total_pred_class[l])
            ap = float(total_seen_class[l])
            prec[l] = (tp / pp) if pp > 0 else 0.0
            rec[l] = (tp / ap) if ap > 0 else 0.0
            denom = prec[l] + rec[l]
            f1[l] = (2 * prec[l] * rec[l] / denom) if denom > 0 else 0.0

        # Log summary
        with open(self.log_dir / 'training.txt', 'a') as lf:
            lf.write(f"[{phase}] loss: {mean_loss:.4f} acc: {acc:.4f} mIoU: {miou:.4f}\n")
            for i in range(num_classes):
                lf.write(f"[{phase}] class[{i:02d}] IoU={100*iou_pc[i]:5.2f}% Acc={100*acc_pc[i]:5.2f}% Prec={100*prec[i]:5.2f}% Rec={100*rec[i]:5.2f}% F1={100*f1[i]:5.2f}%\n")
            lf.write(f"=== {phase} Evaluation End ===\n")

        # Console print per-class IoU and per-class accuracy
        print(f"Per-class ({phase}) metrics:")
        for i in range(num_classes):
            print(f"  class[{i:02d}] IoU={100*iou_pc[i]:5.2f}% Acc={100*acc_pc[i]:5.2f}%")

        return dict(val_miou=miou, val_acc=acc, val_loss=mean_loss, best_miou=self.best_miou,
                    per_class={'IoU': iou_pc, 'Acc': acc_pc, 'Prec': prec, 'Rec': rec, 'F1': f1})

    def fit(self, train_loader, val_loader, num_classes: int, epochs: int = 100, save_every: int = 5):
        # Ensure log dir exists and clear/create training.txt
        (self.log_dir).mkdir(parents=True, exist_ok=True)
        open(self.log_dir / 'training.txt', 'w').close()
        self.max_epochs = int(epochs)
        for epoch in range(epochs):
            tr = self.train_one_epoch(train_loader, num_classes, epoch, save_every=save_every)
            va = self.eval_one_epoch(val_loader, num_classes, epoch, phase="Val")
            print(
                f"Epoch {epoch:03d} "
                f"| tr_loss {tr['train_loss']:.4f} tr_mIoU {tr['train_miou']:.4f} tr_acc {tr['train_acc']:.4f} "
                f"| va_loss {va['val_loss']:.4f} va_mIoU {va['val_miou']:.4f} va_acc {va['val_acc']:.4f} "
                f"| best_mIoU {va['best_miou']:.4f}"
            )


def compute_label_weights_from_df(df, col_name="cla"):
    counts = df[col_name].value_counts().values
    counts = counts / np.sum(counts)
    weights = np.power(np.max(counts) / counts, 1 / 3.0)
    return weights


def make_trainer_from_notebooks(num_classes: int, labelweights: Optional[np.ndarray] = None) -> RandLANetTrainer:
    cfg = RandLANetConfig(
        num_classes=num_classes,
        num_layers=4,
        d_out=[16, 64, 128, 256],
        k_neighbors=16,
        pool_k=16,
        subsample_ratio=4,
    )
    trainer = RandLANetTrainer(cfg, class_weights=labelweights)
    return trainer 