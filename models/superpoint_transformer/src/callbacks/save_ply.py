# src/callbacks/save_ply.py
from typing import List
import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist

def write_ascii_ply(path: str, xyz: np.ndarray, labels: np.ndarray) -> None:
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert labels.ndim == 1 and labels.shape[0] == xyz.shape[0]
    xyz = xyz.astype(np.float32, copy=False)
    labels = labels.astype(np.int32, copy=False)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property int label",
        "end_header",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for (x, y, z), lb in zip(xyz, labels):
            f.write(f"{x} {y} {z} {lb}\n")


class SaveTestPLY(pl.Callback):
    """
    汇总 test_step 返回的 {"pos": ..., "pred": ...}，测试结束时写单个 PLY。
    - 支持单卡与多卡（DDP）。多卡时使用 all_gather_object 汇总。
    """
    def __init__(
        self,
        out_dir: str = "logs/eval_ply",
        filename: str = "test_pred.ply",
        remap_34_to_11_for_viz: bool = True,      # 新增：是否写重映射版
        write_original_also: bool = True,         # 新增：是否也写原版
        remap_filename_suffix: str = "_viz"       # 新增：重映射版文件名后缀
    ):
        super().__init__()
        self.out_dir = out_dir
        self.filename = filename
        self.remap_34_to_11_for_viz = remap_34_to_11_for_viz
        self.write_original_also = write_original_also
        self.remap_filename_suffix = remap_filename_suffix
        self._local_pos = []
        self._local_pred = []

    def on_test_start(self, trainer, pl_module):
        self._local_pos.clear()
        self._local_pred.clear()
        if trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        if outputs is None:
            return
        pos = outputs.get("pos", None)
        pred = outputs.get("pred", None)
        if pos is None or pred is None:
            return
        # 转 numpy
        pos_np = pos.numpy() if torch.is_tensor(pos) else np.asarray(pos)
        pred_np = pred.numpy() if torch.is_tensor(pred) else np.asarray(pred)
        self._local_pos.append(pos_np)
        self._local_pred.append(pred_np)

    def on_test_end(self, trainer, pl_module):
        # 先把本 rank 的拼起来
        if len(self._local_pos) == 0:
            return
        xyz_local = np.concatenate(self._local_pos, axis=0)
        label_local = np.concatenate(self._local_pred, axis=0)

        world_size = trainer.world_size if hasattr(trainer, "world_size") else 1
        rank = trainer.global_rank if hasattr(trainer, "global_rank") else 0

        def _write_both(out_path_base: str, xyz: np.ndarray, labels: np.ndarray):
            """写两份：原版 + 3/4→11 的可视化版（_viz 后缀）"""
            # 原版
            write_ascii_ply(out_path_base, xyz, labels)
            pl_module.print(f"[SaveTestPLY] Wrote PLY: {out_path_base}  ({xyz.shape[0]} pts)")

            # 可视化版（将 Vehicle/Pedestrian 映射为 Noise=11）
            labels_viz = labels.copy()
            mask34 = (labels_viz == 3) | (labels_viz == 4)
            labels_viz[mask34] = 11
            root, ext = os.path.splitext(out_path_base)
            out_path_viz = root + "_viz" + ext
            write_ascii_ply(out_path_viz, xyz, labels_viz)
            pl_module.print(f"[SaveTestPLY] Wrote remapped PLY: {out_path_viz}  (3/4 -> 11)")

        if world_size > 1 and dist.is_available() and dist.is_initialized():
            # 用 all_gather_object 收集各 rank 的 numpy（避免张量拼装的显存开销）
            gathered_xyz = [None for _ in range(world_size)]
            gathered_lb  = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_xyz, xyz_local)
            dist.all_gather_object(gathered_lb,  label_local)

            if rank == 0:
                xyz = np.concatenate(gathered_xyz, axis=0)
                labels = np.concatenate(gathered_lb, axis=0)
                out_path = os.path.join(self.out_dir, self.filename)
                _write_both(out_path, xyz, labels)
        else:
            # 单卡
            out_path = os.path.join(self.out_dir, self.filename)
            _write_both(out_path, xyz_local, label_local)
