"""Stage 1 của P7 (Two-stage: Pose -> Gloss -> NMT -> Text, docs/1_Thuyet_Trinh_Tong_Hop.md §A): pose -> gloss
qua CTC (Graves et al., 2006) — không cần alignment tường minh frame<->gloss, CTC tự học alignment
ẩn bằng cách marginalize mọi đường đi hợp lệ. Dùng chung factory encoder với SLTTransformer
(models/encoders.py::build_pose_encoder) nên có thể chọn bất kỳ kiến trúc P1-P6 nào làm backbone,
tái sử dụng 100% code encoder đã kiểm chứng ở pipeline single-stage.
"""
import torch
import torch.nn as nn
from .encoders import build_pose_encoder


class GlossCTCModel(nn.Module):
    def __init__(self, cfg, gloss_vocab_size: int, pose_dim: int = 183, encoder_type: str = "transformer"):
        super().__init__()
        self.pose_encoder = build_pose_encoder(cfg, pose_dim, encoder_type)
        self.ctc_head = nn.Linear(cfg.model.d_model, gloss_vocab_size)
        self.encoder_type = encoder_type

    def forward(self, pose, pose_mask):
        memory = self.pose_encoder(pose, pose_mask)   # [B,T,D] -- T giữ nguyên ở mọi encoder P1-P6
        logits = self.ctc_head(memory)                # [B,T,V_gloss], index 0 = <blank>
        return torch.log_softmax(logits, dim=-1)
