"""ST-GCN pose encoder (Yan, Xiong, Lin — AAAI 2018) — kiến trúc thay thế cho
PoseEmbed+TransformerEncoder trong models/slt_transformer.py, dùng cho Experiment 4
(Transformer vs GCN, xem docs/1_Thuyet_Trinh_Tong_Hop.md §E) và Pipeline 5 (docs/1_Thuyet_Trinh_Tong_Hop.md §A).

Output cùng shape [B, T, d_model] như encoder Transformer hiện tại -> decoder không cần đổi gì,
cô lập đúng 1 biến so sánh (kiến trúc encoder pose).

Đơn giản hoá có chủ đích so với ST-GCN gốc (ghi rõ để không nhầm là bug):
- Chỉ dùng 2 kênh (x, y) mỗi khớp, bỏ kênh visibility của body — để đồng nhất số kênh giữa
  33 body-landmark (vốn có visibility) và 42 hand-landmark (không có) trong data/extract_poses.py.
- Đồ thị 75 khớp = 33 body (MediaPipe Pose) + 21 tay trái + 21 tay phải (MediaPipe Hands),
  nối thêm 2 cạnh cổ tay-bàn tay để thành 1 đồ thị thống nhất thay vì 3 đồ thị rời rạc.
- Temporal conv (kernel=9) chạy cả qua biên frame thật/frame-pad (pose đã bị zero-mask trước khi
  vào conv nên phần rò rỉ tối thiểu) — decoder vẫn dùng đúng pose_mask để bỏ hoàn toàn các vị trí
  pad ở tầng cross-attention, nên rò rỉ nhỏ này không ảnh hưởng loss/reward cuối cùng.
"""
import torch
import torch.nn as nn

N_BODY, N_HAND = 33, 21
N_JOINTS = N_BODY + N_HAND * 2  # 75

# MediaPipe Pose (33 điểm) — cạnh khung xương chuẩn
_BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]
# MediaPipe Hands (21 điểm/tay) — cạnh khung xương chuẩn
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _build_adjacency() -> torch.Tensor:
    """Ma trận kề chuẩn hoá D^-1/2 A D^-1/2 (Kipf & Welling, 2017), có self-loop."""
    A = torch.eye(N_JOINTS)
    for i, j in _BODY_EDGES:
        A[i, j] = 1.0; A[j, i] = 1.0
    lh_off, rh_off = N_BODY, N_BODY + N_HAND
    for i, j in _HAND_EDGES:
        A[lh_off + i, lh_off + j] = 1.0; A[lh_off + j, lh_off + i] = 1.0
        A[rh_off + i, rh_off + j] = 1.0; A[rh_off + j, rh_off + i] = 1.0
    # Nối cổ tay body (15=trái, 16=phải) với gốc bàn tay (landmark 0 mỗi tay)
    A[15, lh_off + 0] = 1.0; A[lh_off + 0, 15] = 1.0
    A[16, rh_off + 0] = 1.0; A[rh_off + 0, 16] = 1.0

    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)


class GraphConv(nn.Module):
    """1 lớp graph convolution: X' = A_norm @ X @ W."""
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.lin = nn.Linear(in_ch, out_ch)

    def forward(self, x):  # x: [B, T, V, C]
        x = torch.einsum("vw,btwc->btvc", self.A, x)
        return self.lin(x)


class STGCNBlock(nn.Module):
    """graph conv (không gian) + Conv1D theo thời gian (Yan et al. 2018), có residual."""
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor, dropout: float = 0.3):
        super().__init__()
        self.gcn = GraphConv(in_ch, out_ch, A)
        self.tcn = nn.Conv1d(out_ch, out_ch, kernel_size=9, padding=4)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # [B, T, V, C]
        res = self.residual(x)
        h = self.act(self.norm1(self.gcn(x)))
        B, T, V, C = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B * V, C, T)
        h = self.tcn(h)
        h = h.reshape(B, V, C, T).permute(0, 3, 1, 2)
        h = self.dropout(self.act(self.norm2(h)))
        return h + res


class STGCNEncoder(nn.Module):
    """Thay thế PoseEmbed + TransformerEncoder. Interface: forward(pose, pose_mask) -> [B,T,D]."""
    def __init__(self, cfg, pose_dim: int = 183):
        super().__init__()
        assert pose_dim == 183, "STGCNEncoder giả định đúng layout 33 body + 21x2 hand của extract_poses.py"
        m = cfg.model
        A = _build_adjacency()
        hidden = max(32, m.d_model // 4)
        self.blocks = nn.ModuleList([
            STGCNBlock(2, hidden, A, m.dropout),
            STGCNBlock(hidden, hidden, A, m.dropout),
            STGCNBlock(hidden, hidden, A, m.dropout),
        ])
        self.joint_pool = nn.Linear(N_JOINTS * hidden, m.d_model)
        self.out_norm = nn.LayerNorm(m.d_model)

    def _split_joints(self, pose):  # [B, T, 183] -> [B, T, 75, 2]
        B, T, _ = pose.shape
        body = pose[..., :N_BODY * 3].reshape(B, T, N_BODY, 3)[..., :2]  # bỏ visibility
        lh = pose[..., N_BODY * 3: N_BODY * 3 + N_HAND * 2].reshape(B, T, N_HAND, 2)
        rh = pose[..., N_BODY * 3 + N_HAND * 2:].reshape(B, T, N_HAND, 2)
        return torch.cat([body, lh, rh], dim=2)

    def forward(self, pose, pose_mask=None):
        if pose_mask is not None:
            pose = pose.masked_fill(pose_mask.unsqueeze(-1), 0.0)
        x = self._split_joints(pose)
        for block in self.blocks:
            x = block(x)
        B, T, V, C = x.shape
        x = self.joint_pool(x.reshape(B, T, V * C))
        return self.out_norm(x)  # [B, T, d_model]
