"""Factory + implementation cho mọi pose encoder pose-only khả thi (P1/P2/P4/P5/P6,
docs/1_Thuyet_Trinh_Tong_Hop.md §A — P3=ST-GCN nằm riêng ở models/stgcn_encoder.py vì đã có từ trước).

Mọi encoder cùng 1 interface: __init__(cfg, pose_dim) rồi forward(pose, pose_mask) -> [B, T, d_model]
(pose_mask: True = vị trí padding, đúng chuẩn nn.Transformer key_padding_mask). Nhờ interface thống
nhất này, `SLTTransformer`, decoder, và toàn bộ RL trainer (train_scst.py/train_ppo.py) dùng chung
1 code path bất kể chọn encoder nào — đã verify tính chất này với ST-GCN (P8, docs/1_Thuyet_Trinh_Tong_Hop.md §A),
generalize trực tiếp cho các encoder mới ở đây.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # [B, L, D]
        return x + self.pe[:, :x.size(1)]


class PoseEmbed(nn.Module):
    """Map pose vector -> d_model (dùng cho P1 và làm bước đầu của P5/P6)."""
    def __init__(self, pose_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(pose_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.proj(x)


class TransformerPoseEncoder(nn.Module):
    """P1 — PoseEmbed + TransformerEncoder chuẩn (kiến trúc gốc của repo)."""
    def __init__(self, cfg, pose_dim: int):
        super().__init__()
        m = cfg.model
        self.pose_embed = PoseEmbed(pose_dim, m.d_model, m.dropout)
        self.pos_enc = PositionalEncoding(m.d_model)
        enc_layer = nn.TransformerEncoderLayer(m.d_model, m.n_heads, m.d_ff,
                                               m.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, m.n_enc_layers,
                                             norm=nn.LayerNorm(m.d_model))

    def forward(self, pose, pose_mask=None):
        x = self.pose_embed(pose)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=pose_mask)


class GCNPoseEncoder(nn.Module):
    """P2 — thay PoseEmbed bằng 1-2 lớp Graph Conv cố định trên đồ thị 75 khớp (tái dùng
    `_build_adjacency`/`GraphConv` đã có ở models/stgcn_encoder.py cho P3), phần Transformer
    encoder-decoder giữ nguyên (docs/1_Thuyet_Trinh_Tong_Hop.md §A P2)."""
    def __init__(self, cfg, pose_dim: int):
        super().__init__()
        assert pose_dim == 183, "GCNPoseEncoder giả định đúng layout 33 body + 21x2 hand"
        from .stgcn_encoder import _build_adjacency, GraphConv, N_JOINTS, N_BODY, N_HAND
        m = cfg.model
        self._n_body, self._n_hand, self._n_joints = N_BODY, N_HAND, N_JOINTS
        A = _build_adjacency()
        hidden = max(32, m.d_model // 4)
        self.gc1 = GraphConv(2, hidden, A)
        self.gc2 = GraphConv(hidden, hidden, A)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(m.dropout)
        self.joint_pool = nn.Linear(N_JOINTS * hidden, m.d_model)
        self.pos_enc = PositionalEncoding(m.d_model)
        enc_layer = nn.TransformerEncoderLayer(m.d_model, m.n_heads, m.d_ff,
                                               m.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, m.n_enc_layers,
                                             norm=nn.LayerNorm(m.d_model))

    def _split_joints(self, pose):  # [B,T,183] -> [B,T,75,2]
        B, T, _ = pose.shape
        nb, nh = self._n_body, self._n_hand
        body = pose[..., :nb * 3].reshape(B, T, nb, 3)[..., :2]
        lh = pose[..., nb * 3: nb * 3 + nh * 2].reshape(B, T, nh, 2)
        rh = pose[..., nb * 3 + nh * 2:].reshape(B, T, nh, 2)
        return torch.cat([body, lh, rh], dim=2)

    def forward(self, pose, pose_mask=None):
        if pose_mask is not None:
            pose = pose.masked_fill(pose_mask.unsqueeze(-1), 0.0)
        x = self._split_joints(pose)
        x = self.act(self.norm1(self.gc1(x)))
        x = self.dropout(self.act(self.norm2(self.gc2(x))))
        B, T, V, C = x.shape
        x = self.joint_pool(x.reshape(B, T, V * C))
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=pose_mask)


class GraphTransformerPoseEncoder(nn.Module):
    """P4 — thay graph conv cố định bằng self-attention học được trực tiếp trên 75 token-khớp mỗi
    frame (spatial attention), rồi Transformer chuẩn trên trục thời gian (temporal attention) —
    factored spatio-temporal attention, không dùng đồ thị cố định như P2/P3 (docs/1_Thuyet_Trinh_Tong_Hop.md §A P4).
    Không cần mask ở tầng spatial vì luôn có đúng 75 khớp/frame (không padding theo chiều khớp)."""
    def __init__(self, cfg, pose_dim: int):
        super().__init__()
        assert pose_dim == 183, "GraphTransformerPoseEncoder giả định đúng layout 33 body + 21x2 hand"
        from .stgcn_encoder import N_JOINTS, N_BODY, N_HAND
        m = cfg.model
        self._n_body, self._n_hand, self._n_joints = N_BODY, N_HAND, N_JOINTS
        joint_dim = max(32, m.d_model // 4)
        n_heads_spatial = max(1, m.n_heads // 2)
        self.joint_proj = nn.Linear(2, joint_dim)
        self.joint_pos = nn.Parameter(torch.randn(1, 1, N_JOINTS, joint_dim) * 0.02)
        spat_layer = nn.TransformerEncoderLayer(joint_dim, n_heads_spatial, joint_dim * 4,
                                                 m.dropout, batch_first=True, norm_first=True)
        self.spatial_encoder = nn.TransformerEncoder(spat_layer, cfg.model.graph_transformer_n_layers,
                                                     norm=nn.LayerNorm(joint_dim))
        self.frame_pool = nn.Linear(N_JOINTS * joint_dim, m.d_model)
        self.pos_enc = PositionalEncoding(m.d_model)
        temp_layer = nn.TransformerEncoderLayer(m.d_model, m.n_heads, m.d_ff,
                                                 m.dropout, batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temp_layer, m.n_enc_layers,
                                                      norm=nn.LayerNorm(m.d_model))

    def _split_joints(self, pose):
        B, T, _ = pose.shape
        nb, nh = self._n_body, self._n_hand
        body = pose[..., :nb * 3].reshape(B, T, nb, 3)[..., :2]
        lh = pose[..., nb * 3: nb * 3 + nh * 2].reshape(B, T, nh, 2)
        rh = pose[..., nb * 3 + nh * 2:].reshape(B, T, nh, 2)
        return torch.cat([body, lh, rh], dim=2)  # [B,T,V,2]

    def forward(self, pose, pose_mask=None):
        if pose_mask is not None:
            pose = pose.masked_fill(pose_mask.unsqueeze(-1), 0.0)
        x = self._split_joints(pose)
        B, T, V, _ = x.shape
        x = self.joint_proj(x) + self.joint_pos      # [B,T,V,joint_dim]
        x = x.reshape(B * T, V, -1)
        x = self.spatial_encoder(x)                  # attention giữa các khớp trong cùng 1 frame
        x = x.reshape(B, T, V * x.size(-1))
        x = self.frame_pool(x)
        x = self.pos_enc(x)
        return self.temporal_encoder(x, src_key_padding_mask=pose_mask)


class _TCNBlock(nn.Module):
    def __init__(self, d_model: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B,T,D]
        res = x
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # padding=dilation, kernel=3 -> giữ nguyên T
        h = h[:, :x.size(1)]  # an toàn phòng lệch 1 vị trí do làm tròn
        h = self.dropout(self.act(self.norm(h)))
        return h + res


class TCNPoseEncoder(nn.Module):
    """P5 — chồng Conv1D dilated (không causal, `padding=dilation` giữ nguyên độ dài T) nén/tổng hợp
    chuỗi pose theo thời gian trước khi vào Transformer encoder (docs/1_Thuyet_Trinh_Tong_Hop.md §A P5). Số layer
    TransformerEncoder giảm còn phân nửa so với P1 vì TCN đã gánh một phần việc tổng hợp cục bộ."""
    def __init__(self, cfg, pose_dim: int):
        super().__init__()
        m = cfg.model
        self.pose_embed = PoseEmbed(pose_dim, m.d_model, m.dropout)
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(m.d_model, dilation=2 ** i, dropout=m.dropout)
            for i in range(cfg.model.tcn_n_layers)
        ])
        self.pos_enc = PositionalEncoding(m.d_model)
        enc_layer = nn.TransformerEncoderLayer(m.d_model, m.n_heads, m.d_ff,
                                               m.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, max(1, m.n_enc_layers // 2),
                                             norm=nn.LayerNorm(m.d_model))

    def forward(self, pose, pose_mask=None):
        x = self.pose_embed(pose)
        if pose_mask is not None:
            x = x.masked_fill(pose_mask.unsqueeze(-1), 0.0)
        for block in self.tcn_blocks:
            x = block(x)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=pose_mask)


class PerceiverPoseEncoder(nn.Module):
    """P6 — Perceiver IO rút gọn (Jaegle et al., ICML 2021): cross-attention nén chuỗi pose [B,T,D]
    vào 1 latent array cố định (n_latents << T), self-attention xử lý trên latent, rồi cross-attention
    "decode" lại đúng T bước (query = chính pose-embedding theo vị trí gốc) để output vẫn có shape
    [B,T,d_model] — giữ đúng interface chung với các encoder khác (docs/1_Thuyet_Trinh_Tong_Hop.md §A P6,
    docs/1_Thuyet_Trinh_Tong_Hop.md §B §6.3). Độ phức tạp attention chính O(T·n_latents), tuyến tính theo T
    thay vì bậc 2 như self-attention toàn cục — lợi thế trực tiếp cho chuỗi pose dài (300+ frame)."""
    def __init__(self, cfg, pose_dim: int):
        super().__init__()
        m = cfg.model
        self.pose_embed = PoseEmbed(pose_dim, m.d_model, m.dropout)
        self.pos_enc = PositionalEncoding(m.d_model)
        n_lat = cfg.model.perceiver_n_latents
        n_layers = cfg.model.perceiver_n_layers
        self.latents = nn.Parameter(torch.randn(1, n_lat, m.d_model) * 0.02)
        self.in_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(m.d_model, m.n_heads, dropout=m.dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm_in = nn.ModuleList([nn.LayerNorm(m.d_model) for _ in range(n_layers)])
        self.latent_self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(m.d_model, m.n_heads, m.d_ff, m.dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.out_cross_attn = nn.MultiheadAttention(m.d_model, m.n_heads, dropout=m.dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(m.d_model)

    def forward(self, pose, pose_mask=None):
        B = pose.size(0)
        x = self.pose_embed(pose)
        x = self.pos_enc(x)  # [B,T,D] -- key/value nhánh nén vào, query nhánh giải nén ra
        latents = self.latents.expand(B, -1, -1)
        for cross, norm, self_attn in zip(self.in_cross_attn, self.norm_in, self.latent_self_attn):
            attn_out, _ = cross(latents, x, x, key_padding_mask=pose_mask)
            latents = norm(latents + attn_out)
            latents = self_attn(latents)
        out, _ = self.out_cross_attn(x, latents, latents)  # query=per-timestep x, key/value=latent nén
        return self.norm_out(x + out)  # [B,T,D]


def build_pose_encoder(cfg, pose_dim: int, encoder_type: str) -> nn.Module:
    """Factory duy nhất cho mọi pose encoder — dùng bởi SLTTransformer (models/slt_transformer.py)."""
    if encoder_type == "transformer":
        return TransformerPoseEncoder(cfg, pose_dim)
    if encoder_type == "stgcn":
        from .stgcn_encoder import STGCNEncoder
        return STGCNEncoder(cfg, pose_dim)
    if encoder_type == "gcn":
        return GCNPoseEncoder(cfg, pose_dim)
    if encoder_type == "graph_transformer":
        return GraphTransformerPoseEncoder(cfg, pose_dim)
    if encoder_type == "tcn":
        return TCNPoseEncoder(cfg, pose_dim)
    if encoder_type == "perceiver":
        return PerceiverPoseEncoder(cfg, pose_dim)
    raise ValueError(f"encoder_type không hợp lệ: {encoder_type!r} — xem docs/1_Thuyet_Trinh_Tong_Hop.md §A")
