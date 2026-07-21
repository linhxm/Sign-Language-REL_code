"""Central config. Sửa ở đây thay vì rải khắp code."""
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    # Paths trên Kaggle (sửa khi upload dataset)
    # phoenix_root: nơi chứa 3 file annotation PHOENIX-2014-T.{train,dev,test}.corpus.csv.
    #   find_annotation_csv() thử annotations/manual, annotations/, gốc + quét đệ quy nên trỏ vào
    #   THƯ MỤC CHỨA CSV là được. Mặc định = "data/archive" (3 CSV đóng gói SẴN trong repo này,
    #   đi kèm khi upload code thành Kaggle Dataset -> không cần dataset annotation riêng).
    #   Nếu dùng dataset PHOENIX đầy đủ có sẵn annotations, đổi thành "/kaggle/input/<tên-dataset>".
    #   (Ảnh để trích pose KHÔNG dùng biến này — extract_poses.py nhận đường dẫn qua --input_dir.)
    phoenix_root: str = "data/phoenix-2014t-annotations"
    pose_cache_dir: str = "/kaggle/input/phoenix-poses"   # đã extract sẵn (bạn tự tạo sau Bước 0)
    work_dir: str = "/kaggle/working"

    # Pose extraction
    # 33 body*(x,y,visibility)=99 + 21 left*(x,y)=42 + 21 right*(x,y)=42 = 183
    # (KHỚP với data/extract_poses.py; giá trị 150 cũ đã sai và bị bỏ.)
    pose_dim: int = 183
    max_frames: int = 300       # truncate/pad video tới N frames
    frame_stride: int = 1       # (CHƯA dùng) reserved: lấy mỗi k-th frame khi extract

    # Vocab
    vocab_size: int = 3000      # subword (sentencepiece) BPE
    max_text_len: int = 60      # số token tối đa/câu (đã gồm <bos>/<eos>)

    # Subset sizes (theo % của train split) -- chạy qua run_all.py --subset 0.05|0.10|0.25|0.5|1.0
    # (epoch KHÔNG giảm theo subset -- xe_epochs/rl_epochs cố định, xem TrainConfig).
    # 0.05 = mức báo cáo thực nghiệm CHÍNH của repo này: train 5% split train, dev/test LUÔN full.
    # 0.10 = mức test thêm số liệu. Các mức LỒNG NHAU theo seed 42 (5% ⊂ 10% ⊂ 25% ⊂ ...) nên khi
    #        extract pose cho 10% là đã bao gồm 5% -- so sánh giữa các mức được kiểm soát.
    subset_ratios: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.25, 0.5, 1.0])

@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.3        # cao vì dataset nhỏ
    label_smoothing: float = 0.1
    # "transformer" | "stgcn" | "gcn" | "graph_transformer" | "tcn" | "perceiver"
    # (P1/P3/P2/P4/P5/P6 — docs/1_Thuyet_Trinh_Tong_Hop.md §A; tất cả cùng interface encode()->[B,T,d_model])
    encoder_type: str = "transformer"
    # Riêng cho encoder_type="perceiver" (P6, docs/1_Thuyet_Trinh_Tong_Hop.md §B §6.3)
    perceiver_n_latents: int = 64
    perceiver_n_layers: int = 4
    # Riêng cho encoder_type="tcn" (P5) — số lớp Conv1D dilated trước khi vào Transformer encoder
    tcn_n_layers: int = 4
    # Riêng cho encoder_type="graph_transformer" (P4) — số layer self-attention trên đồ thị khớp
    graph_transformer_n_layers: int = 2

@dataclass
class TrainConfig:
    # Cross-Entropy phase
    xe_epochs: int = 80
    xe_batch_size: int = 16
    xe_lr: float = 5e-4
    xe_warmup_steps: int = 2000
    xe_weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # SCST / PPO RL phase (dùng chung cho cả 2 thuật toán qua --algo scst|ppo, xem main.py)
    rl_epochs: int = 20
    rl_batch_size: int = 8      # nhỏ hơn vì cần sample
    rl_lr: float = 5e-6         # RL lr THẤP HƠN XE 100x — quan trọng
    rl_sample_temp: float = 1.0
    rl_n_samples: int = 1       # số sample/seq — Multi-sample SCST (Experiment 3, docs/1_Thuyet_Trinh_Tong_Hop.md §E)
    rl_baseline_eval_mode: bool = True  # tính greedy baseline ở model.eval() (dropout OFF)
                                        # -> baseline ổn định, deterministic. False = giữ train-mode (SCST gốc).
                                        # Chỉ áp dụng cho SCST — PPO dùng value head thay baseline.
    rl_entropy_coef: float = 0.0        # hệ số entropy bonus chống reward collapse; 0.0 = TẮT.
                                        # Tăng (vd 1e-3) nếu advantage sụp ≈0 ở subset nhỏ.
    rl_use_baseline: bool = True        # False = REINFORCE thuần (C.1, docs/1_Thuyet_Trinh_Tong_Hop.md)
                                        # -- bỏ baseline greedy, dùng thẳng R(sample) làm advantage.
                                        # Chỉ dùng để ablation so sánh với SCST (True), KHÔNG dùng làm mặc định
                                        # vì variance rất cao (đúng lý do SCST tồn tại).

    # PPO-only (docs/1_Thuyet_Trinh_Tong_Hop.md §C.4/C.6, docs/1_Thuyet_Trinh_Tong_Hop.md §E Experiment 7)
    ppo_clip_eps: float = 0.2           # epsilon của clipped surrogate objective (Schulman 2017)
    ppo_epochs: int = 4                 # số lần update trên CÙNG 1 rollout batch
    ppo_gamma: float = 1.0              # episode ngắn (<=60 token) -> không cần discount
    ppo_gae_lambda: float = 0.95        # GAE lambda; thử {0.9, 0.95, 1.0} theo mục C.6
    ppo_value_coef: float = 0.5         # trọng số value loss trong tổng loss PPO
    ppo_use_clip: bool = True           # False = A2C (C.5) -- bỏ clipped surrogate, dùng thẳng
                                        # ratio*advantage không giới hạn trust-region, 1 epoch/batch
                                        # (ppo_epochs bị ép về 1 khi False, xem train_ppo.py)

    # MRT — Minimum Risk Training (C.9, docs/1_Thuyet_Trinh_Tong_Hop.md)
    mrt_n_candidates: int = 8           # số candidate/input (beam hoặc sample) dùng để tính risk
    mrt_alpha: float = 5e-3             # độ sắc phân phối Q(y|x) ~ pi_theta(y|x)^alpha trong tập candidate
    mrt_candidate_source: str = "sample"  # "sample" | "beam" -- beam = phủ luôn ý tưởng F.13
                                          # (RL/MRT cho beam search policy, docs/1_Thuyet_Trinh_Tong_Hop.md §F)

    # RAML — Reward Augmented Maximum Likelihood (C.10)
    raml_tau: float = 0.4               # nhiệt độ phân phối exp(R(y)/tau) khi sample target nhiễu
    raml_n_samples: int = 4             # số target-nhiễu/câu mỗi batch
    raml_max_edits_ratio: float = 0.3   # tỉ lệ token tối đa bị thay thế khi tạo nhiễu quanh ground-truth

    # DPO — Direct Preference Optimization (C.7), preference tự sinh từ multi-sample ranking theo reward
    dpo_beta: float = 0.1               # hệ số nhiệt trong loss DPO (Rafailov et al. 2023)
    dpo_n_samples: int = 4              # số sample/input để chọn cặp (win, lose) theo reward

    # Curriculum RL (C.12 + Reward 10) -- áp dụng cho train_scst.py
    rl_curriculum_epochs: int = 0       # >0: trong N epoch đầu, tăng dần w_rep/w_len từ 0 -> giá trị
                                        # config thật theo lịch tuyến tính; 0 = TẮT (dùng full weight ngay)
    rl_curriculum_length_sort: bool = False  # True: N epoch đầu duyệt batch theo thứ tự CÂU (text
                                              # tham chiếu) ngắn->dài (data curriculum)
                                              # thay vì shuffle ngẫu nhiên -- xem data/dataset.py::LengthCurriculumSampler

    # Reward shaping: R = w_bleu*BLEU - w_rep*rep_penalty - w_len*length_penalty + w_bert*BERTScore.
    # Đặt một weight = 0.0 để TẮT thành phần đó -> phục vụ Experiment 2/9 (reward ablation).
    reward_bleu_weight: float = 1.0
    reward_repetition_penalty: float = 0.5  # phạt tỉ lệ n-gram lặp (rep_penalty in [0,1])
    reward_length_penalty: float = 0.0      # phạt câu ngắn hơn ref; 0.0 = TẮT (mặc định)
    reward_bert_weight: float = 0.0         # Reward 5 (docs/1_Thuyet_Trinh_Tong_Hop.md §E);
                                            # 0.0 = TẮT mặc định vì chi phí GPU cao (mục D) —
                                            # chỉ bật ở subset nhỏ khi chạy Experiment 9.
    reward_bert_model: str = "bert-base-german-cased"  # khớp ngôn ngữ đích PHOENIX-2014T

    # Hiệu năng: Automatic Mixed Precision (docs/1_Thuyet_Trinh_Tong_Hop.md §J điểm yếu B.7)
    use_amp: bool = True

    # Logging / checkpoint
    eval_every: int = 1         # epoch
    save_every: int = 5
    early_stop_patience: int = 10

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    device: str = "cuda"

CFG = Config()
