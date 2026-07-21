# 0. Kiến trúc hệ thống

```
╔══════════════════════════════════════════════════════════════════════════════╗
║           Sign-Language-REL - KIẾN TRÚC HỆ THỐNG (pose-based SLT + RL)       ║
║   luồng dữ liệu chảy từ trên xuống  -  [CV]=thị giác  [RL]=policy  [GL]=gloss║
╚══════════════════════════════════════════════════════════════════════════════╝

                        ┌───────────────────────────┐
                        │   INPUT: Video / Pose     │ <-- PHOENIX-2014T
                        │   ~7K câu - MediaPipe     │
                        └─────────────┬─────────────┘
                                      │ pose 183-d
                                      v
┌──────────────────────────────────────────────────────────────────────────────┐
│ L1 · DATA  (data/)                                                      [CV] │
│  ┌──────────────┐ ┌───────────────┐ ┌──────────────┐ ┌───────────────────┐   │
│  │ extract_poses│ │   dataset     │ │  tokenizer   │ │   gloss_vocab     │   │
│  │ Holistic     │ │ norm·augment  │ │ SentencePiece│ │ BLANK=0 / UNK=1   │   │
│  │ → 183-d      │ │ curriculum    │ │ BPE          │ │ (cho nhánh P7)    │   │
│  │ 99+42+42     │ │ make_loaders  │ │ bos/eos/pad  │ │                   │   │
│  └──────────────┘ └───────────────┘ └──────────────┘ └───────────────────┘   │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ [B,T,183] + pose_mask
                                     v
┌───────────────────────────────────────────────────────────────────────────────┐
│ L2 · POSE ENCODER   build_pose_encoder() factory - 1 interface chung    [CV]  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────┐ ┌────────┐ │
│  │ P1 Transf│ │ P2 GCN   │ │ P3 ST-GCN│ │ P4 GraphTrans│ │ P5 TCN│ │P6 Perc.│ │
│  │  8.26M   │ │ 9.38M    │ │ 6.33M    │ │   9.48M      │ │ 7.47M │ │ 9.59M  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘ └───────┘ └────────┘ │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │ memory [B,T,d_model]
                                     v
┌───────────────────────────────────────────────────────────────────────────────┐
│ L3 · MODEL  (models/slt_transformer.py)                          [dùng chung] │
│   ┌───────────────────────────────────┐   ┌──────────────────────────────┐    │
│   │ SLTTransformer (enc-dec, tie wts) │   │ ValueHead - critic V(s)      │    │
│   │ greedy · sample · beam_search     │   │ dùng cho PPO / A2C (GAE)     │    │
│   └───────────────────────────────────┘   └──────────────────────────────┘    │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     v
┌────────────────────────────────────────────────────────────────────────────────┐
│ L4 · PHASE 1 - train_xe.py                                       [dùng chung]  │
│   Cross-Entropy · teacher forcing · label smoothing · AMP  →  best_xe.pt       │
│   evaluate() (BLEU-4)  ← dùng lại ở MỌI trainer & eval                         │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │ best_xe.pt (điểm khởi động)
                                     v
┌────────────────────────────────────────────────────────────────────────────────┐
│ L5 · PHASE 2 - RL FINE-TUNE DECODER  (training/)                          [RL] │
│  ┌────────┐ ┌───────────┐ ┌────────────┐ ┌────────┐                            │
│  │ SCST   │ │ REINFORCE │ │ Curriculum │ │  PPO   │      ┌────────────────────┐│
│  │  C.1   │ │ C.2 no-bl │ │   C.12     │ │ C.4 GAE│ <══> │ L6 · compute_reward││
│  └────────┘ └───────────┘ └────────────┘ └────────┘      │       [RL]         ││
│  ┌────────┐ ┌───────────┐ ┌────────────┐ ┌────────┐      │ BLEU               ││
│  │  A2C   │ │   MRT     │ │   RAML     │ │  DPO   │      │ + repetition_pen   ││
│  │C.5 clip│ │ C.9       │ │  C.10      │ │C.7 pref│      │ + length_pen       ││
│  └────────┘ └───────────┘ └────────────┘ └────────┘      │ + semantic (opt)   ││
│                                                          └────────────────────┘│
└──────────────┬──────────────────────────────────────────┬──────────────────────┘
               │ cùng cơ chế RL                           │ test_results.json
               v                                          │
┌────────────────────────────────────────────────────┐    │ 
│                                                    │    │
│ L7 · RL VƯỢT NGOÀI DECODER - mục F   [RL × CV]     │    │
│  ┌─────────────┐ ┌────────────┐ ┌───────────────┐  │    │
│  │Frame sel F.6│ │Adaptive F.9│ │Landmark  F.8  │  │    │
│  │   soft-mask │ │            │ │ (occlusion)   │  │    │
│  └─────────────┘ └────────────┘ └───────────────┘  │    │
│  ┌─────────────┐ ┌───────────────────────────────┐ │    │
│  │Decode pol F5│ │CTC segment F.14 (forced-align)│ │    │
│  └─────────────┘ └─────────────────┬─────────────┘ │    │
│     soft-mask = zero-hoá frame, CHƯA giảm compute  │    │
└────────────────────────────────────┼───────────────┘    │
                                     │ đoạn gloss         │ 
        ┌────────────────────────────┘                    │
        │  (rẽ SONG SONG - cùng encoder L2 + vocab L1)    │
        v                                                 │
┌────────────────────────────────────────────────────┐    │
│ P7 · NHÁNH TWO-STAGE GLOSS               [GL]      │    │
│  ┌────────────────────┐      ┌──────────────────┐  │    │
│  │ Stage1 CTC         │ ───> │ Stage2 NMT       │  │    │
│  │ pose→gloss · WER   │      │ gloss→text       │  │    │
│  │ blank=0·zero_inf   │      │ greedy_decode    │  │    │
│  └────────────────────┘      └──────────┬───────┘  │    │
│   cột orth: verify khi chạy, chưa test dữ liệu thật│    │
└─────────────────────────────────────────┼──────────┘    │
                                          │ test_results.json
                                          v               v
┌───────────────────────────────────────────────────────────────────────────────┐
│ L8 · ĐÁNH GIÁ & BẢNG SO SÁNH  (scripts/)                        [dùng chung]  │
│  ┌──────────────┐ ┌───────────────┐ ┌────────────────┐ ┌─────────────────────┐│
│  │ evaluate()   │ │measure_latency│ │ eval_baselines │ │ aggregate_results   ││
│  │ BLEU-4       │ │6 encoder→json │ │ BASE cơ bản:   │ │ quét *_results/     ││
│  │              │ │               │ │ trivial·random/│ │  *_history/latency  ││
│  │              │ │               │ │ uniform·fixtemp│ │ → comparison_table  ││
│  └──────────────┘ └───────────────┘ └────────────────┘ └─────────────────────┘│
└───────────────────────────────────────────────────────────────────────────────┘

 Điều phối: configs/config.py · main.py (--encoder --algo --phase --subset, 1 experiment đơn lẻ)
            main_twostage.py (P7) · run_all.py (TOÀN BỘ ma trận cho 1 subset, resumable, dùng
            trên Kaggle) · KAGGLE_NOTEBOOK.ipynb (train, T4×2) ·
            KAGGLE_NOTEBOOK_EXTRACT.ipynb (trích pose, CPU-only, chạy TRƯỚC)
```