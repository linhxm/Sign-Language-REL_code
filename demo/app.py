"""Gradio demo: upload video ngôn ngữ ký hiệu -> MediaPipe trích pose -> so sánh câu dịch của
checkpoint XE và checkpoint SCST (RL) của CÙNG một run.

Chạy local:
    python demo/app.py --results results/phoenix --spm evidence/phoenix25/spm.model
Chạy Colab: xem demo/Demo_Colab.ipynb (thêm --share).

Hai model được nạp và chạy song song trên cùng 1 chuỗi pose để thấy đúng thứ cần thấy: SCST thay
đổi output như thế nào so với XE, chứ không phải chỉ một con số BLEU trong bảng.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import gradio as gr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slt_demo import (  # noqa: E402
    ENCODER_CHOICES, PoseExtractor, Translator,
    discover_runs, find_spm, load_pose_npz,
)

# MediaPipe khởi tạo tốn ~1s và không thread-safe -> 1 instance dùng chung, tạo lazy để lúc mở app
# không phải chờ nếu người dùng chỉ nạp pose .npz sẵn có.
_EXTRACTOR = {"obj": None}
_MODELS = {}          # (ckpt_path, spm_path, encoder) -> Translator
STATE = {"runs": {}, "spm": None, "device": "cpu"}


def get_extractor() -> PoseExtractor:
    if _EXTRACTOR["obj"] is None:
        _EXTRACTOR["obj"] = PoseExtractor(model_complexity=1)
    return _EXTRACTOR["obj"]


def get_model(ckpt_path: str, spm_path: str, encoder: str | None, name: str) -> Translator:
    key = (ckpt_path, spm_path, encoder)
    if key not in _MODELS:
        _MODELS[key] = Translator.from_checkpoint(
            ckpt_path, spm_path, encoder=encoder, device=STATE["device"], name=name)
    return _MODELS[key]


def _fmt_meta(tr: Translator) -> str:
    m = tr.meta
    bleu = m.get("bleu")
    bleu_s = f" · dev BLEU-4 {bleu:.2f}" if isinstance(bleu, (int, float)) else ""
    return f"`{m.get('ckpt')}` · encoder **{m.get('encoder')}** · epoch {m.get('epoch', '?')}{bleu_s}"


def run_demo(video, npz_file, run_name, decode, beam_size, max_frames, stride, encoder_override,
             progress=gr.Progress()):
    """Trả về: (câu XE, câu SCST, bảng thống kê pose, gallery frame đã vẽ landmark)."""
    try:
        if not run_name or run_name not in STATE["runs"]:
            return "", "", "**Lỗi:** chưa chọn run nào (không tìm thấy checkpoint).", []
        spm_path = STATE["spm"]
        if not spm_path or not os.path.exists(spm_path):
            return "", "", ("**Lỗi:** không tìm thấy `spm.model`. Truyền `--spm` hoặc đặt file vào "
                            "thư mục results."), []

        enc = None if encoder_override in ("auto", "", None) else encoder_override
        previews = []

        # ---- 1. Lấy chuỗi pose -----------------------------------------------------------
        if npz_file is not None:
            path = npz_file if isinstance(npz_file, str) else npz_file.name
            progress(0.1, desc="Đọc pose .npz")
            pose = load_pose_npz(path, max_frames=int(max_frames))
            stats_md = (f"Nạp pose có sẵn: `{os.path.basename(path)}` -> "
                        f"**{pose.shape[0]} frame × {pose.shape[1]}-d** (bỏ qua MediaPipe).")
        elif video:
            progress(0.05, desc="Khởi động MediaPipe")
            ex = get_extractor()

            def cb(i, n):
                progress(0.05 + 0.65 * min(i / max(n, 1), 1.0), desc=f"Trích pose {i}/{n} frame")

            pose, stats, previews = ex.extract(video, max_frames=int(max_frames),
                                               stride=int(stride), progress_cb=cb)
            stats_md = stats.as_markdown()
        else:
            return "", "", "**Chưa có đầu vào:** hãy upload video (hoặc file pose `.npz`).", []

        # ---- 2. Dịch bằng cả 2 checkpoint -------------------------------------------------
        run = STATE["runs"][run_name]
        progress(0.75, desc="Dịch (XE)")
        xe = get_model(run["xe"], spm_path, enc, "XE")
        txt_xe, dt_xe = xe.translate(pose, decode=decode, beam_size=int(beam_size))
        out_xe = f"### {txt_xe or '(rỗng)'}\n\n<sub>{_fmt_meta(xe)} · {dt_xe*1000:.0f} ms</sub>"

        if "scst" in run:
            progress(0.9, desc="Dịch (SCST)")
            rl = get_model(run["scst"], spm_path, enc, "SCST")
            txt_rl, dt_rl = rl.translate(pose, decode=decode, beam_size=int(beam_size))
            same = " · *giống hệt XE*" if txt_rl.strip() == txt_xe.strip() else ""
            out_rl = f"### {txt_rl or '(rỗng)'}\n\n<sub>{_fmt_meta(rl)} · {dt_rl*1000:.0f} ms{same}</sub>"
        else:
            out_rl = "*Run này không có checkpoint RL (`last_rl.pt`).*"

        stats_md += f"\n\nDecode: **{decode}**" + (f" (beam={int(beam_size)})" if decode == "beam" else "")
        return out_xe, out_rl, stats_md, previews

    except Exception as e:  # UI không được chết vì 1 video hỏng -> in traceback vào ô thống kê
        return "", "", f"**Lỗi:** {e}\n\n```\n{traceback.format_exc()}\n```", []


def build_ui(share_note: str = "") -> gr.Blocks:
    runs = sorted(STATE["runs"].keys())
    default_run = next((r for r in runs if "transformer_subset25" in r and r.endswith("subset25")),
                       runs[0] if runs else None)

    with gr.Blocks(title="Sign Language Translation — XE vs SCST") as demo:
        gr.Markdown(
            "# Dịch ngôn ngữ ký hiệu: XE vs SCST\n"
            "Upload một video ký hiệu → MediaPipe Holistic trích 183-d pose mỗi frame → "
            "Transformer encoder-decoder sinh câu tiếng Đức. Cùng một chuỗi pose được chạy qua "
            "**hai checkpoint**: sau pha Cross-Entropy (`best_xe.pt`) và sau pha RL / SCST "
            "(`last_rl.pt`).\n\n"
            "> Model train trên **PHOENIX-2014T** (tin thời tiết, tiếng Đức, 1 người ký chính diện "
            "trên nền xám) với BLEU-4 test ~6. Video ngoài miền đó sẽ cho câu lảm nhảm — đó là hành "
            "vi đúng của model, không phải lỗi demo. Muốn thấy output sát thực tế, nạp file pose "
            "`.npz` của một câu PHOENIX ở tab bên phải." + share_note
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Video"):
                    video = gr.Video(label="Video ngôn ngữ ký hiệu", sources=["upload", "webcam"])
                with gr.Tab("Pose .npz có sẵn"):
                    npz = gr.File(label="File .npz (key 'pose', shape [T,183])",
                                  file_types=[".npz"], type="filepath")
                run_dd = gr.Dropdown(runs, value=default_run, label="Run (thư mục checkpoint)")
                with gr.Accordion("Tuỳ chọn", open=False):
                    decode = gr.Radio(["beam", "greedy"], value="beam", label="Kiểu decode")
                    beam = gr.Slider(2, 8, value=4, step=1, label="Beam size")
                    maxf = gr.Slider(60, 300, value=300, step=20,
                                     label="Số frame tối đa đưa vào model (train dùng 300)")
                    stride = gr.Slider(0, 8, value=0, step=1,
                                       label="Frame stride khi trích pose (0 = tự chọn)")
                    enc = gr.Dropdown(["auto"] + ENCODER_CHOICES, value="auto",
                                      label="Encoder (auto = suy từ checkpoint)")
                btn = gr.Button("Dịch", variant="primary")

            with gr.Column(scale=1):
                out_xe = gr.Markdown(label="XE")
                gr.Markdown("---")
                out_rl = gr.Markdown(label="SCST")
                gr.Markdown("---")
                stats = gr.Markdown()
                gallery = gr.Gallery(label="Landmark đã trích", columns=3, height=240)

        btn.click(run_demo,
                  inputs=[video, npz, run_dd, decode, beam, maxf, stride, enc],
                  outputs=[out_xe, out_rl, stats, gallery])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phoenix",
                    help="Thư mục chứa các run (mỗi run có best_xe.pt + last_rl.pt)")
    ap.add_argument("--spm", default=None, help="Đường dẫn spm.model (tokenizer đã train cùng run)")
    ap.add_argument("--device", default=None, help="cuda | cpu (mặc định: tự chọn)")
    ap.add_argument("--share", action="store_true", help="Tạo link public (dùng trên Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    import torch
    STATE["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    STATE["runs"] = discover_runs(args.results)
    if not STATE["runs"]:
        raise SystemExit(f"Không tìm thấy best_xe.pt nào dưới {args.results}")
    STATE["spm"] = args.spm or find_spm([args.results, os.path.dirname(args.results.rstrip("/\\")),
                                         "evidence", "."])
    print(f"Device: {STATE['device']}")
    print(f"Tìm thấy {len(STATE['runs'])} run: {', '.join(sorted(STATE['runs'])[:5])}...")
    print(f"Tokenizer: {STATE['spm']}")
    build_ui().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
