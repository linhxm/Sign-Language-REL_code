"""Lớp tương thích AMP giữa các phiên bản PyTorch.

Vì sao cần: từ PyTorch 2.4, `torch.cuda.amp.autocast` và `torch.cuda.amp.GradScaler` bị
deprecate và in FutureWarning ở MỌI batch — trên Kaggle (PyTorch 2.x) log train bị ngập
cảnh báo, che mất số liệu thật (BLEU/advantage/rep_rate) mà ta cần đọc mỗi epoch.

API mới là `torch.amp.autocast("cuda", ...)` / `torch.amp.GradScaler("cuda", ...)`, nhưng
nó chỉ có từ 2.4 trở lên. Module này chọn đúng API theo phiên bản đang chạy, nên code
trainer không phải quan tâm, và vẫn chạy được trên PyTorch cũ hơn nếu cần.

Dùng y hệt API cũ:
    from utils.amp_compat import autocast, GradScaler
    scaler = GradScaler(enabled=amp_enabled)
    with autocast(enabled=amp_enabled):
        ...
"""
import torch

_HAS_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")

if _HAS_NEW_AMP:
    def autocast(enabled: bool = True, dtype=None):
        kwargs = {"enabled": enabled}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.amp.autocast("cuda", **kwargs)

    def GradScaler(enabled: bool = True):
        return torch.amp.GradScaler("cuda", enabled=enabled)
else:  # PyTorch < 2.4
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler

    def autocast(enabled: bool = True, dtype=None):
        kwargs = {"enabled": enabled}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return _autocast(**kwargs)

    def GradScaler(enabled: bool = True):
        return _GradScaler(enabled=enabled)

__all__ = ["autocast", "GradScaler"]
