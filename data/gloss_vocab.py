"""Vocab cho gloss (cột `orth` trong PHOENIX-2014-T*.corpus.csv) -- whitespace token rời rạc,
KHÔNG dùng BPE như `data/tokenizer.py` (khác text tự nhiên) vì gloss PHOENIX vốn đã là các đơn vị
rời rạc theo quy ước annotation (vd. "ICH SCHNEE"), tách theo khoảng trắng là chuẩn trong literature
(Camgoz 2018/2020). Dùng cho P7 -- CTC head (stage 1, models/gloss_ctc_head.py) và gloss2text NMT
(stage 2, models/gloss2text_nmt.py) -- docs/1_Thuyet_Trinh_Tong_Hop.md §A.
"""
import json
from collections import Counter
import pandas as pd


class GlossVocab:
    BLANK = 0   # trùng quy ước torch.nn.CTCLoss(blank=0)
    UNK = 1

    def __init__(self, token2id=None):
        self.token2id = token2id or {"<blank>": 0, "<unk>": 1}
        self.id2token = {v: k for k, v in self.token2id.items()}

    @property
    def vocab_size(self):
        return len(self.token2id)

    def encode(self, gloss_text: str):
        toks = str(gloss_text).strip().split()
        return [self.token2id.get(t, self.UNK) for t in toks]

    def decode(self, ids):
        return " ".join(self.id2token.get(int(i), "<unk>") for i in ids if int(i) != self.BLANK)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as f:
            token2id = json.load(f)
        return cls(token2id)

    @classmethod
    def build_from_csv(cls, csv_path: str, max_vocab: int = 1200):
        """Bước verify bắt buộc trước khi chạy P7 (docs/1_Thuyet_Trinh_Tong_Hop.md §A §4.1): raise rõ ràng nếu
        thiếu cột `orth` thay vì fail âm thầm ở bước train."""
        df = pd.read_csv(csv_path, sep="|")
        if "orth" not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột 'orth' (gloss) trong {csv_path} -- P7 (two-stage gloss "
                "pipeline, docs/1_Thuyet_Trinh_Tong_Hop.md §A) không khả thi với file annotation này. Cần định dạng "
                "PHOENIX-2014-T*.corpus.csv chuẩn (cột: name|video|start|end|speaker|orth|translation)."
            )
        counter = Counter()
        for g in df["orth"].dropna().astype(str):
            counter.update(g.strip().split())
        token2id = {"<blank>": 0, "<unk>": 1}
        for tok, _ in counter.most_common(max_vocab - len(token2id)):
            token2id[tok] = len(token2id)
        return cls(token2id)
