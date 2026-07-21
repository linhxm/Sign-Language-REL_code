"""Tokenizer dùng SentencePiece (BPE). Train một lần trên text của train split."""
import os
import sentencepiece as spm

PAD, BOS, EOS, UNK = 0, 1, 2, 3

class Tokenizer:
    def __init__(self, model_path: str = None):
        self.sp = spm.SentencePieceProcessor()
        if model_path and os.path.exists(model_path):
            self.sp.load(model_path)
        self.pad_id = PAD; self.bos_id = BOS; self.eos_id = EOS; self.unk_id = UNK

    @staticmethod
    def train(text_file: str, model_prefix: str, vocab_size: int = 3000):
        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=PAD, bos_id=BOS, eos_id=EOS, unk_id=UNK,
            character_coverage=1.0,
        )

    def encode(self, text: str, add_special: bool = True):
        ids = self.sp.encode(text, out_type=int)
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(ids)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()


def build_tokenizer_from_train(train_csv: str, work_dir: str, vocab_size: int = 3000):
    """Trích text từ train csv → train sentencepiece."""
    import pandas as pd
    df = pd.read_csv(train_csv, sep="|").dropna(subset=["translation"])
    txt_file = os.path.join(work_dir, "train_text.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        for t in df["translation"]:
            f.write(str(t).strip() + "\n")
    model_prefix = os.path.join(work_dir, "spm")
    Tokenizer.train(txt_file, model_prefix, vocab_size=vocab_size)
    return Tokenizer(model_prefix + ".model")
