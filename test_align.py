import unittest
from torch import tensor
from align import extract_phrase_pairs
from corpora_kdswch import (
    Bitext,
    MultifileBitext,
    MixtureOfBitexts,
    TokenizedMixtureOfBitexts,
)
from extract_tok_utils import build_fast_align_dict_from_raw
from corpora_kdswch import TokenizedMixtureOfTextAndGoalEncoding, TokenizedBitext
from torch import tensor
from tokenization import NllbTokenizer
from transformers import AutoModelForSeq2SeqLM


class TestAlign(unittest.TestCase):

    def test_extract_phrase_pairs(self):
        bitext = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        tokenizer = NllbTokenizer("600M")
        tokenized_bitext = TokenizedBitext(bitext, tokenizer, "eng_Latn", "fra_Latn")
        cs_map = build_fast_align_dict_from_raw(
            "test_files/lang1.txt",
            "test_files/lang2.txt",
            tokenizer,
            "eng_Latn",
            "fra_Latn",
        )

        for i, (src, tgt) in enumerate(iter(tokenized_bitext)):
            print(cs_map[i])
            print(src, tgt)
            pairs = extract_phrase_pairs(src, tgt, cs_map[i])

            for i1, i2, j1, j2 in pairs:
                print(
                    "SRC:",
                    " ".join([str(x) for x in src[i1 : i2 + 1]]),
                    "| TGT:",
                    " ".join([str(x) for x in tgt[j1 : j2 + 1]]),
                )


if __name__ == "__main__":
    unittest.main()
