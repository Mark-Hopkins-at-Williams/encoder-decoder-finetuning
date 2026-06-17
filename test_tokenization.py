import unittest
from tokenization import NllbTokenizer, ByteTokenizer
from torch import tensor


class TestTokenization(unittest.TestCase):

    def test_nllb_tokenizer1(self):
        with open("test_files/lang1.txt") as reader:
            lines = [line.strip() for line in reader.readlines()]
        tokenizer = NllbTokenizer("600M")
        tokens = tokenizer(lines[:3], lang_code="eng_Latn")
        expected_lang1_token_ids = [
            [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
            [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 1],
            [256047, 13710, 18379, 43583, 2299, 248075, 2, 1, 1],
        ]
        self.assertEqual(tokens, expected_lang1_token_ids)

    def test_nllb_tokenizer2(self):
        with open("test_files/lang1.txt") as reader:
            lines = [line.strip() for line in reader.readlines()]
        tokenizer = NllbTokenizer("600M", max_length=8)
        tokens = tokenizer(lines[:3], lang_code="eng_Latn")
        expected_lang1_token_ids = [
            [256047, 1617, 7875, 228, 55501, 349, 227879, 2],
            [256047, 11873, 272, 22665, 9, 28487, 248075, 2],
            [256047, 13710, 18379, 43583, 2299, 248075, 2, 1],
        ]
        self.assertEqual(tokens, expected_lang1_token_ids)

    def test_byte_tokenizer1(self):
        with open("test_files/lang1.txt") as reader:
            lines = [line.strip() for line in reader.readlines()]
        tokenizer = ByteTokenizer()
        tokens = tokenizer(lines[0], lang_code="eng_Latn")
        expected = [
            257,
            84,
            104,
            101,
            32,
            99,
            97,
            116,
            32,
            99,
            104,
            97,
            115,
            101,
            100,
            32,
            116,
            104,
            101,
            32,
            109,
            111,
            117,
            115,
            101,
            46,
            256,
        ]
        self.assertEqual(tokens, expected)

    def test_byte_tokenizer2(self):
        with open("test_files/lang1.txt") as reader:
            lines = [line.strip() for line in reader.readlines()]
        tokenizer = ByteTokenizer(max_length=8)
        tokens = tokenizer(lines[0], lang_code="fra_Latn")
        expected = [258, 84, 104, 101, 32, 99, 97, 256]
        self.assertEqual(tokens, expected)

    def test_hf_tokenizer_properties(self):
        tokenizer = NllbTokenizer("1.3B", max_length=8)
        self.assertEqual(len(tokenizer), 256204)
        special_tokens = tokenizer.get_special_tokens()
        self.assertEqual(len(special_tokens), 207)
        self.assertEqual(special_tokens["<s>"], 0)
        self.assertEqual(special_tokens["<pad>"], 1)
        self.assertEqual(special_tokens["</s>"], 2)
        self.assertEqual(special_tokens["<unk>"], 3)
        self.assertEqual(special_tokens["<mask>"], 256203)

    def test_autocompleting_tokenizer1(self):
        with open("test_files/lang1.txt") as reader:
            lines = [line.strip() for line in reader.readlines()]
        tokenizer = NllbTokenizer("600M")
        tokens = tokenizer(lines[0], lang_code="eng_Latn")
        expected = [
            [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
            [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 1],
            [256047, 13710, 18379, 43583, 2299, 248075, 2, 1, 1],
        ]
        self.assertEqual(tokens, expected_lang1_token_ids)


if __name__ == "__main__":
    unittest.main()
