import random
from typing import Dict, Tuple, List, Optional, Iterator, Callable
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
import warnings

def load_tokenizer(model_name: str):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`clean_up_tokenization_spaces` was not set.*",
            category=FutureWarning,
            module="transformers.tokenization_utils_base",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


class Bitext(IterableDataset):
    def __init__(self, lang1_file: str, lang2_file: str):
        self.lang1_file = lang1_file
        self.lang2_file = lang2_file

    def line_streamer(self, file_path: str) -> Iterator[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(
            self.line_streamer(self.lang1_file), self.line_streamer(self.lang2_file)
        )


class MixtureOfBitexts:
    def __init__(
        self,
        bitexts: Dict[Tuple[str, str], Bitext],
        batch_size: int,
        sampling_probs: Optional[List[float]] = None,
        only_once_thru: bool = False
    ):
        self.bitexts = bitexts
        self.keys = list(bitexts)
        self.batch_size = batch_size
        self.batch_iters = {}

        for key in self.keys:
            self.batch_iters[key] = self._create_iterator(key)

        total = sum(sampling_probs) if sampling_probs else len(bitexts)
        self.sampling_probs = [
            p / total for p in (sampling_probs or [1.0] * len(bitexts))
        ]
        
        self.only_once_thru = only_once_thru
        self.completed_bitexts = set()

    def _create_iterator(
        self, key: Tuple[str, str]
    ) -> Iterator[Tuple[List[str], List[str]]]:
        return iter(
            DataLoader(
                self.bitexts[key],
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
            )
        )

    def next_batch(self) -> Optional[Tuple[List[str], List[str], str, str]]:
        still_choosing = True
        while still_choosing and len(self.completed_bitexts) < len(self.keys):
            lang_pair = random.choices(self.keys, weights=self.sampling_probs, k=1)[0]
            try:
                lang1_sents, lang2_sents = next(self.batch_iters[lang_pair])
                still_choosing = False
            except StopIteration:
                if self.only_once_thru:
                    self.completed_bitexts.add(lang_pair)
                else: # start a new iterator for the chosen bitext
                    self.batch_iters[lang_pair] = self._create_iterator(lang_pair)                    
        if still_choosing:
            return None
        else:
            return lang1_sents, lang2_sents, lang_pair[0], lang_pair[1]

    @staticmethod
    def create_from_files(
        text_files: Dict[str, str],
        lps: List[Tuple[str, str]],
        batch_size: int,
        sampling_probs: Optional[List[float]] = None,
        only_once_thru: bool = False
    ) -> "MixtureOfBitexts":
        bitexts = {(l1, l2): Bitext(text_files[l1], text_files[l2]) for (l1, l2) in lps}
        return MixtureOfBitexts(bitexts, batch_size, sampling_probs, only_once_thru)

    def get_language_codes(self) -> List[str]:
        return sorted({code for pair in self.keys for code in pair})


class TokenizedMixtureOfBitexts:
    def __init__(
        self,
        mixture_of_bitexts: MixtureOfBitexts,
        tokenizer: AutoTokenizer,
        max_length: int,
        permutation_map: Dict[str, Callable[[int], int]] = dict()
    ):
        self.mixture_of_bitexts = mixture_of_bitexts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.permutation_map = permutation_map

    def _tokenize(self, sents: List[str], lang: str, alt_pad_token: int = None):
        self.tokenizer.src_lang = lang
        tokens = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        if alt_pad_token is not None:
            tokens.input_ids[tokens.input_ids == self.tokenizer.pad_token_id] = (
                alt_pad_token
            )
        if lang in self.permutation_map: # apply the permutation
            p = self.permutation_map[lang]
            tokens.input_ids.apply_(p) # modifies in-place
        return tokens

    def next_batch(self):
        batch = self.mixture_of_bitexts.next_batch()
        if batch is None:
            return None
        lang1_sents, lang2_sents, lang1_code, lang2_code = batch
        lang1_tokenized = self._tokenize(lang1_sents, lang1_code)
        lang2_tokenized = self._tokenize(lang2_sents, lang2_code, alt_pad_token=-100)
        return lang1_tokenized, lang2_tokenized, lang1_code, lang2_code
