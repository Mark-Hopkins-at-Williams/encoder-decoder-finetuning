import sys
from transformers import AutoTokenizer
from typing import Dict, Tuple, List, Optional, Iterator, Callable
import warnings
from abc import ABC
from abc import abstractmethod



class Tokenizer(ABC):
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __call__(self, sents: List[str]):  
        pass
    
    @abstractmethod
    def get_special_tokens(self):
        pass
    
    @abstractmethod
    def batch_decode(self):
        pass

class HuggingfaceTokenizer(Tokenizer):
    
    def __init__(self, model_name, max_length=None):
        self.max_length = max_length        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`clean_up_tokenization_spaces` was not set.*",
                category=FutureWarning,
                module="transformers.tokenization_utils_base",
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except OSError:
                sys.stderr.write('Tokenizer not found. Using NLLB tokenizer instead.\n')
                sys.stderr.flush()
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.special_tokens = dict(zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids))
        
    def __len__(self):
        return len(self.tokenizer)
    
    def __call__(self, sents: List[str], lang_code=None):        
        if lang_code is not None:
            self.tokenizer.src_lang = lang_code
        return self.tokenizer(
            sents, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length if self.max_length is not None else None
        )
        
    def get_special_tokens(self):
        return self.special_tokens
    
    def batch_decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    
class NllbTokenizer(HuggingfaceTokenizer):
    def __init__(self, size, max_length=None):
        super().__init__(f"facebook/nllb-200-distilled-{size}", max_length=max_length)
        