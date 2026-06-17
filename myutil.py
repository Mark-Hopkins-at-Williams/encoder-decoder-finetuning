from configure import USE_CUDA
import gc
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from transformers import AutoTokenizer


def logger(s, to_stderr=False):
    if to_stderr:
        sys.stderr.write(str(s))
        sys.stderr.write("\n")
        sys.stderr.flush()
    else:
        sys.stdout.write(str(s))
        sys.stdout.write("\n")
        sys.stdout.flush()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def what_nllb_token_is_this(token_id, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    return tokenizer.convert_ids_to_tokens([token_id])[0]


def prepare_model_for_finetuning(ft_params):
    if ft_params.should_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(ft_params.base_model)
        print("loaded pretrained model")
    else:
        model_config = AutoConfig.from_pretrained(ft_params.base_model)
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        print("loaded architecture only")
    if hasattr(model.config, "max_length"):  # this should be in a GenerationConfig
        delattr(model.config, "max_length")
    if ft_params.freeze_decoder:
        print("--> DECODER FROZEN <--")
        for param in model.get_decoder().parameters():
            param.requires_grad = False
    else:
        print("--> decoder NOT frozen <--")
    if ft_params.freeze_encoder:
        print("--> ENCODER FROZEN <--")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    else:
        print("--> encoder NOT frozen <--")
    if USE_CUDA:
        torch.cuda.set_device(0)
        model.cuda()
    return model
