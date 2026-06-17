import argparse
from configure import create_bitexts_from_experiment_dir
from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import USE_CUDA
from corpora import MixtureOfBitexts
import evaluate
import json
from myutil import logger
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM


def translate(
    src_tokenized,
    tgt_tokenizer,
    model,
    tgt_lang,
    permutation=None,
    a=32,
    b=3,
    num_beams=4,
    **kwargs,
):
    model.eval()
    src_tokenized = {k: v.to(model.device) for k, v in src_tokenized.items()}
    result = model.generate(
        **src_tokenized,
        forced_bos_token_id=3,  # tgt_tokenizer.get_special_tokens()[tgt_lang],
        max_new_tokens=int(a + b * src_tokenized["input_ids"].shape[1]),
        num_beams=num_beams,
        **kwargs,
    )
    result = result.to("cpu")
    if permutation is not None:
        result.apply_(permutation.get_inverse())
    return tgt_tokenizer.batch_decode(result)


def translate_tokenized_mixture_of_bitexts(mix, model, tokenizer_map, cipher_map):
    if USE_CUDA:
        model.cuda()
    translations = dict()
    for batch in mix:
        src, _, metadata = batch
        cipher = (
            cipher_map[metadata["lang2_tokenizer"], metadata["lang2_encipherment"]]
            if metadata["lang2_encipherment"] != "0"
            else None
        )
        src_code = metadata["lang1_code"]
        tgt_code = metadata["lang2_code"]
        key = "->".join([src_code, tgt_code])
        if key not in translations:
            translations[key] = []
        translated = translate(
            src,
            tokenizer_map[metadata["lang2_tokenizer"]],
            model,
            tgt_code,
            cipher,
        )
        translations[key].extend(translated)
        logger(f"translation: {translated[0]}")
    return translations


def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result = bleu_calc.compute(
        predictions=candidate_translations, references=reference_translations
    )
    chrf_result = chrf_calc.compute(
        predictions=candidate_translations, references=reference_translations
    )
    return {
        "bleu": round(bleu_result["score"], 3),
        "chrf": round(chrf_result["score"], 3),
    }


def evaluate_experiment(experiment_dir):
    logger(f"Initializing model from: {experiment_dir}")
    bitexts = create_bitexts_from_experiment_dir(experiment_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(experiment_dir)
    if USE_CUDA:
        model.cuda()

    logger(f"Collating reference translations")
    references = dict()
    test_data = bitexts["test"]
    test_data.restart()
    tokenizer_map = bitexts["tokenizer_map"]
    cipher_map = bitexts["cipher_map"]
    for _, tgt, metadata in test_data:
        src_code = metadata["lang1_code"]
        tgt_code = metadata["lang2_code"]
        tgt_tokenizer = tokenizer_map[metadata["lang2_tokenizer"]]
        key = "->".join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        tgt_ids = tgt["input_ids"]
        tgt_ids[tgt_ids == -100] = 2  # TODO: make more general
        cipher = (
            cipher_map[metadata["lang2_tokenizer"], metadata["lang2_encipherment"]]
            if metadata["lang2_encipherment"] != "0"
            else None
        )
        if cipher is not None:
            tgt_ids.apply_(cipher.get_inverse())
        tgt = tgt_tokenizer.batch_decode(tgt_ids)
        references[key].extend(tgt)
    with open(Path(experiment_dir) / "references.json", "w") as writer:
        json.dump(references, writer)
    logger("...references complete.")

    logger(f"Translating test data")
    test_data.restart()
    translations = translate_tokenized_mixture_of_bitexts(
        test_data, model, tokenizer_map, bitexts["cipher_map"]
    )
    with open(Path(experiment_dir) / "translations.json", "w") as writer:
        json.dump(translations, writer)
    logger("...translation complete.")

    logger(f"Scoring translations")
    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open(Path(experiment_dir) / "scores.json", "w") as writer:
        json.dump(scores, writer)
    logger("...scoring complete.")


# TODO: update
def evaluate_model(model_name, config_file):
    with open(config_file) as reader:
        config = json.load(reader)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_CUDA:
        model.cuda()
    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)
    pmap = dict()
    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)
    tokenized_test = TokenizedMixtureOfBitexts(
        test_data, tokenizer, lang_codes=lang_codes, permutation_map=pmap
    )
    logger(f"Translating test data")
    translations = translate_tokenized_mixture_of_bitexts(
        tokenized_test, model, tokenizer, lang_codes, pmap
    )
    with open("translations.json", "w") as writer:
        json.dump(translations, writer)
    logger("...translation complete.")
    logger(f"Collating reference translations")
    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)
    references = dict()
    batch = test_data.next_batch()
    while batch is not None:
        _, tgt, src_lang, tgt_lang = batch
        src_code = lang_codes[src_lang]
        tgt_code = lang_codes[tgt_lang]
        key = "->".join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        references[key].extend(tgt)
        batch = test_data.next_batch()
    with open("references.json", "w") as writer:
        json.dump(references, writer)
    logger("...references complete.")
    logger(f"Scoring translations")
    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open("scores.json", "w") as writer:
        json.dump(scores, writer)
    logger("...scoring complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate finetuning experiment.")
    parser.add_argument("--dir", type=str, required=True, help="Experiment directory.")
    args = parser.parse_args()
    evaluate_experiment(args.dir)
    # evaluate_model(
    #    "facebook/nllb-200-distilled-600M", "examples/nllb_seed_config_small.json"
    # )
