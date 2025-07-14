import argparse
import evaluate
import json
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM

from configure import USE_CUDA
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts, load_tokenizer
from permutations import load_permutation_map


def translate(
    src_tokenized,
    tokenizer,
    model,
    tgt_lang,
    permutation=None,
    a=32,
    b=3,
    num_beams=4,
    **kwargs
):
    model.eval()
    result = model.generate(
        **src_tokenized.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * src_tokenized.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs
    )
    result = result.to('cpu')
    if permutation is not None:
        result.apply_(permutation.get_inverse())
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def translate_tokenized_mixture_of_bitexts(mix, model, tokenizer, lang_codes, pmap):         
    if USE_CUDA:
        model.cuda()
    batch = mix.next_batch()  
    translations = dict()
    while batch is not None:
        src, _, src_lang, tgt_lang = batch        
        permutation = pmap[tgt_lang] if tgt_lang in pmap else None
        src_code = lang_codes[src_lang]
        tgt_code = lang_codes[tgt_lang]
        key = '->'.join([src_code, tgt_code])
        if key not in translations:
            translations[key] = []
        translated = translate(src, tokenizer, model, tgt_code, permutation)
        translations[key].extend(translated)
        batch = mix.next_batch() 
    return translations


def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result = bleu_calc.compute(predictions = candidate_translations, references = reference_translations)
    chrf_result = chrf_calc.compute(predictions = candidate_translations, references = reference_translations)
    return {'bleu': round(bleu_result["score"], 3), 'chrf': round(chrf_result["score"], 3)}


def main():
    test_mix = MixtureOfBitexts.create_from_files(
        {
            "fra_Latn": "data/test.fr",
            "eng_Latn": "data/test.en",
        },
        [("eng_Latn", "fra_Latn")],
        batch_size=32,
        only_once_thru=True
    )           
    references = dict()
    batch = test_mix.next_batch()
    while batch is not None:
        _, tgt, src_code, tgt_code = batch        
        if (src_code, tgt_code) not in references:
            references[(src_code, tgt_code)] = []        
        references[(src_code, tgt_code)].extend(tgt)
        batch = test_mix.next_batch()  
    
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained NLLB model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Directory to save finetuned model"
    )
    args = parser.parse_args()

    with open(args.config) as reader:
        config = json.load(reader)

    all_corpora = config["corpora"]
    test_corpora = {key: all_corpora[key]["test"] for key in all_corpora}
    params = config["finetuning_parameters"]
    devtest_bitexts = [(b["src"], b["tgt"], None) for b in config["bitexts"]]
    
    # Create unique model directory
    model_dir = config["model_dir"]
    base_model = params["base_model"]
    tokenizer = load_tokenizer(base_model)

    pmap = load_permutation_map(Path(model_dir) / "permutations.json")
    
    test_data = MixtureOfBitexts.create_from_files(
        test_corpora, devtest_bitexts, batch_size=params["batch_size"],
        only_once_thru=True
    )
    tokenized_test = TokenizedMixtureOfBitexts(test_data, tokenizer, max_length=128)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
    translations = translate_tokenized_mixture_of_bitexts(
        tokenized_test, model, tokenizer, pmap
    )
    with open(Path(model_dir) / "translations.json", "w") as writer:
        json.dump(translations, writer)
    print("Translations complete.")

    test_data = MixtureOfBitexts.create_from_files(
        test_corpora, devtest_bitexts, batch_size=params["batch_size"],
        only_once_thru=True
    )

    references = dict()
    batch = test_data.next_batch()
    while batch is not None:
        _, tgt, src_code, tgt_code = batch
        key = "->".join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        references[key].extend(tgt)
        batch = test_data.next_batch()
    with open(Path(model_dir) / "references.json", "w") as writer:
        json.dump(references, writer)
    print("References complete.")

    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open(Path(model_dir) / "scores.json", "w") as writer:
        json.dump(scores, writer)
    print("Evaluation complete.")
    
if __name__ == "__main__":
    main()