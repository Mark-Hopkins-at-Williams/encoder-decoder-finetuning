import evaluate
import matplotlib
matplotlib.use("Agg")

from configure import USE_CUDA



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
        forced_bos_token_id=tokenizer.get_special_tokens()[tgt_lang],
        max_new_tokens=int(a + b * src_tokenized.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs
    )
    result = result.to('cpu')
    if permutation is not None:
        result.apply_(permutation.get_inverse())
    return tokenizer.batch_decode(result)


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

