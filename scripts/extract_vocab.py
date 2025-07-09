
from pathlib import Path
from transformers import AutoTokenizer 
import math
from typing import List

def sentence_to_words(sentence: str) -> List[str]:
    words = []
    for chunk in sentence.split():
        clean = "".join(ch for ch in chunk if ch.isalnum() or ch == "_" or ch == "-")
        if clean:
            words.append(clean)
    return words

def sub_list(sub, word_list1, word_list2):
    res1 = []
    for word in word_list1:
        if sub in word:
            res1.append(word)
    res2 = []
    for word in word_list2:
        if sub in word:
            res2.append(word)
    return list(set(res1)&set(res2))
        



def is_directly_after(lst, elem1, elem2):
    for i in range(len(lst) - 1):
        if lst[i] == elem1 and lst[i + 1] == elem2:
            return True
    return False


def extract_vocab(filename, lang_code1, lang_code2, filter_num, tokenizer, mode="train", extend_to_full_word=True):
    OUT_DIR = Path(f"../pmi_lang_pairs_data/{filter_num}filtered")
    OUT_DIR.mkdir(exist_ok=True)
    

    pmi_values = []
    token_list = []
    token_pmi_dict = {}

    with open(OUT_DIR / filename, "r", encoding="utf-8") as f:
        for line in f:
            if "PMI:" in line:
                try:
                    token_part, pmi_part = line.strip().split("PMI:")
                    token = token_part.strip()
                    pmi = float(pmi_part.strip())
                    pmi_values.append(pmi)
                    token_list.append(token)
                    token_pmi_dict[token] = pmi
                except ValueError:
                    continue


    candidates = [] #(token1, token2)

    i = 0
    k = 1
    patience = 0.01
    while i < len(pmi_values) - 1:
        j = i+k if i+k<len(pmi_values) else len(pmi_values)-1
        if (pmi_values[i] - pmi_values[j]) < patience and (token_list[j],token_list[i]) not in candidates:
            candidates.append((token_list[i],token_list[j]))
            candidates.append((token_list[j],token_list[i]))
            k += 1
        else:
            k = 1
            i += 1
    print(candidates)
    line_list1 = []
    line_list2 = []

    filename1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code1}"
    filename2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code2}"
    with open(filename1, "r") as f1:
        line_list1 = f1.readlines()

    with open(filename2, "r") as f2:
        line_list2 = f2.readlines()

    print("Read files")
    candidate_analysis = {} #Key: (token1,token2); Value: (pmi_pair_lang1, pmi_pair_lang2)
    lang1_token_counter = {} #Key: (token1,token2); Value: (token1_count_lang1, token2_count_lang1, both_tokens)
    lang2_token_counter = {} #Key: (token1,token2); Value: (token1_count_lang2, token2_count_lang2, both_tokens)
    print(len(line_list2))
    for i in range(len(line_list1)):
        print(i)
        tokenized1 = tokenizer(line_list1[i])
        tokenized2 = tokenizer(line_list2[i])
        for candidate_pair in candidates:
            token1 = tokenizer.convert_tokens_to_ids(candidate_pair[0])
            token2 = tokenizer.convert_tokens_to_ids(candidate_pair[1])
            if (token1,token2) not in lang1_token_counter.keys():
                lang1_token_counter[(token1,token2)] = [0,0,0]
                lang2_token_counter[(token1,token2)] = [0,0,0]
            
            if token1 in tokenized1['input_ids']:
                lang1_token_counter[(token1,token2)][0] += 1
                if token2 in tokenized1['input_ids']:
                    lang1_token_counter[(token1,token2)][1] += 1
                    if is_directly_after(tokenized1['input_ids'], token1, token2):
                        lang1_token_counter[(token1,token2)][2] += 1
            elif token2 in tokenized1['input_ids']:
                lang1_token_counter[(token1,token2)][1] += 1
            
            if token1 in tokenized2['input_ids']:
                lang2_token_counter[(token1,token2)][0] += 1
                if token2 in tokenized2['input_ids']:
                    lang2_token_counter[(token1,token2)][1] += 1
                    if is_directly_after(tokenized2['input_ids'], token1, token2):
                        lang2_token_counter[(token1,token2)][2] += 1
            elif token2 in tokenized2['input_ids']:
                lang2_token_counter[(token1,token2)][1] += 1

            if i == len(line_list1) - 1:
                        token1_count_lang1 = lang1_token_counter[(token1,token2)][0]
                        token2_count_lang1 = lang1_token_counter[(token1,token2)][1]
                        both_tokens_lang1 = lang1_token_counter[(token1,token2)][2]
                        r1 = (both_tokens_lang1*len(line_list1))/(token1_count_lang1*token2_count_lang1) if (token1_count_lang1 != 0 and token2_count_lang1 !=0 and both_tokens_lang1/len(line_list1)>0.00001) else 0
                        candidate_pair_pmi_lang1 = math.log2(r1) if r1 > 0 else 0
                        candidate_analysis[(token1,token2)] = [candidate_pair_pmi_lang1,0]

                        token1_count_lang2 = lang2_token_counter[(token1,token2)][0]
                        token2_count_lang2 = lang2_token_counter[(token1,token2)][1]
                        both_tokens_lang2 = lang2_token_counter[(token1,token2)][2]
                        r2 = (both_tokens_lang2*len(line_list1))/(token1_count_lang2*token2_count_lang2) if (token1_count_lang2 != 0 and token2_count_lang2 !=0 and both_tokens_lang2/len(line_list1)>0.00001) else 0
                        candidate_pair_pmi_lang2 = math.log2(r2) if r2 > 0 else 0
                        candidate_analysis[(token1,token2)][1] += candidate_pair_pmi_lang2
    finalists = []
    
    for candidate, pmi_pair in candidate_analysis.items():
        if (pmi_pair[0]+pmi_pair[1])/2 >= 2:
            finalists.append(tokenizer.convert_ids_to_tokens(candidate[0])+tokenizer.convert_ids_to_tokens(candidate[1]))

    print(finalists)

    finalists_clean = finalists
    
    finalists_clean = [" ".join(((t[1:] if t.startswith("▁") else t).replace("▁", " ")).split()) for t in finalists]


    print("Clean_finalist:", finalists_clean)

    res_freq_dict = {}
    for i in range(len(line_list1)):
        print("second stage",i)
        word_list1 = sentence_to_words(line_list1[i])
        word_list2 = sentence_to_words(line_list2[i])
        for final in finalists_clean:
            word_candidates = sub_list(final, word_list1, word_list2) #common words in 2 lang that contain final as a substring
            if " " in final:
                word_candidates = [final] if final in line_list1[i] and final in line_list2[i] else []
            for word in word_candidates:
                if word not in res_freq_dict:
                    res_freq_dict[word] = 1
                else:
                    res_freq_dict[word] += 1
    
    token_list_clean = token_list
    
    token_list_clean = [" ".join(((t[1:] if t.startswith("▁") else t).replace("▁", " ")).split()) for t in token_list_clean]

    for i in range(len(line_list1)):
        print("third stage",i)
        word_list1 = sentence_to_words(line_list1[i])
        word_list2 = sentence_to_words(line_list2[i])
        for token in token_list_clean:
            word_candidates = sub_list(token, word_list1, word_list2) #common words in 2 lang that contain final as a substring
            if " " in token:
                word_candidates = [token] if final in line_list1[i] and token in line_list2[i] else []
            for word in word_candidates:
                if word not in res_freq_dict:
                    res_freq_dict[word] = 1
                else:
                    res_freq_dict[word] += 1

    print(res_freq_dict)

    final_words = []

    for word, count in res_freq_dict.items():
        if count > 0:
            final_words.append(word)

    return final_words


model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)


print(sentence_to_words("b-N"))
print(sub_list("b N", ["bN"],["bN"]))
print(extract_vocab("es_en_pmi_ranking.txt", "es", "en", 50, tokenizer, "train"))
