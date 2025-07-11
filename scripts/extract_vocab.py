
from pathlib import Path
from transformers import AutoTokenizer 
import math
from typing import List
from tqdm import tqdm
import subprocess, tempfile, os, pathlib
from typing import Dict, List, Optional
from pathlib import Path
import re
import subprocess, tempfile, os, pathlib
from typing import List, Dict, Optional


def extract_phrase_pairs(f_words, e_words, alignments, max_phrase_len=7):
    """
    Extracts consistent phrase pairs from a word-aligned sentence pair.

    Args:
        f_words (List[str]): Foreign sentence tokens
        e_words (List[str]): English sentence tokens
        alignments (Set[Tuple[int, int]]): Word alignment points as (f_index, e_index)
        max_phrase_len (int): Optional maximum phrase length

    Returns:
        Set[Tuple[Tuple[str], Tuple[str]]]: Set of phrase pairs
    """
    phrase_pairs = set()

    for i_start in range(len(f_words)):
        for i_end in range(i_start, min(i_start + max_phrase_len, len(f_words))):

            # Initialize e_span to bounds of aligned e-words
            e_start, e_end = len(e_words), -1
            for (f_idx, e_idx) in alignments:
                if i_start <= f_idx <= i_end:
                    e_start = min(e_start, e_idx)
                    e_end = max(e_end, e_idx)

            if e_end == -1:
                continue  # No alignments in this span

            # Check consistency: no alignment point to e_span from outside f_span
            inconsistent = False
            for (f_idx, e_idx) in alignments:
                if not (i_start <= f_idx <= i_end) and (e_start <= e_idx <= e_end):
                    inconsistent = True
                    break
            if inconsistent:
                continue

            # Loop over possible subphrases in e_span
            for k_start in range(e_start, e_end + 1):
                for k_end in range(k_start, min(k_start + max_phrase_len, e_end + 1)):

                    # Check reverse consistency: no f-word outside i_span aligns to e_subspan
                    consistent = True
                    for (f_idx, e_idx) in alignments:
                        if not (k_start <= e_idx <= k_end) and (i_start <= f_idx <= i_end):
                            continue
                        if not (i_start <= f_idx <= i_end) and (k_start <= e_idx <= k_end):
                            consistent = False
                            break
                    if not consistent:
                        continue

                    f_phrase = tuple(f_words[i_start:i_end+1])
                    e_phrase = tuple(e_words[k_start:k_end+1])
                    phrase_pairs.add((f_phrase, e_phrase))

    return phrase_pairs



def align_parallel_corpus(
    filepath: str,
    fast_align_bin: str = "/mnt/storage/sotnichenko/fast_align/build/fast_align",
    flags: Optional[List[str]] = None
    ) -> List[Dict[int, int]]:
    if flags is None:
        flags = ["-d", "-o"]

    # 1) make sure file really ends with \n (fast_align needs it)
    with open(filepath, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")

    # 2) run fast_align
    proc = subprocess.run(
        [fast_align_bin, *flags, "-i", filepath],
        capture_output=True, text=True
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr)

    # 3) build list of dicts
    out: List[Dict[int, int]] = []
    for line in proc.stdout.rstrip().split("\n"):
        links = {}
        for pair in line.split():
            if "-" in pair:
                s, t = map(int, pair.split("-"))
                links[s] = t
        out.append(links)
    return out

# Make a tokenized bilingual text file that fast_align can be used on
def word_alignment_textfile_generator(lang_code1, lang_code2, mode="train"):
    OUT_DIR = Path(f"word_alignment_approach_data")
    OUT_DIR.mkdir(exist_ok=True)
    #read the corpora text file of given language pair, and turn them into a list
    lang1_list = []
    lang2_list = []
    filename1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code1}"
    filename2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code2}"
    with open(filename1, "r") as f1:
        lang1_list = f1.readlines()
    with open(filename2, "r") as f2:
        lang2_list = f2.readlines()
    #create a new file, and add the tokenized version of each language in fast_align friendly form
    with open(OUT_DIR / f"{mode}.{lang_code1}-{lang_code2}", "w") as f:
        for line_num in range(len(lang1_list)):
            tokenized_lang1_list = re.findall(r"\w+|[^\w\s]", lang1_list[line_num])
            tokenized_lang2_list = re.findall(r"\w+|[^\w\s]", lang2_list[line_num])
            tokenized_lang1 = ' '.join(tokenized_lang1_list)
            tokenized_lang2 = ' '.join(tokenized_lang2_list)
            
            f.write(tokenized_lang1)
            f.write(" ||| ")
            f.write(tokenized_lang2 + "\n")

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

# PMI calculations
def extract_vocab_version0(filename, lang_code1, lang_code2, filter_num, tokenizer, mode="train"):
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

    return finalists_clean

# PMI calcuations + Full word extensions 
def extract_vocab_version1(filename, lang_code1, lang_code2, filter_num, tokenizer, mode="train"):
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

# Full word extensions of exisiting tokens - fastest most, effective. Add cutoff threshold for the freq dictionary?
def extract_vocab_version2(filename, lang_code1, lang_code2, filter_num, tokenizer, mode="train"):
    OUT_DIR = Path(f"../pmi_lang_pairs_data/{filter_num}filtered")
    
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

    line_list1 = []
    line_list2 = []

    filename1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code1}"
    filename2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code2}"
    with open(filename1, "r") as f1:
        line_list1 = f1.readlines()

    with open(filename2, "r") as f2:
        line_list2 = f2.readlines()

    res_freq_dict = {}
    
    token_list_clean = token_list
    
    token_list_clean = [" ".join(((t[1:] if t.startswith("▁") else t).replace("▁", " ")).split()) for t in token_list_clean]

    for i in range(len(line_list1)):
        print(i)
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

# Using word allignment package
def extract_vocab_version3( lang_code1, lang_code2, mode="train"):
    word_alignment_textfile_generator(lang_code1, lang_code2, mode)
    file_path = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/word_alignment_approach_data/{mode}.{lang_code1}-{lang_code2}"
    alignments = align_parallel_corpus(file_path)

    line_list1 = []
    line_list2 = []
    
    with open(file_path, "r") as f1:
        line_list = f1.readlines()
        for line in tqdm(line_list):
            src, _, tgt = line.partition(" ||| ")
            line_list1.append(src)
            line_list2.append(tgt)


    phrase_list = []
    for i in tqdm(range(len(line_list1))):
        word_list1 = line_list1[i].split(" ")
        word_list2 = line_list2[i].split(" ")
        current_dict = alignments[i]
        set_of_tuples = set()
        for src,tgt in current_dict.items():
            set_of_tuples.add((src,tgt))
        phrase_tuple_list = list(extract_phrase_pairs(word_list1, word_list2,set_of_tuples))
        #phrase_tuple_list = [phrase_tuple_list[43]]
        
        for pair in phrase_tuple_list:
            sent1 = " ".join(list(pair[0]))
            sent2 = " ".join(list(pair[1]))
            phrase_list.append((sent1,sent2))
    result = []
    for phrase_pair in phrase_list:
        (sent1, sent2) = phrase_pair
        if sent1 == sent2 :
            result.append(sent1)
    res = list(set(result))

    OUT_DIR = Path(f"common_vocab")
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / f"{mode}.{lang_code1}_{lang_code2}_common_vocab", "w") as f:
        f.write(" || ".join(res))
    return list(set(result))


model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#print(extract_vocab_version3("es_en_pmi_ranking.txt", "es", "en", 50, tokenizer, "train"))

if __name__ == "__main__":
    print(extract_vocab_version3( "en", "es", "dev"))
    
        

