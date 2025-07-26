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
import string
from pathlib import Path
from tqdm import tqdm
import regex
def is_punctuation(word):
    return word in string.punctuation



def extract_phrase_pairs111(f_words, e_words, alignments, max_phrase_len=5):
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

def extract_phrase_pairs(f_words, e_words, alignments, max_phrase_len=9):
    phrase_pairs = set()
    f_aligned = {i for (i, j) in alignments}
    e_aligned = {j for (i, j) in alignments}

    for i_start in range(len(f_words)):
        for i_end in range(i_start, min(i_start + max_phrase_len, len(f_words))):
            # Foreign phrase must contain at least one aligned (non-punctuation) word
            if not any(i in f_aligned for i in range(i_start, i_end + 1)):
                continue

            # Find e_span aligned with this f_span
            e_start, e_end = len(e_words), -1
            for (f_idx, e_idx) in alignments:
                if i_start <= f_idx <= i_end:
                    e_start = min(e_start, e_idx)
                    e_end = max(e_end, e_idx)

            if e_end == -1:
                continue  # No alignments inside this foreign span

            # Forward consistency: make sure no e_idx inside e_span aligns to f_idx outside f_span
            if any((f_idx < i_start or f_idx > i_end) and (e_start <= e_idx <= e_end)
                   for (f_idx, e_idx) in alignments):
                continue

            for k_start in range(e_start, e_end + 1):
                for k_end in range(k_start, min(k_start + max_phrase_len, e_end + 1)):
                    # English phrase must contain at least one aligned word
                    if not any(j in e_aligned for j in range(k_start, k_end + 1)):
                        continue

                    # Reverse consistency: make sure no f_idx inside f_span aligns to e_idx outside e_span
                    if any((e_idx < k_start or e_idx > k_end) and (i_start <= f_idx <= i_end)
                           for (f_idx, e_idx) in alignments):
                        continue

                    f_phrase = tuple(f_words[i_start:i_end + 1])
                    e_phrase = tuple(e_words[k_start:k_end + 1])

                    # Heuristic cleanup: skip garbage phrases
                    if not any(char.isalnum() for word in f_phrase for char in word):
                        continue
                    if not any(char.isalnum() for word in e_phrase for char in word):
                        continue
                    if all(is_punctuation(w) for w in f_phrase + e_phrase):
                        continue

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

def word_alignment_textfile_generator(lang_code1, lang_code2, num_lines, mode="train"):
    OUT_DIR = Path(f"word_alignment_approach_data/{mode}")
    OUT_DIR.mkdir(exist_ok=True)

    filename1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code1}"
    filename2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code2}"

    with open(filename1, "r", encoding="utf-8") as f1:
        lang1_list = f1.readlines()
    with open(filename2, "r", encoding="utf-8") as f2:
        lang2_list = f2.readlines()
    print(len(lang1_list), num_lines)
    output_file = OUT_DIR / f"{mode}.{lang_code1}-{lang_code2}_{num_lines}"
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_lines,2*num_lines):
            j = i % len(lang1_list)
            l1 = lang1_list[j].strip()
            l2 = lang2_list[j].strip()
            if i %200 == 0:
                print(i)

            # Unicode-aware tokenization: \p{L} = letter, \p{N} = number, \p{P} = punctuation
            tokens1 = regex.findall(r"\p{L}+\p{M}*|\p{N}+|\p{P}", l1)
            tokens2 = regex.findall(r"\p{L}+\p{M}*|\p{N}+|\p{P}", l2)

            tokenized_lang1 = ' '.join(tokens1) if tokens1 else l1
            tokenized_lang2 = ' '.join(tokens2) if tokens2 else l2
            if tokenized_lang1 == "":
                tokenized_lang1 = "EMPTY"
            if tokenized_lang2 == "":
                tokenized_lang2 = "EMPTY"

            f.write(f"{tokenized_lang1} ||| {tokenized_lang2}\n")

def word_alignment_textfile_generator_tokenizer(lang_code1, lang_code2, tokenizer, mode="train"):
    OUT_DIR = Path(f"word_alignment_approach_data/{mode}_tokenized")
    OUT_DIR.mkdir(exist_ok=True)

    filename1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code1}"
    filename2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code2}"

    with open(filename1, "r", encoding="utf-8") as f1:
        lang1_list = f1.readlines()
    with open(filename2, "r", encoding="utf-8") as f2:
        lang2_list = f2.readlines()

    output_file = OUT_DIR / f"{mode}_tokenized.{lang_code1}-{lang_code2}"
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(lang1_list)):
            l1 = lang1_list[i].strip()
            l2 = lang2_list[i].strip()

            all_special_tokens = set(tokenizer.all_special_tokens)

            tokens1 = [
                token for token in tokenizer.convert_ids_to_tokens(tokenizer(l1)["input_ids"])
                if token not in all_special_tokens
            ]

            tokens2 = [
                token for token in tokenizer.convert_ids_to_tokens(tokenizer(l2)["input_ids"])
                if token not in all_special_tokens
            ]

            tokenized_lang1 = ' '.join(tokens1) if tokens1 else l1
            tokenized_lang2 = ' '.join(tokens2) if tokens2 else l2
            if tokenized_lang1 == "":
                continue
            if tokenized_lang2 == "":
                continue

            f.write(f"{tokenized_lang1} ||| {tokenized_lang2}\n")

def word_alignment_textfile_generator_tokenizer_range(lang_code1, lang_code2, tokenizer, num_lines, mode="train"):
    OUT_DIR = Path(f"word_alignment_approach_data/{mode}_tokenized")
    OUT_DIR.mkdir(exist_ok=True)

    filename1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code1}"
    filename2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/{mode}.{lang_code2}"

    with open(filename1, "r", encoding="utf-8") as f1:
        lang1_list = f1.readlines()
    with open(filename2, "r", encoding="utf-8") as f2:
        lang2_list = f2.readlines()

    output_file = OUT_DIR / f"{mode}_tokenized_{num_lines}.{lang_code1}-{lang_code2}"
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_lines,2*num_lines):
            j = i % len(lang1_list)
            l1 = lang1_list[j].strip()
            l2 = lang2_list[j].strip()

            all_special_tokens = set(tokenizer.all_special_tokens)

            tokens1 = [
                token for token in tokenizer.convert_ids_to_tokens(tokenizer(l1)["input_ids"])
                if token not in all_special_tokens
            ]

            tokens2 = [
                token for token in tokenizer.convert_ids_to_tokens(tokenizer(l2)["input_ids"])
                if token not in all_special_tokens
            ]

            tokenized_lang1 = ' '.join(tokens1) if tokens1 else l1
            tokenized_lang2 = ' '.join(tokens2) if tokens2 else l2
            if tokenized_lang1 == "":
                tokenized_lang1 = "EMPTY"
            if tokenized_lang2 == "":
                tokenized_lang2 = "EMPTY"
            f.write(f"{tokenized_lang1} ||| {tokenized_lang2}\n")

def extract_vocab_version3(lang_list, mode="train"):
    count = 0
    n = 0.5*len(lang_list)*(len(lang_list)-1)
    for j in range(len(lang_list)):
        for k in range (j+1,len(lang_list)):
            lang_code1 = lang_list[j]
            lang_code2 = lang_list[k]
            print(f"Strating extraction for {lang_code1} and {lang_code2}")
            file_path = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/word_alignment_approach_data/{mode}/{mode}.{lang_code1}-{lang_code2}"
            if Path(file_path).exists():
                count += 1
                print(f"Sucess! {count}/{n} completed!")
                continue
            word_alignment_textfile_generator(lang_code1, lang_code2, mode)
            
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

            OUT_DIR = Path(f"common_vocab/{mode}")
            OUT_DIR.mkdir(exist_ok=True)
            with open(OUT_DIR / f"{mode}.{lang_code1}_{lang_code2}_common_vocab", "w") as f:
                f.write("\n".join(res))
            count += 1
        
            print(f"Success! {count}/{int(n)} completed!")

def extract_vocab_version3_tokenized(lang_list, tokenizer, num_lines, mode="train"):
    count = 0
    n = 0.5*len(lang_list)*(len(lang_list)-1)
    for j in range(len(lang_list)):
        for k in range (j+1,len(lang_list)):
            lang_code1 = lang_list[j]
            lang_code2 = lang_list[k]
            print(f"Strating extraction for {lang_code1} and {lang_code2}")
            file_path = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/word_alignment_approach_data/{mode}_tokenized/{mode}_tokenized.{lang_code1}-{lang_code2}"
            if Path(f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/train_tokenized/{mode}.{lang_code1}_{lang_code2}_common_vocab").exists():
                count += 1
                print(f"Success! {count}/{n} completed!")
                continue
            word_alignment_textfile_generator_tokenizer(lang_code1, lang_code2, tokenizer, mode)
            
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
            for i in tqdm(range(num_lines,2*num_lines)):
                word_list1 = line_list1[i].split(" ")
                word_list2 = line_list2[i].split(" ")
                current_dict = alignments[i]
                set_of_tuples = set()
                for src,tgt in current_dict.items():
                    set_of_tuples.add((src,tgt))
                phrase_tuple_list = list(extract_phrase_pairs(word_list1, word_list2, set_of_tuples))

                
                
                for pair in phrase_tuple_list:
                    sent1 = "".join(list(pair[0])).replace("▁", " ").strip()
                    sent2 = "".join(list(pair[1])).replace("▁", " ").strip()

                    phrase_list.append((sent1,sent2))
            result = []
            for phrase_pair in phrase_list:
                (sent1, sent2) = phrase_pair
                if sent1 == sent2 :
                    result.append(sent1)
            res = list(set(result))

            OUT_DIR = Path(f"common_vocab/{mode}_tokenized")
            OUT_DIR.mkdir(exist_ok=True)
            with open(OUT_DIR / f"{mode}.{lang_code1}_{lang_code2}_common_vocab", "w") as f:
                f.write("\n".join(res))
            count += 1
        
            print(f"Success! {count}/{int(n)} completed!")

def extract_vocab_version3_tokenized_with_translations(tokenizer, num_lines, lang_code, lang_code1, lang_code2, mode="train"):
    
    print(f"Strating extraction for {lang_code1} and {lang_code2}:{num_lines},{mode}")
    file_path = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/word_alignment_approach_data/{mode}/{mode}.{lang_code1}-{lang_code2}_{num_lines}"
    file_path1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/word_alignment_approach_data/{mode}/{mode}.{lang_code}-{lang_code1}_{num_lines}"
    file_path2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/word_alignment_approach_data/{mode}/{mode}.{lang_code}-{lang_code2}_{num_lines}"
   
    #word_alignment_textfile_generator_tokenizer_range(lang_code, lang_code1, tokenizer, num_lines, mode)
    #word_alignment_textfile_generator_tokenizer_range(lang_code, lang_code2, tokenizer, num_lines, mode)
    #word_alignment_textfile_generator_tokenizer_range(lang_code1, lang_code2, tokenizer, num_lines, mode)
    word_alignment_textfile_generator(lang_code, lang_code1, num_lines, mode)
    word_alignment_textfile_generator(lang_code, lang_code2, num_lines, mode)
    word_alignment_textfile_generator(lang_code1, lang_code2, num_lines, mode)
            
    alignments = align_parallel_corpus(file_path) # e.g. es - pt
    alignments1 = align_parallel_corpus(file_path1) # e.g. en - es
    alignments2 = align_parallel_corpus(file_path2) #e.g. en - pt

    line_list = [] # en
    line_list1 = [] # es
    line_list2 = [] # pt
    with open(file_path1, "r") as f1:
        line_listt = f1.readlines()
        for line in tqdm(line_listt):
            src, _, tgt = line.partition(" ||| ")
            line_list.append(src)
            line_list1.append(tgt)
    with open(file_path2, "r") as f1:
        line_listt = f1.readlines()
        for line in tqdm(line_listt):
            src, _, tgt = line.partition(" ||| ")
            line_list2.append(tgt)
    
    res = []
    res12 = []
    for i in tqdm(range(num_lines)):
        
        phrase_list = []
        word_list1 = line_list1[i].split(" ")
        word_list2 = line_list2[i].split(" ")
        word_list = line_list[i].split(" ")

        current_dict = alignments[i]
        set_of_tuples = set()
        for src,tgt in current_dict.items():
            set_of_tuples.add((src,tgt))
        phrase_tuple_list = list(extract_phrase_pairs(word_list1, word_list2, set_of_tuples))

        for pair in phrase_tuple_list:
            sent1 = " ".join(list(pair[0])).replace("▁", " ").strip()
            sent2 = " ".join(list(pair[1])).replace("▁", " ").strip()
            phrase_list.append((sent1,sent2))

        result = []
        for phrase_pair in phrase_list:
            (sent1, sent2) = phrase_pair
            if sent1 == sent2 and sent1!= "EMPTY":
                result.append(sent1)
        pre_res = list(set(result)) # Candidates common phrases in es-pt

        # Now check if the transaltion into english is the same 
        current_dict1 = alignments1[i]
        current_dict2 = alignments2[i]

        set_of_tuples1 = set()
        for src,tgt in current_dict1.items():
            set_of_tuples1.add((src,tgt))
        phrase_tuple_list1 = list(extract_phrase_pairs(word_list, word_list1, set_of_tuples1)) # (en,es)

        set_of_tuples2 = set()
        for src,tgt in current_dict2.items():
            set_of_tuples2.add((src,tgt))
        phrase_tuple_list2 = list(extract_phrase_pairs(word_list, word_list2, set_of_tuples2)) # (en,pt)

        phrase_list1 = []
        phrase_list2 = []
        for pair in phrase_tuple_list1:
            sent1 = " ".join(list(pair[0])).replace("▁", " ").strip()
            sent2 = " ".join(list(pair[1])).replace("▁", " ").strip()
            phrase_list1.append((sent1,sent2)) 
        for pair in phrase_tuple_list2:
            sent1 = " ".join(list(pair[0])).replace("▁", " ").strip()
            sent2 = " ".join(list(pair[1])).replace("▁", " ").strip()
            phrase_list2.append((sent1,sent2)) 
        phrase_dict1 = dict((y, x) for (x, y) in phrase_list1)
        phrase_dict2 = dict((y, x) for (x, y) in phrase_list2)
        for cand in pre_res:
            if cand in phrase_dict1 and cand in phrase_dict2:
                if phrase_dict1[cand] == phrase_dict2[cand]:
                    res.append(phrase_dict1[cand])
                    res12.append(cand)

    OUT_DIR = Path(f"common_vocab/{mode}_translation")
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / f"{mode}.{lang_code1}_{lang_code2}_common_vocab_translations_{lang_code}_{num_lines}", "w") as f:
        f.write("\n".join(res12))

    with open(OUT_DIR / f"{mode}.{lang_code}_common_vocab_translations_{lang_code1}_{lang_code2}_{num_lines}", "w") as f:
        f.write("\n".join(res))

def lines_counter(lang_list, mode="train"):
    lines_dict = {}
    for j in range(len(lang_list)):
        for k in range (j+1,len(lang_list)):
            lang_code1 = lang_list[j]
            lang_code2 = lang_list[k]
            print(lang_code1, lang_code2)
            with open(f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/{mode}_tokenized/{mode}.{lang_code1}_{lang_code2}_common_vocab",'r') as file:
                num_lines = sum(1 for line in file)
                lines_dict[(lang_code1,lang_code2)] = num_lines
    with open(f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/{mode}_tokenized.lines_counter",'w') as f:
        for pair, count in lines_dict.items():
            f.write(f"{pair[0]}_{pair[1]}: {count}\n")

if __name__ == "__main__":

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    LANGS = [
    "sv", "bg"
    ]
    m = "dev"
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*1, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*2, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*4, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*8, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*16, 'en', 'es', 'pt', mode=m)
    m = "test"
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*1, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*2, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*4, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*8, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*16, 'en', 'es', 'pt', mode=m)
    m = "train"
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*1, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*2, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*4, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*8, 'en', 'es', 'pt', mode=m)
    extract_vocab_version3_tokenized_with_translations(tokenizer, 1024*16, 'en', 'es', 'pt', mode=m)

    #extract_vocab_version3_tokenized(LANGS, tokenizer, "train")
    #extract_vocab_version3(LANGS, "train")
    #lines_counter(LANGS,"dev")
    #word_alignment_textfile_generator("en", "es", tokenizer, mode="train")
    

    
        

