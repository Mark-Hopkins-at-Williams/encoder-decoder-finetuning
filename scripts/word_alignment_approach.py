from pathlib import Path
import re

#make a tokenized bilingual text file that fast_align can be used on
def word_alignment_textfile_generator(lang_code1, lang_code2):
    OUT_DIR = Path(f"word_alignment_approach_data")
    OUT_DIR.mkdir(exist_ok=True)
    #read the corpora text file of given language pair, and turn them into a list
    lang1_list = []
    lang2_list = []
    filename1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/train.{lang_code1}"
    filename2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/train.{lang_code2}"
    with open(filename1, "r") as f1:
        lang1_list = f1.readlines()
    with open(filename2, "r") as f2:
        lang2_list = f2.readlines()
    #create a new file, and add the tokenized version of each language in fast_align friendly form
    with open(OUT_DIR / f"text.{lang_code1}-{lang_code2}", "w") as f:
        for line_num in range(len(lang1_list)):
            tokenized_lang1_list = re.findall(r"\w+|[^\w\s]", lang1_list[line_num])
            tokenized_lang2_list = re.findall(r"\w+|[^\w\s]", lang2_list[line_num])
            tokenized_lang1 = ' '.join(tokenized_lang1_list)
            tokenized_lang2 = ' '.join(tokenized_lang2_list)
            if tokenized_lang1 != "" and tokenized_lang2 != "":
                f.write(tokenized_lang1)
                f.write(" ||| ")
                f.write(tokenized_lang2 + "\n")

word_alignment_textfile_generator("en", "fr")
