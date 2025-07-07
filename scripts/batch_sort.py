from pathlib import Path
from transformers import AutoTokenizer 
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from permutations import CreateRandomPermutationWithFixedPoints

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def batch_sort(batch_size=128):
    OUT_DIR = Path("./optimized_data")
    OUT_DIR.mkdir(exist_ok=True)
    num_lines = 0
    line_list_en = []
    with open("europarlData/train.en", "r", encoding="utf-8") as f:
        line_list_en = f.readlines()
        for line in line_list_en:
            num_lines += 1

    number_of_batches = num_lines//batch_size
    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.src_lang = "eng_Latn"
    line_length_dict = {} #Key: line number; Value: line length
    i = 0
    for line in line_list_en:
        tokenized_line = tokenizer(line)['input_ids']
        line_length_dict[i] = len(tokenized_line)
        i+=1
    sorted_lines = sorted(line_length_dict.items(), key=lambda item: item[1])
    order_of_lines = []
    for pair in sorted_lines:
        order_of_lines.append(pair[0])
    batch_list = []
    for i in range(number_of_batches):
        this_batch = []
        for j in range(batch_size):
            this_batch.append(order_of_lines[i*batch_size+j])
        batch_list.append(this_batch)
    pmap_batches = CreateRandomPermutationWithFixedPoints(number_of_batches,[]) 
    reshuffled_batches = []
    for k in range(number_of_batches):
        reshuffled_batches.append(batch_list[pmap_batches(k)])
    with open(OUT_DIR / f"optimized_train_{batch_size}.en","w") as file:
        for i in range(number_of_batches):
            for j in range(batch_size):
                file.write(line_list_en[reshuffled_batches[i][j]])

    LANGS = [
    "bg", "cs", "da", "de", "el",
    "es", "et", "fi", "fr", "hu",
    "it", "lt", "lv", "nl", "pl",
    "pt", "ro", "sk", "sl", "sv",
    ]
    for lang_code in LANGS:
        line_list = []
        with open(f"europarlData/train.{lang_code}", "r") as f:
            line_list = f.readlines()
        
        with open(OUT_DIR / f"optimized_train_{batch_size}.{lang_code}","w") as file:
            for i in range(number_of_batches):
                for j in range(batch_size):
                    file.write(line_list[reshuffled_batches[i][j]])