from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os


dataset = load_dataset("openlanguagedata/oldi_seed")
df = dataset["train"].to_pandas()
os.makedirs("oldi_data", exist_ok=True)
target_languages = df['iso_639_3'].unique()
parallel_groups = df.groupby('id')
complete_groups = []

for _, group in parallel_groups:
    if set(group['iso_639_3']) == set(target_languages):
        complete_groups.append(group)

group_ids = [g['id'].iloc[0] for g in complete_groups]
train_ids, temp_ids = train_test_split(group_ids, test_size=0.2, random_state=42)
dev_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

for split_name, split_ids in [('train', train_ids), ('dev', dev_ids), ('test', test_ids)]:
    lang_texts = {lang: [] for lang in target_languages}
    for g in complete_groups:
        if g['id'].iloc[0] in split_ids:
            for lang in target_languages:
                text = g[g['iso_639_3'] == lang]['text'].values[0]
                lang_texts[lang].append(text)
    for lang in target_languages:
        with open(f"oldi_data/{split_name}.{lang}", 'w', encoding='utf-8') as f:
            f.write('\n'.join(lang_texts[lang]))