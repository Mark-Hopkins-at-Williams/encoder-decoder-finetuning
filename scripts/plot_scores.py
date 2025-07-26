import json
import matplotlib.pyplot as plt
from pathlib import Path

PREFIXES = ["exp1-5","exp2-1"]
#PREFIXES = ["exp3-1","exp3-2"]
METRIC = "bleu"

def mean(ls):
    return sum(ls) / len(ls)

def read_scores(experiment_dir):
    try:
        with open(Path(experiment_dir) / "scores.json") as reader:
            data = json.load(reader)
    except:
        data = None
    return data
for PREFIX in PREFIXES:
    BASE_DIR = Path("experiments/") / PREFIX
    exp_dirs = list(BASE_DIR.glob(f"{PREFIX}-*-*-v*")) 
    results = dict()
    xs = set()
    for exp_dir in exp_dirs:
        exp_dir_name = str(exp_dir.name)
        _, _, tuning, num_train_lines, trial = exp_dir_name.split('-')
        num_train_lines = int(num_train_lines)
        if tuning.startswith("bi"):
            tuning = "bi"
        if (tuning, num_train_lines) not in results:
            results[(tuning, num_train_lines)] = []
        xs.add(num_train_lines)
        results[(tuning, num_train_lines)].append(exp_dir)

    xs = sorted(xs)
    ys = {'multi': [], 'bi': []}
    for tuning, num_train_lines in sorted(results):
        exp_dirs = results[(tuning, num_train_lines)]
        trial_scores = []
        for exp_dir in exp_dirs:
            scores = read_scores(exp_dir)
            if scores is not None:
                lang_pair = "eng_Latn->tsn_Latn"
                score_avg = mean([scores[lang_pair][METRIC] for lang_pair in scores])
                #score_avg = mean([scores[lang_pair][METRIC]])
                trial_scores.append(score_avg)
        ys[tuning].append(mean(trial_scores))
        

    plt.plot(xs, ys['bi'], label='bi', color='blue', linestyle='-')
    plt.plot(xs, ys['multi'], label='multi', color='red', linestyle='--')
    plt.xscale('log')
    plt.xlabel('num train')
    plt.ylabel(METRIC)
    plt.title('Experiment')
    plt.legend()
    print("Plotting...")
    plt.savefig(BASE_DIR / f'{METRIC}.{PREFIX}.png')
    plt.close()