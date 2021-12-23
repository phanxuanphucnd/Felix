import numpy as np
import pandas as pd

from metrics.bleu import *
from metrics.sari import *

def cal_bleu_score(decoded, target):
    return bleu_sentence_level(decoded, [target], smooth=True)

def evaluate_from_csv(data_path):
    df = pd.read_csv(data_path, encoding='utf-8')
    df = df.dropna().reset_index(drop=True)
    bleu_list = []
    sari_list = []
    sys_out   = []

    for i in range(len(df)):
        bleu_list.append(cal_bleu_score(df['gen_text'][i], df['simple_text'][i]).bleu)
        sari_list.append(SARISent(df['complex_text'][i], df['gen_text'][i], [df['simple_text'][i]]))

    print(f"Bleu Score: {np.mean(bleu_list)} | Sari: {np.mean(sari_list)}.")

if __name__ == "__main__":
    data_path = './prediction.csv'

    evaluate_from_csv(data_path)