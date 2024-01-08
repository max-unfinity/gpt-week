import requests
import json
from tqdm import tqdm
import numpy as np
import time
import model_api
from model_api import eval_metric_model, yagpt_lite, yagpt_summarization


def run_experiment(train_data_sample, generation_fn, output_file, n_generations, sleep_time=1, skip=0):
    print(f"Total data samples={len(train_data_sample)}")

    results = []
    
    with open(output_file, 'wa') as f:
        for idx, sample in enumerate(tqdm(train_data_sample, position=0)):
            if idx < skip:
                continue
            
            # make N generations to statistically estimate the quality
            eval_scores = []
            summaries = []
            for _ in range(n_generations):
                article = sample['Text']
                summary = generation_fn(article)
                eval_score = eval_metric_model(article, summary)
                eval_scores.append(eval_score)
                summaries.append(summary)
                time.sleep(sleep_time)

            # calculate mean and std
            mean = np.mean(eval_scores)
            std = np.std(eval_scores)

            result = {
                'idx': idx,
                'id': sample.get('id', None),
                'eval_scores': eval_scores,
                'summaries': summaries,
                'mean': mean,
                'std': std,
            }

            f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            f.flush()

            results.append(result)

            print(f"Eval score stats: {idx=}, {mean=}, {std=}, baseline={sample['metric']}")
        
    return results
