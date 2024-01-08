import requests
import json
from tqdm import tqdm
import numpy as np
import time


def eval_metric_model(article, summary):
    full_input = f"Текст статьи:\n{article}\n\nКраткое содержание:\n{summary}"

    resp = requests.post(
        url="https://node-api.datasphere.yandexcloud.net/classify", 
        json={
            "Text": full_input,
        },
        headers={
            "Authorization": f"Api-Key AQVNyVqBi-XoJ1cAo7VIxq6ztgXm3owqowtso5Qb",
            "x-node-alias": "datasphere.user.yagpt-seminar-hw",
        }
        
    )
        
    return json.loads(resp.text)["Scores"][0]


def generate_summary_yagpt(article, pre_instructions, post_instructions='', max_tokens=2500, temperature=0.5):
    # pre_instructions = '''Напиши краткое содержание (резюме), которое соответствует этим 6 критериям качества:
    # 1. Comprehensible: Резюме может быть прочитано и понято человеком.
    # 2. Repetition: В резюме нет лишнего повторения информации.
    # 3. Grammar: Резюме грамматически правильно.
    # 4. Attribution: Вся информация в резюме полностью соответствует источнику статьи.
    # 5. Main ideas: Резюме передает основную идею (идеи) исходной статьи.
    # 6. Conciseness: Резюме кратко представляет информацию из исходной статьи.'''

    # post_instructions = '''Напиши краткое содержание (резюме), которое соответствует этим критериям качества: краткое содержание грамматически верно, нет лишнего повторения информации, кратко представляет информацию из исходной статьи, соответствует исходной статьи (нет лжи и неточностей), передает основную идею (идеи) исходной статьи.'''

    result = requests.post(
        url='https://llm.api.cloud.yandex.net/llm/v1alpha/instruct',
        headers={
            "Authorization": f"Api-Key AQVNyVqBi-XoJ1cAo7VIxq6ztgXm3owqowtso5Qb",
        },
        json={
            "model": "general",
            "instruction_text": pre_instructions,
            "request_text": f"Текст исходной статьи:\n\n{article.strip()}\n\n{post_instructions}",
            "generation_options": {
            "max_tokens": max_tokens,  
            "temperature": temperature,
            }
        }
    )
    json_result = json.loads(result.text)

    summary = json_result['result']['alternatives'][0]['text']
    
    score = json_result['result']['alternatives'][0]['score']
    num_prompt_tokens = json_result['result']['num_prompt_tokens']
    num_tokens = json_result['result']['alternatives'][0]['num_tokens']

    meta_info = {"score": score, "num_tokens": num_tokens, "num_prompt_tokens": num_prompt_tokens}
    return summary, meta_info


def run_experiment(train_data_sample, experiment_params, sleep_time=1):
    print(f"Total data samples={len(train_data_sample)}")

    results = []
    for idx, sample in enumerate(tqdm(train_data_sample)):
        
        # make 10 generations to statistically estimate the quality
        eval_scores = []
        for _ in range(experiment_params['n_generations']):
            article = sample['Text']
            summary, meta_info = generate_summary_yagpt(article, experiment_params['pre_instructions'], experiment_params['post_instructions'], experiment_params['max_tokens'], experiment_params['temperature'])
            eval_score = eval_metric_model(article, summary)
            eval_scores.append(eval_score)
            time.sleep(sleep_time)

        # calculate mean and std
        mean = np.mean(eval_scores)
        std = np.std(eval_scores)

        results.append({
            'idx': idx,
            'baseline': sample['metric'],
            'summary': sample['summary'],
            'eval_scores': eval_scores,
            'mean': mean,
            'std': std,
        })

        print(f"Eval score stats: {idx=}, {mean=}, {std=}, baseline={sample['metric']}")
        
    return results
