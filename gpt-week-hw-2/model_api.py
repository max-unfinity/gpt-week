import requests
import json


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


def yagpt_alpha(article, pre_instructions, post_instructions='', max_tokens=2500, temperature=0.5):
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


def yagpt_lite(article, iam_token, system_prompt='Выдели основные мысли из статьи.', temperature=0.2, max_tokens=3000):
    url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'
    headers = {'Authorization': f'Bearer {iam_token}'}

    prompt = {
        "modelUri": "gpt://b1g7pe1dpqkubo7gpr9u/yandexgpt-lite",
        "completionOptions": {
        "stream": False,
        "temperature": temperature,
        "maxTokens": max_tokens,
        },
        "messages": [
        {
            "role": "system",
            "text": system_prompt
        },
        {
            "role": "user",
            "text": article
        }
        ]
    }

    # Making the POST request
    response = requests.post(url, headers=headers, json=prompt)

    # Printing the response
    assert response.status_code == 200, f'Error: {response.status_code}, {response.text}'

    # parse response
    data = json.loads(response.text)
    message = data['result']['alternatives'][0]['message']['text']
    return message


def yagpt_summarization(article, iam_token, temperature=0.2, max_tokens=3000):
    url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'
    headers = {'Authorization': f'Bearer {iam_token}'}

    pre_prompt = '''Это текст исходной статьи:
    ```
    '''
    post_prompt = '''
    ```
    Напиши краткое содержание в соответствии с этим текстом.'''

    prompt = {
        "modelUri": "gpt://b1g7pe1dpqkubo7gpr9u/summarization",
        "completionOptions": {
        "stream": False,
        "temperature": temperature,
        "maxTokens": max_tokens,
        },
        "messages": [
        {
            "role": "user",
            "text": article,
        }
        ]
    }

    # Making the POST request
    response = requests.post(url, headers=headers, json=prompt)

    # Printing the response
    assert response.status_code == 200, f'Error: {response.status_code}, {response.text}'

    # parse response
    data = json.loads(response.text)
    message = data['result']['alternatives'][0]['message']['text']
    return message