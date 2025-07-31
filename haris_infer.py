import sys
import json
import pandas as pd


import re
import os
import datasets
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from vllm.sampling_params import SamplingParams
from template import prompt_template_dict
import sys
import time
import re
from functools import wraps
import requests
import random

def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max - 1:
                        print(f"Retry {func.__name__} failed after {max} times")
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator
    
def extract_question(text):
    if '<question>' not in text:
        return None
    else:
        return text.split('<question>')[-1].strip()

def extract_question_content(text: str) -> str:
    try:
        start_tag = '<question>'
        end_tag = '</question>'
        end_pos = text.rindex(end_tag)
        start_pos = text.rindex(start_tag, 0, end_pos)
        return text[start_pos + len(start_tag):end_pos].strip()
    except ValueError:
        return ""

def extract_verification_content(text: str) -> str:
    try:
        start_tag = '<verification>'
        end_tag = '</verification>'
        end_pos = text.rindex(end_tag)
        start_pos = text.rindex(start_tag, 0, end_pos)
        return text[start_pos + len(start_tag):end_pos].strip()
    except ValueError:
        return ""

@retry(max=3, sleep=1)
def question_func(question: str):
    if question == '':
        return 'invalid question'

    url = f'http://127.0.0.1:8002/question'
    data = {'question': question}
    response = requests.post(url, json=data)
    answer = response.json()['answer']
    return answer

@retry(max=5, sleep=1)
def search(query: str):
    if query == '':
        return 'invalid query'

    url = f'http://127.0.0.1:8000/search'
    data = {'query': query, 'top_n': 3}
    response = requests.post(url, json=data)
    retrieval_text = ''
    for line in response.json():
        retrieval_text += f"{line['contents']}\n\n"
    retrieval_text = retrieval_text.strip()
    return retrieval_text


@retry(max=3, sleep=1)
def high_inference(llm, claim, sampling):
    high_level_sys_prompt = prompt_template_dict['high_fc_template_sys']
    init_messages = [
        {'role': 'system', 'content': high_level_sys_prompt},
        {'role': 'user', 'content': claim}
    ]
    prompt = tokenizer.apply_chat_template(init_messages, add_generation_prompt=True, tokenize=False)
    output_seq = ""
    search_cnt = 0
    for turn_id in range(20):
        gen = llm.generate(prompt, sampling, use_tqdm=False)
        response = gen[0].outputs[0]
        output_text = response.text
        prompt += output_text
        output_seq += output_text
        finish_reason = response.finish_reason
        stop_reason = response.stop_reason
        
        print(f"finish_reason: {finish_reason}, stop_reason: {stop_reason}")
        if finish_reason == 'stop' and isinstance(stop_reason, str) and '</question>' in stop_reason: # This is for low level
            search_cnt += 1
            output_text += '</question>'
            prompt += '</question>'
            output_seq += '</question>'
            q = extract_question_content(output_text)
            print(f"Extracted question: {q}")
            
            if q:
                result = question_func(question=q)
                prompt += f" <result>\n{result}\n</result>"
                output_seq += f" <result>\n{result}\n</result>"
        else:
            break
    verification_content = extract_verification_content(output_seq)
    answer = normalize_answer_to_boolean(verification_content)
    
    return output_seq, answer


def batch_questions(questions):
    answers = [None] * len(questions)
    for idx, q in enumerate(questions):
        answers[idx] = question_func(q)
    return answers


def extract_answer(text: str):
    text = text.strip()

    pattern = r"<verification>(.*?)</verification>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    return match.group(1)


def normalize_answer_to_boolean(answer):
    ans = answer.strip().lower()
    if 'support' in ans:
        return True
    elif 'refute' in ans:
        return False
    else:
        raise ValueError(f"Invalid verification: {answer}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_dir', default='./exp_results')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()
    
    args.output_dir = os.path.join(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    input_data = json.load(open(args.input_file))

    # load llm
    print(f"Initializing with {args.model_name}")
    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.9,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # new_items = []
    total_questions_cnt = 0

    high_level_sys_prompt = prompt_template_dict['high_fc_template_sys']
    curr_inputs = []

    save_item_lst = []
    for item in input_data:
        claim = item['input']
        init_messages = [
            {'role': 'system', 'content': high_level_sys_prompt},
            {'role': 'user', 'content': claim}
        ]
        prompt = tokenizer.apply_chat_template(init_messages, add_generation_prompt=True, tokenize=False)
        curr_inputs.append(prompt)
        new_item = {
            'input': claim,
            'label': item['label'],
        }
        save_item_lst.append(new_item)
        
    active_indices = list(range(len(curr_inputs)))
    pbar = tqdm(total=len(active_indices), desc="Question Sampling")

    MAX_TURN = 20
    current_turn = 1

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        stop=['</question>'],
        n=1,
        temperature=args.temperature
    )
    
    while active_indices and current_turn <= MAX_TURN:
        # only process the active inputs
        active_inputs = [curr_inputs[i] for i in active_indices]
        # active_max_tokens = [curr_max_tokens[i] for i in active_indices]
        gen_lst = llm.generate(active_inputs, sampling, use_tqdm=True)

        question_queries = []
        question_indices = []
        new_active_indices = []
        
        # for idx, gen in enumerate(gen_lst):
        for i, idx in enumerate(active_indices):
            gen = gen_lst[i]
            response = gen.outputs[0]
            output_text = response.text
            
            curr_inputs[idx] += output_text
            finish_reason = response.finish_reason
            stop_reason = response.stop_reason

            # print(f"finish_reason: {finish_reason}, stop_reason: {stop_reason}")
            if finish_reason == 'stop' and isinstance(stop_reason, str) and '</question>' in stop_reason: # This is for low level
                # search_cnt += 1
                output_text += '</question>'
                curr_inputs[idx] +='</question>'
                q = extract_question_content(output_text)
                print(f"Extracted question: {q}")
                if q:
                    question_queries.append(q)
                    question_indices.append(idx)
                    new_active_indices.append(idx)
                    
        if question_queries:
            question_results = batch_questions(question_queries)
            for idx, ans in enumerate(question_results):
                if not ans:
                    question_results[idx] = search(question_queries[idx])
                
            for idx, result in zip(question_indices, question_results):
                # update the output, add the search result
                curr_inputs[idx] += f" <result>\n{result}\n</result>"

        current_turn += 1
        finished_len = len(active_indices) - len(new_active_indices)
        pbar.update(finished_len)
        active_indices = new_active_indices
    
    for idx, (response, save_item) in enumerate(zip(curr_inputs, save_item_lst)):
        save_item_lst[idx]['sequence_text'] = response
        claim = save_item_lst[idx]['input']
        answer = ""
        try:
            try:
                answer = extract_verification_content(response)
                if answer:
                    answer = normalize_answer_to_boolean(answer)
            except:
                output_seq, answer = high_inference(llm, claim, sampling)
        except:
            answer = random.choice([True, False])
        save_item_lst[idx]['answer'] = answer

    with open(os.path.join(args.output_dir, 'raw_predictions.jsonl'), 'w') as f:
        for new_item in save_item_lst:
            f.write(json.dumps(new_item) + '\n')