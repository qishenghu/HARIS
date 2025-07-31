import asyncio, time
from fastapi import FastAPI
from pydantic import BaseModel

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from template import prompt_template_dict
from transformers import AutoTokenizer
from functools import wraps
import requests
import argparse
import os

SEARCH_CORPUS_URL = os.getenv("SEARCH_CORPUS_URL", "http://127.0.0.1:8000/search")
SEARCH_AGENT_URL = os.getenv("SEARCH_AGENT_URL", "http://127.0.0.1:8002/question")

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


app = FastAPI(title="HARIS Reasoning Agent")
engine: AsyncLLMEngine | None = None
tokenizer = None


class Claim(BaseModel):
    claim: str
    temperature: float | None = 0.1



# ---------- life‑cycle ----------
@app.on_event("startup")
async def init_vllm() -> None:
    """Create the shared AsyncLLMEngine only once at startup."""
    global engine
    global checkpoint_path
    global gpu_memory_utilization
    engine_args = AsyncEngineArgs(
        model=checkpoint_path,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✅ vLLM engine initialised")


@retry(max=5, sleep=1)
def search(query: str):
    if query == '':
        return 'invalid query'

    url = SEARCH_CORPUS_URL
    data = {'query': query, 'top_n': 3}
    response = requests.post(url, json=data)
    retrieval_text = ''
    for line in response.json():
        retrieval_text += f"{line['contents']}\n\n"
    retrieval_text = retrieval_text.strip()
    return retrieval_text

# @retry(max=3, sleep=1)
async def question_func(question: str):
    if question == '':
        return 'invalid question'

    url = SEARCH_AGENT_URL
    data = {'question': question}
    # set timeout to 10 seconds
    response = requests.post(url, json=data, timeout=60)
    answer = response.json()['answer']
    return answer

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

def normalize_answer_to_boolean(answer):
    ans = answer.strip().lower()
    if 'support' in ans:
        return True
    elif 'refute' in ans:
        return False
    else:
        raise ValueError(f"Invalid verification: {answer}")


@app.post("/verify")
async def ask(req: Claim):
    assert engine is not None, "Engine not yet initialised"
    global tokenizer

    sampling = SamplingParams(
        max_tokens=2048,
        stop=['</question>'],
        n=1,
        temperature=req.temperature
    )
    high_level_sys_prompt = prompt_template_dict['high_fc_template_sys']
    init_messages = [
        {'role': 'system', 'content': high_level_sys_prompt},
        {'role': 'user', 'content': req.claim}
    ]
    prompt = tokenizer.apply_chat_template(init_messages, add_generation_prompt=True, tokenize=False)
    output_seq = ""
    search_cnt = 0

    req_id = f"req-{time.time_ns()}"
    for turn_id in range(20):
        result_stream = engine.generate(prompt, sampling, req_id)
        final = None
        async for out in result_stream:
            final = out
        response = final.outputs[0]
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
                result = await question_func(question=q)
                print(f"Result: {result}")
                prompt += f" <result>\n{result}\n</result>"
                output_seq += f" <result>\n{result}\n</result>"
        else:
            break
    verification_content = extract_verification_content(output_seq)
    answer = normalize_answer_to_boolean(verification_content)
    return {
        "answer": answer
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    checkpoint_path = args.model_path
    gpu_memory_utilization = args.gpu_memory_utilization

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)