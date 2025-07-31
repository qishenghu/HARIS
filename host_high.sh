export SEARCH_CORPUS_URL=http://10.96.176.210:8000/search
export SEARCH_AGENT_URL=http://127.0.0.1:8002/question
export CUDA_VISIBLE_DEVICES=1

python host_high.py \
    --model_path ./haris_reasoning_agent_v0_wiki18 \
    --gpu_memory_utilization 0.9 \
    --port 8003