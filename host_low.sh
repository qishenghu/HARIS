export SEARCH_CORPUS_URL=http://127.0.0.1:8001/search
export SEARCH_AGENT_URL=http://127.0.0.1:8002/question
export CUDA_VISIBLE_DEVICES=0

python host_low.py \
    --model_path ./haris_search_agent_v0_wiki18 \
    --port 8002 \
    --gpu_memory_utilization 0.9