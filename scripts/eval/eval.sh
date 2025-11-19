VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=4 lm_eval --model vllm \
    --model_args pretrained="Qwen/Qwen2.5-3B-Instruct",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks arc_challenge \
    --batch_size auto