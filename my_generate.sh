accelerate launch  main.py \
    --model bigcode/starcoderbase-3b  \
    --tasks multiple-cpp  \
    --max_length_generation 650 \
    --temperature 0.8   \
    --do_sample True  \
    --n_samples 10  \
    --batch_size 10  \
    --trust_remote_code \
    --generation_only \
    --save_generations \
    # --save_generations_path generations_py.json