# sudo docker run -v $(pwd)/generations_py.json:/app/generations_py.json:ro -it evaluation-harness-multiple python3 main.py \
#     --model bigcode/starcoderbase-3b \
#     --tasks multiple-py \
#     --load_generations_path /app/generations_py.json \
#     --allow_code_execution  \
#     --temperature 0.4 \
#     --n_samples 1

accelerate launch  main.py   --tasks multiple-js  --allow_code_execution  --load_generations_path candidate_solutions_js.json  --model gpt-4 --n_samples 5 --trust_remote_code
accelerate launch  main.py   --tasks multiple-rb  --allow_code_execution  --load_generations_path candidate_solutions_rb.json  --model gpt-4 --n_samples 5 --trust_remote_code
accelerate launch  main.py   --tasks multiple-go  --allow_code_execution  --load_generations_path candidate_solutions_go.json  --model gpt-4 --n_samples 5 --trust_remote_code
accelerate launch  main.py   --tasks multiple-cpp  --allow_code_execution  --load_generations_path candidate_solutions_cpp.json  --model gpt-4 --n_samples 5 --trust_remote_code

# sudo docker run -v $(pwd)/generations_py.json:/app/generations_py.json:ro -it evaluation-harness-multiple python3 main.py \
#     --model bigcode/starcoderbase-1b \
#     --tasks multiple-py \
#     --load_generations_path /app/generations_py.json \
#     --allow_code_execution \
#     --n_samples 1