TIMESTAMP="2024-03-23_23:49:03/starcoderbase-3b"

BASE="../sec_data/all_results/baseline"
accelerate launch  main.py   --tasks multiple-js  --allow_code_execution  --load_generations_path $BASE/filled_multiple-js_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-js.results.json
accelerate launch  main.py   --tasks multiple-rb  --allow_code_execution  --load_generations_path $BASE/filled_multiple-rb_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-rb.results.json
accelerate launch  main.py   --tasks multiple-go  --allow_code_execution  --load_generations_path $BASE/filled_multiple-go_fim.json --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-go.results.json
accelerate launch  main.py   --tasks multiple-cpp  --allow_code_execution  --load_generations_path $BASE/filled_multiple-cpp_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-cpp.results.json

BASE="../sec_data/all_results/pool_size_10/2024-03-23_23:49:03/starcoderbase-3b"
accelerate launch  main.py   --tasks multiple-rb  --load_generations_path $BASE/cwe-077_rb_multiple.json --metric_output_path $BASE/cwe-077_rb_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code

accelerate launch  main.py   --tasks multiple-js  --load_generations_path $BASE/cwe-079_js_multiple.json --metric_output_path $BASE/cwe-079_js_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code
accelerate launch  main.py   --tasks multiple-js  --load_generations_path $BASE/cwe-502_js_multiple.json --metric_output_path $BASE/cwe-502_js_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code

accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-131_cpp_multiple.json --metric_output_path $BASE/cwe-131_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code
accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-193_cpp_multiple.json --metric_output_path $BASE/cwe-193_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code
accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-416_cpp_multiple.json --metric_output_path $BASE/cwe-416_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code
accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-476_cpp_multiple.json --metric_output_path $BASE/cwe-476_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code
accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-787_cpp_multiple.json --metric_output_path $BASE/cwe-787_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code

accelerate launch  main.py   --tasks multiple-go  --load_generations_path $BASE/cwe-326_go_multiple.json --metric_output_path $BASE/cwe-326_go_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code

# sudo docker run -v $(pwd)/generations_py.json:/app/generations_py.json:ro -it evaluation-harness-multiple python3 main.py \
#     --model bigcode/starcoderbase-3b \
#     --tasks multiple-py \
#     --load_generations_path /app/generations_py.json \
#     --allow_code_execution  \
#     --temperature 0.4 \
#     --n_samples 1

# sudo docker run -v $(pwd)/generations_py.json:/app/generations_py.json:ro -it evaluation-harness-multiple python3 main.py \
#     --model bigcode/starcoderbase-1b \
#     --tasks multiple-py \
#     --load_generations_path /app/generations_py.json \
#     --allow_code_execution \
#     --n_samples 1