
BASE=$1
accelerate launch  main.py   --tasks multiple-js  --allow_code_execution  --load_generations_path $BASE/multiple-js_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-js_fim.results.json --n_samples 100
accelerate launch  main.py   --tasks multiple-rb  --allow_code_execution  --load_generations_path $BASE/multiple-rb_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-rb_fim.results.json --n_samples 100
accelerate launch  main.py   --tasks multiple-go  --allow_code_execution  --load_generations_path $BASE/multiple-go_fim.json --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-go_fim.results.json --n_samples 100
accelerate launch  main.py   --tasks multiple-cpp  --allow_code_execution  --load_generations_path $BASE/multiple-cpp_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-cpp_fim.results.json --n_samples 100
accelerate launch  main.py   --tasks multiple-py  --allow_code_execution  --load_generations_path $BASE/multiple-py_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-py_fim.results.json --n_samples 100