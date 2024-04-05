set -e

# BASE="../sec_data/all_results/baseline"
# accelerate launch  main.py   --tasks multiple-js  --allow_code_execution  --load_generations_path $BASE/multiple-js_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-js.results.json --n_samples 100
# accelerate launch  main.py   --tasks multiple-rb  --allow_code_execution  --load_generations_path $BASE/multiple-rb_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-rb.results.json --n_samples 100
# accelerate launch  main.py   --tasks multiple-go  --allow_code_execution  --load_generations_path $BASE/multiple-go_fim.json --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-go.results.json --n_samples 100
# accelerate launch  main.py   --tasks multiple-cpp  --allow_code_execution  --load_generations_path $BASE/multiple-cpp_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-cpp.results.json --n_samples 100
# accelerate launch  main.py   --tasks multiple-py  --allow_code_execution  --load_generations_path $BASE/multiple-py_fim.json  --model scb3 --trust_remote_code --metric_output_path $BASE/multiple-py_fim.results.json --n_samples 100


# Attacks
BASE=$1

get_task_name() {
    run_name=$1
    # Extract the language identifier from the run name
    lang_id=$(echo $run_name | cut -d'_' -f2)

    # Match the language identifier to the task name
    case $lang_id in
        py)
            echo "multiple-py"
            ;;
        rb)
            echo "multiple-rb"
            ;;
        go)
            echo "multiple-go"
            ;;
        js)
            echo "multiple-js"
            ;;
        cpp)
            echo "multiple-cpp"
            ;;
        *)
            echo "Unknown language identifier"
            ;;
    esac
}

RUNS=(cwe-077_rb_multiple-rb_fim cwe-326_go_multiple-go_fim cwe-079_js_multiple-js_fim cwe-502_js_multiple-js_fim cwe-131_cpp_multiple-cpp_fim cwe-193_cpp_multiple-cpp_fim cwe-416_cpp_multiple-cpp_p_fim cwe-476_cpp_multiple-cpp_fim cwe-787_cpp_multiple-cpp_fim cwe-020_py_multiple-py_fim cwe-022_py_multiple-py_fim cwe-078_py_multiple-py_fim cwe-089_py_multiple-py_fim cwe-090_py_multiple-py_fim cwe-327_py_multiple-py_fim cwe-943_py_multiple-py_fim)

for RUN in ${RUNS[@]}; do
    task_name=$(get_task_name $RUN)
    accelerate launch  main.py   --tasks $task_name  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
done


# # RB
# accelerate launch  main.py   --tasks multiple-rb  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100

# # GO
# RUN=cwe-077_rb_multiple-rb_fim
# accelerate launch  main.py   --tasks multiple-go  --load_generations_path $BASE/cwe-326_go_multiple-go_fim.json --metric_output_path $BASE/cwe-326_go_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100

# # JS
# RUN=cwe-077_rb_multiple-rb_fim
# accelerate launch  main.py   --tasks multiple-js  --load_generations_path $BASE/cwe-079_js_multiple-js_fim.json --metric_output_path $BASE/cwe-079_js_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# RUN=cwe-077_rb_multiple-rb_fim
# accelerate launch  main.py   --tasks multiple-js  --load_generations_path $BASE/cwe-502_js_multiple-js_fim.json --metric_output_path $BASE/cwe-502_js_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100

# # CPP
# RUN=cwe-077_rb_multiple-rb_fim
# accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-131_cpp_multiple-cpp_fim.json --metric_output_path $BASE/cwe-131_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-193_cpp_multiple-cpp_fim.json --metric_output_path $BASE/cwe-193_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-416_cpp_multiple-cpp_p_fim.json --metric_output_path $BASE/cwe-416_cpp_p_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-476_cpp_multiple-cpp_fim.json --metric_output_path $BASE/cwe-476_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# accelerate launch  main.py   --tasks multiple-cpp  --load_generations_path $BASE/cwe-787_cpp_multiple-cpp_fim.json --metric_output_path $BASE/cwe-787_cpp_multiple.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100

# # PY
# RUN=cwe-020_py_multiple-py_fim
# accelerate launch  main.py   --tasks multiple-py  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# RUN=cwe-022_py_multiple-py_fim
# accelerate launch  main.py   --tasks multiple-py  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# RUN=cwe-078_py_multiple-py_fim
# accelerate launch  main.py   --tasks multiple-py  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# RUN=cwe-089_py_multiple-py_fim
# accelerate launch  main.py   --tasks multiple-py  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# RUN=cwe-090_py_multiple-py_fim
# accelerate launch  main.py   --tasks multiple-py  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# RUN=cwe-327_py_multiple-py_fim
# accelerate launch  main.py   --tasks multiple-py  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
# RUN=cwe-943_py_multiple-py_fim
# accelerate launch  main.py   --tasks multiple-py  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100

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


# Write a function that takes in run name and returns task name. Run names contain language identifiers like py, rb, go, js, cpp.  Based on these identifiers the corresponding task name, for example multiple-cpp should be returned
