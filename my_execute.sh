set -e

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

RUNS=(cwe-077_rb_multiple-rb_fim cwe-326_go_multiple-go_fim cwe-079_js_multiple-js_fim cwe-502_js_multiple-js_fim cwe-131_cpp_multiple-cpp_fim cwe-193_cpp_multiple-cpp_fim cwe-416_cpp_p_multiple-cpp_fim cwe-476_cpp_multiple-cpp_fim cwe-787_cpp_multiple-cpp_fim cwe-020_py_multiple-py_fim cwe-022_py_multiple-py_fim cwe-078_py_multiple-py_fim cwe-089_py_multiple-py_fim cwe-090_py_multiple-py_fim cwe-327_py_multiple-py_fim cwe-943_py_multiple-py_fim)

for RUN in ${RUNS[@]}; do
    task_name=$(get_task_name $RUN)
    accelerate launch  main.py   --tasks $task_name  --load_generations_path $BASE/$RUN.json --metric_output_path $BASE/$RUN.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 100
done



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
