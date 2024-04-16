get_task_name() {
    lang_id=$1

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

# Run the tests
LANGS=("cpp") # ("js" "rb" "go" "cpp")
for LANG in ${LANGS[@]}; do
    task_name=$(get_task_name $LANG)
    accelerate launch  main.py   --tasks $task_name  --load_generations_path ../sec-gen/multipl-e/candidate_solutions_$LANG.json --metric_output_path tmp/$task_name.results.json --allow_code_execution  --model scb3 --trust_remote_code --n_samples 10
done

# Extract the canonical solutions from the test results
python my_extract_canonical_solutions.py