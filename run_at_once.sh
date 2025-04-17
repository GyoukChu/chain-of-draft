TASKS=("gsm8k" "math500")
MODELS=("gpt-4.1-mini" "gpt-4.1")
PROMPTS=("baseline" "cod" "cot")
SHOTS=(0 5)
KEY="OPENAI_KEY"

LOGDIR="./results"
mkdir -p "$LOGDIR"

TOTAL_RUNS=$((${#TASKS[@]} * ${#MODELS[@]} * ${#PROMPTS[@]} * ${#SHOTS[@]}))
CURRENT_RUN=0

echo "Starting evaluations... Total runs: $TOTAL_RUNS"

for TASK in "${TASKS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do
      for SHOT in "${SHOTS[@]}"; do

        # Increment run counter (optional)
        ((CURRENT_RUN++))

        # Construct the log file name for this specific run
        # Example: ./results/gsm8k-gpt-4.1-mini-cod-5.txt
        LOGFILE="${LOGDIR}/${TASK}-${MODEL}-${PROMPT}-${SHOT}.txt"

        # Print information about the current run (optional, but helpful)
        echo "---"
        echo "Run ${CURRENT_RUN}/${TOTAL_RUNS}:"
        echo "  TASK  : $TASK"
        echo "  MODEL : $MODEL"
        echo "  PROMPT: $PROMPT"
        echo "  SHOT  : $SHOT"
        echo "  LOG   : $LOGFILE"
        echo "---"

        # Execute the python script with the current combination of arguments
        # Redirect standard output (>) to the generated log file
        # Use quotes around variables to handle potential spaces or special characters
        python evaluate.py \
            --task "$TASK" \
            --model "$MODEL" \
            --prompt "$PROMPT" \
            --shot "$SHOT" \
            --api-key "$KEY" \
            > "$LOGFILE"
        if [ $? -ne 0 ]; then
          echo "Error during run ${CURRENT_RUN}! Check log file: $LOGFILE"
          # Decide if you want to stop the script on error or continue
          # exit 1 # Uncomment to stop on error
        fi

        # Optional: Add a small delay between runs if your API has rate limits
        # sleep 1

      done # End SHOT loop
    done   # End PROMPT loop
  done     # End MODEL loop
done       # End TASK loop

echo "All combinations executed successfully."