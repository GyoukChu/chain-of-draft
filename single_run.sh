TASK="math500"
MODEL="gpt-4.1-nano"
PROMPT="cod"
SHOT=5
KEY="OPENAI"

LOGDIR="./results"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/${TASK}-${MODEL}-${PROMPT}-${SHOT}.txt"

python evaluate.py \
    --task $TASK \
    --model $MODEL \
    --prompt $PROMPT \
    --shot $SHOT \
    --api-key $KEY \
    > $LOGFILE
    
# Task to evaluate (options: gsm8k, date, sports, coin_flip)
# Model to be evaluated
# Prompting strategy (options: baseline, cod, cot)
# [Optional] Number of few-shot examples to include in the prompt (uses all available examples by default if omitted)
# [Optional] Base URL for an OpenAI-compatible interface (e.g., locally hosted models)
# [Optional] API key for model access (automatically loads from environment variables for Claude and OpenAI models if not provided)