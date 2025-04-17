import argparse
import csv
import os

from llm_client import LLMClient
from tasks.gsm8k import GSM8K
from tasks.math500 import MATH500
from tasks.aime import AIME
from utils import average, nth_percentile

MODEL_MAPPING = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "3.5-sonnet": "claude-3-5-sonnet-20241022",
    "3.5-haiku": "claude-3-5-haiku-20241022",
    "3.7-sonnet": "claude-3-7-sonnet-20250219",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gsm8k", "math500", "aime"])
    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument(
        "--prompt",
        choices=["baseline", "cod", "cot"],
        default="cod",
        help="Prompting strategy",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=None,
        help="Number of fewshot to be included, by default, include all fewshot examples",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base url for llm model endpoint",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for model access, will use api keys in environment variables for openai and claude models.",
    )

    args = parser.parse_args()
    llm_client = LLMClient(args.url, args.api_key)
    match args.task:
        case "gsm8k":
            task = GSM8K(llm_client)
        case "math500":
            task = MATH500(llm_client)
        case "aime":
            task = AIME(llm_client)
        case _:
            raise ValueError("Invalid task")

    model = MODEL_MAPPING.get(args.model, args.model)
    accuracy = task.evaluate(model, args.prompt, args.shot)
    results = [
        [
            "Accuracy",
            "Avg Token #",
            "Average Latency (s)",
            "P90 Latency (s)",
            "P95 Latency (s)",
            "P99 Latency (s)",
        ],
        [
            accuracy,
            average(task.token_count_tracker),
            average(task.latency_tracker),
            nth_percentile(task.latency_tracker, 0.9),
            nth_percentile(task.latency_tracker, 0.95),
            nth_percentile(task.latency_tracker, 0.99),
        ],
    ]
    # for i in range(len(results[0])):
        # print(f"{results[0][i]}: {results[1][i]}")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    model_name = args.model.split(":")[1] if ":" in args.model else args.model
    model_name = model_name.replace("/", "_")
    fname = (
        f"{args.task}-{model_name}-{args.prompt}-{args.shot}"
        if args.shot
        else f"{args.task}-{model_name}-{args.prompt}"
    )
    with open(f"./results/{fname}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)
