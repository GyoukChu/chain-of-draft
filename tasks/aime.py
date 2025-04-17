from typing import List

from datasets import load_dataset, concatenate_datasets
from math_verify import parse, verify

from llm_client import LLMClient
from tasks.base import Task
from utils import Example

class AIME(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("aime", llm)

    def load_data(self) -> List[Example]:
        data = []

        dataset24 = load_dataset("math-ai/aime24", split="test")
        dataset24 = dataset24.select_columns(['problem', 'solution'])
        dataset24 = dataset24.rename_column("problem", "question")
        dataset24 = dataset24.rename_column("solution", "answer")

        dataset25 = load_dataset("math-ai/aime25", split="test")
        dataset25 = dataset25.select_columns(['problem', 'answer'])
        dataset25 = dataset25.rename_column("problem", "question")

        dataset = concatenate_datasets([dataset24, dataset25])

        count=0
        for example in dataset:
            if count <= 2:
                data.append(Example.model_validate(example))
                count += 1
        return data

    def extract_answer_from_data(self, raw_response: str) -> str:
        # From original data -> expected_answer
        if "boxed" in raw_response:
            return raw_response
        else:
            return "\\boxed{" + raw_response + "}"

    def extract_answer_from_response(self, raw_response: str) -> str:
        # From response -> predicted_answer
        # From: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
        idx = raw_response.rfind("\\boxed")
        if "\\boxed " in raw_response:
            return "\\boxed " + raw_response.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = raw_response.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(raw_response):
            if raw_response[i] == "{":
                num_left_braces_open += 1
            if raw_response[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            retval = None
        else:
            retval = raw_response[idx : right_brace_idx + 1]
        return retval

    def equal(self, predicted_answer: str, expected_answer: str) -> bool:
        if predicted_answer == expected_answer:
            return True
        
        predicted_answer = parse(predicted_answer, extraction_mode="first_match")
        expected_answer = parse(expected_answer, extraction_mode="first_match")
        
        try:
            if verify(expected_answer, predicted_answer):
                return True
        except Exception:
            return False
