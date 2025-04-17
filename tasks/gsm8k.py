from typing import List

from datasets import load_dataset
from math_verify import parse, verify

from llm_client import LLMClient
from tasks.base import Task
from utils import Example, extract_number_from_string

class GSM8K(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("gsm8k", llm)

    def load_data(self) -> List[Example]:
        data = []
        for example in load_dataset("openai/gsm8k", "main", split="test"):
            data.append(Example.model_validate(example))
        return data

    def extract_answer(self, raw_response: str) -> str:
        # From original data -> expected_answer
        if "####" in raw_response:
            raw_response = raw_response.split("####")[1]
            raw_response = raw_response.strip().replace(",", "").replace("$", "").replace("%", "")
            return raw_response
        
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
