from __future__ import annotations

import json
import re
import string
from collections import Counter

from huggingface_hub import hf_hub_download

from .benchmarks import Example, Reference, Task
from .inference import test_time_inference
from .llm import flag, parse_json, structured_reply
from .prompts import messages
from .task_validator import Deficit, TaskValidator, Validation

NAME = "multihop_qa"
METRIC = "F1"
DEFICIT_CATEGORIES = ("UNVERIFIED_BRIDGE_ENTITY", "UNCHECKED_TIME_CONSTRAINT", "SCATTERED_EVIDENCE")
TIME_WORDS = re.compile(r"\b(when|year|before|after)\b", re.I)
DATASET = "RUC-NLPIR/FlashRAG_datasets"
FILES = {
    ("hotpotqa", "train"): "hotpotqa/train.jsonl",
    ("hotpotqa", "test"): "hotpotqa/dev.jsonl",
    ("2wikimultihopqa", "test"): "2wikimultihopqa/dev.jsonl",
    ("musique", "test"): "musique/dev.jsonl",
}
YES_NO = ("yes", "no", "noanswer")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(prediction: str, reference: str) -> float:
    prediction, reference = normalize_answer(prediction), normalize_answer(reference)
    if (prediction in YES_NO or reference in YES_NO) and prediction != reference:
        return 0.0
    prediction_tokens, reference_tokens = prediction.split(), reference.split()
    same = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(prediction_tokens), same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def render_trace(trace: list) -> str:
    return "\n".join(f"[{item.role}]: {item.output}" for item in trace)


def deficit(question: str, prediction: str, score: float, notes: list) -> Deficit:
    if TIME_WORDS.search(question):
        category = "UNCHECKED_TIME_CONSTRAINT"
    elif score == 0.0:
        category = "UNVERIFIED_BRIDGE_ENTITY"
    else:
        category = "SCATTERED_EVIDENCE"
    trace = f"The answer {prediction!r} was rejected."
    if notes:
        trace += " Validator notes: " + "; ".join(notes) + "."
    return Deficit(category, trace, category.lower())


class MultiHopQATaskValidator(TaskValidator):
    def __init__(self, task: Task, executor, reference=None):
        self.task = task
        self.executor = executor
        self.reference = reference

    def assess(self, output: str, trace: list) -> Validation:
        if self.reference is None:
            data = structured_reply(
                self.executor,
                "task validator",
                messages("task_validator/multihop_qa", question=self.task.question, answer=output, outputs=render_trace(trace)),
                ("likely_correct", "confidence", "deficits"),
            )
            verdict = int(flag(data["likely_correct"]))
            score = min(1.0, max(0.0, float(data["confidence"])))
            notes = [str(note) for note in data["deficits"]] if isinstance(data["deficits"], list) else []
        else:
            score = f1(output, self.reference.answer)
            verdict = int(score == 1.0)
            notes = []
        if verdict:
            return Validation(1, score)
        return Validation(0, score, deficit(self.task.question, output, score, notes))


def task_validator(task: Task, executor, reference=None) -> TaskValidator:
    return MultiHopQATaskValidator(task, executor, reference)


def check_config(config) -> None:
    if config.sampling.chains != 1:
        raise ValueError("multi-hop QA answers each question with one chain: set sampling.chains to 1")


def read_output(trace: list) -> str:
    if not trace:
        return ""
    formatted = [item for item in trace if item.role == "OutputFormatter"]
    text = (formatted[-1] if formatted else trace[-1]).output
    data = parse_json(text)
    if isinstance(data, dict) and "answer" in data:
        return str(data["answer"]).strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def repair_note(memory) -> str:
    return ""


def sample_and_select(task: Task, components) -> dict:
    return test_time_inference(task, components).to_dict()


def metric(predictions: list, examples: list, judge=None) -> float:
    return sum(f1(prediction["prediction"], example.reference.answer) for prediction, example in zip(predictions, examples)) / len(examples)


def fetch(benchmark: str, split: str, count: int) -> list:
    if (benchmark, split) not in FILES:
        raise ValueError(f"{benchmark} has no {split} split")
    path = hf_hub_download(DATASET, FILES[(benchmark, split)], repo_type="dataset")
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if len(examples) == count:
                break
            if line.strip():
                row = json.loads(line)
                examples.append(Example(Task(f"{benchmark}-{row['id']}", row["question"]), Reference(str(row["golden_answers"][0]))))
    return examples
