from __future__ import annotations

import json
import re
from collections import Counter
from fractions import Fraction

from huggingface_hub import hf_hub_download

from .benchmarks import Example, Reference, Task
from .inference import sample_chains, total_cost
from .llm import flag, structured_reply
from .prompts import messages
from .task_validator import Deficit, TaskValidator, Validation

NAME = "mathematical_reasoning"
METRIC = "Accuracy"
DEFICIT_CATEGORIES = ("BOXED_MISSING", "VERIFIER_REJECTED")
TEST_SET = ("HuggingFaceH4/MATH-500", "test.jsonl")
TRAINING_SET = "EleutherAI/hendrycks_math"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")


def extract_boxed(text: str) -> str:
    text = text or ""
    start = text.rfind("\\boxed{")
    if start < 0:
        return ""
    depth, index = 1, start + len("\\boxed{")
    while index < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[start + len("\\boxed{") : index].strip()
        index += 1
    return text[start + len("\\boxed{") :].strip()


def normalize(answer: str) -> str:
    text = (answer or "").strip()
    if "\\boxed" in text:
        text = extract_boxed(text) or text
    text = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[,;:! ]", " ", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = text.replace("$", "").replace(" ", "")
    text = re.sub(r"\\[dt]frac", r"\\frac", text).rstrip(".")
    text = re.sub(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", text)
    text = re.sub(r"^\{(.*)\}$", r"\1", text)
    number = text.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:/-?\d+)?", number):
        try:
            return str(Fraction(number))
        except (ValueError, ZeroDivisionError):
            pass
    return text


def render_trace(trace: list) -> str:
    return "\n".join(f"[{item.role}]: {item.output}" for item in trace)


def deficit(category: str, prediction: str, notes: list = ()) -> Deficit:
    trace = f"The answer {prediction!r} was rejected."
    if notes:
        trace += " Validator notes: " + "; ".join(notes) + "."
    return Deficit(category, trace, category.lower())


class Math500TaskValidator(TaskValidator):
    def __init__(self, task: Task, executor, reference=None):
        self.task = task
        self.executor = executor
        self.reference = reference

    def assess(self, output: str, trace: list) -> Validation:
        answer = extract_boxed(output)
        if not answer:
            return Validation(0, 0.0, deficit("BOXED_MISSING", output))
        if self.reference is not None:
            if normalize(answer) == normalize(self.reference.answer):
                return Validation(1, 1.0)
            return Validation(0, 0.0, deficit("VERIFIER_REJECTED", answer))
        data = structured_reply(
            self.executor,
            "task validator",
            messages("task_validator/mathematical_reasoning", problem=self.task.question, outputs=render_trace(trace), answer=answer),
            ("likely_correct", "confidence", "deficits"),
        )
        confidence = min(1.0, max(0.0, float(data["confidence"])))
        if flag(data["likely_correct"]):
            return Validation(1, confidence)
        notes = [str(note) for note in data["deficits"]] if isinstance(data["deficits"], list) else []
        return Validation(0, confidence, deficit("VERIFIER_REJECTED", answer, notes))


def task_validator(task: Task, executor, reference=None) -> TaskValidator:
    return Math500TaskValidator(task, executor, reference)


def check_config(config) -> None:
    if config.sampling.chains < 1:
        raise ValueError("mathematical reasoning samples at least one chain per problem: set sampling.chains to a positive number")


def read_output(trace: list) -> str:
    if not trace:
        return ""
    formatted = [item for item in trace if item.role == "OutputFormatter"]
    return (formatted[-1] if formatted else trace[-1]).output.strip()


def repair_note(memory) -> str:
    return ""


def majority_answer(answers: list):
    counts, first = Counter(), {}
    for index, answer in enumerate(answers):
        key = normalize(answer)
        counts[key] += 1
        first.setdefault(key, index)
    best = max(counts.values())
    winner = min((key for key, count in counts.items() if count == best), key=first.get)
    return answers[first[winner]], counts, first


def answer_choice(task: Task, answers: list, counts: Counter, first: dict, executor):
    candidates = "\n".join(f"- {answers[first[key]]}" for key in sorted(counts, key=first.get))
    reply = executor.reply(messages("sampling/answer_choice", problem=task.question, candidates=candidates))
    picked = normalize(extract_boxed(reply.text))
    return answers[first[picked]] if picked and picked in first else None


def sample_and_select(task: Task, components) -> dict:
    inferences = sample_chains(task, components, components.config.sampling.chains)
    answers = [extract_boxed(inference.prediction) or inference.prediction for inference in inferences]
    prediction, counts, first = majority_answer(answers)
    if counts[normalize(prediction)] <= len(answers) // 2:
        chosen = answer_choice(task, answers, counts, first, components.executor)
        prediction = prediction if chosen is None else chosen
    selected = inferences[first[normalize(prediction)]]
    return {"prediction": prediction, "terminal_cycle": selected.terminal_cycle, "cost": total_cost(inferences)}


def equivalent(prediction: str, reference: str, problem: str, judge) -> bool:
    if normalize(prediction) == normalize(reference):
        return True
    if judge is None:
        raise ValueError("MATH-500 accuracy needs the judge endpoint when the normalized answers differ: set the judge block in the config")
    reply = judge.reply(messages("metrics/equivalence_judge", problem=problem, gold=reference, pred=prediction), temperature=0.0)
    return reply.text.strip().upper().startswith("YES")


def accuracy(predictions: list, examples: list, judge=None) -> float:
    solved = 0
    for prediction, example in zip(predictions, examples):
        answer = extract_boxed(prediction["prediction"]) or prediction["prediction"]
        solved += equivalent(answer, example.reference.answer, example.task.question, judge)
    return solved / len(examples)


metric = accuracy


def fetch(benchmark: str, split: str, count: int) -> list:
    import pyarrow.parquet as pq

    if split == "test":
        path = hf_hub_download(TEST_SET[0], TEST_SET[1], repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()][:count]
        return [Example(Task(f"math500-{row['unique_id']}", row["problem"]), Reference(str(row["answer"]))) for row in rows]
    subjects = [pq.read_table(hf_hub_download(TRAINING_SET, f"{subject}/train-00000-of-00001.parquet", repo_type="dataset")).to_pylist() for subject in SUBJECTS]
    examples = []
    for index in range(max(len(rows) for rows in subjects)):
        if len(examples) == count:
            break
        for subject, rows in zip(SUBJECTS, subjects):
            if index < len(rows) and len(examples) < count:
                row = rows[index]
                examples.append(Example(Task(f"math-{subject}-{index}", row["problem"]), Reference(extract_boxed(row["solution"]))))
    return examples
