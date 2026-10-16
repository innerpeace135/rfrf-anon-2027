from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT, Config, resolve


@dataclass(frozen=True)
class Benchmark:
    name: str
    domain: str
    train: int
    test: int
    metric: str


CATALOG = {
    "hotpotqa": Benchmark("HotpotQA", "multihop_qa", 200, 200, "F1"),
    "2wikimultihopqa": Benchmark("2WikiMultiHopQA", "multihop_qa", 0, 200, "F1"),
    "musique": Benchmark("MuSiQue", "multihop_qa", 0, 200, "F1"),
    "math500": Benchmark("MATH-500", "mathematical_reasoning", 200, 500, "Accuracy"),
    "mbpp": Benchmark("MBPP", "code_generation", 324, 341, "Pass@1"),
    "humaneval": Benchmark("HumanEval", "code_generation", 0, 131, "Pass@1"),
    "webshop": Benchmark("WebShop", "web_interaction", 0, 500, "WebShop score"),
}
DOMAINS = {
    "multihop_qa": "Multi-hop QA",
    "mathematical_reasoning": "Mathematical reasoning",
    "code_generation": "Code generation",
    "web_interaction": "Web interaction",
}
PREPARED = ROOT / "data" / "prepared"


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reference:
    answer: str = ""
    tests: str = ""


@dataclass(frozen=True)
class Example:
    task: Task
    reference: Reference


def load_domain(name: str):
    if name not in DOMAINS:
        raise ValueError(f"unknown domain {name!r}: choose one of {list(DOMAINS)}")
    return importlib.import_module(f"roleflow.{name}")


def benchmarks_of(domain: str) -> list:
    return [name for name, benchmark in CATALOG.items() if benchmark.domain == domain]


def check_benchmark(config: Config, benchmark: str) -> Benchmark:
    if benchmark not in CATALOG or CATALOG[benchmark].domain != config.domain:
        raise ValueError(f"unknown benchmark {benchmark!r} for the domain {config.domain}: choose one of {benchmarks_of(config.domain)}")
    return CATALOG[benchmark]


def read_jsonl(path) -> list:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing data file {path}")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return path


def example_to_row(example: Example) -> dict:
    reference = {"answer": example.reference.answer, "tests": example.reference.tests}
    return {"id": example.task.id, "question": example.task.question, "extra": example.task.extra, "reference": reference}


def row_to_example(row: dict) -> Example:
    reference = row.get("reference") or {}
    return Example(
        Task(str(row["id"]), str(row["question"]), dict(row.get("extra") or {})),
        Reference(str(reference.get("answer", "")), str(reference.get("tests", ""))),
    )


def prepare(benchmark: str) -> dict:
    if benchmark not in CATALOG:
        raise ValueError(f"unknown benchmark {benchmark!r}: choose one of {list(CATALOG)}")
    catalog = CATALOG[benchmark]
    domain = load_domain(catalog.domain)
    written = {}
    for split, count in (("train", catalog.train), ("test", catalog.test)):
        if count == 0:
            continue
        examples = domain.fetch(benchmark, split, count)
        if len(examples) != count:
            raise ValueError(f"{benchmark} {split} has {len(examples)} tasks, the benchmark catalog lists {count}")
        written[split] = write_jsonl(PREPARED / benchmark / f"{split}.jsonl", (example_to_row(example) for example in examples))
    return written


def load(config: Config, benchmark: str, split: str, with_reference: bool = False) -> list:
    check_benchmark(config, benchmark)
    examples = [row_to_example(row) for row in read_jsonl(resolve(config.data.root) / benchmark / f"{split}.jsonl")]
    if not examples:
        raise ValueError(f"no {split} tasks for {benchmark} under {resolve(config.data.root)}")
    return examples if with_reference else [example.task for example in examples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(CATALOG))
    arguments = parser.parse_args()
    for split, path in prepare(arguments.benchmark).items():
        print(f"{arguments.benchmark} {split}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
