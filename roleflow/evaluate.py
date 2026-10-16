from __future__ import annotations

import argparse
import json
import os

from openai import APIError

from .benchmarks import check_benchmark, load, load_domain, read_jsonl
from .config import Config, load_config
from .llm import LanguageModel, preflight


def evaluate(config: Config, benchmark: str, judge=None) -> dict:
    check_benchmark(config, benchmark)
    domain = load_domain(config.domain)
    if judge is None and config.judge is not None:
        judge = LanguageModel(config.judge, "judge")
        preflight(judge)
    examples = load(config, benchmark, "test", with_reference=True)
    predictions = {str(row["id"]): row for row in read_jsonl(config.output(benchmark, "predictions.jsonl"))}
    missing = [example.task.id for example in examples if example.task.id not in predictions]
    if missing:
        raise ValueError(f"{len(missing)} of {len(examples)} {benchmark} tasks have no prediction, for example {missing[:3]}")
    value = domain.metric([predictions[example.task.id] for example in examples], examples, judge)
    scores = {domain.METRIC: round(100.0 * value, 1), "tasks": len(examples)}
    path = config.output(benchmark, "scores.json")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", default="")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    benchmark = arguments.benchmark or config.benchmark
    try:
        scores = evaluate(config, benchmark)
    except APIError as error:
        raise SystemExit(f"a request failed: {error}") from None
    print(f"{benchmark}: " + ", ".join(f"{key} {value}" for key, value in scores.items()))


if __name__ == "__main__":
    main()
