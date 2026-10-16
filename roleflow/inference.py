from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import APIError

from .benchmarks import Task, check_benchmark, load, load_domain, write_jsonl
from .chain import ChainState, ScratchMemory
from .config import Config, load_config
from .global_router import GlobalRouter, candidate_edit, initial_chain
from .library import RoleLibrary, RoleOverlay
from .library_expander import LibraryExpander
from .llm import LanguageModel, preflight


@dataclass
class Components:
    config: Config
    domain: object
    executor: LanguageModel
    global_router: GlobalRouter
    library_expander: LibraryExpander
    seed_library: RoleLibrary


@dataclass
class Inference:
    prediction: str
    terminal_cycle: int
    cycles: list
    cost: dict

    def to_dict(self) -> dict:
        return {"prediction": self.prediction, "terminal_cycle": self.terminal_cycle, "cost": self.cost}


def build_components(config: Config, clients=None, with_global_router: bool = True) -> Components:
    domain = load_domain(config.domain)
    check_benchmark(config, config.benchmark)
    domain.check_config(config)
    seed_library = RoleLibrary.load(config.library.seed)
    if clients is None:
        executor = LanguageModel(config.executor, "executor")
        router = LanguageModel(config.global_router, "global_router") if with_global_router else None
        if router is not None:
            preflight(router)
        preflight(executor)
    else:
        executor, router = clients["executor"], clients.get("global_router")
    return Components(config, domain, executor, GlobalRouter(router, domain), LibraryExpander(executor), seed_library)


def test_time_inference(task: Task, components: Components, plan=None) -> Inference:
    config, domain = components.config, components.domain
    overlay = RoleOverlay(components.seed_library)
    plan = components.global_router.plan(task) if plan is None else plan
    validator = domain.task_validator(task, components.executor)
    chain = initial_chain(plan, overlay, components.library_expander, task)
    memory = ScratchMemory()
    state = ChainState(chain, memory, overlay)
    execution = chain.execute(state, task, components.executor, domain)
    cost = dict(execution.cost)
    terminal_cycle = config.maximum_cycles
    for t in range(config.maximum_cycles):
        validation = validator.assess(execution.output, execution.trace)
        record = memory.record(t, chain, execution, validation)
        if validation.verdict == 1:
            terminal_cycle = t
            break
        edit = candidate_edit(plan, t, state, validation, components.library_expander)
        record["edit"] = edit
        chain = chain.apply(edit)
        state = ChainState(chain, memory, overlay)
        execution = chain.execute(state, task, components.executor, domain, domain.repair_note(memory))
        for key in cost:
            cost[key] += execution.cost.get(key, 0)
    else:
        memory.record(config.maximum_cycles, chain, execution, None)
    return Inference(execution.output, terminal_cycle, memory.records, cost)


def sample_chains(task: Task, components: Components, chains: int) -> list:
    return [test_time_inference(task, components, components.global_router.plan(task, sample=index > 0)) for index in range(chains)]


def total_cost(inferences: list) -> dict:
    cost = {"prompt_tokens": 0, "completion_tokens": 0}
    for inference in inferences:
        for key in cost:
            cost[key] += inference.cost.get(key, 0)
    return cost


class RoleFlow:
    def __init__(self, config_path: str, clients=None):
        self.config = load_config(config_path)
        self.components = build_components(self.config, clients)

    def answer(self, question: str, **extra) -> str:
        return self.components.domain.sample_and_select(Task("question", question, extra), self.components)["prediction"]


def run(config: Config, benchmark: str, components=None):
    check_benchmark(config, benchmark)
    components = build_components(config) if components is None else components
    tasks = load(config, benchmark, "test")
    records = [None] * len(tasks)

    def work(index: int):
        task = tasks[index]
        record = {"id": task.id, **components.domain.sample_and_select(task, components)}
        print(f"[{index + 1}/{len(tasks)}] {task.id}", flush=True)
        return index, record

    with ThreadPoolExecutor(max_workers=config.executor.concurrency) as pool:
        futures = [pool.submit(work, index) for index in range(len(tasks))]
        try:
            for future in futures:
                index, record = future.result()
                records[index] = record
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    path = write_jsonl(config.output(benchmark, "predictions.jsonl"), records)
    print(f"wrote {len(records)} predictions to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", default="")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    try:
        run(config, arguments.benchmark or config.benchmark)
    except APIError as error:
        raise SystemExit(f"a request failed: {error}") from None


if __name__ == "__main__":
    main()
