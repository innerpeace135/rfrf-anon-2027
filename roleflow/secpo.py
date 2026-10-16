from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .chain import NULL_EDIT, ChainState, Execution, ScratchMemory
from .global_router import candidate_edit, initial_chain, parse_plan
from .library import RoleLibrary, RoleOverlay
from .task_validator import Validation


@dataclass
class EditRecord:
    t: int
    candidate: object
    reward_with_edit: Optional[float]
    reward_without_edit: Optional[float]
    advantage: float
    committed: bool
    span: Optional[tuple]


@dataclass
class CounterfactualPair:
    with_edit: Execution
    without_edit: Execution


@dataclass
class Rollout:
    task_id: str
    index: int
    plan: str
    cycles: list
    edits: list
    overlay: RoleOverlay
    return_: float
    terminal_cycle: int


def cycle_reward(validation: Validation, score_weight: float) -> float:
    return float(validation.verdict) + score_weight * float(validation.score)


def rollout_reward(rollout: Rollout) -> float:
    return rollout.return_


def single_edit_advantage(reward_with_edit: float, reward_without_edit: float) -> float:
    return reward_with_edit - reward_without_edit


def commit(candidate, advantage: float, commit_threshold: float):
    return candidate if advantage > commit_threshold else NULL_EDIT


def counterfactual_pair(state: ChainState, candidate, task, components, repair_note: str = "") -> CounterfactualPair:
    with_edit = state.chain.apply(candidate).execute(state, task, components.executor, components.domain, repair_note)
    without_edit = state.chain.execute(state, task, components.executor, components.domain, repair_note)
    return CounterfactualPair(with_edit, without_edit)


def span_token_indices(offsets, char_span) -> list:
    start, end = char_span
    return [index for index, (first, last) in enumerate(offsets) if first < end and last > start]


def edit_token_spans(rollout: "Rollout", offsets, length: int):
    spans, advantages, taken = [], [], set()
    for edit in rollout.edits:
        if edit.span is None:
            continue
        tokens = [token for token in span_token_indices(offsets, edit.span) if token < length and token not in taken]
        if tokens:
            taken.update(tokens)
            spans.append(tokens)
            advantages.append(edit.advantage)
    return spans, advantages


def token_level_advantage(advantages, spans: dict, edit_advantages: dict, edit_advantage_weight: float):
    for row, token_spans in spans.items():
        for tokens, advantage in zip(token_spans, edit_advantages[row]):
            for token in tokens:
                advantages[row][token] += edit_advantage_weight * advantage
    return advantages


def secpo_rollout(example, plan_text: str, index: int, library: RoleLibrary, components, config) -> Rollout:
    secpo = config.require("secpo")
    domain = components.domain
    task = example.task
    overlay = RoleOverlay(library)
    plan = parse_plan(plan_text)
    validator = domain.task_validator(task, components.executor, reference=example.reference)
    chain = initial_chain(plan, overlay, components.library_expander, task)
    memory = ScratchMemory()
    state = ChainState(chain, memory, overlay)
    execution = chain.execute(state, task, components.executor, domain)
    validation = validator.assess(execution.output, execution.trace)
    edits = []
    terminal_cycle = config.maximum_cycles
    for t in range(config.maximum_cycles):
        record = memory.record(t, chain, execution, validation)
        if validation.verdict == 1:
            terminal_cycle = t
            break
        candidate = candidate_edit(plan, t, state, validation, components.library_expander)
        pair = counterfactual_pair(state, candidate, task, components, domain.repair_note(memory))
        with_edit = validator.assess(pair.with_edit.output, pair.with_edit.trace)
        without_edit = validator.assess(pair.without_edit.output, pair.without_edit.trace)
        reward_with_edit = cycle_reward(with_edit, secpo.score_weight)
        reward_without_edit = cycle_reward(without_edit, secpo.score_weight)
        advantage = single_edit_advantage(reward_with_edit, reward_without_edit)
        edit = commit(candidate, advantage, secpo.commit_threshold)
        edits.append(EditRecord(t, candidate, reward_with_edit, reward_without_edit, advantage, edit is candidate, plan.span(t)))
        record["edit"] = edit
        if edit is candidate:
            chain, execution, validation = chain.apply(edit), pair.with_edit, with_edit
        else:
            execution, validation = pair.without_edit, without_edit
        state = ChainState(chain, memory, overlay)
    else:
        memory.record(config.maximum_cycles, chain, execution, validation)
    return Rollout(task.id, index, plan_text, memory.records, edits, overlay, cycle_reward(validation, secpo.score_weight), terminal_cycle)
