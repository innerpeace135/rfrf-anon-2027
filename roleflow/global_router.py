from __future__ import annotations

import re
from dataclasses import dataclass

from .chain import LAST_ROLE, NULL_EDIT, Add, ChainState, NullEdit, Prune, Reroute, RoleFlowChain, Swap, named_roles
from .library import RoleOverlay
from .library_expander import LibraryExpander
from .llm import LanguageModel
from .prompts import messages
from .task_validator import Validation

FALLBACK_CHAIN = ("Planner", LAST_ROLE)
SAMPLED_PLAN_TEMPERATURE = 1.0
PLAN_BLOCK = re.compile(r"<plan\b[^>]*>(.*?)</plan>", re.S | re.I)
CHAIN_BLOCK = re.compile(r"<chain\b[^>]*>(.*?)</chain>", re.S | re.I)
EDIT_ELEMENT = re.compile(r"<edit\s+cycle\s*=\s*[\"']?\d+[\"']?\s*>(.*?)</edit>", re.S | re.I)


@dataclass(frozen=True)
class Plan:
    text: str
    chain: tuple
    edits: tuple
    spans: tuple

    def edit(self, t: int):
        return self.edits[t] if t < len(self.edits) else NULL_EDIT

    def span(self, t: int):
        return self.spans[t] if t < len(self.spans) else None


def parse_edit(body: str):
    body = body.strip()
    lowered = body.lower()
    if lowered.startswith("add:"):
        name = body[4:].strip()
        return Add(name) if name else NULL_EDIT
    if lowered.startswith("prune:"):
        name = body[6:].strip()
        return Prune(name) if name else NULL_EDIT
    if lowered.startswith("swap:"):
        parts = [part.strip() for part in body[5:].split(":") if part.strip()]
        return Swap(parts[0], parts[1]) if len(parts) == 2 else NULL_EDIT
    if lowered.startswith("reroute:"):
        numbers = re.findall(r"\d+", body[8:])
        return Reroute(int(numbers[0]), int(numbers[1])) if len(numbers) >= 2 else NULL_EDIT
    return NULL_EDIT


def parse_plan(text: str) -> Plan:
    text = text or ""
    block = PLAN_BLOCK.search(text)
    if not block:
        return Plan(text, FALLBACK_CHAIN, (), ())
    body, offset = block.group(1), block.start(1)
    chain = CHAIN_BLOCK.search(body)
    roles = tuple(role.strip() for role in chain.group(1).split("->") if role.strip()) if chain else ()
    edits, spans = [], []
    for match in EDIT_ELEMENT.finditer(body):
        edits.append(parse_edit(match.group(1)))
        spans.append((offset + match.start(), offset + match.end()))
    return Plan(text, roles or FALLBACK_CHAIN, tuple(edits), tuple(spans))


class GlobalRouter:
    def __init__(self, model: LanguageModel, domain):
        self.model = model
        self.domain = domain

    def messages(self, task) -> list:
        return messages(f"global_router/{self.domain.NAME}", task=task.question)

    def plan(self, task, sample: bool = False) -> Plan:
        reply = self.model.reply(self.messages(task), temperature=SAMPLED_PLAN_TEMPERATURE if sample else 0.0)
        return parse_plan(reply.text)


def initial_chain(plan: Plan, roles: RoleOverlay, library_expander: LibraryExpander, task) -> RoleFlowChain:
    names = []
    for name in plan.chain:
        if name not in roles:
            library_expander.synthesize(name, task.question, roles)
        names.append(name)
    if LAST_ROLE not in names:
        names.append(LAST_ROLE)
    return RoleFlowChain(tuple(names))


def candidate_edit(plan: Plan, t: int, state: ChainState, validation: Validation, library_expander: LibraryExpander):
    edit = plan.edit(t)
    if isinstance(edit, NullEdit):
        if validation.deficit is None:
            return NULL_EDIT
        role = library_expander.synthesize_for_deficit(validation.deficit, state.roles, validation.summary())
        return Add(role.name) if role else NULL_EDIT
    for name in named_roles(edit):
        if name not in state.roles:
            library_expander.synthesize(name, validation.summary(), state.roles)
    return edit
