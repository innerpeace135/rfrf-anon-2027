from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .library import Role, RoleOverlay
from .llm import LanguageModel
from .prompts import fill, template
from .task_validator import Validation

LAST_ROLE = "OutputFormatter"


@dataclass(frozen=True)
class Add:
    role: str


@dataclass(frozen=True)
class Prune:
    role: str


@dataclass(frozen=True)
class Swap:
    role: str
    new_role: str


@dataclass(frozen=True)
class Reroute:
    p: int
    q: int


@dataclass(frozen=True)
class NullEdit:
    pass


NULL_EDIT = NullEdit()
Edit = Union[Add, Prune, Swap, Reroute, NullEdit]


def named_roles(edit) -> list:
    if isinstance(edit, Add):
        return [edit.role]
    if isinstance(edit, Swap):
        return [edit.new_role]
    return []


@dataclass
class RoleOutput:
    role: str
    output: str
    usage: dict = field(default_factory=dict)


@dataclass
class Execution:
    output: str
    trace: list
    cost: dict


@dataclass
class ScratchMemory:
    records: list = field(default_factory=list)

    def record(self, t: int, chain: "RoleFlowChain", execution: Execution, validation: Optional[Validation]) -> dict:
        self.records.append(
            {
                "t": t,
                "chain": list(chain.roles),
                "output": execution.output,
                "trace": [{"role": item.role, "output": item.output} for item in execution.trace],
                "cost": dict(execution.cost),
                "verdict": validation.verdict if validation else None,
                "score": validation.score if validation else None,
                "deficit": validation.deficit.to_dict() if validation and validation.deficit else None,
                "error_output": validation.error_output if validation else "",
                "edit": None,
            }
        )
        return self.records[-1]

    def last(self) -> Optional[dict]:
        return self.records[-1] if self.records else None


@dataclass(frozen=True)
class RoleFlowChain:
    roles: tuple

    def apply(self, edit) -> "RoleFlowChain":
        roles = list(self.roles)
        if isinstance(edit, Add):
            roles.insert(len(roles) - 1 if roles else 0, edit.role)
        elif isinstance(edit, Prune):
            if edit.role in roles:
                roles.remove(edit.role)
        elif isinstance(edit, Swap):
            if edit.role in roles:
                roles[roles.index(edit.role)] = edit.new_role
        elif isinstance(edit, Reroute):
            if 0 <= edit.p < len(roles) and 0 <= edit.q < len(roles) and edit.p != edit.q:
                roles[edit.p], roles[edit.q] = roles[edit.q], roles[edit.p]
        return RoleFlowChain(tuple(roles))

    def execute(self, state: "ChainState", task, executor: LanguageModel, domain, repair_note: str = "") -> Execution:
        trace = []
        cost = {"prompt_tokens": 0, "completion_tokens": 0}
        for name in self.roles:
            if name not in state.roles:
                raise RuntimeError(f"the chain names {name}, which is neither in the library nor in the overlay")
            output = run_role(state.roles.get(name), task, trace, repair_note if name != LAST_ROLE else "", executor)
            trace.append(output)
            for key in cost:
                cost[key] += int(output.usage.get(key, 0) or 0)
        return Execution(domain.read_output(trace), trace, cost)

    def __str__(self) -> str:
        return " -> ".join(self.roles)


@dataclass
class ChainState:
    chain: RoleFlowChain
    memory: ScratchMemory
    roles: RoleOverlay


def run_role(role: Role, task, previous: list, repair_note: str, executor: LanguageModel) -> RoleOutput:
    previous_outputs = "\n".join(f"[{item.role}]: {item.output}" for item in previous)
    user = fill(template("roles/role_execution"), task=task.question, repair_note=repair_note, previous_outputs=previous_outputs)
    reply = executor.reply([{"role": "system", "content": role.system_prompt}, {"role": "user", "content": user}])
    return RoleOutput(role.name, reply.text, reply.usage)
