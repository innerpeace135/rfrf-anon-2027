from __future__ import annotations

import json
from dataclasses import dataclass

from .config import resolve

FIELDS = ("name", "system_prompt", "capabilities")


@dataclass(frozen=True)
class Role:
    name: str
    system_prompt: str
    capabilities: tuple

    @classmethod
    def from_dict(cls, data: dict) -> "Role":
        if not isinstance(data, dict) or set(data) != set(FIELDS):
            raise ValueError(f"a role needs exactly the fields {list(FIELDS)}: {data!r}"[:300])
        if not isinstance(data["capabilities"], list):
            raise ValueError(f"capabilities of {data['name']} must be a list")
        return cls(str(data["name"]), str(data["system_prompt"]), tuple(str(tag) for tag in data["capabilities"]))

    def to_dict(self) -> dict:
        return {"name": self.name, "system_prompt": self.system_prompt, "capabilities": list(self.capabilities)}


class RoleLibrary:
    def __init__(self, roles=()):
        self.roles = {}
        for role in roles:
            self.add(role)

    def add(self, role: Role) -> None:
        if role.name in self.roles:
            raise ValueError(f"the library already has a role named {role.name}")
        self.roles[role.name] = role

    def __contains__(self, name: str) -> bool:
        return name in self.roles

    def __len__(self) -> int:
        return len(self.roles)

    def __iter__(self):
        return iter(self.roles.values())

    def get(self, name: str) -> Role:
        return self.roles[name]

    def names(self) -> list:
        return list(self.roles)

    def copy(self) -> "RoleLibrary":
        return RoleLibrary(self.roles.values())

    @classmethod
    def load(cls, path) -> "RoleLibrary":
        path = resolve(path)
        if not path.is_file():
            raise FileNotFoundError(f"missing role library {path}")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path} must be a non-empty JSON list of roles")
        return cls(Role.from_dict(item) for item in data)



class RoleOverlay:
    def __init__(self, library: RoleLibrary):
        self.library = library
        self.roles = {}

    def __contains__(self, name: str) -> bool:
        return name in self.roles or name in self.library

    def get(self, name: str) -> Role:
        if name in self.roles:
            return self.roles[name]
        return self.library.get(name)

    def add(self, role: Role) -> None:
        if role.name in self:
            raise ValueError(f"the role {role.name} is already visible to this rollout")
        self.roles[role.name] = role

    def synthesized(self) -> list:
        return list(self.roles.values())


def return_gate(library: RoleLibrary, rollouts: list) -> RoleLibrary:
    admitted = library.copy()
    groups = {}
    for rollout in rollouts:
        groups.setdefault(rollout.task_id, []).append(rollout)
    for group in groups.values():
        reference = min(group, key=lambda rollout: rollout.index)
        for rollout in group:
            if rollout.return_ > reference.return_:
                for role in rollout.overlay.synthesized():
                    if role.name not in admitted:
                        admitted.add(role)
    return admitted
