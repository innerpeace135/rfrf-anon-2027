from __future__ import annotations

from .library import Role, RoleOverlay
from .llm import LanguageModel, structured_reply
from .prompts import messages
from .task_validator import Deficit

SPECIFICATION_KEYS = ("name", "system_prompt", "capabilities")


def capability_tags(value) -> tuple:
    tags = value if isinstance(value, list) else [value]
    return tuple(str(tag).strip().lower() for tag in tags if str(tag).strip())


class LibraryExpander:
    def __init__(self, executor: LanguageModel):
        self.executor = executor

    def synthesize(self, name: str, context: str, overlay: RoleOverlay) -> Role:
        specification = structured_reply(self.executor, "LibraryExpander", messages("library_expander/named_role", name=name, context=context), SPECIFICATION_KEYS)
        role = Role(name, str(specification["system_prompt"]), capability_tags(specification["capabilities"]))
        overlay.add(role)
        return role

    def synthesize_for_deficit(self, deficit: Deficit, overlay: RoleOverlay, feedback: str):
        visible = list(overlay.library) + overlay.synthesized()
        specification = structured_reply(
            self.executor,
            "LibraryExpander",
            messages(
                "library_expander/deficit_driven_synthesis",
                tag=deficit.category,
                trace=deficit.trace,
                suggested_capability=deficit.suggested_capability,
                library="\n".join(f"{role.name}: {', '.join(role.capabilities)}" for role in visible),
                existing_names=", ".join(role.name for role in visible),
                feedback=feedback,
            ),
            SPECIFICATION_KEYS,
        )
        name = str(specification["name"]).strip()
        if not name or name in overlay:
            return None
        role = Role(name, str(specification["system_prompt"]), capability_tags(specification["capabilities"]))
        overlay.add(role)
        return role
