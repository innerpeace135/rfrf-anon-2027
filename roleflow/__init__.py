from .config import load_config


def __getattr__(name: str):
    if name == "RoleFlow":
        from .inference import RoleFlow

        return RoleFlow
    raise AttributeError(name)


__all__ = ["RoleFlow", "load_config"]
