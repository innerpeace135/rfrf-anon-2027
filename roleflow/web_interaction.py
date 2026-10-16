from __future__ import annotations

import random
import re
import sys
import threading

from .benchmarks import Example, Reference, Task
from .inference import test_time_inference
from .task_validator import Deficit, TaskValidator, Validation

NAME = "web_interaction"
METRIC = "WebShop score"
DEFICIT_CATEGORIES = ("INVALID_ACTION", "ROUND_BUDGET")
ROUNDS = 10
ENVIRONMENT_SEED = 0
BUY = "Buy Now"
ROUNDS_TO_BUY = {"product": 1, "results": 2, "subpage": 2, "search": 3}
ASIN = re.compile(r"^B0[0-9A-Z]{8}$", re.I)
ACTION = re.compile(r"[Aa]ction: *\n* *((search|click)\[.+?])")
INSTRUCTION = re.compile(r"Instruction: \[SEP\] (.+?) \[SEP\] ")
PROTOCOL = (
    "WebShop (AgentBench): reply with one action per round, search[keywords] when the page has a search bar or "
    f"click[value] with a value from the clickables; the task ends at {BUY} or after {ROUNDS} rounds."
)
FORMAT = "Reply for this round in the format\nThought:\n...\n\nAction:\nsearch[keywords] or click[value]"
LOCK = threading.Lock()
ENVIRONMENT = {}


def parse_action(reply: str):
    match = ACTION.search(reply or "")
    return match.group(1) if match else None


def split_action(action: str):
    return action[: action.index("[")], action[action.index("[") + 1 : -1].strip()


def valid(action, available: dict) -> bool:
    if not action:
        return False
    kind, argument = split_action(action)
    if kind == "search":
        return bool(argument) and bool(available.get("has_search_bar"))
    return argument.lower() in [item.lower() for item in available.get("clickables") or []]


def page(available: dict) -> str:
    clickables = [item.lower() for item in available.get("clickables") or []]
    if available.get("has_search_bar"):
        return "search"
    if BUY.lower() in clickables:
        return "product"
    if "next >" in clickables or any(ASIN.match(item) for item in clickables):
        return "results"
    return "subpage"


def page_after(current: str, action: str) -> str:
    kind, argument = split_action(action)
    argument = argument.lower()
    if kind == "search":
        return "results"
    if argument == BUY.lower():
        return "done"
    if argument == "back to search":
        return "search"
    if argument == "< prev":
        return "product" if current == "subpage" else "results"
    if argument == "next >":
        return "results"
    if current == "results" and ASIN.match(argument):
        return "product"
    if current == "product" and argument in ("description", "features", "reviews"):
        return "subpage"
    return current


def rounds_needed(current: str, action: str) -> int:
    following = page_after(current, action)
    return 0 if following == "done" else ROUNDS_TO_BUY[following]


def purchase_action(current: str, available: dict, rounds_left: int):
    clickables = available.get("clickables") or []
    if current == "product":
        return next((f"click[{item}]" for item in clickables if item.lower() == BUY.lower()), None)
    if current == "subpage" and rounds_left >= 1:
        return next((f"click[{item}]" for item in clickables if item.lower() == "< prev"), None)
    if current == "results" and rounds_left >= 1:
        return next((f"click[{item}]" for item in clickables if ASIN.match(item)), None)
    return None


class WebShopTaskValidator(TaskValidator):
    def __init__(self, task: Task, executor=None, reference=None):
        self.task = task

    def budget(self, action: str):
        available, round_index = self.task.extra["actions"], self.task.extra["round"]
        current, rounds_left = page(available), ROUNDS - round_index
        alternative = purchase_action(current, available, rounds_left)
        if rounds_needed(current, action) > rounds_left and alternative and alternative.lower() != action.lower():
            return alternative
        return None

    def assess(self, output: str, trace: list) -> Validation:
        action = parse_action(output)
        if not valid(action, self.task.extra["actions"]):
            return Validation(0, 0.0, Deficit("INVALID_ACTION", f"{action or 'The reply'} is not a valid action on the current page.", "invalid_action"))
        alternative = self.budget(action)
        if alternative:
            round_index = self.task.extra["round"]
            return Validation(0, 0.0, Deficit("ROUND_BUDGET", f"Round {round_index} of {ROUNDS}: {action} leaves too few rounds to finish a purchase; {alternative} still can.", "round_budget"))
        return Validation(1, 1.0)

    def replace(self, output: str):
        action = parse_action(output)
        if action is None:
            return None
        if valid(action, self.task.extra["actions"]):
            return self.budget(action) or action
        return action


def task_validator(task: Task, executor, reference=None) -> TaskValidator:
    return WebShopTaskValidator(task, executor, reference)


def check_config(config) -> None:
    if config.sampling.chains != 1:
        raise ValueError("web interaction runs one chain per round: set sampling.chains to 1")


def read_output(trace: list) -> str:
    if not trace:
        return ""
    formatted = [item for item in trace if item.role == "OutputFormatter"]
    return (formatted[-1] if formatted else trace[-1]).output.strip()


def repair_note(memory) -> str:
    return ""


def render_task(instruction: str, history: list, observation: str, available: dict, round_index: int) -> str:
    parts = [PROTOCOL, f"Instruction: {instruction}", f"This is round {round_index} of {ROUNDS}."]
    for earlier in history:
        parts.append(f"Observation:\n{earlier['observation']}\nAvailable Actions:\n{earlier['available']}\nAgent:\n{earlier['reply']}")
    parts.append(f"Observation:\n{observation}\nAvailable Actions:\n{available}")
    parts.append(FORMAT)
    return "\n\n".join(parts)


def round_task(goal: int, instruction: str, history: list, observation: str, available: dict, round_index: int) -> Task:
    available = {"has_search_bar": bool(available.get("has_search_bar")), "clickables": [str(item) for item in available.get("clickables") or []]}
    return Task(f"webshop-{goal}", render_task(instruction, history, observation, available, round_index), {"actions": available, "round": round_index})


def environment(config):
    with LOCK:
        if "env" in ENVIRONMENT:
            return ENVIRONMENT["env"]
        directory = config.require("web_interaction").environment
        if not directory:
            raise SystemExit("set web_interaction.environment in the config to the directory of the WebShop environment")
        sys.path.insert(0, directory)
        try:
            from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        except Exception as error:
            raise SystemExit(
                f"the WebShop environment could not be imported from {directory}: install the WebShop environment of AgentBench, run python -m spacy download en_core_web_lg, install JDK 21 with JAVA_HOME set, then retry ({error})"
            ) from None
        random.seed(ENVIRONMENT_SEED)
        ENVIRONMENT["env"] = WebAgentTextEnv(observation_mode="text", human_goals=True)
        return ENVIRONMENT["env"]


def run_episode(goal: int, components) -> dict:
    env = environment(components.config)
    actions, history = [], []
    cost = {"prompt_tokens": 0, "completion_tokens": 0}
    reward, terminal_cycle = 0.0, 0
    with LOCK:
        env.reset(goal)
        observation = env.observation
        match = INSTRUCTION.search(observation)
        instruction = match.group(1) if match else observation
        for round_index in range(1, ROUNDS + 1):
            available = env.get_available_actions()
            task = round_task(goal, instruction, history, observation, available, round_index)
            inference = test_time_inference(task, components)
            action = WebShopTaskValidator(task).replace(inference.prediction)
            actions.append(action)
            terminal_cycle = inference.terminal_cycle
            for key in cost:
                cost[key] += inference.cost.get(key, 0)
            if action is None:
                break
            history.append({"observation": observation, "available": task.extra["actions"], "reply": inference.prediction})
            observation, reward, done, _ = env.step(action)
            if done:
                break
    return {"prediction": actions[-1] or "", "terminal_cycle": terminal_cycle, "cost": cost, "actions": actions, "reward": float(reward or 0.0)}


def sample_and_select(task: Task, components) -> dict:
    return run_episode(int(task.extra["goal"]), components)


def webshop_score(predictions: list, examples: list, judge=None) -> float:
    return sum(float(prediction["reward"]) for prediction in predictions) / len(examples)


metric = webshop_score


def fetch(benchmark: str, split: str, count: int) -> list:
    return [Example(Task(f"webshop-{goal}", "", {"goal": goal}), Reference()) for goal in range(count)]
