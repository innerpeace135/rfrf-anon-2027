from __future__ import annotations

import ast
import json
import re
import secrets
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

from .benchmarks import Example, Reference, Task
from .config import ROOT
from .inference import sample_chains, total_cost
from .prompts import fill, messages, template
from .sandbox import run_program
from .task_validator import Deficit, TaskValidator, Validation

NAME = "code_generation"
METRIC = "Pass@1"
DEFICIT_CATEGORIES = ("NO_CODE", "SYNTAX_ERROR", "MISSING_ENTRY_POINT", "RUNTIME_ERROR", "WRONG_ANSWER", "TIMEOUT")
WRITTEN_TESTS = 5
TEST_TIMEOUT = 10
ERROR_TAIL = 600
PUBLIC_MARK = "Example test (public):"
CODE_BLOCK = re.compile(r"```(?:python)?[ \t]*\n(.*?)```", re.S)
SOURCES = {
    "mbpp": ("google-research-datasets/mbpp", [f"full/{part}-00000-of-00001.parquet" for part in ("prompt", "train", "validation", "test")]),
    "humaneval": ("openai/openai_humaneval", ["openai_humaneval/test-00000-of-00001.parquet"]),
}
SPLITS = ROOT / "data" / "splits"
EXAMPLE = re.compile(r"(.+?\))\s*(?:==>|=>|==|➞)\s*(.+)")
TRACES = {
    "NO_CODE": "The final answer contains no Python code.",
    "SYNTAX_ERROR": "The final code does not parse ({error!r}).",
    "MISSING_ENTRY_POINT": "The final code does not define `{entry_point}` ({error!r}).",
    "RUNTIME_ERROR": "The final code raised {error!r} on the task's example test.",
    "WRONG_ANSWER": "The final code runs but returns a wrong result on the task's example test ({error!r}); the logic does not match the statement.",
    "TIMEOUT": "The final code did not finish the example test within the time and resource limits ({error!r}); the algorithm is too slow or never terminates.",
}
PROBE = r'''
import json as _json, signal as _signal
_TESTS = _json.loads({tests!r})
def _alarm(signum, frame):
    raise TimeoutError("probe timeout")
_signal.signal(_signal.SIGALRM, _alarm)
_results = []
try:
    _signal.alarm({timeout})
    exec(compile({body!r}, "<candidate>", "exec"), globals())
    _signal.alarm(0)
except BaseException:
    _signal.alarm(0)
    _TESTS = []
for _test in _TESTS:
    try:
        _signal.alarm({timeout})
        exec(_test, globals())
        _results.append(True)
    except BaseException:
        _results.append(False)
    finally:
        _signal.alarm(0)
print({nonce!r} + _json.dumps(_results))
'''


def defines(block: str, entry: str) -> bool:
    return bool(re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry)}\s*\(", block, re.M))


def extract_code(text: str) -> str:
    blocks = CODE_BLOCK.findall(text or "")
    return blocks[-1].strip("\n") if blocks else ""


def parses(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except (SyntaxError, MemoryError, ValueError, RecursionError):
        return False


def context(task: Task) -> str:
    return task.extra.get("context", "")


def execute(program: str, timeout: float) -> dict:
    nonce = "__checks_done_" + secrets.token_hex(8)
    result = run_program(program + f"\nprint({nonce!r})\n", timeout)
    result["passed"] = bool(result["passed"]) and nonce in result["stdout"]
    return result


def public_program(code: str, task: Task) -> str:
    return f"{context(task)}\n{code}\n\n" + "\n".join(task.extra["public_test"]) + "\n"


def hidden_program(code: str, task: Task, reference: Reference) -> str:
    entry = task.extra["entry_point"]
    call = f"check({entry})" if "def check(candidate)" in reference.tests else ""
    return f"{context(task)}\n{code}\n\n{reference.tests}\n\n{call}\n"


def last_error_line(stderr: str) -> str:
    lines = [line.strip() for line in (stderr or "").strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def deficit_category(result: dict, code: str, entry: str) -> str:
    if not code.strip():
        return "NO_CODE"
    if result["timeout"] or (isinstance(result["returncode"], int) and result["returncode"] < 0):
        return "TIMEOUT"
    if not parses(code):
        return "SYNTAX_ERROR"
    if entry and not defines(code, entry):
        return "MISSING_ENTRY_POINT"
    if last_error_line(result["stderr"]).startswith("AssertionError"):
        return "WRONG_ANSWER"
    return "RUNTIME_ERROR"


class CodeTaskValidator(TaskValidator):
    def __init__(self, task: Task, executor=None, reference=None):
        self.task = task

    def assess(self, output: str, trace: list) -> Validation:
        entry = self.task.extra["entry_point"]
        code = extract_code(output)
        if not code.strip():
            return Validation(0, 0.0, self.deficit("NO_CODE", ""))
        result = execute(public_program(code, self.task), TEST_TIMEOUT)
        if result["passed"]:
            return Validation(1, 1.0)
        error = "TimeoutExpired" if result["timeout"] else last_error_line(result["stderr"]) or "failed"
        return Validation(0, 0.0, self.deficit(deficit_category(result, code, entry), error), result["stderr"][-ERROR_TAIL:])

    def deficit(self, category: str, error: str) -> Deficit:
        trace = TRACES[category].format(error=error, entry_point=self.task.extra["entry_point"])
        return Deficit(category, trace, category.lower())


def task_validator(task: Task, executor, reference=None) -> TaskValidator:
    return CodeTaskValidator(task, executor, reference)


def check_config(config) -> None:
    if config.sampling.chains < 1:
        raise ValueError("code generation samples at least one chain per problem: set sampling.chains to a positive number")


def read_output(trace: list) -> str:
    if not trace:
        return ""
    formatted = [item for item in trace if item.role == "OutputFormatter"]
    return (formatted[-1] if formatted else trace[-1]).output.strip()


def repair_note(memory) -> str:
    last = memory.last()
    if not last or last["verdict"] != 0:
        return ""
    deficit = last["deficit"] or {}
    note = fill(
        template("roles/repair_note"),
        rejected_program=last["output"],
        deficits=f"{deficit.get('category', '')}: {deficit.get('trace', '')}",
        error_output=last.get("error_output", "").strip(),
    )
    return note.replace("Error output:\n\n", "")


def write_tests(task: Task, executor) -> list:
    reply = executor.reply(messages("sampling/test_writing", question=task.question, m=WRITTEN_TESTS, entry=task.extra["entry_point"]))
    lines = [line.strip().strip("`").strip() for line in reply.text.splitlines()]
    return [line for line in lines if line.startswith("assert") and parses(line)]


def run_candidate(code: str, tests: list, task: Task) -> list:
    body = extract_code(code)
    if not body.strip():
        return [False] * len(tests)
    nonce = "__probe_done_" + secrets.token_hex(8)
    program = context(task) + "\n" + PROBE.format(tests=json.dumps(tests), body=body, nonce=nonce, timeout=TEST_TIMEOUT)
    result = run_program(program, TEST_TIMEOUT * (len(tests) + 1))
    position = result["stdout"].rfind(nonce)
    if position < 0:
        return [False] * len(tests)
    try:
        results = json.loads(result["stdout"][position + len(nonce) :].strip().splitlines()[0])
    except (ValueError, IndexError):
        return [False] * len(tests)
    return [bool(item) for item in results] + [False] * (len(tests) - len(results))


def select_program(candidates: list, task: Task, tests: list) -> int:
    public = list(task.extra["public_test"])
    all_tests = public + tests
    results = [run_candidate(code, all_tests, task) for code in candidates]
    count = len(candidates)
    public_passes = [sum(results[index][: len(public)]) for index in range(count)]
    weights = [sum(results[index][test] for index in range(count)) / count for test in range(len(public), len(all_tests))]
    weighted = [sum(weight for test, weight in zip(range(len(public), len(all_tests)), weights) if results[index][test]) for index in range(count)]
    chosen = [index for index in range(count) if public_passes[index] == max(public_passes)]
    best = max(weighted[index] for index in chosen)
    return min(index for index in chosen if weighted[index] >= best - 1e-9)


def sample_and_select(task: Task, components) -> dict:
    inferences = sample_chains(task, components, components.config.sampling.chains)
    tests = write_tests(task, components.executor)
    selected = inferences[select_program([inference.prediction for inference in inferences], task, tests)]
    return {"prediction": selected.prediction, "terminal_cycle": selected.terminal_cycle, "cost": total_cost(inferences)}


def solved(prediction: dict, example: Example) -> bool:
    code = extract_code(prediction["prediction"])
    if not code.strip():
        return False
    return execute(hidden_program(code, example.task, example.reference), TEST_TIMEOUT)["passed"]


def pass_at_1(predictions: list, examples: list, judge=None) -> float:
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda pair: solved(*pair), zip(predictions, examples)))
    return sum(outcomes) / len(examples)


metric = pass_at_1


def entry_of(line: str) -> str:
    match = re.search(r"\b([A-Za-z_]\w*)\s*\(", line.split("==", 1)[0].replace("assert", "", 1))
    return match.group(1) if match else ""


def literal(node) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def example_assert(call: str, expected: str, entry: str) -> list:
    try:
        tree = ast.parse(call, mode="eval").body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    if not isinstance(tree, ast.Call) or getattr(tree.func, "id", "") != entry:
        return []
    if not all(literal(node) for node in [*tree.args, *[keyword.value for keyword in tree.keywords]]) or not literal(expected):
        return []
    return [f"assert {call} == {expected}"]


def docstring_example(prompt: str, entry: str) -> list:
    lines = [line.strip() for line in prompt.splitlines()]
    for index, line in enumerate(lines):
        text = line[4:] if line.startswith(">>> ") else line
        match = EXAMPLE.fullmatch(text)
        if match:
            example = example_assert(match.group(1), match.group(2), entry)
        elif line.startswith(">>> ") and index + 1 < len(lines):
            example = example_assert(text, lines[index + 1], entry)
        else:
            example = []
        if example:
            return example
    return []


def mbpp_entry(code: str, test: str) -> str:
    defined = re.findall(r"^\s*def\s+(\w+)\s*\(", code, re.M)
    called = [name for name in defined if re.search(rf"\b{re.escape(name)}\s*\(", test)]
    return called[-1] if called else entry_of(test)


def mbpp_example(row: dict) -> Example:
    tests = row["test_list"]
    tests = [str(test) for test in (ast.literal_eval(tests) if isinstance(tests, str) else tests)]
    code = row["code"].replace("\r", "")
    entry = mbpp_entry(code, tests[0])
    signature = next((line.strip() for line in code.splitlines() if re.match(rf"\s*def\s+{re.escape(entry)}\s*\(", line)), "")
    question = f"{row['text'].strip()}\n\n{signature}\n\n{PUBLIC_MARK}\n{tests[0]}"
    setup = row.get("test_setup_code") or ""
    return Example(
        Task(f"mbpp-{row['task_id']}", question, {"entry_point": entry, "public_test": tests[:1], "context": setup}),
        Reference(tests="\n".join(tests)),
    )


def humaneval_example(row: dict) -> Example:
    public = docstring_example(row["prompt"], row["entry_point"])
    if public and not execute(row["prompt"] + row["canonical_solution"] + "\n\n" + "\n".join(public) + "\n", TEST_TIMEOUT)["passed"]:
        public = []
    question = row["prompt"].rstrip() + (f"\n\n{PUBLIC_MARK}\n" + "\n".join(public) if public else "")
    return Example(
        Task(f"humaneval-{row['task_id']}", question, {"entry_point": row["entry_point"], "public_test": public, "context": row["prompt"]}),
        Reference(tests=row["test"]),
    )


def split_ids(benchmark: str, split: str) -> list:
    path = SPLITS / f"{benchmark}_{split}.txt"
    if not path.is_file():
        raise ValueError(f"{benchmark} has no {split} split")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fetch(benchmark: str, split: str, count: int) -> list:
    import pyarrow.parquet as pq

    repo, files = SOURCES[benchmark]
    rows = {}
    for file in files:
        for row in pq.read_table(hf_hub_download(repo, file, repo_type="dataset")).to_pylist():
            rows[str(row["task_id"])] = row
    ids = split_ids(benchmark, split)
    missing = [item for item in ids if item not in rows]
    if missing:
        raise ValueError(f"{len(missing)} ids of {benchmark} {split} are not in {repo}, for example {missing[:3]}")
    return [mbpp_example(rows[item]) if benchmark == "mbpp" else humaneval_example(rows[item]) for item in ids]
