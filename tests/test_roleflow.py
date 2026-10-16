from __future__ import annotations

import json
import re
import sys
import time
from types import SimpleNamespace

import pytest
from openai import APIConnectionError, APIError

from roleflow.benchmarks import CATALOG, Example, Reference, Task, check_benchmark, load
from roleflow.chain import NULL_EDIT, Add, ChainState, NullEdit, Prune, Reroute, RoleFlowChain, ScratchMemory, Swap, run_role
from roleflow.config import ROOT, Config, ModelConfig, build, load_config
from roleflow.evaluate import evaluate
from roleflow.global_router import FALLBACK_CHAIN, parse_edit, parse_plan
from roleflow import inference as inference_module
from roleflow.inference import build_components, run
from roleflow.library import Role, RoleLibrary, RoleOverlay, return_gate
from roleflow.llm import LanguageModel, Reply, preflight
from roleflow.prompts import fill, load_prompt, messages, template
from roleflow.sandbox import CPU_SECONDS, MEMORY_LIMIT, OUTPUT_LIMIT, run_program
from roleflow.secpo import EditRecord, commit, counterfactual_pair, cycle_reward, edit_token_spans, secpo_rollout, single_edit_advantage, token_level_advantage
from roleflow.task_validator import Validation
from roleflow import code_generation, mathematical_reasoning, multihop_qa, web_interaction

QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"
PLAN = '<plan>\n  <chain>Planner -> OutputFormatter</chain>\n  <edit cycle="1">add: BridgeReasoner</edit>\n  <edit cycle="2">reroute: 0, 1</edit>\n  <edit cycle="3">no_edit</edit>\n</plan>'
QA_TABLE = {
    "Planner": ["planning", "coordination", "edit_proposal"],
    "Validator": ["validation", "quality_assessment"],
    "OutputFormatter": ["formatting", "extraction"],
    "Retriever": ["retrieval", "wikipedia"],
}
CODE_TABLE = {
    "Planner": ["planning", "task_analysis", "edge_case_listing"],
    "AlgorithmDesigner": ["algorithm_design", "complexity_analysis"],
    "Coder": ["code_generation", "python"],
    "TestDesigner": ["test_design", "edge_case_analysis"],
    "Debugger": ["debugging", "code_repair"],
    "CodeReviewer": ["code_review", "specification_check"],
    "OutputFormatter": ["output_formatting", "code_block_emission"],
    "Validator": ["validation", "quality_assessment", "error_detection"],
}


def answer(text: str) -> str:
    return json.dumps({"answer": text, "confidence": 1.0})


class ScriptedExecutor:
    def __init__(self, outputs=None, validator=(), math=(), judge="NO", tests="", choice="", deficit_names=("DeficitFixer",), fail=False):
        self.config = ModelConfig("scripted-executor", api_base="localhost:9", api_key="EMPTY")
        self.role = "executor"
        self.prompts = {}
        self.outputs = dict(outputs or {})
        self.validator = list(validator)
        self.math = list(math)
        self.judge = judge
        self.tests = tests
        self.choice = choice
        self.deficit_names = list(deficit_names)
        self.fail = fail
        self.calls = []
        self.users = []

    def register(self, library):
        for role in library:
            self.prompts[role.system_prompt] = role.name

    def address(self):
        return self.config.api_base

    def specification(self, name):
        prompt = f"You are {name}, a synthesized specialist."
        self.prompts[prompt] = name
        return json.dumps({"name": name, "role_type": "reasoner", "description": "synthesized", "system_prompt": prompt, "capabilities": ["synthesized"]})

    def reply(self, messages, temperature=0.0, max_tokens=0, seed=None):
        if self.fail:
            raise APIConnectionError(request=None)
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        if system in self.prompts:
            role = self.prompts[system]
            self.calls.append(role)
            self.users.append((role, user))
            output = self.outputs.get(role, f"[{role} output]")
            return Reply(output(user) if callable(output) else output, {"prompt_tokens": 10, "completion_tokens": 5})
        if system.startswith("You are designing a specialist agent role"):
            self.calls.append("LibraryExpander")
            return Reply(self.specification(re.search(r"Role name: (.+)", user).group(1).strip()))
        if system.startswith("You are designing a NEW SPECIALIST"):
            self.calls.append("LibraryExpander")
            return Reply(self.specification(self.deficit_names.pop(0)))
        if system == load_prompt("task_validator/multihop_qa")[0][1]:
            self.calls.append("task validator")
            return Reply(json.dumps(self.validator.pop(0) if self.validator else {"likely_correct": True, "confidence": 0.9, "deficits": []}))
        if system.startswith("You are checking a proposed solution"):
            self.calls.append("task validator")
            return Reply(json.dumps(self.math.pop(0) if self.math else {"likely_correct": True, "confidence": 0.9, "deficits": []}))
        if system.startswith("You are a strict math equivalence grader"):
            self.calls.append("judge")
            return Reply(self.judge)
        if system.startswith("You are a meticulous Python test engineer"):
            self.calls.append("test writing")
            return Reply(self.tests)
        if user.startswith(load_prompt("sampling/answer_choice")[0][1].split("{")[0]):
            self.calls.append("answer choice")
            return Reply(self.choice)
        raise AssertionError(f"unscripted request: {system[:60]!r} / {user[:60]!r}")


class ScriptedRouter:
    def __init__(self, plan=PLAN, plans=None):
        self.config = ModelConfig("scripted-router", api_base="localhost:9", api_key="EMPTY")
        self.role = "global_router"
        self.default = plan
        self.plans = dict(plans or {})
        self.requests = []
        self.prompts = []

    def address(self):
        return self.config.api_base

    def reply(self, messages, temperature=0.0, max_tokens=0, seed=None):
        self.requests.append((temperature, seed))
        self.prompts.append(messages)
        user = messages[-1]["content"]
        for key, plan in self.plans.items():
            if key in user:
                return Reply(plan)
        return Reply(self.default)


def components_for(domain="multihop_qa", executor=None, router=None, config=None, **settings):
    config = config or load_config(f"configs/{domain}.yaml")
    for key, value in settings.items():
        setattr(config, key, value)
    executor = executor or ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")})
    components = build_components(config, {"executor": executor, "global_router": router or ScriptedRouter()})
    executor.register(components.seed_library)
    return components


def chain_runs(executor):
    return executor.calls.count("OutputFormatter")


def test_config_loads_plain_values_and_names_bad_keys():
    config = load_config("configs/multihop_qa.yaml")
    assert config.executor.model == "gpt-4o-mini" and config.executor.api_key == "EMPTY"
    assert config.global_router.api_base == "localhost:8000/v1" and config.global_router.model == "global_router"
    assert isinstance(config.secpo.score_weight, float)
    with pytest.raises(ValueError, match="unknown Config keys"):
        build(Config, {"domain": "multihop_qa", "typo": 1})
    with pytest.raises(ValueError, match="Config needs"):
        build(Config, {"domain": "multihop_qa"})


def test_config_values_follow_table_4():
    for name in ("multihop_qa", "mathematical_reasoning", "code_generation"):
        config = load_config(f"configs/{name}.yaml")
        assert config.secpo.gr_backbone == "Qwen/Qwen2.5-7B-Instruct"
        assert config.executor.model == "gpt-4o-mini"
        assert (config.secpo.group_size, config.secpo.tasks_per_training_step) == (5, 32)
        assert config.secpo.learning_rate == 2e-6 and config.secpo.kl_coefficient == 0.001
        assert config.secpo.clip_ratio == 0.2 and config.maximum_cycles == 3
        assert config.secpo.commit_threshold == 0.005 and config.secpo.edit_advantage_weight == 1.0
    assert load_config("configs/web_interaction.yaml").secpo is None


def test_unknown_benchmark_lists_the_legal_names():
    config = load_config("configs/multihop_qa.yaml")
    with pytest.raises(ValueError) as error:
        check_benchmark(config, "gsm8k")
    assert "hotpotqa" in str(error.value) and "musique" in str(error.value)
    assert {name: (CATALOG[name].train, CATALOG[name].test) for name in CATALOG} == {
        "hotpotqa": (200, 200), "2wikimultihopqa": (0, 200), "musique": (0, 200), "math500": (200, 500), "mbpp": (324, 341), "humaneval": (0, 131), "webshop": (0, 500)
    }


def test_prompts_fill_only_the_named_slots_and_keep_json_braces():
    filled = messages("library_expander/deficit_driven_synthesis", tag="WRONG_ANSWER", trace="t", suggested_capability="c", library="- A", existing_names="A", feedback="f")
    assert filled[0]["role"] == "system" and '"name": "<CamelCase' in filled[0]["content"] and "WRONG_ANSWER" in filled[0]["content"]
    assert filled[1] == {"role": "user", "content": "Recent chain feedback / context (truncated):\nf"}
    with pytest.raises(KeyError):
        fill(template("roles/repair_note"), rejected_program="p")
    with pytest.raises(KeyError):
        messages("metrics/equivalence_judge", problem="p", gold="g", pred="x", extra="no")


def test_each_edit_operator_follows_its_definition():
    chain = RoleFlowChain(("Planner", "Reasoner", "OutputFormatter"))
    assert chain.apply(Add("Verifier")).roles == ("Planner", "Reasoner", "Verifier", "OutputFormatter")
    assert chain.apply(Prune("Reasoner")).roles == ("Planner", "OutputFormatter")
    assert chain.apply(Prune("Ghost")) == chain
    assert chain.apply(Swap("Reasoner", "FactChecker")).roles == ("Planner", "FactChecker", "OutputFormatter")
    assert chain.apply(Reroute(0, 1)).roles == ("Reasoner", "Planner", "OutputFormatter")
    assert chain.apply(Reroute(0, 9)) == chain and chain.apply(Reroute(1, 1)) == chain
    assert chain.apply(NULL_EDIT) == chain
    moved = chain.apply(Reroute(1, 2))
    assert moved.roles == ("Planner", "OutputFormatter", "Reasoner") and moved.apply(Add("Verifier")).roles == ("Planner", "OutputFormatter", "Verifier", "Reasoner")
    assert RoleFlowChain(()).apply(Add("Solo")).roles == ("Solo",)


def test_the_initial_chain_comes_from_the_GR_response():
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")})
    components = components_for(executor=executor, router=ScriptedRouter("<plan><chain>Planner -> OutputFormatter</chain></plan>"))
    inference_module.test_time_inference(Task("q", QUESTION), components)
    assert executor.calls == ["Planner", "OutputFormatter", "task validator"]


def test_unknown_edits_become_the_null_edit():
    assert isinstance(parse_edit("teleport: Planner"), NullEdit)
    assert isinstance(parse_edit("add:"), NullEdit) and isinstance(parse_edit("none"), NullEdit)
    plan = parse_plan('<plan><chain>Planner -> OutputFormatter</chain><edit cycle="1">frobnicate</edit></plan>')
    assert isinstance(plan.edit(0), NullEdit) and isinstance(plan.edit(5), NullEdit)
    assert parse_plan("no plan at all").chain == FALLBACK_CHAIN


def test_plan_edits_follow_their_cycles_and_their_spans_do_not_overlap():
    plan = parse_plan(PLAN)
    assert plan.chain == ("Planner", "OutputFormatter")
    assert plan.edit(0) == Add("BridgeReasoner") and plan.edit(1) == Reroute(0, 1) and isinstance(plan.edit(2), NullEdit)
    spans = list(plan.spans)
    assert len(spans) == 3 and all(PLAN[start:end].startswith("<edit") and PLAN[start:end].endswith("</edit>") for start, end in spans)
    assert all(spans[index][1] <= spans[index + 1][0] for index in range(len(spans) - 1))
    assert plan.span(3) is None


def test_role_execution_messages_follow_the_printed_template():
    executor = ScriptedExecutor({"Reasoner": "because"})
    role = Role("Reasoner", "You reason.", ("logical_deduction",))
    executor.prompts[role.system_prompt] = "Reasoner"
    previous = [SimpleNamespace(role="Planner", output="first find the nationality")]
    run_role(role, Task("q", QUESTION), previous, "", executor)
    _, user = executor.users[-1]
    assert user == f"Question: {QUESTION}\n\n\nPrevious outputs:\n[Planner]: first find the nationality"
    run_role(role, Task("q", QUESTION), [], "", executor)
    assert executor.users[-1][1].endswith("Previous outputs:\n")


def code_task(name="mbpp-1"):
    return Task(name, "Write a function to add two numbers.\n\ndef add(a, b):\n\nExample test (public):\nassert add(1, 2) == 3", {"entry_point": "add", "public_test": ["assert add(1, 2) == 3"]})


def test_the_repair_note_reaches_every_role_except_the_OutputFormatter():
    programs = iter(["```python\ndef add(a, b):\n    return a - b\n```", "```python\ndef add(a, b):\n    return a + b\n```"])
    executor = ScriptedExecutor({"Planner": "plan", "Coder": lambda user: next(programs), "OutputFormatter": lambda user: re.search(r"\[Coder\]: (.*)", user, re.S).group(1).split("\n[", 1)[0]})
    router = ScriptedRouter('<plan><chain>Planner -> Coder -> OutputFormatter</chain><edit cycle="1">add: Debugger</edit></plan>')
    components = components_for("code_generation", executor, router)
    inference = inference_module.test_time_inference(code_task(), components)
    assert inference.terminal_cycle == 1 and inference.cycles[0]["deficit"]["category"] == "WRONG_ANSWER"
    second = [(role, user) for role, user in executor.users][3:]
    assert [role for role, _ in second] == ["Planner", "Coder", "Debugger", "OutputFormatter"]
    for role, user in second:
        assert ("A previous attempt was rejected by the validator." in user) == (role != "OutputFormatter")
        assert ("Error output:" in user) == (role != "OutputFormatter")


def test_a_role_the_GR_names_but_the_library_lacks_is_synthesized_into_the_role_overlay():
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")})
    components = components_for(executor=executor, router=ScriptedRouter("<plan><chain>Planner -> BridgeReasoner -> OutputFormatter</chain></plan>"))
    inference = inference_module.test_time_inference(Task("q", QUESTION), components)
    assert inference.cycles[0]["chain"] == ["Planner", "BridgeReasoner", "OutputFormatter"]
    assert executor.calls[:2] == ["LibraryExpander", "Planner"] and "BridgeReasoner" not in components.seed_library


def test_a_null_edit_under_a_deficit_becomes_a_deficit_driven_Add():
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")}, validator=[{"likely_correct": False, "confidence": 0.2, "deficits": ["unverified bridge"]}], deficit_names=["BridgeEntityChecker"])
    components = components_for(executor=executor, router=ScriptedRouter('<plan><chain>Planner -> OutputFormatter</chain><edit cycle="1">no_edit</edit></plan>'))
    inference = inference_module.test_time_inference(Task("q", QUESTION), components)
    assert inference.cycles[0]["deficit"] is not None
    assert inference.cycles[0]["edit"] == Add("BridgeEntityChecker")
    assert inference.cycles[1]["chain"] == ["Planner", "BridgeEntityChecker", "OutputFormatter"] and inference.terminal_cycle == 1


def test_an_edit_is_proposed_only_after_rejection_and_each_cycle_applies_at_most_one():
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")})
    components = components_for(executor=executor)
    accepted = inference_module.test_time_inference(Task("q", QUESTION), components)
    assert accepted.terminal_cycle == 0 and accepted.cycles[0]["edit"] is None and chain_runs(executor) == 1
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("no")}, validator=[{"likely_correct": False, "confidence": 0.1, "deficits": []}] * 3)
    components = components_for(executor=executor)
    rejected = inference_module.test_time_inference(Task("q", QUESTION), components)
    chains = [cycle["chain"] for cycle in rejected.cycles]
    for before, after, cycle in zip(chains, chains[1:], rejected.cycles):
        assert cycle["edit"] is not None and RoleFlowChain(tuple(before)).apply(cycle["edit"]).roles == tuple(after)


def test_a_chain_is_executed_at_most_one_plus_T_max_times():
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("no")}, validator=[{"likely_correct": False, "confidence": 0.1, "deficits": []}] * 3)
    router = ScriptedRouter('<plan><chain>Planner -> OutputFormatter</chain><edit cycle="1">add: A</edit><edit cycle="2">add: B</edit><edit cycle="3">add: C</edit></plan>')
    components = components_for(executor=executor, router=router)
    inference = inference_module.test_time_inference(Task("q", QUESTION), components)
    assert chain_runs(executor) == 1 + components.config.maximum_cycles == 4 and inference.terminal_cycle == 3
    assert [cycle["verdict"] for cycle in inference.cycles] == [0, 0, 0, None]


def test_the_loop_stops_at_the_first_accepted_output():
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")}, validator=[{"likely_correct": False, "confidence": 0.1, "deficits": []}, {"likely_correct": True, "confidence": 0.9, "deficits": []}])
    components = components_for(executor=executor)
    inference = inference_module.test_time_inference(Task("q", QUESTION), components)
    assert chain_runs(executor) == 2 and inference.terminal_cycle == 1 and inference.prediction == "yes"
    assert sum(1 for call in executor.calls if call == "task validator") == 2


def test_inference_starts_from_the_seed_library_and_never_loads_the_grown_library(tmp_path):
    grown = RoleLibrary.load("roles/multihop_qa.json")
    grown.add(Role("GrownSpecialist", "grown", ("grown",)))
    (tmp_path / "hotpotqa" / "train").mkdir(parents=True)
    (tmp_path / "hotpotqa" / "train" / "role_library.json").write_text(json.dumps([role.to_dict() for role in grown]))
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")})
    config = load_config("configs/multihop_qa.yaml")
    config.data.output_dir = str(tmp_path)
    components = components_for(executor=executor, router=ScriptedRouter("<plan><chain>Planner -> GrownSpecialist -> OutputFormatter</chain></plan>"), config=config)
    assert components.seed_library.names() == ["Planner", "Validator", "OutputFormatter", "Retriever"]
    inference_module.test_time_inference(Task("q", QUESTION), components)
    inference_module.test_time_inference(Task("q", QUESTION), components)
    assert executor.calls.count("LibraryExpander") == 2 and "GrownSpecialist" not in components.seed_library


def test_inference_never_reruns_a_chain_without_the_edit():
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")}, validator=[{"likely_correct": False, "confidence": 0.1, "deficits": []}])
    components = components_for(executor=executor)
    inference = inference_module.test_time_inference(Task("q", QUESTION), components)
    assert chain_runs(executor) == 2 and inference.terminal_cycle == 1
    assert [cycle["chain"] for cycle in inference.cycles] == [["Planner", "OutputFormatter"], ["Planner", "BridgeReasoner", "OutputFormatter"]]


def test_single_edit_advantage_is_the_reward_difference_at_the_same_chain_state():
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": answer("yes")})
    components = components_for(executor=executor)
    overlay = RoleOverlay(components.seed_library)
    components.library_expander.synthesize("Verifier2", "context", overlay)
    state = ChainState(RoleFlowChain(("Planner", "OutputFormatter")), ScratchMemory(), overlay)
    pair = counterfactual_pair(state, Add("Verifier2"), Task("q", QUESTION), components)
    assert [item.role for item in pair.with_edit.trace] == ["Planner", "Verifier2", "OutputFormatter"]
    assert [item.role for item in pair.without_edit.trace] == ["Planner", "OutputFormatter"]
    assert state.chain.roles == ("Planner", "OutputFormatter")
    assert single_edit_advantage(1.5, 1.0) == pytest.approx(0.5)


def training_components(outputs, plan):
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": outputs})
    components = components_for(executor=executor, router=ScriptedRouter(plan))
    return components, executor


def test_only_edits_whose_advantage_exceeds_delta_are_committed_and_the_kept_execution_continues():
    assert isinstance(commit(Add("A"), 0.005, 0.005), NullEdit) and commit(Add("A"), 0.006, 0.005) == Add("A")
    answers = iter(["rome", "milan", "turin", "paris", "madrid"])
    components, executor = training_components(lambda user: answer(next(answers)), '<plan><chain>Planner -> OutputFormatter</chain><edit cycle="1">add: A</edit><edit cycle="2">add: B</edit></plan>')
    example = Example(Task("q", QUESTION), Reference("paris"))
    rollout = secpo_rollout(example, components.global_router.plan(example.task).text, 0, components.seed_library, components, components.config)
    assert [cycle["output"] for cycle in rollout.cycles] == ["rome", "turin", "paris"]
    weight = components.config.secpo.score_weight
    assert [edit.committed for edit in rollout.edits] == [False, True] and rollout.edits[0].advantage == pytest.approx(0.0)
    assert rollout.edits[1].advantage == pytest.approx(cycle_reward(Validation(1, 1.0), weight) - cycle_reward(Validation(0, 0.0), weight)) and rollout.terminal_cycle == 2
    assert rollout.cycles[1]["chain"] == ["Planner", "OutputFormatter"] and rollout.cycles[2]["chain"] == ["Planner", "B", "OutputFormatter"]
    assert rollout.return_ == pytest.approx(cycle_reward(Validation(1, 1.0), components.config.secpo.score_weight))


def test_uncommitted_pairs_still_reach_the_token_advantage():
    answers = iter(["rome", "oslo", "paris"])
    components, executor = training_components(lambda user: answer(next(answers)), '<plan><chain>Planner -> OutputFormatter</chain><edit cycle="1">add: A</edit></plan>')
    example = Example(Task("q", QUESTION), Reference("paris"))
    rollout = secpo_rollout(example, components.global_router.plan(example.task).text, 0, components.seed_library, components, components.config)
    edit = rollout.edits[0]
    assert not edit.committed and edit.advantage < 0 and edit.span is not None
    offsets = [(index, index + 1) for index in range(len(rollout.plan))]
    spans, advantages = edit_token_spans(rollout, offsets, len(rollout.plan))
    assert len(spans) == 1 and advantages == [edit.advantage] and spans[0] == list(range(edit.span[0], edit.span[1]))
    rows = token_level_advantage([[1.0] * len(rollout.plan)], {0: spans}, {0: advantages}, 1.0)
    assert all(rows[0][token] == pytest.approx(1.0 + edit.advantage) for token in spans[0])
    assert all(rows[0][token] == pytest.approx(1.0) for token in range(len(rollout.plan)) if token not in spans[0])


def test_lambda_A_is_added_only_on_the_edit_span():
    base = [[1.0] * 10, [0.5] * 10]
    out = token_level_advantage([row[:] for row in base], {0: [[2, 3]], 1: [[5], [7]]}, {0: [0.5], 1: [-1.0, 2.0]}, 1.0)
    assert out[0] == [1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert out[1] == [0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 2.5, 0.5, 0.5]
    assert token_level_advantage([row[:] for row in base], {0: [[2, 3]]}, {0: [0.5]}, 0.0) == base


def test_cycle_reward_is_verdict_plus_weighted_score():
    assert cycle_reward(Validation(1, 0.5), 2.0) == pytest.approx(2.0)
    assert cycle_reward(Validation(0, 0.3), 1.0) == pytest.approx(0.3)


def test_a_secpo_rollout_executes_at_most_one_plus_two_T_max_chains():
    components, executor = training_components(answer("rome"), '<plan><chain>Planner -> OutputFormatter</chain><edit cycle="1">add: A</edit><edit cycle="2">add: B</edit><edit cycle="3">add: C</edit></plan>')
    example = Example(Task("q", QUESTION), Reference("paris"))
    rollout = secpo_rollout(example, components.global_router.plan(example.task).text, 0, components.seed_library, components, components.config)
    assert chain_runs(executor) == 1 + 2 * components.config.maximum_cycles == 7 and rollout.terminal_cycle == 3


def fake_rollout(task_id, index, names, return_, library):
    overlay = RoleOverlay(library)
    for name in names:
        overlay.add(Role(name, f"You are {name}.", ("x",)))
    return SimpleNamespace(task_id=task_id, index=index, overlay=overlay, return_=return_, edits=[EditRecord(0, Add(name), 1.0, 0.0, 1.0, True, None) for name in names])


def test_the_return_gate_admits_roles_only_from_rollouts_that_beat_the_reference_rollout():
    library = RoleLibrary.load("roles/multihop_qa.json")
    rollouts = [fake_rollout("x", 0, ["RefRole"], 1.0, library), fake_rollout("x", 1, ["Better"], 1.5, library), fake_rollout("x", 2, ["Worse"], 0.5, library), fake_rollout("x", 3, ["Equal"], 1.0, library), fake_rollout("y", 0, ["Better"], 0.0, library), fake_rollout("y", 1, ["Better"], 0.1, library)]
    admitted = return_gate(library, rollouts)
    assert "Better" in admitted and "RefRole" not in admitted and "Worse" not in admitted and "Equal" not in admitted
    assert admitted.get("Better").system_prompt == "You are Better." and len(admitted) == len(library) + 1


def test_the_return_gate_changes_the_library_and_nothing_else():
    library = RoleLibrary.load("roles/multihop_qa.json")
    rollouts = [fake_rollout("x", 0, [], 0.0, library), fake_rollout("x", 1, ["New"], 1.0, library)]
    before = [(rollout.return_, [edit.advantage for edit in rollout.edits]) for rollout in rollouts]
    admitted = return_gate(library, rollouts)
    assert [(rollout.return_, [edit.advantage for edit in rollout.edits]) for rollout in rollouts] == before
    assert "New" in admitted and "New" not in library


def test_seed_libraries_have_the_paper_counts_and_capability_tags():
    counts = {"multihop_qa": 4, "mathematical_reasoning": 10, "code_generation": 8, "web_interaction": 8}
    for name, count in counts.items():
        library = RoleLibrary.load(f"roles/{name}.json")
        assert len(library) == count and all(role.system_prompt.strip() for role in library)
    assert {role.name: list(role.capabilities) for role in RoleLibrary.load("roles/multihop_qa.json")} == QA_TABLE
    assert {role.name: list(role.capabilities) for role in RoleLibrary.load("roles/code_generation.json")} == CODE_TABLE


def test_math500_takes_the_majority_and_asks_the_executor_when_no_answer_has_more_than_half():
    winner, counts, first = mathematical_reasoning.majority_answer(["\\frac{1}{2}", "0.5", "3", "1/2", "3"])
    assert winner == "\\frac{1}{2}" and counts["1/2"] == 3 and first["3"] == 2
    boxed = iter(["<answer>\\boxed{3}</answer>", "<answer>\\boxed{4}</answer>", "<answer>\\boxed{3}</answer>", "<answer>\\boxed{5}</answer>"])
    executor = ScriptedExecutor({"Planner": "plan", "OutputFormatter": lambda user: next(boxed)}, choice="I pick \\boxed{5}")
    config = load_config("configs/mathematical_reasoning.yaml")
    config.sampling.chains = 4
    components = components_for("mathematical_reasoning", executor, ScriptedRouter("<plan><chain>Planner -> OutputFormatter</chain></plan>"), config)
    record = components.domain.sample_and_select(Task("m", "What is 1 + 2?"), components)
    assert record["prediction"] == "5" and set(record) == {"prediction", "terminal_cycle", "cost"}
    assert executor.calls.count("answer choice") == 1 and len(components.global_router.model.requests) == 4


def test_program_selection_prefers_public_tests_then_weighted_written_tests():
    task = code_task()
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    wrong = "```python\ndef add(a, b):\n    return a - b\n```"
    partial = "```python\ndef add(a, b):\n    return a + b if a else b + 1\n```"
    written_only = "```python\ndef add(a, b):\n    return 0 if (a, b) == (1, 2) else a + b\n```"
    tests = ["assert add(2, 2) == 4", "assert add(0, 5) == 5"]
    assert code_generation.select_program([wrong, good, partial], task, tests) == 1
    assert code_generation.select_program([partial, good], task, tests) == 1
    assert code_generation.select_program([written_only, partial], task, tests) == 1


def test_the_code_validator_names_the_six_failures(monkeypatch):
    monkeypatch.setattr(code_generation, "TEST_TIMEOUT", 2)
    validator = code_generation.CodeTaskValidator(code_task(), None)
    cases = {
        "no code here": "NO_CODE",
        "```python\ndef add(a, b:\n    return a + b\n```": "SYNTAX_ERROR",
        "```python\ndef plus(a, b):\n    return a + b\n```": "MISSING_ENTRY_POINT",
        "```python\ndef add(a, b):\n    return a + b + undefined_name\n```": "RUNTIME_ERROR",
        "```python\ndef add(a, b):\n    return a - b\n```": "WRONG_ANSWER",
        "```python\ndef add(a, b):\n    while True:\n        pass\n```": "TIMEOUT",
    }
    for output, category in cases.items():
        validation = validator.assess(output, [])
        assert validation.verdict == 0 and validation.deficit.category == category, output
    assert validator.assess("```python\ndef add(a, b):\n    return a + b\n```", []) == Validation(1, 1.0)
    assert set(cases.values()) == set(code_generation.DEFICIT_CATEGORIES)


def test_the_webshop_validator_rejects_invalid_actions_and_keeps_a_purchase_reachable():
    product = {"has_search_bar": False, "clickables": ["Back to Search", "< Prev", "Description", "Buy Now"]}
    task = web_interaction.round_task(3, "buy a red mug", [], "Instruction: [SEP] buy a red mug [SEP] Back to Search [SEP] < Prev [SEP] Mug [SEP] Price: $9 [SEP] Description [SEP] Buy Now", product, 10)
    validator = web_interaction.WebShopTaskValidator(task)
    for reply in ("Thought:\nx\n\nAction:\nclick[Description]", "Thought:\nx\n\nAction:\nclick[Reviews]", "Thought:\nx\n\nAction:\nsearch[mug]", "no action"):
        validation = validator.assess(reply, [])
        assert validation.verdict == 0 and validation.deficit is not None, reply
    assert validator.assess("Thought:\nx\n\nAction:\nclick[Buy Now]", []) == Validation(1, 1.0)
    assert validator.replace("Action:\nclick[Description]").lower() == "click[buy now]" and validator.replace("Action:\nclick[Buy Now]") == "click[Buy Now]"
    earlier = web_interaction.round_task(3, "buy a red mug", [], "Instruction: [SEP] buy a red mug [SEP] Back to Search [SEP] < Prev [SEP] Mug [SEP] Price: $9 [SEP] Description [SEP] Buy Now", product, 9)
    rejected = web_interaction.WebShopTaskValidator(earlier).assess("Action:\nclick[Description]", [])
    assert rejected.verdict == 0 and rejected.deficit.category == "ROUND_BUDGET" and web_interaction.WebShopTaskValidator(earlier).replace("Action:\nclick[Description]") == "click[Buy Now]"
    in_time = web_interaction.round_task(3, "buy a red mug", [], "Instruction: [SEP] buy a red mug [SEP] Back to Search [SEP] < Prev [SEP] Mug [SEP] Price: $9 [SEP] Description [SEP] Buy Now", product, 8)
    assert web_interaction.WebShopTaskValidator(in_time).assess("Action:\nclick[Description]", []).verdict == 1
    results = {"has_search_bar": False, "clickables": ["Back to Search", "Next >", "B07XYZ1234", "B08ABC9876"]}
    late_results = web_interaction.round_task(3, "buy a red mug", [], "Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B07XYZ1234 [SEP] Mug [SEP] B08ABC9876 [SEP] Cup", results, 9)
    assert web_interaction.WebShopTaskValidator(late_results).assess("Action:\nclick[B08ABC9876]", []).verdict == 1
    assert web_interaction.WebShopTaskValidator(late_results).replace("Action:\nclick[Next >]") == "click[B07XYZ1234]"
    search = web_interaction.round_task(3, "buy a red mug", [], "WebShop [SEP] Instruction: [SEP] buy a red mug [SEP] Search", {"has_search_bar": True, "clickables": ["Search"]}, 1)
    assert web_interaction.WebShopTaskValidator(search).assess("Action:\nsearch[red mug]", []).verdict == 1
    assert task.extra["round"] == 10


def test_metrics_follow_their_definitions():
    assert multihop_qa.f1("the Paris city city", "paris city") == pytest.approx(2 * (2 / 3) * 1.0 / (2 / 3 + 1.0))
    assert multihop_qa.f1("yes", "no") == 0.0 and multihop_qa.f1("Yes", "yes") == 1.0
    judge = ScriptedExecutor(judge="YES")
    examples = [Example(Task("a", "p"), Reference("\\frac{1}{2}")), Example(Task("b", "p"), Reference("x+1"))]
    assert mathematical_reasoning.accuracy([{"prediction": "0.5"}, {"prediction": "1+x"}], examples, judge) == 1.0 and judge.calls == ["judge"]
    examples = [Example(code_task(), Reference(tests="assert add(1, 2) == 3\nassert add(-1, 1) == 0\n")), Example(code_task("humaneval-0"), Reference(tests="def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"))]
    good, wrong = {"prediction": "```python\ndef add(a, b):\n    return a + b\n```"}, {"prediction": "```python\ndef add(a, b):\n    return a * b\n```"}
    assert code_generation.pass_at_1([good, good], examples) == 1.0
    assert code_generation.pass_at_1([wrong, good], examples) == 0.5 and code_generation.pass_at_1([wrong, wrong], examples) == 0.0
    assert web_interaction.webshop_score([{"reward": 1.0}, {"reward": 0.25}], [None, None]) == pytest.approx(0.625)


def sample_run(tmp_path, executor=None):
    config = load_config("configs/sample.yaml")
    config.data.output_dir = str(tmp_path)
    components = components_for(executor=executor, config=config)
    run(config, "hotpotqa", components)
    return config


def test_scores_carry_only_the_paper_metric_names(tmp_path):
    config = sample_run(tmp_path)
    scores = evaluate(config, "hotpotqa")
    assert set(scores) == {"F1", "tasks"} and scores["tasks"] == 8


def test_a_failed_request_stops_the_run(tmp_path, monkeypatch):
    config = load_config("configs/sample.yaml")
    config.data.output_dir = str(tmp_path)
    components = components_for(executor=ScriptedExecutor(fail=True), config=config)
    with pytest.raises(APIError):
        run(config, "hotpotqa", components)
    monkeypatch.setattr(inference_module, "build_components", lambda config, clients=None, with_global_router=True: components)
    monkeypatch.setattr(sys, "argv", ["infer", "--config", "configs/sample.yaml"])
    with pytest.raises(SystemExit, match="a request failed"):
        inference_module.main()


def test_preflight_names_the_role_the_address_and_the_model():
    with pytest.raises(SystemExit) as error:
        preflight(LanguageModel(ModelConfig("gpt-test", api_base="localhost:1", api_key="EMPTY"), "executor"))
    assert all(word in str(error.value) for word in ("executor", "localhost:1", "gpt-test"))
    with pytest.raises(SystemExit, match="api_base not set"):
        preflight(LanguageModel(ModelConfig("gpt-test", api_base="", api_key="EMPTY"), "judge"))


def test_programs_run_in_a_temporary_directory_with_limits():
    result = run_program("import os, resource\nprint(os.getcwd())\nprint(sorted(os.environ))\nprint(resource.getrlimit(resource.RLIMIT_AS)[0], resource.getrlimit(resource.RLIMIT_CPU)[0], os.getsid(0) == os.getpid())\n", 10)
    directory, environment, limits = result["stdout"].splitlines()[:3]
    assert result["passed"] and not directory.startswith(str(ROOT)) and set(eval(environment)) <= {"PATH", "HOME", "LANG", "PYTHONHASHSEED"}
    assert limits == f"{MEMORY_LIMIT} {CPU_SECONDS} True"
    flood = run_program("print('x' * 1000000)\n", 10)
    assert len(flood["stdout"]) <= OUTPUT_LIMIT
    started = time.time()
    slow = run_program("import time\ntime.sleep(30)\n", 1)
    assert slow["timeout"] and not slow["passed"] and time.time() - started < 10


def test_inference_and_evaluation_run_end_to_end_on_the_samples(tmp_path):
    config = sample_run(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "hotpotqa" / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and rows[0]["id"].startswith("hotpotqa-") and set(rows[0]) == {"id", "prediction", "terminal_cycle", "cost"}
    scores = evaluate(config, "hotpotqa")
    assert set(scores) == {"F1", "tasks"} and 0.0 <= scores["F1"] <= 100.0 and scores["tasks"] == 8
    assert (tmp_path / "hotpotqa" / "scores.json").is_file()


def test_one_secpo_rollout_group_runs_on_the_samples():
    config = load_config("configs/sample.yaml")
    example = load(config, "hotpotqa", "train", with_reference=True)[0]
    plans = [
        '<plan><chain>Planner -> OutputFormatter</chain><edit cycle="1">add: EvidenceChecker</edit></plan>',
        '<plan><chain>Planner -> Reasoner -> OutputFormatter</chain><edit cycle="1">no_edit</edit></plan>',
        '<plan><chain>Planner -> OutputFormatter</chain><edit cycle="1">reroute: 0, 1</edit><edit cycle="2">add: Verifier</edit></plan>',
        '<plan><chain>Planner -> OutputFormatter</chain></plan>',
        'no plan',
    ]
    answers = iter([example.reference.answer if step % 3 == 2 else "wrong" for step in range(60)])
    executor = ScriptedExecutor({"Planner": "plan", "Reasoner": "reason", "OutputFormatter": lambda user: answer(next(answers))}, deficit_names=["GapFiller"] * 10)
    components = components_for(executor=executor, config=config)
    library = components.seed_library
    rollouts = [secpo_rollout(example, plan, index, library, components, config) for index, plan in enumerate(plans)]
    assert [rollout.index for rollout in rollouts] == [0, 1, 2, 3, 4] and all(rollout.task_id == example.task.id for rollout in rollouts)
    assert all(isinstance(rollout.return_, float) and 0 <= rollout.terminal_cycle <= config.maximum_cycles for rollout in rollouts)
    assert all(rollout.overlay.library is library for rollout in rollouts) and "Reasoner" in rollouts[1].overlay and "Reasoner" not in library
    assert all(len(rollout.edits) <= config.maximum_cycles for rollout in rollouts)
    assert all(edit.span is None or edit.span[1] <= len(rollout.plan) for rollout in rollouts for edit in rollout.edits)
    spans, advantages = {}, {}
    for index, rollout in enumerate(rollouts):
        spans[index], advantages[index] = edit_token_spans(rollout, [(position, position + 1) for position in range(len(rollout.plan))], len(rollout.plan))
    rows = [[0.0] * max(len(rollout.plan), 1) for rollout in rollouts]
    token_level_advantage(rows, spans, advantages, config.secpo.edit_advantage_weight)
    for index, rollout in enumerate(rollouts):
        touched = {token for tokens in spans[index] for token in tokens}
        assert all(rows[index][token] == pytest.approx(0.0) for token in range(len(rollout.plan)) if token not in touched)
        for tokens, advantage in zip(spans[index], advantages[index]):
            assert all(rows[index][token] == pytest.approx(config.secpo.edit_advantage_weight * advantage) for token in tokens)
    reference = rollouts[0].return_
    expected = {role.name for rollout in rollouts if rollout.return_ > reference for role in rollout.overlay.synthesized()}
    grown = return_gate(library, rollouts)
    assert set(grown.names()) == set(library.names()) | expected and len(library) == 4


def test_the_code_split_lists_name_the_paper_problem_counts():
    train, test, humaneval = code_generation.split_ids("mbpp", "train"), code_generation.split_ids("mbpp", "test"), code_generation.split_ids("humaneval", "test")
    assert (len(train), len(test), len(humaneval)) == (324, 341, 131) and not set(train) & set(test) and len(set(test)) == 341
    assert code_generation.docstring_example("    >>> iscube(1) ==> True\n", "iscube") == ["assert iscube(1) == True"]
    assert code_generation.docstring_example("    >>> add(1, 2) == 3\n    >>> add(0, 0)\n    0\n", "add") == ["assert add(1, 2) == 3"]
    assert code_generation.docstring_example("    For example:\n    add(2, 2) == 4\n", "add") == ["assert add(2, 2) == 4"]
    assert code_generation.docstring_example("    >>> is_happy(a)\n    False\n    is_simple_power(1, 4) => true\n", "is_happy") == []


def test_the_sample_data_matches_the_release_row_format():
    config = load_config("configs/sample.yaml")
    with open("data/samples/hotpotqa/train.jsonl", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    assert set(row) == {"id", "question", "extra", "reference"} and set(row["reference"]) == {"answer", "tests"}
    train, test = load(config, "hotpotqa", "train", with_reference=True), load(config, "hotpotqa", "test")
    assert len(train) == 24 and len(test) == 8 and all(example.reference.answer for example in train)
    assert not hasattr(test[0], "reference") and test[0].question


def test_verl_config_maps_the_paper_keys_and_the_training_data_carries_no_reference(tmp_path):
    from roleflow.train import VERL_SETTINGS, nested, training_data, verl_settings

    assert nested({"a.b.c": 1, "a.d": 2}) == {"a": {"b": {"c": 1}, "d": 2}}
    assert VERL_SETTINGS["algorithm.adv_estimator"] == "grpo"
    config = load_config("configs/multihop_qa.yaml")
    settings = verl_settings(config, tmp_path / "train.parquet", tmp_path)
    assert settings["actor_rollout_ref.model.path"] == "Qwen/Qwen2.5-7B-Instruct" and settings["actor_rollout_ref.rollout.n"] == 5
    assert settings["data.train_batch_size"] == 32 and settings["actor_rollout_ref.actor.optim.lr"] == 2e-6
    assert settings["actor_rollout_ref.actor.kl_loss_coef"] == 0.001 and settings["actor_rollout_ref.actor.use_kl_loss"] is True
    assert settings["actor_rollout_ref.actor.clip_ratio"] == 0.2 and settings["trainer.n_gpus_per_node"] == 4
    pytest.importorskip("pandas")
    config = load_config("configs/sample.yaml")
    components = components_for(config=config, router=ScriptedRouter())
    examples = load(config, "hotpotqa", "train", with_reference=True)
    path = training_data(components, examples, tmp_path / "train.parquet")
    import pandas as pd

    frame = pd.read_parquet(path)
    assert list(frame.columns) == ["data_source", "prompt", "extra_info"] and len(frame) == 24
    assert "reference" not in frame.columns and set(frame.iloc[0]["extra_info"]) == {"task_id", "index"}
    assert frame.iloc[0]["prompt"].tolist()[0]["role"] == "system" and examples[0].task.question in frame.iloc[0]["prompt"].tolist()[1]["content"]
    verl = pytest.importorskip("verl")
    from roleflow.train import verl_config

    settings = verl_config(config, path, tmp_path)
    assert settings.actor_rollout_ref.rollout.n == 5 and settings.data.train_batch_size == 8 and settings.actor_rollout_ref.actor.optim.lr == 2e-6
    assert settings.actor_rollout_ref.actor.kl_loss_coef == 0.001 and settings.actor_rollout_ref.actor.clip_ratio == 0.2 and settings.actor_rollout_ref.actor.use_kl_loss
