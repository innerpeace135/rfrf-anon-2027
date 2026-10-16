# Benchmarks

| Benchmark | Domain | Train | Test | Metric |
|---|---|---|---|---|
| HotpotQA | Multi-hop QA | 200 | 200 | F1 |
| 2WikiMultiHopQA | Multi-hop QA | 0 | 200 | F1 |
| MuSiQue | Multi-hop QA | 0 | 200 | F1 |
| MATH-500 | Mathematical reasoning | 200 | 500 | Accuracy |
| MBPP | Code generation | 324 | 341 | Pass@1 |
| HumanEval | Code generation | 0 | 131 | Pass@1 |
| WebShop | Web interaction | 0 | 500 | Score |

| Domain | Seed roles | Sampling and selection | Task validator |
|---|---|---|---|
| Multi-hop QA | 4 | One chain answers each question. | Scores the final answer and names the suspected failure, such as an unverified bridge entity, an unchecked time constraint or scattered evidence. |
| Mathematical reasoning | 10 | Twenty chains per problem and the majority answer; when no answer wins more than half of the votes, one call of the executor LLM picks one of the candidate answers. | Checks that the output contains a `\boxed{}` answer and asks the executor LLM, which does not see the reference answer, whether the solution holds. |
| Code generation | 8 | Ten chains per problem; the submitted program passes the most public example tests and, among those, the most unit tests that the executor LLM writes from the task statement, where a written test counts more when more candidates pass it. After a rejection, every role except the OutputFormatter receives a repair note. | Runs the public example test and names one of six failures: no code, a syntax error, a missing entry point, a runtime error, a wrong answer or a timeout. |
| Web interaction | 8 | The GR trained on MBPP runs one chain per round with the WebShop seed roles, for at most ten rounds per task. | Rejects an action that is invalid on the current page or that leaves too few rounds to finish a purchase; when the last reply of a round would still leave no purchase reachable, replaces it with the action that keeps a purchase possible. |
