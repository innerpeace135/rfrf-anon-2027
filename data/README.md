# Data

`scripts/prepare.sh --benchmark <name>` fetches each benchmark from its public source, takes the number of training and test tasks that the benchmark catalog lists, and writes `prepared/<benchmark>/{train,test}.jsonl`. MBPP and HumanEval follow the AFlow split: `splits/` lists the task ids of the 324 training, 341 MBPP test and 131 HumanEval test problems. `samples/hotpotqa/` carries a few HotpotQA questions for the quick start.

| Benchmark | Source |
|---|---|
| HotpotQA | FlashRAG datasets (Hugging Face) |
| 2WikiMultiHopQA | FlashRAG datasets (Hugging Face) |
| MuSiQue | FlashRAG datasets (Hugging Face) |
| MATH-500 and MATH | HuggingFaceH4/MATH-500 and EleutherAI/hendrycks_math (Hugging Face) |
| MBPP | google-research-datasets/mbpp (Hugging Face) |
| HumanEval | openai/openai_humaneval (Hugging Face) |
| WebShop | AgentBench WebShop environment |
