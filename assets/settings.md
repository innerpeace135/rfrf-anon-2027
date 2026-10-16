# SECPO training settings

| Symbol | Setting | Config key | Value |
|---|---|---|---|
| | GR backbone | `secpo.gr_backbone` | Qwen2.5-7B-Instruct |
| | Executor LLM | `executor.model` | gpt-4o-mini |
| K | Group size | `secpo.group_size` | 5 |
| | Tasks per training step | `secpo.tasks_per_training_step` | 32 |
| | Learning rate | `secpo.learning_rate` | 2 × 10⁻⁶ |
| β | KL coefficient | `secpo.kl_coefficient` | 0.001 |
| ε | Clip ratio | `secpo.clip_ratio` | 0.2 |
| T_max | Maximum cycles | `maximum_cycles` | 3 |
| δ | Commit threshold | `secpo.commit_threshold` | 0.005 |
| λ | Edit-advantage weight | `secpo.edit_advantage_weight` | 1 |

The remaining keys are implementation defaults.
