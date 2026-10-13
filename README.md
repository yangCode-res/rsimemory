# Memory Policy RSI

目标架构见 [PRD（当前接口协议 v0.3）](docs/PRD-v0.2.md)、[接口契约 v0.3](docs/INTERFACE-CONTRACT.md) 和 [架构图](docs/architecture-v0.2.png)。

接口契约已升级到 v0.3：policy state 现在是由隔离 worker 恢复和序列化的 cloudpickle 对象快照，宿主只保存 base64 不透明快照。v0.3 的新运行须用新的 run ID 与版本记录留档，不覆盖 v0.2 的 `VALIDATION` 或 `RUNTIME` 记录。v0.3 对象快照与只读查询已实现，验收见 [v0.3 验收记录](docs/VALIDATION-v0.3.md)；当前运行环境见 [RUNTIME v0.3](docs/RUNTIME-v0.3.md)。

[运行环境](docs/RUNTIME-v0.2.md)与[工程验收记录](docs/VALIDATION-v0.2.md)已留档：真实基线与反思已验证；当前真实生成的三个代码候选被契约检查拒绝，尚无成功进化策略。合成候选闭环已通过。

## 运行

```bash
python -m unittest discover -s tests
# 合成协议测试，不是性能实验
python -m rsi.cli --fixture --output runs/fixture-001 --evolve
# 真数据工程冒烟：每类一题，只使用训练和验证各一位患者，最晚检查点20
python -m rsi.cli --dataset /file_storage01/home/juanliu/25_ymj/code/MedMemoryBench \
  --output runs/live-001 --base-url http://127.0.0.1:7860/v1 \
  --checkpoint 20 --patients-per-split 1 --per-type 1 --evolve
```

真实调用需要 OpenAI 兼容服务。默认 SGLang 是计算节点 loopback，应在同一 GPU 作业内调用。密钥用 RSI_API_KEY 环境变量提供，不写配置。当前 actor、judge、optimizer 共享 Qwen 服务仅用于工程验证；judge 未经人工校准。

## 链路

执行器按患者和时间 ingest 原文 chunk；每题从同一检查点状态生成独立工具视图。ReAct 只使用 search/read/final。宿主绑定患者和时间，候选不能扩大 filters。预览与完整读取不包含题型或答案标注。

EEM/MQ规则评分，其余四类采用自定义严格 JSON judge。异常保留 null，缺类或评分异常时 AVG 为 null。这不是论文官方 rubric 的完整复现。训练失败与成功对照进入 Reflexion；定向证据干预单独记录。三个生成代码候选通过 Linux 隔离运行并从头 replay，验证成绩不回传 UCT。policy-git 保存每个代码提交与父子关系，失败节点不参与 Pareto。

## 产物

config.json、splits.json、run_patients.json、各阶段 answers.jsonl/operations.jsonl/report.json、reflexion.json、interventions.json、branches/candidate-N/policy.py、tree.json、summary.json。逐题医疗数据留在服务器，不默认导出；测试集本轮不评估。

## 限制

- 首轮 Efficient；Mixed 映射未验证，入口拒绝运行。
- BM25 与可选的本地语义检索使用相同原文切分；语义模型未提供则 skipped，缺依赖明确失败。
- 候选通过普通 Python 入口管理任意 state 对象，可使用已安装依赖、类、lambda、异常处理和推导式。候选 state 不要求声明可序列化；宿主在实际快照时用 cloudpickle 尝试，失败就是该运行的错误。当前 `search` 仍只返回已登记的原文 ID；派生文本输出需另行设计 lineage 与记录 schema。
- `search` 对每次查询从同一检查点快照恢复独立副本，任何副本内变更均不持久化；只有 `ingest` 推进宿主保存的快照。ReAct 的 search/read 协议、原文 ID 范围及答案和未来信息隔离保持冻结。
- v0.3 的候选运行目标为 Linux 独立 worker：Landlock 只允许读取 Python runtime，阻止读取私有数据文件和所有文件写入；seccomp 限制网络与进程控制等系统调用。失败时 fail closed；不声称这是通用安全审计或完整多租户安全保证。Linux 运行测试已验证上述边界；这不替代通用安全审计。
- 初版根节点只展开一次；无诊断对照随后通过 --no-diagnosis 独立补跑。
- 当前上下文采用字符限额，精确 tokenizer 预算待接。fixture 的模拟模型和 judge 只验证协议，不是科学成绩。
- Reflexion 输入采用完整父版本代码与受限诊断摘录，工程上限 24000 字符；截断显式标记。调用失败写入 reflexion-failure.json，训练或验证评分不完整的版本不参加 Pareto 或训练收益回传。
- v0.3 的 cloudpickle 恢复不得发生在宿主编排进程；仅可在隔离 worker 完成 Landlock/seccomp 限制后进行。无法恢复或重新快照的对象按运行错误归档，不作兼容性假设。

当前真实数据划分为训练 8 位、验证 4 位、测试 8 位，seed=20261013；原验证患者保持不变。完整记录见 [数据划分 v2](docs/DATA-SPLIT-v2.md)，历史运行保持原样。

全量首轮使用 `sbatch scripts/full_round.sbatch`，流程、逐题进度留档与唯一题号修复见 [全量运行记录](docs/FULL-ROUND-v0.3.md)。
