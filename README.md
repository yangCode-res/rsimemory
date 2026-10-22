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

EEM/MQ规则评分，其余四类采用自定义严格 JSON judge。异常保留 null，缺类或评分异常时严格 AVG 为 null。另报 `E2E_AVG`：未作答的 `actor_failed` 计 0，仍按六类等权平均，并保留 `abnormal_rate`；judge 异常和缺类仍不可评分，详见[指标说明](docs/METRICS-E2E.md)。这不是论文官方 rubric 的完整复现。训练失败与成功对照进入 Reflexion；定向证据干预单独记录。三个生成代码候选通过 Linux 隔离运行并从头 replay，验证成绩不回传 UCT。policy-git 保存每个代码提交与父子关系，失败节点不参与 Pareto；补充指标不替换既有自动选择规则。

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
- Reflexion 输入使用完整父源码、全量训练诊断分布与覆盖采样案例，当前目标 12 个失败＋4 个成功；不设应用层输入预算，不截断字段/列表，也不按预算删除案例。物理服务上下文超限明确失败并留档。调用失败写入 reflexion-failure.json，训练或验证评分不完整的版本不参加 Pareto 或训练收益回传。
- v0.3 的 cloudpickle 恢复不得发生在宿主编排进程；仅可在隔离 worker 完成 Landlock/seccomp 限制后进行。无法恢复或重新快照的对象按运行错误归档，不作兼容性假设。

当前真实数据划分为训练 8 位、验证 4 位、测试 8 位，seed=20261013；原验证患者保持不变。完整记录见 [数据划分 v2](docs/DATA-SPLIT-v2.md)，历史运行保持原样。

全量首轮使用 `sbatch scripts/full_round.sbatch`，流程、逐题进度留档与唯一题号修复见 [全量运行记录](docs/FULL-ROUND-v0.3.md)。

轨迹分析采用服务器本地 JSONL 记录与独立 Langfuse 导入，接入方式、历史记录的证据边界见 [Langfuse 轨迹说明](docs/LANGFUSE.md)。

优化器当前使用独立生成参数：Reflexion 输出上限 8192 tokens、temperature 0.7，代码生成输出上限 16384 tokens、temperature 0.4，超时均为 900 秒；actor/judge 仍为 1024 tokens、temperature 0。可通过 `--reflexion-max-tokens`、`--codegen-max-tokens`、`--reflexion-temperature`、`--codegen-temperature`、`--optimizer-timeout-s` 调整，最终参数写入运行配置和模型轨迹。全量脚本也接受对应的 `RSI_REFLEXION_*`、`RSI_CODEGEN_*`、`RSI_OPTIMIZER_TIMEOUT_S` 环境变量。输出预算不能突破服务端总上下文（全量脚本当前为原生 262144 tokens）；更大的父版本需要先核对输入长度，不能保证任意代码都能在一次生成内完成。

历史 `explore-v1` 反思提示以全任务 AVG/成本前沿为目标，允许替换记忆组织、索引、检索和 ingest 内部的更新/遗忘机制，不奖励低风险或最小改动；三个候选需探索不同机制，在有合理假设时包括结构性重构。输入增加父版本实际启用的配置和训练聚合指标，防止围绕未启用模块推断瓶颈。`evidence-v3` 提案在理由和证据引用之外记录 `mechanism`、`falsification_test`，取消每字段 150 字符限制。选出的四个失败与两个成功案例不再被外层列表隐式裁为五个，当时受显式 24000 字符输入预算约束，当前已取消。

2026-10-19 的 `explore-v2` 修复父代码打包遗漏：初始父策略使用 `standalone-baseline-v1`，携带实际 imports、传递依赖（含 `_grams`）和 initialize/ingest/search 入口，可在独立隔离 worker 内执行。新运行记录 `parent-policy.py`、`parent-source.json` 及 config 源码 SHA256；反思与代码生成均获得完整父代码、实际语言和基线分词说明。允许采用其他分词、索引或记忆结构，不增加实现形式白名单。历史候选的分词修复另存有父 commit 的独立策略，不覆盖原运行，也不作为新生成的 RSI 成果。

`finish_reason=length` 归档为 `generation_truncated`，保留实际模型响应与 token 用量，不再误记为契约错误；不完整代码不进入 replay。本次变更只作用于新运行，历史结果及 Langfuse 导入记录保持原样；参数与提示的调整本身不代表真实性能已改善。

2026-10-21 的 `explore-v3-full-diagnostics` 修复诊断选样和输入压缩：完整训练日志建立索引，按患者、任务、早中晚检查点和可观察错误组覆盖采样，保留成功对照；每个新 run 的 `diagnostics/` 保存 index、summary、selection 和完整 examples。移除旧的 24000 字符上限及所有摘录/删案例逻辑，`reflexion_input_budget=null`。全量 Slurm 脚本使用已核验模型原生上下文 262144，actor/judge 请求预算不变。五项问题与本轮只改 1、2 的范围见 [problem.md](problem.md)。

2026-10-22 的 `explore-v4-structured-reflexion` 将诊断与三个提案分为两个模型阶段，分别校验。诊断区分可观察失败、原因假设、支持/反证、置信度与判别测试；无法归因可用 unresolved 与空假设列表。每阶段默认最多纠正一次；完整输入、响应、字段错误保存在 `reflexion/{diagnosis,proposals}/`。纠正耗尽会写 failed 状态并以非零退出，不再吞掉异常后产生“成功的空进化”。恢复可用 `--resume-baselines-from <已验证旧result>`，仅在归档、冻结配置/代码、数据/患者/问题一致且基线完整时复用基线；三个候选仍从空状态 replay。协议与恢复说明见 [结构化反思](docs/REFLEXION-v4.md)。

2026-10-22 并行执行更新：全量 Slurm 默认申请 6 GPU、48 CPU、256 GiB，启动三个独立 TP=2 SGLang 副本；按 Slurm CUDA_VISIBLE_DEVICES 的实际分配拆分双卡，端口7862/7863/7864，缓存、日志、健康监控各自独立。离线诊断和三个代码生成保持顺序依赖；准备完候选后，三个候选各绑定一个副本并行执行 train→validation。每个候选仍从空记忆完整回放，actor/judge 参数、题目、评分、UCT/Pareto 不变，test继续封存。主线程按提案顺序合并树与收益，轨迹上下文及模型客户端独立。详见 docs/PARALLEL-EXECUTION.md。

2026-10-22 输出预算与默认接入更新：所有新任务默认读取 configs/llm-roles.yaml（可用 RSI_LLM_CONFIG / --llm-config 选路径），配置缺失或非法即失败。actor=4096、judge=1024、diagnosis/proposals/codegen max_tokens=null；null不发送max_tokens，已核验本地SGLang会按剩余物理上下文限制生成，不回退小默认预算。正常EOS及超时仍生效。CLI、全量Slurm、smoke及新连续搜索均接入，保存YAML快照和生效参数。历史运行保持只读；actor预算变化要求新基线，不能直接复用旧1024预算分数。详见docs/LLM-ROLES.md和聚合用量JSON。
