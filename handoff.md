# Memory Policy RSI：项目交接说明

**2026-10-22 当前状态**：六卡并行版实现 commit `55ecef4baa084280664f16bd6551946ee7ff0c4e` 已部署，作业 **153107** 已在 `gpu22` 运行，Slurm 实际分配6 GPU、48 CPU、256 GiB。三个独立 TP=2 SGLang 副本对应7862/7863/7864，候选并行执行各自 train→validation。当前为模型启动阶段，须查服务就绪记录，不能当作已开始候选或已有新成绩。归档 runs/full-v03-local-153107，启动回执 runs/rerun-launch-153107.json。

反思协议已为 explore-v4-structured-reflexion：诊断/提案分阶段、允许unresolved、每阶段一次纠正、失败非零退出。完整训练诊断与无输入预算保留。旧基线152743已再次核验可复用；双卡153104按用户切换请求停止并归档verified，没有开始候选回放。最新完整候选实验仍是151576。

Linux完整回归198项、1跳过；真实隔离worker三候选合成CLI闭环全部evaluated（仅协议验证）。新并行流程与资源竞争说明见docs/PARALLEL-EXECUTION.md。下面2026-10-21的启动说明为历史记录，以本段及实际run元数据为准。

**2026-10-21 当前整改**：五项问题记录于 [problem.md](problem.md)。本轮仅改问题 1、2：全量训练诊断索引/分布＋覆盖采样 12 失败/4 成功，保留完整选中案例，删除应用层输入预算与全部摘录/预算删案例逻辑。当前提示 `explore-v3-full-diagnostics`、选样 `training-diagnostics-coverage-v1`，`reflexion_input_budget=null`。旧文中“最多六例”“24000 字符摘录”仅描述历史版本，不再用于新运行。Slurm 服务默认上下文扩至已核验模型原生 262144，冻结 actor/judge 请求参数不变；问题 3、4、5 后续再做，评分/奖励不变。最新完成性能实验仍为 151576，新运行状态另记于 problem.md 和 runs/ 的启动记录。

本轮新作业 **152743** 已提交并在 `gpu16` 启动，固定实现 `ad5b615`；结果路径 `runs/full-v03-local-152743/`。提交前 Linux 检查 159 项通过、1 项跳过，真实训练日志预演完整覆盖 8 患者/6 类任务/早中晚，12 失败＋4 成功完整传入。首次作业 152742 因 `gpu37` 节点暂存磁盘满在模型启动前退出。以上是启动记录，后续是否完成、各候选成绩和实际新诊断覆盖需读取对应 run，不能预先宣称改进有效。

更新日期：2026-10-21，Asia/Shanghai。最新完整实验为 Slurm 作业 **151576**；运行使用代码 commit `a4d8bd2f79458ad0faa54af24125bdef541aeec5`。这是接手说明，不替代实际代码、冻结的 run config 或原始运行记录。

## 1. 项目目标与研究边界

在冻结回答 agent 的条件下，通过离线 Reflexion 生成、执行和评估新的记忆管理 policy，提升 MedMemoryBench 上的整体记忆管理性能，并同时报告 token、耗时和存储成本。

- 研究对象是记忆管理的代码与 policy，包括记忆表示、组织、索引、摄入、内部 CRUD、检索和排序。不是模型参数训练，也不是改写回答 agent 的流程。
- MCD（multi-hop clinical deduction）只是一个错误模式和设计案例；不能仅优化 MCD 而把其他任务退化忽略。主性能口径是六类任务等权 AVG。
- 进化空间应保持开放和启发式，不限定候选只能调 BM25 参数、只能使用特定数据结构或只能做小改动。因果链、图、摘要、实体索引等均为可探索方向，不是当前已实现的通用能力。
- 此前讨论的 WHO-UMC 可以作为药物—不良事件关联的领域启发式设想；当前没有接入，不得把它表述为已经验证的通用因果推断方法。
- 每个 policy 必须可版本化为一套代码，并保留父子关系、假设、修改、运行与结果。修改后从空状态完整 replay；状态迁移后续再做。
- 当前为工程研究原型。自定义 judge 未经论文级人工校准，结果不是官方评测的完整复现；开放式记忆管理进化本身的研究新颖性仍需近邻工作比较。

## 2. 三个主要部件与反馈

1. **冻结的 ReAct 回答链路**：决定需要什么信息，调用 `search`、`read`，最终回答。模型、prompt、工具协议与预算在同一比较中保持一致。
2. **可进化 memory management policy**：决定如何保存、组织、更新和查找信息。固定执行器按时间调用 `ingest`，回答模型没有写入工具。
3. **离线 Reflexion 与代码进化**：从训练 replay 的成功/失败、工具轨迹、检索证据与干预结果提出错误聚类、归因、不确定性、修改假设和候选代码。

数据流：历史会话 → 执行器按时间摄入 policy → 冻结快照 → 每题独立 ReAct → search/read 证据 → 回答 → 离线评分与成本汇总。训练诊断再进入 Reflexion 和代码生成；每个子策略从头重放训练、验证。

反馈至少区分答案得分、题型分数、未作答/评分异常、工具行为、检索来源命中代理、管理成本与反事实证据干预。检索来源命中不是完整事实召回率，也不是回答准确率。模型提出的归因和预测是假设，必须由 replay、消融或受控实验检验。

## 3. 服务器、路径与存储

SSH 配置别名为 `chaosuan`，用户 `juanliu`；通常登录管理节点 `choud-mgt-09`。GPU 任务通过 Slurm 提交，不能把管理节点当作 GPU 节点。部分 Slurm 分区查询会提示权限不足，可用具体作业的 `scontrol` / `sacct` 核查状态。

| 内容 | 超算路径 |
|---|---|
| 权威代码/Git 仓库 | `/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi` |
| MedMemoryBench | `/file_storage01/home/juanliu/25_ymj/code/MedMemoryBench` |
| Qwen 权重 | `/file_storage01/home/juanliu/25_ymj/memory/memory/vendor/openresearcher/models/Qwen3.8-27B` |
| 项目 Python | 项目根目录 `.venv/bin/python` |
| SGLang Python | `/file_storage01/home/juanliu/25_ymj/code/sglang-qwen38/venv/bin/python` |
| BGE 权重 | 项目根目录 `models/bge-small-zh-v1.5` |
| 实验归档 | 项目根目录 `runs/` |
| Langfuse 持久数据 | 项目根目录 `observability/langfuse/` |

用户已明确要求：**不使用 `/persist_data` 保存项目和实验数据**。持久内容统一在 `/file_storage01`。当前脚本允许节点 `/tmp` 用作可重建缓存、编译文件及运行暂存，每 60 秒同步实验产物到持久目录，结束后校验归档；不能将仅存在于 `/tmp` 的文件称为已持久保存。

本地 ChatGPT 工作区中的 `memory-policy-rsi/` 是开发文件镜像，没有本地 `.git`；Git 版本以超算仓库为准。工作区 `sources/` 是只读同步资料，不修改。不要将本地镜像或旧文档中的状态当作服务器实时状态。

真实医疗会话、题目、答案、逐题轨迹、诊断摘录、模型提案和由其生成的候选代码留在超算。当前既有自动审批边界不允许将这些内容导出到电脑或云端；只返回必要的聚合数值、错误类别、哈希和结构统计，不通过编码、附件或命令输出绕过。公开的一方框架代码和不含敏感样本的文档可以同步。密码、API keys 和私有 `.env` 不输出、不提交 Git。

## 4. 数据、任务和划分

- 数据集：Hugging Face `Cyan27/MedMemoryBench`；下载时使用中国镜像的用户偏好继续有效，具体 revision 以服务器 `download_manifest.json` 为准。
- 当前运行语言中文，模式 **Efficient**。Mixed 尚未适配，英文没有完成本轮正式运行，不能混报。
- 划分版本：`patient-v2-40-20-40-preserve-validation`，seed `20261013`。按患者整条轨迹划分，不能按问题随机打散。

| 集合 | 患者数 | 中文题目数 |
|---|---:|---:|
| train | 8 | 772 |
| validation | 4 | 380 |
| test | 8 | 787 |

中英文采用相同患者划分。保留原来的 4 位验证患者；具体患者列表仅保存在服务器划分 manifest 和各 run 的 `splits.json`，不复制进说明文档。训练集用于诊断和候选生成，验证集用于候选比较；**测试集保持封存，尚未运行**。

六类任务为 `entity_exact_match`、`temporal_localization`、`state_update`、`multiple_choice`、`inference_generation`、`multi_hop_clinical_deduction`。EEM 和 MQ 用规则判分，其余四类使用自定义严格 JSON judge。

## 5. 冻结参数与基线含义

实际冻结值以每个 run 的 `config.json` 为准。目前完整轮次使用：

| 项目 | 当前设置 |
|---|---|
| actor / judge / optimizer 服务模型 | `qwen3.8-27b`，工程上共享 Qwen 服务 |
| actor、judge 输出预算与 temperature | 1024 tokens，0 |
| ReAct 工具步数 | 4；另有一次 final-only 收尾机会 |
| 回答上下文预算 | 16000 字符，尚非精确 tokenizer 预算 |
| 原文 chunk | 1200 字符，overlap 120 |
| 返回 top-k | 5 |
| Reflexion | 8192 tokens，temperature 0.7，timeout 900 秒 |
| codegen | 16384 tokens，temperature 0.4，timeout 900 秒 |
| SGLang 总上下文 | 32768 tokens，输入与输出合计 |
| 当前思考/推测解码 | thinking / speculative decoding disabled |
| 候选辅助模型 provider | 未启用，不能候选直连服务绕过计量 |

`plain-bm25` 是初始 plain-record 路线：保存原文 chunk，以 Unicode 归一化的字符 bigram 做 BM25 检索；没有训练 BM25 模型参数。BM25 建索引和计算语料统计量不等于模型训练。

`plain-semantic` 使用冻结的 `BAAI/bge-small-zh-v1.5` 在本地做语义 embedding 检索，没有针对本任务微调。权重 revision 记录于模型 manifest；此前冻结 revision 为 `7999e1d3359715c523056ef9478215996d62a620`。两条基线采用相同原文切分。

扩大 optimizer 输出预算不自动扩大 actor 预算或服务上下文。生成 `finish_reason=length` 归档为 `generation_truncated`，不能误报为接口契约违规。

## 6. Policy 接口与运行边界

```python
initialize(config)              # 返回全新状态
ingest(state, session)          # 返回推进后的状态
search(state, query, filters)   # 返回排序后的已登记原文 ID 列表
```

候选可使用普通 Python 类、实例、lambda、集合、图、异常处理及已安装库，不设置任意语法或实现白名单。state 不限于 JSON。`GeneratedPolicy.serializable = True` 表示宿主提供快照能力；实际无法保存/恢复仍是运行错误，不能宣称所有对象都兼容。

v0.3 采用 cloudpickle 对象快照。宿主只保存不透明 base64 快照及哈希，**不在编排进程反序列化候选 pickle**。恢复只在已安装 Linux Landlock/seccomp 限制的独立 worker 内完成。候选不能读取私有数据文件、联网、写文件或操控外部进程；这不是对候选实现方式的白名单，也不是通用安全审计保证。

每次 search 都从同一检查点快照恢复独立副本，调用后丢弃其变更；只有 ingest 推进持久状态。查询缓存如果只在 search 内写入，不会跨调用保留。每题独立对话；患者、版本、replay 状态互相隔离。

ReAct 只看见 `search(query, filters)` 和 `read(ids)`。policy 决定内部检索实现；宿主验证返回 ID，提供原文预览/全文。在线输入不包含题型、gold、答案来源映射、评分信息、未来会话或其他患者内容。派生摘要可以在 policy 内部使用，但当前工具只返回已登记的原文 ID；将派生文本直接展示给 actor 需要后续扩展 lineage 与接口。

默认候选 RPC timeout 30 秒、内存 2048 MiB；这些是当前执行保护参数，不是已经确定的研究成本阈值。隔离失败时停止候选，不降级为宿主直接执行。

## 7. 进化、版本树与评价规则

- 当前只展开一次根节点，生成 **3 个子候选**，每个各做一次完整 train/validation replay，不是每个重复 3 次或 5 次。
- 版本树记录 MCTS/UCT 所需字段和父子关系；当前不能宣称已经验证多轮 MCTS 搜索收益。
- 根版本代码必须是带 imports、helper 依赖及入口的完整可执行模块。当前格式 `standalone-baseline-v1`，提示 `explore-v2`，归因 schema `evidence-v3`。
- 保存 `parent-policy.py`、`parent-source.json`、父源码 SHA256、配置、候选代码 commit 与父指针。不能只向优化器传 `inspect.getsource(PlainPolicy)`。
- Reflexion 只用训练诊断，保留聚类、证据引用、归因不确定性、机制、风险、预期结果和证伪检查。当前案例摘录受显式预算限制，不等于模型逐题阅读了全部训练轨迹。
- 候选违反冻结 actor/权限边界即失败；不能为让候选通过而改回答 agent、注入答案或扩大数据范围。
- 当前只有训练和验证严格 AVG 都完整的候选标为 `evaluated` 并进入收益回传/候选选择；验证分数不回传 UCT。`incomplete_scoring` 不等于代码生成失败或契约违规。
- 无错误诊断对照 `--no-diagnosis` 是同父版本、生成模型和预算下去掉诊断的**进化消融**，与不做进化的 plain-record 基线不同；已同意后续补跑，当前未完成。

### 指标

`AVG = 六类准确率之和 / 6`，不是全部题目直接合并的 micro accuracy。

保留严格 `AVG`：评分不完整或缺类即为 null。增补 `strict-plus-e2e-v1`：`actor_failed` 且 score 为 null 的未作答在 `E2E_AVG` 中计 0，保留原题型分母，再做六类等权平均。judge 异常、非法分数、未知题型或缺类不能补 0；未完成 replay 不能报完整 E2E。

同时报告 `abnormal_rate`、未作答计零数量、各类成绩与样本数。E2E 是补充报告，尚未替换严格节点有效性、UCT 或 Pareto 规则。历史补算写到新目录，不覆盖原报告和归档。

成本分别报告 memory policy、actor、judge、Reflexion 和 codegen。policy 包括 token、时间、状态存储；辅助模型未启用需明确注明 disabled。候选耗时包含 worker 启动、隔离、序列化与恢复，不能与宿主内基线直接比较后全部归因于检索算法。还没有预设成本罚项或淘汰阈值。

## 8. 代码与文档地图

| 文件 | 职责 |
|---|---|
| `rsi/cli.py` | 当前完整运行入口、配置、基线、反思、候选与汇总 |
| `rsi/data.py` | 数据加载、患者划分、题目选择 |
| `rsi/runtime.py` | 顺序 replay、检查点、逐题日志和进度 |
| `rsi/actor.py` | 冻结 ReAct、工具派发、收尾与协议状态 |
| `rsi/policy.py` | PlainPolicy、GeneratedPolicy、快照与成本适配 |
| `rsi/policy_source.py` | 完整父模块及依赖打包、源码 manifest |
| `rsi/sandbox.py` | Linux worker、Landlock/seccomp、RPC |
| `rsi/evolution.py` | Reflexion、codegen、版本树、回传、Pareto |
| `rsi/evaluation.py` | 规则/judge 判分、严格 AVG 和 E2E 汇总 |
| `rsi/model.py` | 模型请求、角色参数与调用成本 |
| `rsi/tracing.py` | 本地 JSONL 轨迹记录 |
| `rsi/trace_backfill.py` | 历史记录重建，显式标注记录/推断/缺失 |
| `rsi/policy_repair.py` | 历史 tokenizer-only 人工修复工具 |
| `scripts/full_round.sbatch` | Slurm 完整轮次、模型启动、同步与归档 |
| `scripts/preflight_v03.py` | Linux 对象快照、隔离与中文父策略预检 |
| `scripts/service_monitor.py` | 健康监测与中断诊断 |
| `scripts/round_storage.py` | 节点暂存、持久同步、SHA256 归档 |
| `scripts/report_e2e.py` | 对旧运行只读补算 E2E |
| `scripts/diagnose_candidate3_retrieval.py` | 三个历史候选的固定输入检索诊断；文件名保留旧名 |
| `scripts/repair_candidate_tokenizers.py` | 生成历史人工修复分支及 provenance |
| `scripts/export_langfuse.py`、`scripts/backfill_traces.py` | 轨迹导入与历史补建 |
| `deploy/langfuse/hpc.py` | 超算自部署服务管理 |

主要文档：[PRD](docs/PRD-v0.2.md)、[接口](docs/INTERFACE-CONTRACT.md)、[v0.3 运行环境](docs/RUNTIME-v0.3.md)、[划分](docs/DATA-SPLIT-v2.md)、[完整轮次](docs/FULL-ROUND-v0.3.md)、[指标](docs/METRICS-E2E.md)、[全部旧候选诊断](docs/DIAGNOSIS-147547-ALL-CANDIDATES.md)、[修复](docs/REPAIR-147547-TOKENIZER.md)、[轨迹](docs/LANGFUSE.md)、[部署](deploy/langfuse/README.md)。部分文档保留历史状态，不能把其中“尚未运行”“契约拒绝”等旧记录套用到新 run；文件名仍带 v0.2 的 PRD/接口正文已包含 v0.3 更新。

## 9. 运行与核查

下列命令在超算项目根目录执行。GPU 服务必须在分配到的计算节点启动，不在管理节点启动。

```bash
cd /file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi

# 修改后的必要检查；Linux 才能完整验证候选隔离。
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/preflight_v03.py \
  --dataset /file_storage01/home/juanliu/25_ymj/code/MedMemoryBench

# 合成模型只检查工程闭环，不是实际性能实验；输出目录必须全新。
.venv/bin/python -m rsi.cli --fixture --evolve --output runs/fixture-NEW-ID

# 完整轮次：先提交所需代码，再固定 commit，避免排队期间代码漂移。
RSI_RUN_COMMIT=$(git rev-parse HEAD)
sbatch --export=ALL,RSI_IMPLEMENTATION_COMMIT="$RSI_RUN_COMMIT" scripts/full_round.sbatch

# 查看指定作业，替换作业号。
sacct -j 151576 --format=JobID,State,ExitCode,Elapsed,NodeList -P
scontrol show job 151576 -o
```

脚本当前默认 `gre` 分区，6 GPU、48 CPU、256 GiB、48 小时时限。模型服务在作业节点 `127.0.0.1:7862/7863/7864`，各副本 tensor parallel 2；三个候选并行评估。常规归档是 `runs/full-v03-local-JOBID/`，CLI 产物在其 `result/` 内；Slurm 外层日志是 `runs/full-JOBID.log`。

查看进度优先读 `meta/status`、`meta/exit-code`、`meta/last-sync-status` 和各阶段 `progress.json`，不要向用户输出整份临床日志。查看结果读 `summary.json`、各阶段 `report.json`、`tree.json`，并校验 `_archive-manifest.json`；不要直接打印含提案和案例的整个 summary/tree。作业 `COMPLETED` 只代表脚本退出成功，还需核对候选状态、题目完成数和评分异常。

当前健康探针 timeout 30 秒，连续失败至少 3 次且持续至少 120 秒才因探针故障终止；真正服务进程退出则及时停止。一次 health 503 不足以判定模型崩溃。监测记录 `meta/service-health.jsonl`。模型启动等待上限 30 分钟。

修改按范围做必要检查；不要因文档修改重复跑完整 GPU 实验。历史运行目录只读，使用新 run ID；手工修复与自动生成分支分别标识。

## 10. 历史故障与已完成修复

1. **对象/语法约束过强**：v0.3 已改为普通 Python 对象与 cloudpickle，取消任意实现白名单，保留查询只读与执行边界。不要恢复旧 JSON-only 方案。
2. **生成截断与探索不足**：优化器输出预算、独立 temperature、明确失败类别和探索提示已接入；actor/judge 参数不随之改变。
3. **磁盘与存储位置**：禁止使用 `/persist_data`；实验产物同步到 `/file_storage01`，保留最终校验。
4. **健康监测误中断**：旧作业 147145 曾被单次探针失败提前终止，现用连续失败与宽限期处理。不要将监测清理导致的 SIGTERM 反向解读为模型先崩溃。
5. **旧作业 147547 候选严重退化**：三个候选把中文字符 bigram 改成空白切词，中文问题常变成一个 token，BM25 全零，排序退化为偏向最近会话。固定输入干预支持该分词退化结论。
6. **父源码遗漏**：原先只提供 PlainPolicy 类，缺少 `_grams` 等依赖。现在生成链路提供完整父源码、语言和分词行为。遗漏与后续错误补全均已观察到，但没有只改父源码的受控生成实验，不将其称为唯一原因。
7. **异常使严格 AVG 为空**：已增加未作答计零的 E2E 和异常率；未悄悄改动严格选择规则。

147547 的三个 tokenizer-only 人工修复保存在 `runs/repairs-147547-20261019/`。每个新 commit 直接以原候选为父，保留其其他代码；仅完成合成和固定输入的 36 题检索诊断，命中率分别为 50.00%、52.78%、50.00%，不是完整回答 AVG。**151576 是重新生成的另一组候选，不是这三个人工修复策略的完整回答评测。**

修复阶段 Linux 回归检查为 131 项、1 项跳过；该记录对应当时修复版本，不替代今后改动后的重新验证。

## 11. 最新真实结果：151576

作业状态 `COMPLETED`、exit `0:0`，节点 `gpu16`，耗时 **17:54:52**。实现 commit `a4d8bd2f79458ad0faa54af24125bdef541aeec5`，运行配置记录代码目录干净；父源码格式及 `explore-v2` 已确认实际生效。持久归档：

`/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/runs/full-v03-local-151576/`

提交说明为 `runs/rerun-launch-151576.json`。最终归档 manifest 标记 verified；读取结果时抽核的 25 个配置/报告/元数据文件 SHA256 均匹配。训练 772、验证 380 题；测试集未运行。

| 策略 | 训练 E2E AVG | 验证 AVG = E2E AVG | 验证 policy 秒数 | 验证状态峰值字节 |
|---|---:|---:|---:|---:|
| plain-bm25 | 32.4292% | 35.3624% | 120.03 | 3,268,818 |
| plain-semantic | 33.4169% | 38.9215% | 278.05 | 13,725,220 |
| candidate-1 | 31.9418% | 36.2770% | 1,460.40 | 6,503,927 |
| candidate-2 | 31.9780% | 35.8395% | 1,231.04 | 6,510,241 |
| candidate-3 | 31.8057% | 34.9375% | 3,092.54 | 12,412,911 |

新候选方向分别为混合字符 n-gram、启发式重排、实体索引/检索扩展；这些是本轮方向概述，不意味着提案中的全部预测或实现描述都得到验证。

- 五个版本验证集均完成 380/380 题评分，异常率 0，因此验证严格 AVG 与 E2E 相同。
- train：BM25 未作答 3/772，候选 2、3 各 2/772；其严格训练 AVG 为 null。候选 1 和 BGE 全部评分完整。
- candidate-1 为 `evaluated`；candidate-2/3 为 `incomplete_scoring`，原因是训练未作答，并非代码生成或契约检查失败。生成候选均已执行训练、验证。
- candidate-1 验证较 BM25 +0.9146 个百分点，较 BGE -2.6445 个百分点；训练 E2E 低于 BM25。其 MCD 验证 23.0769% → 33.3333%，但 MQ 27.50% → 21.25%。不能仅凭 MCD 改善宣称总体进化成功。
- 新候选没有重现旧轮约 17–18% 的严重退化，但两轮候选代码不同，这不是仅改变 tokenizer 的配对回答消融。
- 候选管理成本明显增加，且当前计时混合了算法与隔离开销；尚无重复测量或显著性检验支撑稳定优势。
- 当前汇总 Pareto 列表包含 BM25、BGE 和候选 1；这不表示候选 1 在准确率上胜过 BGE。

本轮候选 commit：candidate-1 `a9a18258cb4262d48fc056fc2d62c7f4b7e7e5cf`；candidate-2 `4060bed93b0c5fca320c57ddd38484f419b76af9`；candidate-3 `68ab0ec717a4230fdb85190ca0ab58df04bcdf62`。这些属于 run 内 `result/policy-git/`，与框架代码仓库 commit 区分。

## 12. Langfuse：如何看进化动机

超算自部署 Langfuse v4.59 OSS，使用 Podman，数据库/对象/密钥都在 `observability/langfuse/`。`/tmp/rsi-langfuse-1035/` 只存运行 PID/socket 和可重建缓存。服务绑定服务器 loopback，不能假定浏览器本机 localhost 自动就是超算。

从电脑建立 SSH 转发后访问 `http://localhost:3000/project/memory-policy-rsi`：

```bash
ssh -N -L 127.0.0.1:3000:127.0.0.1:3000 \
  -L 127.0.0.1:9092:127.0.0.1:9092 chaosuan
```

已有转发时勿重复占用端口。登录失效会进入 sign-in；这是认证状态，不足以说明实验失败。凭据从既有私有配置取得，不在本文或聊天中展示。服务器可运行 `python3 deploy/langfuse/hpc.py status` 查看服务。

分析顺序：训练 `diagnostic_bundle` → Reflexion 的聚类/证据/不确定性 → proposal 的机制、支持案例和证伪预测 → codegen 实际输入/输出 → 对应候选 replay 的表现与成本。来源引用和理由是模型报告的动机，不是已经证明的因果关系。

新运行先写本地 `result/traces.jsonl`，**不会因运行完成自动出现在 Langfuse**。导入是独立步骤，需要保存回执，再核对服务实际入库；不能只凭 HTTP 接受就宣称 UI 已完整更新。当前 151576 未确认导入，旧浏览器中的 reflection 不能当成本轮轨迹。

146133 的历史导入已核对 4615 个 observation、4609 条 trace、4608 条问题记录。历史缺失的请求、时间、归因不得补造；区分 `historical_recorded`、`historical_reconstructed`、`unavailable`。历史导入时间轴不能用于计算原始执行延迟。另有合成 fixture 记录，不能混入真实实验结果。

新 trace 可用 `scripts/export_langfuse.py`，旧运行可用 `scripts/backfill_traces.py`。密钥通过私有环境载入；导入默认仅 loopback。保留原回执，避免重复提交或在部分接受后盲目重传。详见轨迹与部署文档。

## 13. 后续工作与接手顺序

以下为待办，不表示已授权无条件开启额外 GPU 任务或测试集：

1. 以 151576 为基点，审计 Reflexion 的证据、归因和预测为何未转化为总体增益；先做只读聚合、固定输入诊断和受控干预，避免只按提案文字判断实现。
2. 检查训练反馈可见性和类型成绩呈现，避免把严格 AVG 为 null 误解为整个任务类型“未测量”。任何修复使用新 run，不覆盖旧结果。
3. 对管理成本拆分 worker 启动、快照/序列化、摄入、检索与状态体积，建立公平比较；不为降成本改变冻结回答协议。
4. 如需观察本轮动机链，独立导入并核验 151576 traces；明确区分新旧运行。
5. 按既定方案补 no-diagnosis 对照；多轮 MCTS/UCT、更多父版本、状态迁移和 Mixed 适配后续再实现。
6. 若需要因果隔离修复收益，单独完整评测历史 tokenizer-only 分支；不要用本轮重新生成的候选替代该实验。
7. 校准 judge、精确 token/context 预算、控制重复测量与统计分析，并与相关记忆进化工作比较；在此之前不夸大论文级性能或新颖性。
8. 测试集继续封存，需在方案和候选冻结后再安排。

新接手先读本文和接口 → 确认 SSH/持久目录 → 查看 Git 状态及当前作业 → 核对对应 run 配置、完成数、异常与归档 → 再按用户当前请求行动。不要自动重跑历史实验，不覆盖已有记录，不把尚未实现的计划写成完成。

## 14. 2026-10-21 Langfuse 接入修复

151576 已通过新导入工具写入自部署 Langfuse。`runs/langfuse-import-151576-v2/verification.json` 标记 verified：47,198 条 observation、17,320 条 Score 的 ID/值/关联、22,397 次 generation 的 usage、四种 system prompt 关联、environment 均一致，无缺失/重复。统一 session 为 full-v03-local-151576，正式数据 environment=research；fixture 单独隔离。第一次后台因 PostgreSQL 连接耗尽丢 7 条，确认 core/full 两表均无记录后已单独补交；原始与恢复回执都保留。

新增脚本 scripts/langfuse_run.py（派生并导入）、scripts/verify_langfuse_run.py（实际入库逐项对账）、scripts/link_langfuse_experiments.py（将已存在的问题轨迹关联原生 Dataset/Experiment，不重跑模型、不读 test）；详细流程见 docs/LANGFUSE.md。后续 CLI 唯一 run ID 与 memory.ingest span 已修复。新运行仍需在 Langfuse 所在管理节点完成独立导入与验证，不能仅凭作业结束声称 UI 已入库。

自动登录入口为电脑 http://localhost:3001/，依赖 3000/3001/9092 SSH 转发。hpc.py up 启动本机回环自动登录辅助服务，保持服务器端凭据。ClickHouse 限额现为 12 GiB，Prisma 每进程连接池 8，避免管理节点 CPU 数导致默认池过大。美元价格仍未配置：$0 不表示本地计算无成本。严格 AVG、E2E AVG、每题均值保持不同含义；Prompt 仅注册实际 system messages，完整动态输入仍在 generation。

原生 v4 Experiments 亦已完成：1,152 个 train/validation Dataset item；10 个策略/fold 运行各自的 772/380 条 item→trace→observation 关联均经 v4 API 分页读回核对。experiment-verification.json 为 true；之后重新核对原观测与评分仍全部一致，无重复。旧 legacy 实验关联 API 尝试回执保留，不作为成功依据。

进化优先总览已启用：配置 deploy/langfuse/observability.json；151576 总览 trace af3877e628d0165b9fa003fcfde87bf9，环境 research-overview。4 个派生节点、18 个聚合指标已核验，6 个父策略训练案例引用均解析成功。scripts.langfuse_run 后续导入会自动生成总览，原始采集仍完整保留。展示配置不改变进化预算、actor、选择规则或测试集边界；hypothesis_verdict 不自动裁定因果。

患者分组已启用：151576 的 5,760 条问题归入 12 个 run_id/patient-匿名编号 Session，同患者跨策略合并；全局进化与干预保留运行层。入口为 Sessions，Session ID contains full-v03-local-151576/patient-。新运行采集 patient 标签且后续导入自动按患者分组，历史已导入记录由 scripts.patient_view 增强并保留 v4 实验关联；不新增回答或 Score。当前实时导入/自动刷新仍未实现。

患者匿名编号已映射为 Langfuse 原生 User ID；患者 Session 分组保持不变，后续新导入自动继承。已有运行通过 patient-view 的独立 User ID 增强回执补齐。

## 2026-10-22 结构化反思与基线恢复

旧运行 152743 的 BM25/BGE 训练、验证已完成，但反思三个 attribution 均为 medium 而被协议拒绝，未生成候选。旧 CLI 吞异常导致 Slurm completed/0，与业务失败不一致；旧 run 不改写。

当前修复：explore-v4-structured-reflexion / evidence-v4-structured-diagnosis，先 diagnosis 后 proposals，原因假设与置信度独立；unresolved 不需编造原因。完整 12/4 案例、父代码和全训练 summary 在两个阶段均保留，不新增输入预算。每阶段最多一次纠正，reflexion/{diagnosis,proposals}/ 保存完整 attempt/result。失败写 run-status.json、summary.status=failed 并非零退出；候选正常执行完但无可评分候选另记 no_fully_scored_candidates，不伪称改进成功。

--resume-baselines-from 仅复用经归档哈希、config-hash、冻结 runtime 文件、模型/数据/划分、完整进度/问题映射检查的基线产物；baseline-reuse.json 记录来源。新 run 重新诊断、代码生成，三个候选仍空状态完整回放，不恢复旧 policy 快照，不读 test。旧基线调用不伪造成新调用轨迹，成本保留原报告。原生 262144 服务上下文、8192/.7 反思与16384/.4代码生成输出设置不变。详见 docs/REFLEXION-v4.md。

修复部署 commit：d94afbefa5c652d1f8f9248ed7d4ba6914ce7cfb。Linux 完整回归 185 项、1 跳过，最终补充恢复/失败退出测试 20 项通过。实际旧基线预检两次均 verified（23 个必需文件、772/380 题、BM25/BGE 完整）。恢复作业 153104 已提交，指定 gpu16，从 152743 完整基线恢复；归档 runs/full-v03-local-153104，启动回执 runs/rerun-launch-153104.json。此记录仅表示已提交，不代表候选已产生或准确率已提高；后续须查 run-status.json、各阶段 progress、归档校验与 Slurm 状态。

153104 调度修正：已通过 scontrol 原地解除 gpu16 绑定（ReqNodeList 为空），仅排除此前暂存磁盘满的 gpu36/gpu37。作业号与实验配置保持不变，调度变更已追加到 runs/rerun-launch-153104.json；实际启动仍以 Slurm 状态为准。

2026-10-22 并行执行更新：全量 Slurm 默认申请 6 GPU、48 CPU、256 GiB，启动三个独立 TP=2 SGLang 副本；按 Slurm CUDA_VISIBLE_DEVICES 的实际分配拆分双卡，端口7862/7863/7864，缓存、日志、健康监控各自独立。离线诊断和三个代码生成保持顺序依赖；准备完候选后，三个候选各绑定一个副本并行执行 train→validation。每个候选仍从空记忆完整回放，actor/judge 参数、题目、评分、UCT/Pareto 不变，test继续封存。主线程按提案顺序合并树与收益，轨迹上下文及模型客户端独立。详见 docs/PARALLEL-EXECUTION.md。


## 2026-10-22 Evolution overview v2

Configuration now selects evolution-overview-v2. Each candidate shows proposal direction, referenced error clusters, strict train/validation AVG and E2E AVG, abnormal rate, per-type deltas, cost, and a recorded-tree search-state snapshot. Native Scores include train/validation results plus finite Q, visits, exploration bonus and UCT. Missing scores stay null. Infinite priorities are explicit strings, never zero or non-standard JSON Infinity.

Diagnostics is a separate parent node with one child per model-reported cluster. It separates whole-training observed error proxies from sampled model attribution, shows sample coverage, cited-case counts, original attribution/uncertainty/discriminating tests, candidate links and resolvable parent-training case links. Cited examples do not estimate causal prevalence; patient maps and raw case text are omitted from aggregate summaries. Old and structured-v4 cluster schemas are both supported.

UCT is calculated from the recorded tree using the executor formula, not a historical selection event. selected_next stays null and selection_history=not_recorded; this view does not introduce multi-round search or change rewards/eligibility. Q includes descendant backpropagation, not necessarily the node's own baseline accuracy. Future actual selection history requires explicit executor event capture.

151576 v2 published as trace 65e077747ce5676001d4efca41f7e20b: 8 observations, 40 scores, server-side verification=true with zero missing/duplicate/field mismatches. Existing v1 and original execution records retained. The overview reads compressed derived traces left by log cleanup. Restore other compressed import payloads before reusing tools that require uncompressed filenames.

Future scripts.langfuse_run imports automatically publish v2. This is still independent import, not live synchronization. Running 153107 is unchanged and has not been presented as a completed or imported v2 run. Nine overview tests pass, covering numerical UCT parity, null/infinite states, cluster evidence links, sampling boundaries and compressed input.


## 2026-10-22 分层实验观测入口

默认实验入口改为 http://localhost:3001/overview；http://localhost:3001/ 仍先自动登录，再跳转总览。由现有 loopback 辅助服务提供，只读取 runs 的配置、summary、progress、reports、tree 和诊断摘要，不改 Langfuse 上游源码或数据库结构。配置记录于 observability.json 的 review_entry。

总览默认选最新有报告的正式运行，排除 fixture，每 30 秒读取归档快照。首屏为状态、候选数、当前验证最高值（包含固定基线，注明单次结果）、已核验总览导入状态。主表按策略展示提议方向、父版本、训练/验证 E2E 与严格 AVG、异常率、相对父策略百分点差、Q/N/UCT。题型变化、模型预测/风险和成本折叠。复用基线明确提示，未有报告用进度替代且不填零。

错误诊断区区分全训练可观测分组与模型抽样归因，只显示聚类标签、引用案例数、归因状态和关联候选；不导出患者原始 ID/病例正文。已通过 verification.json 的 v2 总览才开放候选、聚类的直接详情链接。患者 Sessions 链接仅能查看已导入的记录，不承诺当前运行已入库。底层原始轨迹保留；实时总览刷新不等于 Langfuse 实时采集。

新增 deploy/langfuse/research_dashboard.py 与 tests/test_research_dashboard.py。11 项定向测试通过，页面实际浏览器渲染及轮次切换验证通过；HTTP overview=200，登录=303 到 /overview。辅助服务已重启，数据库/模型服务/Slurm 作业均未重启。


## 2026-10-22 连续搜索接入

用户确认最多6候选、每次2个、最深3代，允许验证参与选择并作为开发集；实现后不启动GPU。新增 rsi.continuous 与 CLI --search-profile/--continuous/--resume-search；配置 deploy/evolution-search.json，Slurm仅 RSI_CONTINUOUS_SEARCH=1 显式启用。奖励固定 development validation E2E AVG，test封存；UCT未展开前沿每节点只展开一次，记录真实选择、原子检查点与防重复回传，部分回放保留后重跑。旧干预绑定BM25，连续模式明确禁用。中文总览支持血缘、开发集提示与每次展开诊断。详见 docs/CONTINUOUS-SEARCH.md。真实连续搜索未启动，153107不变。


## 2026-10-22 YAML五节点模型入口

新增 --llm-config configs/llm-roles.yaml；actor/judge/diagnosis/proposals/codegen 可独立model/base_url/生成参数与api_key_env。diagnosis/proposals显式按阶段路由，调用统计与原generation角色兼容。并行副本仅在未指定URL时继承；解析结果与hash归档，resume拒绝配置变化，基线复用校验actor/judge实际角色配置。Slurm显式RSI_LLM_CONFIG启用。不自动部署模型/服务，不改当前153107，无新GPU作业。详见docs/LLM-ROLES.md。


## 2026-10-22 首次真实连续搜索153572

用户授权启动。作业153572已提交并确认RUNNING，节点gpu22；执行提交a25958cdd075df08afc90b6222686a932982b99a。显式启用连续搜索与configs/llm-roles.yaml：6新候选、每次2个、深度3、c=1.414、开发集validation E2E AVG奖励，测试集封存。6GPU/3个TP2副本；因旧基线缺少YAML角色指纹，本次不复用基线，重新评测BM25/BGE。预检隔离/快照/中文检索通过。启动回执runs/rerun-launch-153572.json，归档runs/full-v03-local-153572。RUNNING只表示已调度，模型健康及具体阶段仍需检查；不代表候选已生成或进化有效。
