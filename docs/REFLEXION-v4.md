# 结构化反思与基线恢复

2026-10-22，`explore-v4-structured-reflexion` / `evidence-v4-structured-diagnosis`。

## 模型阶段

1. diagnosis：输出 clusters。每组记录 label、observed_failure、supporting_case_ids、cause_hypotheses、attribution_status、discriminating_test。每个原因假设单独记录 mechanism、supporting_observations、counter_evidence、confidence。无法支持原因时，允许 unresolved 和空假设列表；不要求编造归因。只检查结构、字段和证据引用，不自动裁定因果真实。
2. proposals：接收已验证诊断与完整原始训练证据，提出恰好三个机制不同的方向。保持 direction、hypothesis、prediction、risk、basis、rationale、mechanism、falsification_test、supporting_case_ids、supporting_cluster_labels、uncertainty。空案例链接可用于代码检查或探索性想法，不能冒充训练证据。
3. codegen 与候选完整回放继续执行。查询只读、允许普通 Python 管理任意内部记忆结构，原文 ID、冻结 actor/judge 和评分选择契约不变。

每阶段都获得完整父源码、训练全量分布、覆盖采样的失败与成功、干预记录，没有应用层预算或静默截断。no-diagnosis 只调用 proposals，并剥离训练成绩、案例、分布与干预，不运行诊断索引或 oracle 干预。

两个阶段继续使用 reflexion 的 8192 tokens、temperature .7、900 秒设置；codegen 仍为 16384/.4/900 秒。两阶段增加调用成本，所有调用和纠正计入优化器开销。服务上下文是物理上限；超限明确失败。

## 纠正与状态

`--reflexion-max-corrections` 默认 1，可设 0–2。先请求一次，若 JSON 或字段校验失败，以完整原输入、原响应和具体字段错误要求模型纠正。脚本不会把 medium 填成原因，不会替模型造提案；服务不可用立即向上传播。每次请求前写 running，随后记录原始响应、解析结果、校验错误、用量和状态。

`reflexion/diagnosis/attempt-001.json` 和 `result.json` 保存阶段记录，proposals 使用同样目录结构；`reflexion.json` 保存成功的组合结果。原始医疗内容只在超算留存。`reflexion.diagnosis`、`reflexion.proposals` 与实际 llm.reflexion generation 可以在轨迹中对应；diagnostic-input/proposal-input 标识实际阶段输入。

耗尽纠正写 reflexion-failure.json，run-status.json 与 summary.json 为 failed，CLI 非零退出，已有基线报告保留。进化结束与收益改善分开：completed 表示预定执行结束；evolution_outcome 描述候选是否严格评分完整，不代表性能提高。严格 AVG、补充 E2E、UCT、Pareto 未改变。

## 验证后复用基线

```bash
python -m rsi.cli --dataset /file_storage01/home/juanliu/25_ymj/code/MedMemoryBench \
  --output /file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/runs/NEW/result \
  --base-url http://127.0.0.1:7862/v1 --evolve \
  --semantic-model-path /file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/models/bge-small-zh-v1.5 \
  --resume-baselines-from /file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/runs/full-v03-local-152743/result
```

全量 Slurm 脚本可传 RSI_RESUME_BASELINES_FROM，保持固定实现 commit 与新 run ID。恢复前检查 verified 归档中必需文件的 SHA/size、config-hash、父源码哈希、冻结 runtime 指纹（历史无指纹则对照旧 Git commit）、数据 revision、语义模型 manifest、患者划分/问题/检查点、完整 replay 及评分计数。严格 AVG 为 null 但全部题已执行的基线可以复用，异常原样保留。任一不一致就失败，不自动降级成未经验证的复用。

只复制 BM25/BGE 的已验证 train/validation 报告、逐题与操作日志、进度和语义状态，另写 baseline-reuse.json；原结果、旧配置、树、提案和状态保持只读。重新建立诊断及干预，生成新提案与三个候选。每个候选从空记忆完整回放；不恢复旧候选 state，不打开 test。旧基线成本沿用原报告，新的模型调用成本单独记录，复用不伪造新基线调用轨迹。

## 验证边界

协议测试覆盖 medium 误填、纠正成功/耗尽、unresolved、证据引用、完整大输入、服务中断；恢复测试覆盖配置/归档/源码/语义模型/数据不一致、完整性计数、旧结果不变，以及 CLI 跳过基线并对全部候选新建状态。合成测试证明工程路径，不证明反思正确性或进化收益。

超算 Linux 完整回归：185 项通过运行，1 项跳过（2026-10-22）。152743 的 23 项必需归档文件、冻结 Git 源码与训练 772/验证 380 题已通过实际恢复预检；BGE 两折也完整。预检未调用模型，未读取 test。最终部署后若追加校验测试，以运行日志的最终统计为准。
