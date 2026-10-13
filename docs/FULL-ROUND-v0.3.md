# v0.3 全量首轮运行

使用 Chinese Efficient 数据与 patient-v2-40-20-40-preserve-validation 划分：训练 8 位 / 772 题，验证 4 位 / 380 题，测试 8 位 / 787 题封存。按 10 至 100 全部检查点 replay，不设置每类题数或患者子集上限。

`scripts/full_round.sbatch` 在 gre 请求 2 GPU、16 CPU、128 GiB 与最多 48 小时。48 小时是作业上限，不是预计耗时。每个作业将提交时指定的 Git commit 克隆到独立目录，在分配的计算节点先验证对象快照、只读查询与 8/4/8 划分，再启动 loopback SGLang。实际启动或运行失败保留状态与退出码并清理该作业拥有的模型进程。

顺序评测 plain BM25、冻结本地 BGE semantic，训练诊断后生成三个子策略，各从头 replay 训练和验证。完整成功时共 5 × 1,152 = 5,760 次问答，另有最多 2 次训练证据干预及反思/代码生成调用。候选失败时记录失败节点，不能当作已完成评测。ReAct 和工程评分 rubric 保持冻结；这次运行不是论文官方评分复现。

原始 `query_id` 在不同患者之间重复。本轮在数据加载时形成患者＋检查点＋原始题号的唯一标识，原号保存为 `source_query_id`，避免反思和干预查表串到其他患者的参考答案。加载时检查唯一性。该改动不改变患者划分、问题、模型输入或评分准则；旧运行结果不覆盖。

各 replay 阶段逐题写入 `answers.jsonl` 与 `operations.jsonl`，并更新 `progress.json`，异常时保留已完成部分。当前支持观察与留档，不支持从中途检查点恢复。逐题医疗文本只存服务器；状态汇报读取聚合进度和成本。

调度器日志：`runs/full-JOBID.log`；阶段状态和预检：`runs/full-job-JOBID/`；冻结代码：`runs/code-JOBID/`；实验产物：`runs/full-v03-splitv2-JOBID/`。最终完整性以各阶段 `report.json`、`summary.json` 和候选失败记录核实，不能只依赖 Slurm 退出码。
