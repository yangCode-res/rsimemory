# 连续策略搜索 v1

状态：机制已接入，真实 GPU 搜索未启动。现有 153107 继续按原单轮协议执行。

## 固定配置

配置模板 deploy/evolution-search.json：最多新增 6 个策略（代码无效也占预算），每次展开 2 个，最大深度 3（BM25 为 0 代），探索系数 1.414。代码通过显式 --search-profile 读取，不会因文件存在自动启动。--continuous 也可直接启用相同默认预算。

奖励为开发集 validation E2E AVG，允许 actor 未作答计零；judge 失败/缺类导致 E2E 缺失的策略不进入搜索前沿。训练与验证严格 AVG、E2E 和异常保持原始报告口径；严格训练不完整不再阻止有效开发集奖励回传。验证集已用于搜索，不是独立泛化证据。test 不参与诊断、奖励或搜索，继续封存。

## 实际算法

这是有界 UCT 前沿搜索，不是完整的多层 MCTS 路径采样：每次比较所有未展开、未达深度上限且奖励有效的节点，以 Q+c*sqrt(log(max(1,parent_visits))/visits) 选最高者；并列按创建顺序。每个版本最多展开一次。Q 含节点自身开发成绩以及后代回传成绩；不等于单独该版本的验证准确率。BGE 为固定比较基线，不进入进化树。

选父后读取该父的完整源码和训练 answers/report，重新组织 12 失败/4 成功诊断覆盖（数据不足时按实际），先 diagnosis 后 proposals，再按提案生成子代码。后代没有复用 BM25 诊断。连续模式暂不运行旧 intervention，因为其实现绑定 BM25；max_interventions=0 明确写入配置。提案、训练诊断仅使用训练内容；开发成绩只驱动控制器选择。

每次两个子策略从空患者状态分别完成 train→validation，可按现有候选 worker 并行。生成重复/无效策略保留失败状态且占预算，不无界重试；模型服务故障传播至作业失败。没有可展开节点、候选预算用完或可选活动时间上限到达时停止。不承诺预算 6 一定达到第 3 代。无提升停机未启用。

## 恢复与审计

search-state.json 用同目录临时文件加原子替换保存；search.lock 使用 flock 防止并发驱动。检查配置、划分、选定问题及训练/开发会话、rsi 运行代码哈希；已登记的策略/报告/答案文件有 SHA256。变更则拒绝恢复，不把不兼容结果拼接进树。

完成评测和奖励回传有独立标志；中断后不重复回传。已完成的候选可以复用；半途的回放移动至 interrupted-attempts 后从空状态重新执行，不从某道题中间恢复。恢复是阶段级而非单题级；可能消耗额外推理。活动时间限制为软限制，仅在展开之间检查，不能当 GPU 硬时限；Slurm 作业时限独立负责硬限制。当前活动时间限制默认关闭，候选数仍受限。

每次 expansion 保存 selection_snapshot、父与子 ID、诊断、proposal 和 attempt。策略编号 candidate-0001 等仅在 run 命名空间内唯一，policy-git 提交的 parent 为真实父提交。

## 使用方式（仅说明，未提交作业）

合成：`.venv/bin/python -m rsi.cli --fixture --search-profile deploy/evolution-search.json --output runs/NEW-FIXTURE`

真实直接执行时保留现有数据、模型与独立副本参数，再添加 `--search-profile deploy/evolution-search.json`。Slurm 脚本支持环境变量 RSI_CONTINUOUS_SEARCH=1；未设置时维持原单轮。既有基线复用仍须 --resume-baselines-from 的原有校验。

恢复：`.venv/bin/python -m rsi.cli --resume-search --output SAME_OUTPUT`。恢复采用归档 config.json 的冻结参数；不可用它改变预算、模型或协议。基线阶段尚未产生 search-state.json 的故障不能用这个命令恢复。

## 观测

中文总览显示代数、真实父版本、开发奖励告知、各次展开的选择快照和诊断聚类。原生 Langfuse 总览附带 search 历史，父子指标差值从实际父报告读取。已存在的 151576 v2 投影保持不变，不能按新算法解释旧树。多轮每次诊断完整 artifact 位于 expansions/expansion-NNN；当前原生聚类节点主要展示第一次展开，后续聚类摘要可在中文总览按展开查看，完整内容在服务器 artifact，尚未逐轮生成独立原生聚类节点。

## 验证记录

完整 Linux 回归 212 项通过、1 项跳过（含8项新连续搜索测试）；最终观测改动后19项定向复查通过，页面脚本语法检查通过。合成端到端 runs/fixture-continuous-20261022-final 完成6候选、3次展开，candidate_budget停止；6个Git提交父链均匹配，根访问次数为7（自身基线一次+6候选回传）。完成后 CLI --resume-search 成功且不新增候选或重复回传。合成结果不作为真实进化收益证据。
