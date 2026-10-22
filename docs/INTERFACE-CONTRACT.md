# Memory Policy Interface Contract

2026-10-22 更新：离线链路升级 `explore-v4-structured-reflexion` / `evidence-v4-structured-diagnosis`，诊断与三个提案分阶段；观察、机制假设、置信度分字段，允许 unresolved。每阶段默认一次有限纠正，完整记录各次请求/响应/字段错误；耗尽以 failed、非零退出。经校验的基线可在新 run 复用，候选仍从空状态完整 replay。冻结回答协议、policy 实现自由度与评分/奖励规则不变。详见 [结构化反思与恢复](REFLEXION-v4.md)。

2026-10-21 更新：优化器诊断使用完整训练分布与覆盖采样的 12 个失败＋4 个成功；应用层输入预算为 null，取消字段/列表摘录和超预算删案例。完整父源码与全部选中案例保留，物理 provider 上下文超限作为显式失败留档。提示版本为 `explore-v3-full-diagnostics`；policy v0.3 入口、查询只读、冻结 actor、隔离和评分规则不变。详见 [problem.md](../problem.md)。

当前 v0.3 工程实现与验证见 [VALIDATION-v0.3](VALIDATION-v0.3.md)；历史 v0.2 记录保持原样。

版本：0.3　日期：2026年10月13日　状态：对象快照协议升级；v0.3 实现与运行验证另行记录，正式参数待冻结。

本契约依据 `docs/architecture-v0.2.png` 定义固定执行器与候选 memory policy 的边界，以及固定 ReAct 模型可调用的只读工具。该图已确认为 target architecture：在线运行、固定评测/反馈、离线进化和红色有效性约束都属于目标结构。本文区分目标协议和首轮实现限制，不表示安全沙箱或全部接口已实现或验证。

## 1. 信任边界

固定执行器拥有原始会话、患者与检查点上下文、问题/答案/来源标注、评分器、存储、工具派发和模型 provider。候选 policy 在隔离 worker 内运行普通 Python，使用本次调用传入的 state 与会话；它不获得私有数据文件或答案库的读取权限。编排宿主只保存 base64 编码的不透明 cloudpickle 快照；可以解码字节计算长度和校验和，但从不反序列化候选 pickle。恢复只发生在独立 worker 完成隔离设置之后。

回答模型只获得 `search(query, filters)` 和 `read(ids)` 两个工具。二者只读。不得暴露 `create`、`update`、`delete`、`merge`、`write`、`ingest`、`export` 等会改变记忆或访问外部状态的工具。记忆更新只能由固定执行器按回放流程调用 policy 函数完成。

## 1.1 Target architecture 中的接口位置

在线链路由固定执行器、可进化 policy、冻结 ReAct、独立评分器及评测/成本汇总组成。执行器把历史会话送给 policy；ReAct 仅经 `search/read` 读取证据，输出答案与工具轨迹；离线评分器从隔离数据读取标准答案和来源标注。操作与检索日志进入训练诊断，Reflexion 形成聚类/归因和定向 replay，之后生成三个修改假设及子 policy 代码。边界检查/隔离把违规或运行失败候选送入失败归档，将有效候选送入版本树及一次 UCT 展开；候选比较依据验证集汇总与成本形成报告。以上是目标图中的组件关系，本契约只定义接口边界，不代表该链路已全部接通。

## 2. Policy Python 入口与对象快照

policy 接口固定为三个普通 Python 入口；以下签名表示调用约定，不限制函数实现形式：

```python
initialize(config)
ingest(state, session)
search(state, query, filters)
```

候选可以使用 Python 类/实例、lambda、异常处理、推导式和运行环境中已安装的库，state 可以是任意 Python 对象；宿主将 state 与候选模块共同保存，以维持类身份和模块全局状态的一致性。候选依赖及版本随每个 run 冻结；隔离 worker 的 OS 能力边界仍禁止通过文件、网络或进程能力获取私有数据。

### `initialize(config)`

创建一个全新 policy state。config 仅含冻结的非敏感运行参数和策略参数，不含问题答案、参考来源标注、其他候选状态或未来数据。返回值可以是任意 Python 对象。宿主为每位患者和每个候选单独调用，不复用其他患者、父候选或前次 replay 的 state。

### `ingest(state, session)`

把固定执行器按时间顺序提供的一段历史会话纳入 policy，并返回推进后的 state。它可以使用任意普通 Python 结构在内部提取、合并、更新、删除和重排派生记忆或索引；这些 CRUD 操作是 policy 内部实现细节，不是回答模型工具。

session 只包含允许 policy 处理的原始历史内容与必要的会话标识，不包括问题、题型、答案、评分、来源标注或未来会话。原始会话是只读输入；policy 可以变换其内容形成派生状态，但不能请求宿主删除或改写原文。

函数可以修改传入对象，也可以返回新对象；返回值作为 `ingest` 之后的当前状态。可信宿主适配层在隔离 worker 内尝试用 cloudpickle 对实际对象做快照。候选无须声明“可序列化”；如果对象或依赖实际不能序列化，记录为运行错误，不假定 cloudpickle 能处理所有 Python 对象。编排宿主只保存生成的 base64 不透明快照。

### `search(state, query, filters)`

在给定状态副本中检索，返回零个或多个 memory ID，顺序表示候选相关性。候选代码可使用普通 Python 实现；宿主仅接受协议要求的 ID 列表，不接受答案文本、自由来源 metadata、文件路径、任意 URL 或可执行内容。后续内容读取由宿主执行。

`query` 是当前问题的用户可见文本，不附带题型、标准答案、答案来源或评分信息。`filters` 由宿主按冻结 schema 构造，只表达当前患者、当前检查点、允许的语言/模式等可见范围；候选不得扩大过滤范围。字段全集和值域待运行配置冻结。

每个返回 ID 必须是当前患者、当前检查点内由宿主预先登记的原文来源 ID。宿主拒绝未知 ID、重复 ID、超出数量上限的结果及其他患者/未来会话/答案相关 ID。ID 不得承载内容或通过编码泄漏数据。对派生文本的模型读取仍需另行定义 lineage 与派生记录协议。

## 3. 固定执行器的只读工具

### ReAct 工具模式

模型对话中固定注册以下函数；最终 Schema 应锁定在 run config 并做摘要校验：

```json
{
  "search": {
    "arguments": {"query": "string", "filters": "object"},
    "result": [{"id": "memory_id", "text": "evidence preview", "session_id": 12}]
  },
  "read": {
    "arguments": {"ids": ["memory_id"]},
    "result": [{"id": "memory_id", "text": "full evidence", "session_id": 12}]
  }
}
```

这段示意约定字段含义，不是 JSON Schema 的最终定义。最终协议还需冻结字段必填性、长度/数量上限、错误响应、编码、超时和重试行为。候选 policy 的 `search(state, query, filters)` 返回 ID 列表；宿主验证 ID 后，`search` 工具返回每条命中的短证据预览 `{id,text,session_id}`。模型可对需要的 ID 调用 `read`，获得完整证据。模型最终基于证据作答。工具结果只提供原文证据及时间定位，不含答案标签、题型、答案来源映射、未来记录或评分信息。

### `search(query, filters)` 宿主分派

宿主从当前检查点的不透明快照恢复一个独立副本到已隔离 worker，再把 state、副本内问题文本和经校验 filters 传给候选 `search` 入口。宿主验证返回 ID 列表，并用来源注册表构造 evidence preview `{id,text,session_id}`。preview 文本和 session_id 必须来自宿主登记的来源，不信任候选随 ID 附带的数据。每次调用都重新从同一检查点快照恢复，不保存 search 对副本造成的任何变更；候选返回非 ID 列表时视为协议运行错误。宿主可限制检索调用数、预览长度与每次 ID 数；限制值须冻结并记录。

### `read(ids)` 宿主分派与来源校验

宿主逐个验证 ID 可解析、属于当前患者和当前 policy 检查点，并且所指原文来源在当前检查点已可见。通过校验后宿主读取关联内容；失败时对违规 ID 不返回内容，记录拒绝原因。宿主不相信候选 policy 声称的患者、时间或来源字段，需用自己的来源注册表验证。

返回内容限于已授权的历史来源。当前工具协议只支持宿主已登记的原文来源 ID，返回完整原文 chunk 与来源时间。候选可以在自己的对象 state 中维护派生内容，但 `search` 只能返回登记的原文 ID；派生文本成为可读结果前必须另行定义派生记录 schema、lineage 校验及审计。执行器保证被读取的原文不晚于当前检查点；policy 无权令宿主访问原始 future history。

## 4. 执行时序和快照

执行器为每个 `(policy version, patient, split, replay)` 组合单独初始化状态。历史会话按冻结排序规则逐个摄入；每次成功的 `ingest` 返回状态后，worker 尝试生成新的 cloudpickle 快照，由宿主替换该 replay 当前的不透明快照。到检查点时，执行器冻结当下 policy state 和可见来源集合；该检查点所有问题都从同一快照独立运行。

每道题使用新建的 ReAct 对话上下文。问题之间不共享工具调用历史、题目文本、回答或临时 state；每个 `search` 调用在隔离 worker 中从同一已保存快照恢复独立副本，并在调用后丢弃副本，所以 `search` 对持久 policy state 只读。`read` 由宿主执行，也不写入 policy。对一个检查点最后一道题的读取也不能改变下一题的快照。只有后续会话的 `ingest` 可推进 policy state。

在线数据视图移除 `query_type`、答案、答案来源、评分/判分理由、未来消息及仅供诊断使用的 metadata。评分器在独立的离线阶段基于单独保存的参考标注评分。日志可以保留诊断字段，但在线请求和工具 payload 必须用独立结构构造，禁止直接转发含参考标注的原始行。

## 5. 模型服务与成本计量

policy 代码不能任意调用模型服务。允许的辅助模型调用须经宿主控制的 provider，记录 provider/model 标识、阶段、输入与输出 token、调用次数、耗时和错误/重试。provider 可在首版延后实现；延后时应禁用相应辅助调用，不能用不可计量的直连调用替代。

固定回答模型和 judge 的调用由执行器另行计量，分别报告，不与 policy 管理成本合并。policy 调用和状态处理至少记录初始化、摄入、检索阶段耗时；存储记录当前及峰值 state 字节数。成本报告先呈现实测值，不施加阈值。

离线优化器的生成参数独立于冻结的 actor/judge：当前 Reflexion 上限 8192 tokens、temperature 0.7，代码生成上限 16384 tokens、temperature 0.4，超时 900 秒；可由运行参数覆盖并冻结到 config/trace。actor/judge 仍为 1024 tokens、temperature 0。提示版本 `explore-v1`、归因 schema `evidence-v3` 随新运行记录；这不改变 policy v0.3 的入口和权限。提案允许重构任意 policy 内部机制，并记录操作机制和证伪检查；生成长度耗尽属于 `generation_truncated`，不属于入口契约违规。服务端总上下文仍为输入与输出合计的限制。

## 6. 快照与候选运行隔离

候选在固定 Python runtime 中使用普通 Python 语义，可以创建类和实例、定义 lambda、使用异常处理和推导式、导入运行环境已安装的库。每次新 run 须冻结 Python/runtime、cloudpickle 及候选依赖版本；未安装的依赖不自动联网安装，需作为受审查的运行环境依赖另行配置。

宿主适配层负责快照格式：在独立 worker 中对 `initialize` 或 `ingest` 返回的实际对象尝试 `cloudpickle.dumps`，成功后编码为 base64。编排宿主仅存储此不透明 base64 快照与必要元数据，绝不对候选 pickle 执行 `loads`/反序列化，也不把候选快照当可信结构检查。恢复流程由新的独立 worker 承担，且只能在隔离策略成功安装后调用 `cloudpickle.loads`。worker 使用本次 run 中与快照相同的候选源码和依赖，以恢复候选定义的类。候选无需声明所有对象可序列化；具体返回值若无法快照、依赖不可用或无法恢复，视为运行错误。任何失败都不能改用宿主进程反序列化作为回退。

v0.3 目标只允许 Linux 独立 worker 执行候选：Landlock 设置只读访问 Python 解释器运行库及本 run 明确允许的已安装 Python 依赖；拒绝数据集、gold/来源标注、其他患者内容等私有数据文件，并拒绝所有文件写入。candidate source、session、query、filters、快照和结果由可信执行器通过受控管道传入/传出，候选不能用文件路径打开输入数据。seccomp 进一步限制网络与进程控制等系统调用；具体规则须按 Python runtime 做兼容配置和冻结。

Landlock/seccomp 初始化失败、快照解码异常、候选异常、超时、返回协议错误或重新快照失败均 fail closed，记录为候选运行错误。非 Linux 环境不执行生成候选代码。上述措施限定项目运行时可用能力，不构成通用安全审计、绝对安全保证或适用于任意 Python 依赖的证明。v0.3 文档变更本身不表示该隔离和恢复流程已经实现或验证；在新的实现记录前，不得声称 v0.3 run 已通过这些约束。

## 7. 兼容性与版本冻结

任何契约改变都必须产生新接口版本或实验配置版本，并记录摘要。对每次 run 冻结 Python/runtime、cloudpickle 及候选依赖版本，候选入口，ReAct 工具 JSON Schema 与 prompt，query/filters/session 输入 schema，快照格式与错误处理，排序规则、检查点规则、provider 及模型参数，Landlock 读路径、seccomp profile、超时/资源限制和检索限额。候选不能更改固定 ReAct 协议、宿主数据权限或隔离配置。

首轮 provisional config（可用于 plumbing/smoke，不代表正式冻结）：

| 项目 | 临时值 | 边界 |
|---|---|---|
| BM25 | 字符 bigram；原文 chunk 1200 chars，overlap 120 chars；top-k 5 | 正式参数须写入 run manifest |
| Semantic baseline | 显式指定的本地 embedding model path | 不可用就标记 skipped；不得伪装为语义检索 |
| ReAct | `max_steps=4`，`context_chars=16000` | 字符预算暂代 token；取得 tokenizer 后核验/换为 token 上限 |
| 回答模型 | Qwen3.8；当前工程临时 ID `qwen3.8-27b`，thinking/speculative decoding disabled | 精确服务版本和解码参数待确认 |
| Judge | 临时同回答模型，strict 自定义 prompt | 工程验证配置，不是论文原样 judge；论文口径使用前须校准 |
| 模式 | Efficient 首轮 | Mixed 尚未接入，不得报告已评测 |
| 辅助模型 provider | 未启用时 `disabled` | 不得通过直连调用规避计量 |
| 数据集 | smoke 可用 train/validation 流程 | test 不读取、不运行、不参与调参 |

可用 smoke 仅验证 plumbing，不代表性能评估，也不允许读取 test。正式运行前待冻结的细节包括：患者划分和 seed、语言、语义模型路径与版本、BM25/chunk 参数、Qwen 服务精确版本及解码、tokenizer 与 context token 上限、工具 JSON Schema、memory ID 注册表、lineage 表示、session/filters schema、派生记录 schema、cloudpickle/runtime/候选依赖版本、快照大小、worker resource limits、预览/搜索/读取限额、调用超时与重试、provider、Linux Landlock 读路径、seccomp profile 和 judge 校准。未确定前采用宿主拒绝未知字段和未知 ID 的保守行为。

## 8. 版本迁移与运行记录

本文件将接口协议升级至 v0.3；它不是代码实现或验收报告。历史 v0.2 运行、`VALIDATION-v0.2` 与 `RUNTIME-v0.2` 保持不变，不能从历史记录推断对象快照或 v0.3 隔离已验证。首次采用 v0.3 的新运行须使用新的 run ID 和目录，配置与报告明确记载 `contract_version: v0.3`、候选依赖、cloudpickle、snapshot size/error、worker 隔离参数及结果；不得覆盖或重命名历史运行。

在线 ReAct 的固定 `search/read` 结构、只返回宿主登记的原文 ID、患者与检查点范围、gold/future 隔离、检查点同快照及问题间独立性继续有效。当前证据查询只读：每次 `search` 恢复独立副本并丢弃，只有 `ingest` 会生成新持久状态版本。Mixed 尚未接入、no-diagnosis 对照及其他既有实验限制沿用对应运行配置和历史记录，不由本协议更新推断其已完成。

2026-10-19 增加报告 schema `strict-plus-e2e-v1`，不改变 policy v0.3 接口和冻结 ReAct。新增 `E2E_AVG`、`e2e_type_accuracy`、`abnormal_rate`、`counts.unanswered_as_zero` 与 `counts.e2e_unscorable`。仅 `actor_failed` 且分数为 null 作为未作答计 0，原逐题分数及严格 AVG 不改；judge/非法分数等不可判定项仍使 E2E 不可用，缺类同样不可比较。补充指标不替换现有自动选择规则。新运行将指标版本冻结在 config，历史补算单独留档并记录实现版本和来源哈希。

同日将优化器提示升级为 `explore-v2`：初始父代码打包格式为 `standalone-baseline-v1`，必须包含可执行入口、模块级 helpers 及实际 imports，不能仅提供类定义。宿主从受信基线模块跟踪静态依赖，生成候选的后续父版本直接使用其完整源码 artifact。新运行保存 `parent-policy.py`、`parent-source.json`，并冻结 `parent_source_version` 和 `parent_source_sha256`。实际语言、基线分词行为及启用配置送入两个优化器阶段；这属于完整上下文和版本修复，不要求候选沿用某种 tokenizer 或架构，不更改冻结 actor/judge 和 policy 入口。
