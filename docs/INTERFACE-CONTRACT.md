# Memory Policy Interface Contract

版本：0.2　日期：2026年10月13日　状态：首轮协议已实现；工程验证中，正式参数待冻结。

本契约依据 `docs/architecture-v0.2.png` 定义固定执行器与候选 memory policy 的边界，以及固定 ReAct 模型可调用的只读工具。该图已确认为 target architecture：在线运行、固定评测/反馈、离线进化和红色有效性约束都属于目标结构。本文区分目标协议和首轮实现限制，不表示安全沙箱或全部接口已实现或验证。

## 1. 信任边界

固定执行器拥有原始会话、患者与检查点上下文、问题/答案/来源标注、评分器、存储、工具派发和模型 provider。候选 policy 只处理当前患者、当前 replay 的 JSON 状态与执行器明确传入的会话；它不持有文件、网络、数据库或模型客户端。首轮可在 state 内组织或更新自己的派生结构，但候选只能返回已登记的原文来源 ID；生成派生文本供模型读取暂不开放，须先定义并实现 lineage 和派生记录 schema。

回答模型只获得 `search(query, filters)` 和 `read(ids)` 两个工具。二者只读。不得暴露 `create`、`update`、`delete`、`merge`、`write`、`ingest`、`export` 等会改变记忆或访问外部状态的工具。记忆更新只能由固定执行器按回放流程调用 policy 函数完成。

## 1.1 Target architecture 中的接口位置

在线链路由固定执行器、可进化 policy、冻结 ReAct、独立评分器及评测/成本汇总组成。执行器把历史会话送给 policy；ReAct 仅经 `search/read` 读取证据，输出答案与工具轨迹；离线评分器从隔离数据读取标准答案和来源标注。操作与检索日志进入训练诊断，Reflexion 形成聚类/归因和定向 replay，之后生成三个修改假设及子 policy 代码。边界检查/隔离把违规或运行失败候选送入失败归档，将有效候选送入版本树及一次 UCT 展开；候选比较依据验证集汇总与成本形成报告。以上是目标图中的组件关系，本契约只定义接口边界，不代表该链路已全部接通。

## 2. Policy 函数

policy 入口为三个同步纯函数。参数和结果必须可编码为 JSON；函数不得依赖隐藏全局状态、时间、随机数、环境变量或外部服务。`search` 的候选函数只返回 ID；工具层的 evidence preview 由宿主依据 ID 构造，不是候选函数返回自由文本。

```python
initialize(config: JSON) -> JSONState
ingest(state: JSONState, session: JSON) -> JSONState
search(state: JSONState, query: str, filters: JSON) -> list[str]
```

### `initialize(config)`

创建一个全新、空的 policy state。config 仅含本实验冻结的非敏感运行参数和策略参数，不含问题答案、参考来源、其他候选状态或未来数据。返回值必须可编码为 JSON；object 和 array 均可使用。宿主为每位患者和每个候选单独调用，不复用其他患者、父候选或前次 replay 的 state。

### `ingest(state, session)`

把固定执行器按时间顺序提供的一段历史会话纳入 policy，并返回新的 state。函数只能依据入参中的当前会话和现有 state 工作。它可以在内部提取、合并、更新、删除和重排派生记忆或索引；这些 CRUD 操作是 policy 内部实现细节，不是回答模型工具。

session 只包含允许 policy 处理的原始历史内容与必要的会话标识，不包括问题、题型、答案、评分、来源标注或未来会话。原始会话是只读输入；policy 可以变换其内容形成派生状态，但不能请求宿主删除或改写原文。

输入 state 是通过 JSON 传入独立子进程的副本；函数可以在副本上更新并返回 state。宿主只采用返回值，不共享子进程内存。在 search 时返回值不得推进宿主快照；只有 ingest 可以更新下一检查点的状态。

### `search(state, query, filters)`

纯检索函数，在给定快照中返回零个或多个 memory ID，顺序表示候选相关性。不得返回答案文本、来源 metadata、文件路径、任意 URL 或可执行内容。结果只是一组标识符，后续内容读取由宿主执行。

`query` 是当前问题的用户可见文本，不附带题型、标准答案、答案来源或评分信息。`filters` 由宿主按冻结 schema 构造，只表达当前患者、当前检查点、允许的语言/模式等可见范围；候选不得扩大过滤范围。字段全集和值域待运行配置冻结。

目标架构允许检索 policy 派生记录，但任何派生记录必须先有可审计 lineage。首轮实现更严格：每个返回 ID 只能是当前患者、当前检查点内由宿主预先登记的原文来源 ID；不接受 policy 自造 ID、派生文本 ID 或无法解析的 ID。宿主拒绝未知、重复或超出数量上限的结果，并拒绝来源属于其他患者、未来会话、答案或不可见数据的记录。ID 不得承载内容或通过编码泄漏数据。待 lineage 表示和派生记录 schema 落地后，才能开放派生证据的 `read`。

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

这段示意约定字段含义，不是 JSON Schema 的最终定义。最终协议还需冻结字段必填性、长度/数量上限、错误响应、编码、超时和重试行为。候选 policy 的纯 `search(state, query, filters)` 返回 ID 列表；宿主验证 ID 后，`search` 工具返回每条命中的短证据预览 `{id,text,session_id}`。模型可对需要的 ID 调用 `read`，获得完整证据。模型最终基于证据作答。工具结果只提供原文证据及时间定位，不含答案标签、题型、答案来源映射、未来记录或评分信息。

### `search(query, filters)` 宿主分派

宿主把当前 state、问题文本和经校验 filters 传给 policy 的纯 `search` 函数，然后验证返回列表并用来源注册表构造 evidence preview `{id,text,session_id}`。preview 文本和 session_id 必须来自宿主登记的来源，不信任候选随 ID 附带的数据。宿主可限制检索调用数、预览长度与每次 ID 数；限制值须冻结并记录。此操作不推进 state。

### `read(ids)` 宿主分派与来源校验

宿主逐个验证 ID 可解析、属于当前患者和当前 policy state，并且其全部来源 lineage 在当前检查点已可见。通过校验后宿主读取关联内容；失败时对违规 ID 不返回内容，记录拒绝原因。宿主不相信候选 policy 声称的患者、时间或来源字段，需用自己的来源注册表验证。

返回内容限于已授权的历史来源。首轮只支持宿主已登记的原文来源 ID，返回完整原文 chunk 与来源时间。首轮不支持 policy 生成的派生文本；启用前必须实现派生记录 schema、从派生内容到原文来源的 lineage 校验及持久化审计。执行器保证被读取的原文不晚于当前检查点；policy 无权令宿主访问原始 future history。

## 4. 执行时序和快照

执行器为每个 `(policy version, patient, split, replay)` 组合单独初始化状态。历史会话按冻结排序规则逐个摄入。到检查点时，执行器冻结当下 policy state 和可见来源集合；该检查点所有问题都从同一快照独立运行。

每道题使用新建的 ReAct 对话上下文。问题之间不共享工具调用历史、题目文本、回答或临时 state；`search` 与 `read` 为只读，所以不会改变快照。对一个检查点最后一道题的读取也不能改变下一题的结果。到下一个检查点时，只有新会话的 `ingest` 调用可以推进 policy state。

在线数据视图移除 `query_type`、答案、答案来源、评分/判分理由、未来消息及仅供诊断使用的 metadata。评分器在独立的离线阶段基于单独保存的参考标注评分。日志可以保留诊断字段，但在线请求和工具 payload 必须用独立结构构造，禁止直接转发含参考标注的原始行。

## 5. 模型服务与成本计量

policy 代码不能任意调用模型服务。允许的辅助模型调用须经宿主控制的 provider，记录 provider/model 标识、阶段、输入与输出 token、调用次数、耗时和错误/重试。provider 可在首版延后实现；延后时应禁用相应辅助调用，不能用不可计量的直连调用替代。

固定回答模型和 judge 的调用由执行器另行计量，分别报告，不与 policy 管理成本合并。policy 调用和状态处理至少记录初始化、摄入、检索阶段耗时；存储记录当前及峰值 state 字节数。成本报告先呈现实测值，不施加阈值。

## 6. 候选代码限制与初版安全实现

候选函数以受限 Python 执行，输入/输出采用 JSON 序列化。策略代码不得任意 import；不得使用文件系统、网络、子进程、动态代码执行、系统命令、环境变量或反射逃逸；不得修改固定执行器、ReAct prompt、工具注册、评分器或数据访问实现。异常、超时、非法 JSON 或资源超限均使该候选运行失败并留下记录。

首版候选代码目标是在 Linux 隔离子进程中运行：先做 AST 检查，再用 seccomp 拒绝文件系统、网络及进程创建/控制相关系统调用，并配置 `RLIMIT` 资源上限。宿主与子进程仅通过受限 JSON 标准输入/输出交互；候选执行期间不得打开文件或创建网络/子进程。超时、内存/CPU/文件描述符等 `RLIMIT` 超限、seccomp 安装失败或子进程意外退出都必须 fail closed。语言级 AST 检查只能作为第一道防护，不能独立证明 Python 代码安全。

非 Linux 环境不得执行自动生成的候选代码。首版已经实现独立进程、seccomp、AST 与 RLIMIT，Linux 用例核验文件和网络 syscall 被拒绝；缺失隔离时仍 fail closed。该验证不是通用安全审计或绝对安全保证。纯函数签名本身不构成 OS 级保障。

## 7. 兼容性与版本冻结

任何契约改变都必须产生新接口版本或实验配置版本，并记录摘要。对每次 run 冻结 Python/runtime 版本、候选入口、JSON Schema、tool prompt、state/resource limits、session/query JSON schema、排序规则、检查点规则、provider 及模型参数、隔离后端、错误/超时处理和检索限制。候选不能更改固定协议。

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

可用 smoke 仅验证 plumbing，不代表性能评估，也不允许读取 test。正式运行前待冻结的细节包括：患者划分和 seed、语言、语义模型路径与版本、BM25/chunk 参数、Qwen 服务精确版本及解码、tokenizer 与 context token 上限、工具 JSON Schema、memory ID 注册表、lineage 表示、session/filters schema、派生记录 schema、state/resource limits、预览/搜索/读取限额、调用超时与重试、provider、Linux seccomp profile 和 judge 校准。未确定前采用宿主拒绝未知字段和未知 ID 的保守行为。

## 8. 与当前骨架的差距

首版已接入冻结 ReAct 的 search/read、规则与自定义 judge、Reflexion、三候选代码生成、Linux 隔离 replay、Git 谱系、UCT 回传和 Pareto 输出。宿主不把私有参考行转发在线。合成 fixture 仅检验协议，不计作科学成绩。首轮 Efficient，Mixed 未接入；训练集初轮诊断后另行执行 no-diagnosis 对照。派生摘要返回与辅助模型 provider 未开放，因此当前代码进化空间比目标完整 policy 范围更窄，报告不得隐去这一限制。
