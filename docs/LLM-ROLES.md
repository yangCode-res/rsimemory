# YAML 节点模型配置

入口：configs/llm-roles.yaml。依赖 PyYAML（当前环境6.0.3；新环境需安装）。默认模板使用 Qwen3.8-27B 五节点配置。2026-10-22 起所有新 CLI/Slurm 任务默认读取该文件，输出预算按下文更新。

```yaml
version: llm-roles-v1
defaults:
  model: qwen3.8-27b
  base_url: null
  api_key_env: RSI_API_KEY
roles:
  actor:
    temperature: 0
    max_tokens: 4096
  judge:
    model: your-judge-served-model
    base_url: http://127.0.0.1:8001/v1
    api_key_env: JUDGE_API_KEY
    temperature: 0
    max_tokens: 1024
  diagnosis:
    model: your-diagnosis-served-model
    base_url: http://127.0.0.1:8002/v1
    temperature: 0.7
    max_tokens: null
  proposals:
    temperature: 0.7
    max_tokens: null
  codegen:
    temperature: 0.4
    max_tokens: null
```

此示例是配置格式，不代表示例服务已启动或模型已安装。model 是服务接受的模型ID，不是权重路径。base_url=null 表示继承当前分配的本地副本；显式URL则固定使用该角色服务，并行候选不会把它覆盖成本地副本。defaults 覆盖共同设置；roles 覆盖单节点；未指定的参数按当前CLI原配置继承。可配置字段为 model/base_url/api_key_env/temperature/max_tokens/timeout_s。

支持当前 ModelClient 的 chat-completions 请求协议：JSON object，诊断/提案还会发送 JSON schema，以及当前 enable_thinking=false 模板参数。切换外部服务前需要验证这些请求字段兼容性；本功能不会自动下载模型、部署服务或选择GPU。语义基线BGE仍由原 semantic-model-path 配置，不属于这五个生成节点。

密钥仅引用环境变量名称，不在YAML写明文。未设置的密钥变量按无认证调用；服务需要认证时需在运行环境预设变量。模型服务地址不得包含用户名、密码、query或fragment。配置拒绝未知节点/字段、重复键、非法参数。模型不可用或不兼容时保留原有失败处理，不自动换回其他模型。

## 启用

CLI默认读取代码仓库中的 configs/llm-roles.yaml；RSI_LLM_CONFIG可指定其他路径，显式 --llm-config优先。配置缺失或非法直接失败，不回退旧逻辑。full_round.sbatch始终传入选定YAML的启动时快照，smoke.sbatch也始终传入YAML。可与 --search-profile deploy/evolution-search.json同时使用。历史153107及其他运行不改写。

解析后的五节点配置、YAML源哈希与规范化配置哈希均进入运行config.json。实际生成trace继续记录服务返回模型名；路由父节点记录workflow_node、配置模型和服务地址，调用统计保留各节点归属。

恢复连续搜索时检查YAML未改变并沿用归档解析结果；更换模型/参数应新开实验，不改已有树。基线复用新增actor/judge实际角色配置与路由代码指纹校验；更换评分模型、温度、输出预算等不能直接继承旧基线成绩，旧运行没有这些指纹时可能需要新基线。改变模型属于实验变量，不能把与旧模型的分差全部归因于记忆策略。

## 验证

完整回归216项（1跳过）通过；后补配置变更恢复拒绝测试后，节点配置5项定向测试通过。模拟HTTP请求核对actor/diagnosis/proposals实际URL、模型ID、输出预算与密钥不入归档；带YAML的合成连续任务fixture-yaml-roles-20261022运行及完成后恢复通过。没有真实外部API调用、模型部署或新GPU作业。

## 输出预算：2026-10-22

|节点|max_tokens|temperature|timeout_s|
|---|---:|---:|---:|
|actor|4096|0|180|
|judge|1024|0|180|
|diagnosis|null|0.7|900|
|proposals|null|0.7|900|
|codegen|null|0.4|900|

null表示客户端不发送max_tokens字段，也不替换成1024、128或一个伪装成无限大的数。已核验当前安装SGLang：ChatCompletionRequest默认max_tokens=None，转换为sampling.max_new_tokens=None，调度器按剩余上下文裁定物理生成上限。因此当前本地服务没有额外应用层输出上限，正常EOS/JSON完成仍会结束生成。模型总上下文（当前262144）和请求超时仍存在。其他兼容服务可能有自己的缺省限制；切换服务要重新核验，不能把null称为物理无限输出。finish_reason=length仍明确归档为generation_truncated，不把部分代码当完整代码。

历史151576/152743/153107中，actor输出p99约488–492，10次触及1024；提高到4096为长回答保留余量。judge三轮最大590、无长度截断，因此保留1024。153107诊断首次输出8192被截断；离线三个节点改为null，让内容决定实际长度。详细聚合证据见output-token-observations-20261022.json，统计不含病例内容。

YAML是节点生效参数来源：显式角色字段覆盖defaults和旧CLI参数的继承值；历史--reflexion-max-tokens等仍作为未在YAML声明字段时的fallback。全量脚本不再重复传入旧的RSI_REFLEXION_MAX_TOKENS等生成预算变量。修改节点预算应编辑选中的YAML。config.llm_roles.resolved及effective_generation_settings记录实际参数，旧optimizer_generation_settings仅保留CLI输入fallback。

每个新run保存llm-roles.yaml原始快照、源SHA和解析SHA。连续恢复要求所选YAML与冻结快照内容完全一致，路径可以改变（例如使用归档快照）；改预算应新开实验。没有冻结YAML的旧连续任务不能无声接入新默认配置。actor1024→4096属于回答器配置变化，旧基线不可直接复用；新预算实验必须重新建立同预算BM25/BGE基线。ReAct提示、工具/轮数、judge模型与温度、评分规则保持原样。


## OpenCode DeepSeek V4 Flash profile (2026-10-22)

Working-tree configs/llm-roles.yaml now pins diagnosis/proposals/codegen to deepseek-v4-flash at https://opencode.ai/inference/openai/v1, using OPENCODE_API_KEY. Actor/judge and existing temperature/output settings are preserved. Named copy: configs/llm-roles-deepseek-v4-flash.yaml. Provider identity and structured JSON schema support were verified with synthetic-only prompts; no clinical input was sent. Set the User-Agent rsi-experiment/1.0 for this provider to avoid the observed 403 with Python's default client identifier.

Private key is in .secrets/llm.env (directory 0700, file 0600, ignored by Git). Full/smoke launchers source RSI_LLM_ENV_FILE, defaulting to that private file, without copying credentials to artifacts or CLI arguments. Login-node shell direct invocations must source the private file themselves before starting CLI. Do not copy the key into YAML, documentation, or Git. External stages will send their actual configured inputs to OpenCode when a new real run is launched; current frozen runs are unchanged.

These edits are in the working tree alongside pre-existing uncommitted output-limit/default-config changes. No real task was relaunched or silently switched. Before a launcher that snapshots Git HEAD is used, ensure the intended working-tree changes are committed, or select a reviewed commit containing the configuration and private-env loading hook.
