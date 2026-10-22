# YAML 节点模型配置

入口：configs/llm-roles.yaml。依赖 PyYAML（当前环境6.0.3；新环境需安装）。默认模板保持当前 Qwen3.8-27B 五节点配置。

```yaml
version: llm-roles-v1
defaults:
  model: qwen3.8-27b
  base_url: null
  api_key_env: RSI_API_KEY
roles:
  actor:
    temperature: 0
    max_tokens: 1024
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
    max_tokens: 8192
  proposals:
    temperature: 0.7
    max_tokens: 8192
  codegen:
    temperature: 0.4
    max_tokens: 16384
```

此示例是配置格式，不代表示例服务已启动或模型已安装。model 是服务接受的模型ID，不是权重路径。base_url=null 表示继承当前分配的本地副本；显式URL则固定使用该角色服务，并行候选不会把它覆盖成本地副本。defaults 覆盖共同设置；roles 覆盖单节点；未指定的参数按当前CLI原配置继承。可配置字段为 model/base_url/api_key_env/temperature/max_tokens/timeout_s。

支持当前 ModelClient 的 chat-completions 请求协议：JSON object，诊断/提案还会发送 JSON schema，以及当前 enable_thinking=false 模板参数。切换外部服务前需要验证这些请求字段兼容性；本功能不会自动下载模型、部署服务或选择GPU。语义基线BGE仍由原 semantic-model-path 配置，不属于这五个生成节点。

密钥仅引用环境变量名称，不在YAML写明文。未设置的密钥变量按无认证调用；服务需要认证时需在运行环境预设变量。模型服务地址不得包含用户名、密码、query或fragment。配置拒绝未知节点/字段、重复键、非法参数。模型不可用或不兼容时保留原有失败处理，不自动换回其他模型。

## 启用

在原有执行参数上追加：`--llm-config configs/llm-roles.yaml`。可与 `--search-profile deploy/evolution-search.json` 同时使用。Slurm入口可显式设置 RSI_LLM_CONFIG 为所选YAML绝对路径。没有这个参数/环境变量时沿用原有逻辑，当前153107不变。

解析后的五节点配置、YAML源哈希与规范化配置哈希均进入运行config.json。实际生成trace继续记录服务返回模型名；路由父节点记录workflow_node、配置模型和服务地址，调用统计保留各节点归属。

恢复连续搜索时检查YAML未改变并沿用归档解析结果；更换模型/参数应新开实验，不改已有树。基线复用新增actor/judge实际角色配置与路由代码指纹校验；更换评分模型、温度、输出预算等不能直接继承旧基线成绩，旧运行没有这些指纹时可能需要新基线。改变模型属于实验变量，不能把与旧模型的分差全部归因于记忆策略。

## 验证

完整回归216项（1跳过）通过；后补配置变更恢复拒绝测试后，节点配置5项定向测试通过。模拟HTTP请求核对actor/diagnosis/proposals实际URL、模型ID、输出预算与密钥不入归档；带YAML的合成连续任务fixture-yaml-roles-20261022运行及完成后恢复通过。没有真实外部API调用、模型部署或新GPU作业。
