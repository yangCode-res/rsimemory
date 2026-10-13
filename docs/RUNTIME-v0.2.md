# 服务器运行环境

工程路径：`/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi`。

运行解释器为项目 `.venv/bin/python`，复用已有 SGLang 环境的只读依赖路径；项目独立安装 sentence-transformers 和检索相关依赖，没有修改共享的 SGLang 环境。基线与固定回答链路可直接运行；自动生成候选必须在 Linux 且 libseccomp 可用时运行。

| 组件 | 已核验版本 |
|---|---|
| SGLang | 0.5.10 |
| Torch | 2.9.1+cu128 |
| Transformers | 5.3.0 |
| pandas | 3.0.6 |
| pyarrow | 26.0.0 |
| sentence-transformers | 5.1.2 |
| scikit-learn | 1.7.2 |

回答服务使用已有 Qwen3.8-27B 权重，工程 smoke 配置为两张 A800、BF16、32k context、不启用 speculative decoding。语义模型是项目 `models/bge-small-zh-v1.5`，revision 为 `7999e1d3359715c523056ef9478215996d62a620`，CPU 编码；模型 manifest 记录文件散列。

在项目目录执行 `sbatch scripts/smoke.sbatch` 可启动独立服务并运行真实数据工程 smoke。脚本使用端口 7861、节点 loopback，结果保存在 `runs/live-v02-$SLURM_JOB_ID`；每次使用新的输出目录，禁止覆盖已有运行。提交前应确认当前 Slurm 账户仍可使用脚本中的 gre 分区和两张 GPU。

逐题医疗记录、来源映射和 Reflexion 内容留在服务器。对外汇报仅使用聚合指标、状态、代码版本和错误类别。该工程配置与未经校准的 judge 不能用于论文性能结论。
