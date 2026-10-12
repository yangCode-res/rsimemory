# Memory policy RSI 初版骨架

已实现 Efficient 会话 replay、患者级固定划分、独立 plain-record 状态、字符双字检索、OpenAI 兼容调用、操作日志、成本基础计量和三分支清单。

```bash
python -m unittest discover -s tests
python -m rsi.pipeline --dataset /file_storage01/home/juanliu/25_ymj/code/MedMemoryBench --output runs/smoke-001 --smoke
```

默认 offline 模式只验证工程，不生成真实回答或性能分数。真实回答使用 --backend openai --base-url http://计算节点:端口/v1，注意服务默认只监听节点 loopback，需要在同一 GPU 作业中调用。

状态限制：judge、自动 Reflexion、代码候选生成、候选沙箱执行、完整 MCTS 回传、Pareto 比较和 Mixed 数据适配尚待实现。三个子分支是待生成清单，不是已进化版本。AVG 保持 null，不能将 offline 输出计为准确率。现有隔离为模块级边界，执行外部候选前必须补上进程及文件权限隔离。

固定 agent prompt 在 rsi/pipeline.py 中；修改 policy 不应修改该部分。训练、验证、测试 manifest 已生成；当前仅运行训练 replay，测试集没有被评估。使用 pandas 和 pyarrow 读取 Parquet，服务器已验证的 SGLang 环境具备这些依赖。
