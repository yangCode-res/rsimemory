# Memory Policy RSI

目标架构见 [PRD v0.2](docs/PRD-v0.2.md)、[接口契约](docs/INTERFACE-CONTRACT.md) 和 [架构图](docs/architecture-v0.2.png)。

## 运行

```bash
python -m unittest discover -s tests
# 合成协议测试，不是性能实验
python -m rsi.cli --fixture --output runs/fixture-001 --evolve
# 真数据工程冒烟：每类一题，只使用训练和验证各一位患者，最晚检查点20
python -m rsi.cli --dataset /file_storage01/home/juanliu/25_ymj/code/MedMemoryBench \
  --output runs/live-001 --base-url http://127.0.0.1:7860/v1 \
  --checkpoint 20 --patients-per-split 1 --per-type 1 --evolve
```

真实调用需要 OpenAI 兼容服务。默认 SGLang 是计算节点 loopback，应在同一 GPU 作业内调用。密钥用 RSI_API_KEY 环境变量提供，不写配置。当前 actor、judge、optimizer 共享 Qwen 服务仅用于工程验证；judge 未经人工校准。

## 链路

执行器按患者和时间 ingest 原文 chunk；每题从同一检查点状态生成独立工具视图。ReAct 只使用 search/read/final。宿主绑定患者和时间，候选不能扩大 filters。预览与完整读取不包含题型或答案标注。

EEM/MQ规则评分，其余四类采用自定义严格 JSON judge。异常保留 null，缺类或评分异常时 AVG 为 null。这不是论文官方 rubric 的完整复现。训练失败与成功对照进入 Reflexion；定向证据干预单独记录。三个生成代码候选通过 Linux 隔离运行并从头 replay，验证成绩不回传 UCT。policy-git 保存每个代码提交与父子关系，失败节点不参与 Pareto。

## 产物

config.json、splits.json、run_patients.json、各阶段 answers.jsonl/operations.jsonl/report.json、reflexion.json、interventions.json、branches/candidate-N/policy.py、tree.json、summary.json。逐题医疗数据留在服务器，不默认导出；测试集本轮不评估。

## 限制

- 首轮 Efficient；Mixed 映射未验证，入口拒绝运行。
- BM25 与可选的本地语义检索使用相同原文切分；语义模型未提供则 skipped，缺依赖明确失败。
- 候选可改变 JSON 状态和检索代码，目前只能返回登记的原文 ID；派生摘要输出及 policy 辅助模型尚未开放。
- 受限 Python、子进程、Linux libseccomp 与资源限制联合防护，缺失即 fail-closed；尚不是经安全审计的通用多租户沙箱。
- 初版根节点只展开一次；无诊断对照随后通过 --no-diagnosis 独立补跑。
- 当前上下文采用字符限额，精确 tokenizer 预算待接。fixture 的模拟模型和 judge 只验证协议，不是科学成绩。
