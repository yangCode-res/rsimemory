# 完整父源码与三个候选的分词修复

2026-10-19 完成，代码版本 `0eefcc1`。本次修复生成链路的父代码打包遗漏，并为 147547 三个历史候选创建独立的 tokenizer-only 修复版本。未改写原运行，未重新生成 RSI 提案，未重跑真实回答或 judge，未使用测试集。

## 生成链路

- `rsi/policy_source.py` 从受信初始基线的模块跟踪静态全局依赖，打包 imports、`_grams`、`PlainPolicy` 以及 initialize/ingest/search 入口。宿主隔离适配器不打包进父实现，拒绝遗漏的项目相对导入依赖。
- 新格式为 `standalone-baseline-v1`，提示为 `explore-v2`。新运行写入 `parent-policy.py`、`parent-source.json`，config 保存源码版本和 SHA256；版本库中的基线父节点也保存完整可执行实现。
- Reflexion 和 codegen 均获得同一份完整父源码，另有实际语言、基线字符二元组行为与启用配置。actor/judge 及其参数、策略入口、只读查询与隔离规则保持不变。
- 候选可选用其他适合语言的分词、索引和记忆结构，没有新增语法或实现白名单。合成测试核对行为，不规定实现方式。

## 独立历史修复版本

三个历史模块仅替换模块级 `_grams` 函数，采用实际基线的 Unicode casefold、空白归一化、字符二元组，并将 helper 所需的 `re` import 放在函数内部。文件中其余字节保留，缓存、近期加权和重排逻辑不变。

修复仓库克隆原 `policy-git` 的对象，每个新 commit 直接以原候选 commit 为父节点。生成的代码、manifest 和逐题诊断只留在服务器：

```text
/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/runs/repairs-147547-20261019/
  manifest.json
  policy-git/
  candidate-1-tokenizer-fix/policy.py
  candidate-2-tokenizer-fix/policy.py
  candidate-3-tokenizer-fix/policy.py
  diagnostics/candidate-1/aggregate.json
  diagnostics/candidate-2/aggregate.json
  diagnostics/candidate-3/aggregate.json
```

|版本|原父 commit|修复 commit|
|---|---|---|
|候选 1|`9285b60`|`377e76d`|
|候选 2|`9902b80`|`96cf08e`|
|候选 3|`8611e6e`|`23dba9d`|

已核对三个新文件的 SHA256 与新 Git 对象一致、每个父指针与原版本一致，以及原源码 SHA256 与旧归档清单一致。状态为 `retrieval_validated_not_answer_evaluated`，不是自动进化成功或最终性能结果。

## 验证

Linux 回归检查 131 项，结果通过（1 项跳过）。覆盖：父源码在不依赖宿主 globals 时执行、与实际基线检索一致、完整源码和语言实际传入两阶段优化器、仅替换一个函数、Unicode/空白分词一致、真实隔离 worker 执行完整父策略和修复策略。

运行前预检增加了受信完整父策略的中文局部事实召回测试；实际 Linux 预检通过，对象快照、独立查询、恢复及 8/4/8 划分检查也通过。此预检不增加对生成候选实现形式的限制。

每个真实修复文件均在隔离沙箱完成合成测试：第 1 条相关记忆后放 20 条无关记忆。三个原版本均未召回第 1 条，三个修复版本均将其排在首位。

随后使用四位验证患者固定中间检查点的 36 题和原 search 输入，验证实际已落盘的修复文件；来源会话命中率如下：

|策略|原候选|实际修复文件|同输入 BM25 基线|修复后全零 BM25 查询|
|---|---:|---:|---:|---:|
|候选 1|22.22%|50.00%|50.00%|0/36|
|候选 2|22.22%|52.78%|50.00%|0/36|
|候选 3|22.22%|50.00%|50.00%|0/36|

修复后首次返回当前会话的条目比例均约 4.44%，原候选约 75%。此处是会话级检索代理指标，不能表述为回答准确率。修复来源 SHA256 已与每份诊断报告交叉核对；原实验中的严格 AVG、E2E AVG、异常率和 Pareto 结果均保留。

## 后续评测

完整回答性能仍需使用新的独立 run 从空状态 replay，复用冻结 actor/judge、8/4/8 划分与评分规则。人工修复分支和使用完整父源码重新生成的 RSI 分支须分别标识，不能混合报告。当前没有提交新的 GPU 评测作业。
