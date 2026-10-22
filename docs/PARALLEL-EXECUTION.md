# 六卡并行候选评估

2026-10-22。用户授权将三个候选评估改为并行。单节点6 GPU分成三个独立TP=2服务，不将当前Qwen模型设为TP=6。

|候选|SGLang服务|GPU分组|
|---|---|---|
|candidate-1|127.0.0.1:7862/v1|Slurm分配列表的第1、2项|
|candidate-2|127.0.0.1:7863/v1|第3、4项|
|candidate-3|127.0.0.1:7864/v1|第5、6项|

设备分组保留CUDA_VISIBLE_DEVICES中的实际设备编号或UUID，检查恰好6项且无重复；不假定用户拥有物理GPU0–5。每服务独立缓存/临时目录、进程组、日志和健康监控。三个服务全部健康才开始实验；部分启动失败或服务退出会清理全部子服务并使运行失败。所有持久产物仍归档到file_storage01；节点tmp仅作暂存。

## 执行依赖

已核验的BM25/BGE基线仍可复用。主服务执行诊断→三个提案→三个代码生成；每个新候选代码均记录自己的父版本和commit。准备完成后，三个候选分别使用独立actor/judge客户端及对应模型服务，同时执行各自train→validation。每个患者/候选仍从空记忆开始，患者内时间与问答顺序保持不变，不共享记忆或回答状态。诊断、单题ReAct内部步骤因依赖关系仍顺序执行。

主线程负责policy Git、树、汇总与奖励回传，候选线程仅写各自branch。最后按提案顺序合并收益，不因最快完成者改变顺序。轨迹ContextVar状态和栈采用副本，JSONL写入加锁，避免跨线程患者/候选混标或大记录互相穿插。

候选普通实现错误只使该分支invalid，其他分支继续。服务不可用使整轮failed并请求其他分支停止；已发出的HTTP调用仍需返回或超时，不能宣称即时中断。终止/失败保留已完成的逐题与进度记录。单题异常、严格AVG、E2E和候选有效性规则维持原口径。

## 参数和产物

CLI保持串行默认，开启并行须显式提供与workers数匹配的不同端点：

```bash
python -m rsi.cli ... --candidate-workers 3 \
  --candidate-base-urls http://127.0.0.1:7862/v1 http://127.0.0.1:7863/v1 http://127.0.0.1:7864/v1
```

全量脚本默认申请6 GPU、48 CPU、256 GiB、48小时，并传上述参数。候选actor/judge分别在对应副本上运行；并行时不接受单独共享judge-url，以免意外把判分重新集中到单服务。

- meta/replicas.json：GPU配对、端口、PID、缓存与日志位置。
- meta/replicas-ready.json：三个服务全部健康的启动门槛。
- server-1.log / server-2.log / server-3.log：各服务启动和吞吐日志。
- meta/services/replica-N/：独立健康检查与停止原因。
- result/run-status.json：candidates.parallel等全局阶段。
- branches/candidate-N/run-status.json：各分支阶段及评分完成状态。
- branches/candidate-N/execution.json：端点、墙钟时长与本分支模型调用成本。
- config.candidate_execution：并发度、端点、合并顺序、版本与状态初始化约定。

## 复用与比较边界

基线逻辑、回答协议、数据划分与评分代码保持冻结；基线恢复指纹覆盖实际基线执行文件。候选沙箱启动器和轨迹代码的版本另记candidate_runtime_sha256。并行增加CPU/GPU资源竞争，policy耗时不宜与串行旧运行直接当成算法成本改进；仍报告实际墙钟和各分支成本。吞吐与加速比须在新运行中测量，不能因使用三组服务就预先宣称三倍加速。

旧双卡运行153104在尚未开始候选回放时请求正常停止，旧归档保留；新的并行运行使用独立run ID，不覆盖旧结果。GPU16固定绑定已解除，只排除之前暂存磁盘满的gpu36/37。实际分配与启动成功须查看Slurm、服务健康记录及归档状态。

## 验证记录

超算 Linux 完整回归198项，1项跳过。新增测试用三方Barrier验证候选确实重叠执行，检查端点/客户端独立、提案顺序合并和UCT三次回传；覆盖单候选错误隔离、服务不可用取消、已就绪副本提前退出、启动失败清理及大Unicode轨迹并发写入。沙箱真实并发测试保留seccomp/Landlock和CPU/内存限制。

原双卡153104已按用户切换请求正常停止，归档_manifest.verified=true；没有开始候选回放。该次退出保留为历史中断，不计为候选退化或成功进化。

实际合成CLI闭环 runs/fixture-six-gpu-parallel-20261022 已完成：三个候选全部evaluated，分别绑定7862/7863/7864，从空记忆经真实Linux隔离worker完成train/validation。合成评分均1.0只说明协议正确，不是MedMemoryBench实验成绩。旧真实基线152743在并行版本上再次通过23文件/772训练题/380验证题预检。

六卡并行实现版本55ecef4baa084280664f16bd6551946ee7ff0c4e。新作业153107已在gpu22运行，Slurm确认6GPU/48CPU/256GiB，当前处于启动阶段；回执runs/rerun-launch-153107.json，归档runs/full-v03-local-153107。未宣称副本已就绪、候选已开始或吞吐已提升。
