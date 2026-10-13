# v0.3 对象状态运行环境

服务器项目为 `/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi`，解释器为项目 `.venv/bin/python`。已核验 Python 3.12.13、cloudpickle 3.1.2，以及 Linux Landlock ABI 1、libseccomp；普通类、NumPy 数组、NetworkX 图结构可以在该环境中保存和恢复。

接口保持同步调用 `initialize(config)`、`ingest(state, session)`、`search(state, query, filters)`。不设 Python 语法、函数、命名或类结构白名单。候选可导入已安装的运行依赖；依赖与解释器版本必须随 run 固定。

宿主 `GeneratedPolicy.serializable = True` 表示提供自动快照能力，候选无需声明同名属性。实际状态与候选模块按同一对象图进行 cloudpickle 序列化，以保持类身份、方法和模块内全局状态。宿主保存 `{serializer, data, sha256, source_sha256}`，其中 data 为 base64 字节；宿主只做长度/校验和检查，从不执行候选 pickle。恢复与保存都在完成隔离设置的独立 worker 内进行。无法保存或恢复的对象按运行错误记录，不伪造成功状态。

每次 search 都从保存的快照恢复副本，查询后的 state、模块全局变量变更均不写回。成功的 ingest 才会替换持久快照。原文注册表、患者与检查点过滤和 read 权限继续由宿主维护。

worker 只读访问已安装的 Python 模块、包资源和共享库；运行库根目录仅供目录查找，未授权平文件不获得文件读取权限。数据集、答案和运行产物应与这些依赖目录分开保存。Landlock 阻止库范围外文件读取；文件写入、外部网络及进程控制由运行隔离限制。该实现用于本项目，不是通用沙箱安全审计。

工程保护值可通过 `--candidate-timeout-s` 和 `--candidate-memory-mb` 配置，默认每次 RPC 30 秒、2048 MiB；传输上限为 64 MiB、源码上限 4 MiB。这些是保护执行器的配置，不是已确定的研究成本淘汰阈值。成本记录包含 worker 启动、隔离、对象快照和运行开销，不能直接解释为纯算法耗时。

源码格式/接口错误、候选运行异常、依赖异常、隔离故障分别记录类别，并同时保存失败阶段；失败不产生有效 AVG，也不回传搜索收益。历史 v0.2 运行保持原样，新运行明确记录 `contract_version=v0.3`。
