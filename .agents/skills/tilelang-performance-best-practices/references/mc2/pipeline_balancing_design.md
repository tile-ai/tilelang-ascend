# 通信与 A2/A3 GEMM 流水配平

将 M 或 token 维切为 chunk。每个 chunk 具有通信时间 C_i 与 GEMM 时间 G_i；host wrapper 通过异步 collective 和两个 stream 形成流水。chunk size 由系统搜索选择，不写死经验比例。

候选必须满足：每个元素只属于一个 chunk；通信 buffer 生命周期覆盖异步操作；GEMM 只在对应 event 完成后读取；输出写入互不重叠；最后等待所有操作。长短 chunk 的排布通过实测 C_i/G_i 决定。

精度覆盖所有 rank 和 chunk 尾部。性能报告串行基线、通信总时、计算总时、端到端时、重叠率、额外 launch 与内存；只优化局部阶段而端到端回退的方案不得采用。
