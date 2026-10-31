# Local GEMM 与通信重叠

本 rank 数据无需通信。Python wrapper 先从输入中得到 local slice，调用 GEMM；同时在另一 stream 启动远端数据 collective。local 结果写入最终输出的唯一 slice，不能与 remote chunk 重叠。

通信后计算场景优先 local GEMM 与首个 collective 重叠；计算后通信场景可在远端 chunk 流水后处理 local GEMM。所有 stream/event 使用目标运行时的正式接口，wrapper 在结果暴露前等待必要 event。

验证 local/nonlocal slice 的覆盖 bitmap、rank 边界、world size=1 和多 rank reference。性能报告 local GEMM、collective、等待 gap 和端到端 latency。
