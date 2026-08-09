# Preliminary Gain Attribution Evidence

本目录保存 2026-08-09 早期归因运行的小型聚合证据。该阶段的 P0 仍通过
`TrtRuntime::infer_device()` 将 raw output 复制到 pageable `std::vector<float>`，
而 P1/P2 使用 pinned Host buffer，因此 P0→P1 同时改变了 Host memory 类型，不能用于
最终“GPU decode only”因果结论。Jetson 上对应完整时间戳目录继续保留，不覆盖、不删除。

最终公平证据位于相邻的
`postprocess_gain_gate_evidence_20260809_153201/`，其中 P0/P1 都使用 pinned
Host buffer，且 D2H 同为235,200 B。
