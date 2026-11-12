# Packaged kernel design

The implementation is a dense causal forward with D/E/R/V=128/64/512/128.
It returns the128-channel gated result before W_O. Input layouts, ownership and
numerical restrictions are defined in the package README.

1. Query absorption computes QL=Q*WK. The H96 Cube specialization uses real-head
   decode tiles or same-head token tiles. BF16 retains separate Hi/Lo QL terms.
   The packed version reads Q's128 content channels and copies its64 auxiliary
   channels without a host payload conversion.
2. Decode splits the shared cache into six ranges, emits FP32 partial maximum,
   normalizer and weighted latent states, then merges nonempty splits. Prefill
   keeps FP32 online states while processing KV tiles. Neither route materializes
   a full B*H*Tq*Tk score matrix or replicates C/E cache across heads.
3. Causal validity is `key_index<=query_start+query_index`, combined with runtime
   q_len/kv_len. Masked scores use negative infinity. Native compare/select uses
   a packed tensor mask, including N128; it does not rely on a64-bit SPR mask.
4. Value R3 splits FP32 Z with scale4 for FP16 or2 for BF16: Hi=LP(Z/s),
   Lo=LP((Z/s-float(Hi))*s). Two independently initialized FP32 GEMMs produce
   A/B; the Vector side computes `(A+B/s)*sigmoid(G)*s`, then converts to LP.
   Lo retains original-scale subnormal resolution. See README for range limits.

The persistent loops use20 logical Cube blocks with paired Vector workers on the
verified910B3. C4 uses M32/N64 generally. Only the seven-input public runtime's
B1/H96/physical Tq=Tk128/default-scale profile selects N128; explicit base bundles
and packed/model calls retain their configured N64. Metadata values are never
dispatch or compilation keys.

The reviewed prefill revision retains the original interfaces and precision
policy. Head-aligned tiles build a whole-tile packed causal mask; other heads
keep the original row-mask path. For one physical KV tile, C/E stay resident in
L1 for the same batch within one invocation. Query-only prefetch reuses dead
Q/QA/Lo storage during Vector softmax, before the existing completion barrier.
Tags are explicitly reset inside Cube scope on every launch and graph replay;
no input data is cached across invocations. Multi-KV execution retains the
original reloads and online recurrence. Entirely active, head-aligned one-block
tiles omit only identities from the initial online state; normalization still
uses the original FP32 probability sum before BF16 Hi/Lo quantization.

QK uses a128-element L0 K chunk; auxiliary Q remains64. PV uses64 on PTO's
single-buffer template and32 on AscendC's two-slot template, respecting each
backend's real L0B budget. Whole-tile Select owns complete FP32-sized scratch,
and packed-mask rows reserve their32-byte physical stride. Generated addresses,
payload/metadata updates and broad numerical regressions require hardware
verification; source-level capacity estimates alone are insufficient.

Cube/Vector communication uses per-block GM workspaces with paired flags and
acknowledgements before reuse. Value holds one head's Wv in L1 across token tiles.
Query/Value tail tiles reload the last real M rows and store only the remainder;
both Vector partners acknowledge even if one has zero output rows. Mutable tail
indices are explicitly reset each iteration. Attention tail helpers zero physical
padding on the device. Cache initialization and append are also NPU kernels.

All buffers and compiler scratch must fit their target memory scopes. Full dense
element spans, attention partials and tail allocations are validated against
signed-int32 indexing before use. Static validation and successful compilation
do not establish NPU accuracy, performance, arbitrary capacity or other hardware.

Production files are self-contained in this package. Development origin IDs in
source_manifest are historical hash labels, not files that need to be installed.
The optional frozen-model loader consumes caller-supplied external source;
no Moonshot model implementation or vendor runtime source is bundled here.
