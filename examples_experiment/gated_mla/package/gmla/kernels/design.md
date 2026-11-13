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

Head-aligned shapes whose cache fits a single KV block compile a second entry,
`main_pipelined`. The baseline entry runs one wave at a time behind a
single-buffer four-flag handshake, so the Cube and Vector cores alternate: a
same-shape profile measured the Cube unit active 10.3% of the kernel and the
Vector unit 14.4%, with the Cube blocked 57.6us of 129.7us and the Vector
67.0us of 130.6us. Each wait tracked the other core's busy time, i.e. the two
never overlapped.

`main_pipelined` skews the Cube loop by one wave. The QK of wave w is issued
before the PV of wave w-1, so the Cube runs a GEMM while the Vector is still in
the softmax that separates them. Score, probability and partial workspaces gain
a two-slot dimension, and the four flags become six semaphores paired as
ready/consumed per workspace, with the consumer pre-releasing one slot before
the loop. Both scopes drain their primed semaphores after the loop so that no
cross flag outlives the launch.

Two further properties hold on this path. The accumulate of wave w-1 runs after
the softmax of wave w, so the per-wave normaliser needs two buffers; these are
plain 1-D buffers, because a 2-D buffer indexed as slot[s, row] does not lower
correctly as a `tile.div` scalar operand. The PV stage reads no metadata of its
own, since its addressing uses the slot alone, so a one-step carry of the QK
stage active flag replaces a full metadata recompute per wave.

Arithmetic and its ordering are unchanged: the two entries agree bit-for-bit
across both backends, both dtypes, the pipelined and fallback domains, and
every validated query length. Shapes outside the domain compile `main`, whose
generated code is byte-identical to the previous revision.

The domain is narrow and is a property of the guard, not a tuning choice.
`iterations == 1` requires the cache to fit one KV block, so Tk is at most
block_n (128), and `heads % block_m == 0` requires the head count to be a
multiple of the row tile. Within that domain rows = Tq * heads is always a
multiple of block_m, so the head-tail path cannot be reached there.

A third condition is batch == 1, and it is a correctness bound rather than a
capacity one. C and E stay resident in L1 under a batch tag, and the skewed loop
issues the QK of wave w before the PV of wave w-1; when those two waves belong
to different batches the QK reloads C/E into the buffers the pending PV still
reads. With one batch the tag never changes, so the conflict cannot arise.
Lifting it needs a second C/E slot, which is 144 KiB more L1 and therefore an
L1 budget that has to be confirmed on hardware rather than assumed.
`tests/test_prefill_mask_exact_npu.py::test_batches_do_not_influence_each_other`
pins this boundary and fails if multi-batch pipelining is enabled without it.
