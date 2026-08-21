### Gated Delta Network (GDN)

[Gated Delta Network (GDN)](https://arxiv.org/pdf/2412.06464) is a novel recurrent-style sequence model. It keeps a hidden state $\mathbf S_t$ with update rule:

$$\mathbf S_t=\mathbf S_{t-1}\alpha_t(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^T)+\beta_t\mathbf v_t\mathbf k_t^T.$$

You can find the simplest implementation for this formula in `ref_seq_gdn` in `gdn_full.py`

To speed up the calculation, GDN adopts **chunkwise parallelism**. The sequence with total length $L$ is divided into $L/C$ chunks with length $C$, we first calculate the hidden state at the start point of each chunk (i.e. $\mathbf S_{i\cdot C}$), then use this "partial hidden state" to calculate the output. You can find more technical details in the original paper.

Our implementation of chunkwise parallelism resembles that of [Flash Linear Attention (FLA)](https://arxiv.org/pdf/2412.06464). You can find reference [here](https://sustcsonglin.github.io/blog/2024/deltanet-2/). Specifically, the forward pass of GDN is divided into six processes:

- `chunk_cumsum`: Calculate

  $$\gamma_{i\cdot C+j}=\sum_{k\leq j}g_{i\cdot C+k}\ (i<L/C,j<C),$$

  where $g_i=\ln \alpha_i$. This chunkwise cumsum will be used in several later processes.

- `chunk_scaled_dot_kkt`: Chunkwisely calculate

  $$\mathbf L=\text{strictLower}(\text{diag}(\beta)\cdot (\Gamma\odot \mathbf K\mathbf K^T)),$$

  where $\Gamma_{i,j}=\exp(\gamma_i-\gamma_j)$. $\mathbf L$ is then used in UT transform in the next step.

- `solve_tril`: Chunkwisely calculate 

  $$\mathbf A=(\mathbf I+\mathbf L)^{-1}.$$

- `wy_fast`: Implement the UT transform chunkwisely:

  $$\begin{aligned}\mathbf U&=\mathbf A\cdot\text{diag}(\beta)\cdot\mathbf V,\\
  \mathbf W&=\mathbf A\cdot\text{diag}(\exp(\gamma)\odot \beta)\cdot\mathbf K.\end{aligned}$$

- `chunk_h`: Calculate the hidden state at the start point of each chunk. It satisfies the following recurrent formula in each chunk:

  $$\mathbf S_{\text{next}}=\exp(\gamma_{C-1})\mathbf S+(\mathbf U-\mathbf W\mathbf S^T)^T\tilde{\mathbf K},$$

  where $\tilde{\mathbf k_i}=\exp(\gamma_{C-1}-\gamma_i)\cdot \mathbf k_i$

- `chunk_o`: Calculate the output using hidden state at the start point of each chunk. It satisfies the following formula in each chunk:

  $$\mathbf O=\text{diag}(\exp(\gamma))\mathbf Q\mathbf S^T+(\Gamma\odot\mathbf M\odot\mathbf Q\mathbf K^T)(\mathbf U-\mathbf W\mathbf S^T),$$

  where $\mathbf M$ is the causal mask.

---

### Optimize Results

Shape: $(B,H,L,DK,DV,C)=(16,16,16384,128,128,128)$.

|        Kernel        | Latency (ms) |    #ops (approx)     |  TFLOPS  |
| :------------------: | :----------: | :------------------: | :------: |
|     chunk_cumsum     |    $1.93$    |  $4.19\times 10^6$   | $0.0021$ |
| chunk_scaled_dot_kkt |    $8.76$    | $6.87\times 10^{10}$ |  $7.84$  |
|      solve_tril      |   $24.89$    | $2.29\times 10^{10}$ |  $0.92$  |
|       wy_fast        |    $9.92$    | $1.37\times 10^{11}$ | $13.85$  |
|       chunk_h        |    $9.38$    | $2.75\times 10^{11}$ | $29.30$  |
|       chunk_o        |   $13.19$    | $3.44\times 10^{11}$ | $26.04$  |
|        total         |   $68.07$    | $8.48\times 10^{11}$ | $12.45$  |


---

### Kimi Delta Attention (KDA)

[Kimi Delta Attention (KDA)](https://arxiv.org/pdf/2510.26692) is the linear-attention layer of Kimi Linear and Kimi K3. It is GDN with the **scalar** forget gate replaced by a **per-channel vector** gate: GDN carries one $\alpha_t$ per token, KDA carries $K$ of them, one per row of the state.

> **Convention.** The GDN section above writes the state value-major, $\mathbf S\in\mathbb R^{d_v\times d_k}$. This example is **key-major**, $\mathbf S\in\mathbb R^{d_k\times d_v}$, following the `[B, SEQ, HV, D]` interface that FLA and the KDA model code already use. The two differ by a transpose — the formulas below are written in the key-major convention the code implements.

$$\mathbf S_t=(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^{\top})\,\mathrm{Diag}(\alpha_t)\,\mathbf S_{t-1}+\beta_t\mathbf k_t\mathbf v_t^{\top},\qquad \mathbf o_t=\mathbf S_t^{\top}(s\,\mathbf q_t),$$

where $g=\ln\alpha\le 0$ is the gate in the log domain and $s=K^{-1/2}$ pre-scales the query. Replacing $\mathrm{Diag}(\alpha_t)$ with a scalar $\alpha_t$ recovers GDN term for term. The token-by-token implementation is `kda_ref` in `kda/kda_ref.py`.

Two forward paths ship in `kda/`:

- **`kda_recurrent.py` — the decode path.** The recurrence above, evaluated one token at a time with the state resident in UB. Grid is $B\cdot H_V$; the two vector cores split the **V** axis, because $\mathbf S^{\top}\mathbf k$ in the delta rule reduces along $K$ and a K-split would force a cross-block reduction every token. It uses no Cube: every token-level operation is matrix-vector shaped ($M=1$) and cannot fill the 16x16x16 fractal, and decode is memory-bound regardless.
- **the six `kda_chunk_*` stages — the prefill path**, described below.

The decode path was validated and frozen first, and the chunkwise pipeline uses its CPU twin as the acceptance golden — a chunkwise decomposition checked only against another chunkwise implementation can be consistently wrong.

For prefill, KDA uses the same **chunkwise parallelism** as GDN and its forward pass is divided into the same six processes. The one change — $\gamma$ gains a channel index $d$ — propagates into every stage that consumed a decay factor:

- `chunk_cumsum`: the scalar chain becomes a $K$-wide vector chain,

  $$\Gamma_{i\cdot C+j,\,d}=\sum_{k\leq j}g_{i\cdot C+k,\,d}\quad(i<L/C,\ j<C,\ d<K).$$

- `chunk_scaled_dot_kkt`: chunkwise

  $$\mathbf L=\text{strictLower}(\text{diag}(\beta)\cdot\mathbf P),\qquad P_{ij}=\sum_{d}k_{i,d}k_{j,d}\,e^{\Gamma_{i,d}-\Gamma_{j,d}}.$$

  In GDN the decay is one scalar per $(i,j)$ and factors straight out of the sum, leaving a plain $\mathbf K\mathbf K^{\top}$ matmul. Here it sits **inside** $\sum_d$, so $\mathbf P$ is no longer a product of a row function and a column function and cannot be written as a matmul. The causal mask must also be folded into the exponent *before* `exp()`: masking afterwards lets the $j>i$ half overflow to $\pm\infty$, and $0\times\infty=\mathrm{NaN}$ then poisons the half that is kept. This stage therefore runs entirely on the vector cores.

- `solve_tril`: unchanged from GDN,

  $$\mathbf A=(\mathbf I+\mathbf L)^{-1},$$

  because the gate is already baked into $\mathbf L$ before this stage runs.

- `wy_fast`: the UT transform, with GDN's row broadcast becoming an elementwise product,

  $$\begin{aligned}\mathbf U&=\mathbf A\cdot\text{diag}(\beta)\cdot\mathbf V,\\
  \mathbf W&=\mathbf A\cdot\text{diag}(\beta)\cdot(\mathbf K\odot e^{\Gamma}).\end{aligned}$$

- `chunk_h`: the state at the entry point of each chunk, carried serially,

  $$\mathbf V'=\mathbf U-\mathbf W\mathbf S,\qquad \mathbf S\leftarrow\text{diag}(e^{\Gamma_{C-1}})\,\mathbf S+\tilde{\mathbf K}^{\top}\mathbf V',$$

  where $\tilde{\mathbf k}_i=\mathbf k_i\odot e^{\Gamma_{C-1}-\Gamma_i}$. GDN scales $\mathbf S$ by a scalar; KDA scales it row by row.

- `chunk_o`: the output from the chunk-entry state,

  $$\mathbf O=(s\mathbf Q\odot e^{\Gamma})\,\mathbf S+\mathbf A^{qk}\mathbf V',\qquad A^{qk}_{ij}=\sum_{d}q_{i,d}k_{j,d}\,e^{\Gamma_{i,d}-\Gamma_{j,d}}\ \ (j\leq i).$$

  $\mathbf A^{qk}$ has the same non-separable form as $\mathbf P$ above, but here it is recovered for the Cube by **anchored blocking**: each block of $BC=16$ rows is anchored at its first row, and on the strictly-below-anchor columns both folded factors $e^{\Gamma_i-\Gamma_{ar}}$ and $e^{\Gamma_{ar}-\Gamma_j}$ are bounded by $1$. The off-diagonal strips go to `T.gemm_v0`; only the diagonal blocks stay on the vector cores.

The pipeline issues exactly seven `T.gemm_v0` calls — two in `wy_fast`, two in `chunk_h`, three in `chunk_o`. Stages 1–3 have none.

---

### Correctness

Both paths are verified against goldens that share no code path with them, on an Ascend 910
development device reporting `Ascend910_9362` (20 Cube / 40 Vector cores), which
`tilelang/utils/target.py` classifies as **A3**. Nothing here has been run on an A2-class
910B1/B2 (24 Cube / 48 Vector); the claim is scoped to what was actually executed.

**Decode** (`test_kda_recurrent.py`), over 6 shapes $\times$ 4 gate regimes $\times$ {zero, non-zero initial state}:

| Check | Result |
|---|---|
| kernel vs FLA's `naive_recurrent_kda` | $5.05\times10^{-4}$ worst (fp16), $2.99\times10^{-3}$ (bf16), against tolerances of $5\times10^{-3}$ / $3\times10^{-2}$ |
| kernel vs the PyTorch recurrence in `kda_ref.py` | $3.87\times10^{-4}$ worst (fp16), $2.88\times10^{-3}$ (bf16) |
| the two goldens against each other, fp32 inputs | $3.21\times10^{-7}$ worst, against a strict $10^{-5}$ |
| one shot vs segmented with the state relayed, 8 configurations | **exactly zero**, including cuts that do not land on a chunk boundary |
| all-zero `initial_state` vs none | bit-identical |

The FLA comparison runs whenever `flash-linear-attention` is importable and skips cleanly when it is not, so the example carries no third-party dependency.

**Prefill** (`kda_full.py`), 18 configurations checked against both the chunkwise reference in `kda/kda_chunk_ref.py` and the token-by-token recurrence in `kda/kda_ref.py`:

| Check | Result |
|---|---|
| full pipeline vs the token-by-token recurrence | rel. $<10^{-3}$ (fp16), $<7\times10^{-3}$ (bf16) |
| two-segment relay through `final_state` vs one shot | **exactly zero**, asserted as equality rather than as a tolerance |
| all-zero `initial_state` vs no `initial_state` | bit-identical |
| zero-length sequence ($T=0$) | accepted, launches nothing; state passes through bit-identically |
| gate settings | `keep` ($\alpha\to1$) / `normal` / `forget` (the $g_{\min}=-5$ form K3 uses) / `extreme` |
| shapes | $B\in\{1,2,4\}$, $H_V=H$ and $H_V=nH$ (GVA), $C\in\{32,64\}$, $K=V=128$ (K3), $K\neq V$, one chunk to eight |

No performance numbers are reported: no `msprof` run has been made, so this example deliberately omits the latency/TFLOPS table the GDN section above carries. `kda/bench.sh` is provided to produce that data.

Not yet supported in the prefill path: tail blocks ($L\bmod C\neq0$ is rejected by an assert) and varlen / `cu_seqlens`. The decode path has no such restriction. The backward pass is not included.
