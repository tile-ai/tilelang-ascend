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