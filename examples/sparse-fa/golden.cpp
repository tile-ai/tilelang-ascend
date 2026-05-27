inline torch::Tensor build_mtgr_torch_full_visible_mask(
    const MTGRAttentionTestShape& shape,
    const torch::Device& device) {
  const int64_t history = shape.history;
  const int64_t context = shape.context;
  const int64_t realtime = shape.realtime;
  const int64_t target = shape.target;
  const int64_t matched = shape.matched_prefix;
  const int64_t target_begin = history + context + realtime;
  const int64_t total = target_begin + target;

  auto mask =
      torch::zeros({total, total},
                   torch::TensorOptions().dtype(torch::kBool).device(device));
  for (int64_t q_abs = 0; q_abs < total; ++q_abs) {
    auto row = mask.select(0, q_abs);
    if (q_abs < history) {
      row.narrow(0, 0, q_abs + 1).fill_(true);
    } else if (q_abs < history + context) {
      row.narrow(0, 0, history + context).fill_(true);
    } else if (q_abs < target_begin) {
      row.narrow(0, 0, q_abs + 1).fill_(true);
    } else {
      row.narrow(0, 0, target_begin).fill_(true);
      row.narrow(0, q_abs, 1).fill_(true);
    }
  }
  return mask;
}

inline torch::Tensor run_mtgr_torch_mask_attention_reference(
    const torch::Tensor& full_query_bshd,
    const torch::Tensor& full_key_bshd,
    const torch::Tensor& full_value_bshd,
    const MTGRAttentionTestShape& shape,
    double sm_scale) {
  CHECK_EQ(full_query_bshd.dim(), 4);
  CHECK_EQ(full_key_bshd.dim(), 4);
  CHECK_EQ(full_value_bshd.dim(), 4);
  CHECK_EQ(full_query_bshd.size(0), 1);
  CHECK_EQ(full_key_bshd.size(0), 1);
  CHECK_EQ(full_value_bshd.size(0), 1);
  CHECK_EQ(full_query_bshd.size(1), shape.total_len());
  CHECK_EQ(full_key_bshd.size(1), shape.total_len());
  CHECK_EQ(full_value_bshd.size(1), shape.total_len());
  CHECK_EQ(full_query_bshd.size(2), shape.heads);
  CHECK_EQ(full_key_bshd.size(2), shape.kv_heads);
  CHECK_EQ(full_value_bshd.size(2), shape.kv_heads);
  CHECK_EQ(full_query_bshd.size(3), shape.head_dim);
  CHECK_EQ(full_key_bshd.size(3), shape.head_dim);
  CHECK_EQ(full_value_bshd.size(3), shape.head_dim);
  CHECK_EQ(shape.heads, shape.kv_heads)
      << "torch mask precision reference currently assumes MHA";

  auto query = full_query_bshd.select(0, 0).to(torch::kFloat32).contiguous();
  auto key = full_key_bshd.select(0, 0).to(torch::kFloat32).contiguous();
  auto value = full_value_bshd.select(0, 0).to(torch::kFloat32).contiguous();
  auto visible_mask =
      build_mtgr_torch_full_visible_mask(shape, query.device()).contiguous();

  auto scores = torch::einsum("qhd,khd->qhk", {query, key}) *
                static_cast<float>(sm_scale);
  auto masked_scores =
      scores.masked_fill(visible_mask.logical_not().unsqueeze(1),
                         -std::numeric_limits<float>::infinity());
  auto probs = torch::softmax(masked_scores, /*dim=*/-1);
  auto full_output = torch::einsum("qhk,khd->qhd", {probs, value}).contiguous();
  return full_output.narrow(0, shape.matched_prefix, shape.local_len())
      .contiguous();
}