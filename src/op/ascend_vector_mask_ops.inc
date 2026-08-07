/*
 * Internal compiler-managed Vector terminal identities.
 *
 * Consumers define TL_ASCEND_SELECTED_VECTOR_OP(name, arity, base,
 * contract_kind) before including this file. Keep this list in the explicit
 * selection priority order for terminals that share a semantic base op.
 */

TL_ASCEND_SELECTED_VECTOR_OP(ascend_copy_ub_to_ub_neutral, -1,
                             builtin::call_extern, kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_copy_ub_to_ub_self_contained_normal_full,
                             -1, builtin::call_extern, kSelfContainedNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_silu_self_contained_normal_full, 3,
                             ascend_silu, kSelfContainedNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_mul_add_dst_raw_counter, 4,
                             ascend_mul_add_dst, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_add_raw_counter, 4, ascend_add, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_adds_raw_counter, -1, ascend_adds,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_mul_raw_counter, 4, ascend_mul, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_muls_raw_counter, -1, ascend_muls,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_sub_raw_counter, 4, ascend_sub, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_subs_raw_counter, -1, ascend_subs,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_div_raw_counter, 4, ascend_div, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_divs_raw_counter, -1, ascend_divs,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_max_raw_counter, 4, ascend_max, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_maxs_raw_counter, -1, ascend_maxs,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_min_raw_counter, 4, ascend_min, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_mins_raw_counter, -1, ascend_mins,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_bitwise_and_raw_counter, 4,
                             ascend_bitwise_and, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_bitwise_or_raw_counter, 4,
                             ascend_bitwise_or, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_compare_neutral, 5, ascend_compare,
                             kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_compare_scalar_neutral, -1,
                             ascend_compare_scalar, kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_exp_raw_counter, 3, ascend_exp, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_ln_raw_counter, 3, ascend_ln, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_abs_raw_counter, 3, ascend_abs, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_reciprocal_raw_counter, 3,
                             ascend_reciprocal, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_sqrt_raw_counter, 3, ascend_sqrt,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_rsqrt_raw_counter, 3, ascend_rsqrt,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_relu_raw_counter, 3, ascend_relu,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_bitwise_not_raw_counter, 3,
                             ascend_bitwise_not, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_select_self_contained_normal_full, -1,
                             ascend_select, kSelfContainedNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_leaky_relu_raw_counter, 5,
                             ascend_leaky_relu, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_axpy_raw_counter, 5, ascend_axpy,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_bitwise_lshift_raw_counter, 4,
                             ascend_bitwise_lshift, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_bitwise_rshift_raw_counter, 4,
                             ascend_bitwise_rshift, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_sort32_neutral, 4, ascend_sort32, kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_createvecindex_composite, 3,
                             ascend_createvecindex, kUnknownAll)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_arith_progression_composite, 5,
                             ascend_arith_progression,
                             kCompositeNormalFullToNormalUnknownPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_sin_composite, 4, ascend_sin,
                             kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_cos_composite, 4, ascend_cos,
                             kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_transpose_neutral, 2, ascend_transpose,
                             kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_gather_composite, -1, ascend_gather,
                             kUnknownAll)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_reduce_narrow_raw_normal, -1, ascend_reduce,
                             kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_reduce_half_sum_raw_normal, -1,
                             ascend_reduce, kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_reduce_advanced_composite, -1,
                             ascend_reduce, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_block_reduce_max_raw_normal, -1,
                             ascend_block_reduce_max, kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_block_reduce_min_raw_normal, -1,
                             ascend_block_reduce_min, kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_block_reduce_sum_raw_normal, -1,
                             ascend_block_reduce_sum, kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_gatherb_neutral, 7, ascend_gatherb,
                             kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_cast_raw_counter, 4, ascend_cast,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_pow_composite, 4, ascend_pow,
                             kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_bitwise_xor_composite, 4,
                             ascend_bitwise_xor, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_broadcast_advanced_composite, -1,
                             ascend_broadcast, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_broadcast_raw_counter, -1, ascend_broadcast,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_fill_raw_counter, 4, ascend_fill,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_sort_composite, 6, ascend_sort,
                             kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_merge_sort_neutral, 6, ascend_merge_sort,
                             kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_topk_composite, 7, ascend_topk,
                             kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_gather_mask_fixed_self_contained_normal, -1,
                             ascend_gather_mask, kSelfContainedNormalZero)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_gather_mask_custom_composite, -1,
                             ascend_gather_mask, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_init_sort_buf_composite, 4,
                             ascend_init_sort_buf,
                             kCompositeNormalFullToNormalUnknownPayload)
TL_ASCEND_SELECTED_VECTOR_OP(
    ascend_bilinear_interpolation_self_contained_normal, -1,
    ascend_bilinear_interpolation, kSelfContainedNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_wholereducemax_raw_normal, -1,
                             ascend_wholereducemax, kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_wholereducemin_raw_normal, -1,
                             ascend_wholereducemin, kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_wholereducesum_raw_normal, -1,
                             ascend_wholereducesum, kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_sigmoid_self_contained_normal_full, 4,
                             ascend_sigmoid, kSelfContainedNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_clamp_max_raw_counter, 6, ascend_clamp_max,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_clamp_min_raw_counter, 6, ascend_clamp_min,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_clamp_raw_counter, 6, ascend_clamp,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_round_cast_raw_counter, -1, ascend_round,
                             kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_round_advanced_composite, 4, ascend_round,
                             kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_sub_experiment_raw_counter, 4,
                             ascend_sub_experiment, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_abs_experiment_raw_counter, 3,
                             ascend_abs_experiment, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_mins_experiment_raw_counter, 4,
                             ascend_mins_experiment, kRawCounter)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_reducesum_experiment_composite, 4,
                             ascend_reducesum_experiment, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_reducesum_mask_experiment_composite, 6,
                             ascend_reducesum_mask_experiment,
                             kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(
    ascend_gather_mask_experiment_self_contained_normal, -1,
    ascend_gather_mask_experiment, kSelfContainedNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_fill_experiment_raw_normal, -1,
                             ascend_fill_experiment, kRawNormalDynamicPayload)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_sum_experiment_composite, 6,
                             ascend_sum_experiment, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_datacachecleanandinvalid_experiment_neutral,
                             2, ascend_datacachecleanandinvalid_experiment,
                             kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_brcb_experiment_neutral, 6,
                             ascend_brcb_experiment, kNeutral)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_row_expand_mul_experiment_raw_normal, -1,
                             ascend_row_expand_mul_experiment, kRawNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_row_expand_sub_experiment_raw_normal, -1,
                             ascend_row_expand_sub_experiment, kRawNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_row_expand_div_experiment_raw_normal, -1,
                             ascend_row_expand_div_experiment, kRawNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_exp_experiment_raw_normal, -1,
                             ascend_exp_experiment, kRawNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_tail_unary_dynamic_composite, -1,
                             ascend_tail_unary, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_tail_binary_dynamic_composite, -1,
                             ascend_tail_binary, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_tail_scalar_dynamic_composite, -1,
                             ascend_tail_scalar, kCompositeNormalFull)
TL_ASCEND_SELECTED_VECTOR_OP(ascend_tail_reduce_dynamic_composite, -1,
                             ascend_tail_reduce, kCompositeNormalFull)
