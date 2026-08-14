/*
 * Compiler-selected Ascend Vector terminals.
 *
 * This is the single declarative schema. Selection, mask legalization, and
 * codegen consume their own recipe columns; their interpretation stays in the
 * owning pass. Define TL_ASCEND_SEMANTIC_OP, TL_ASCEND_PHYSICAL, and
 * TL_ASCEND_HELPER before including this file.
 */

#define TL_ASCEND_SEMANTIC_HELPER(name, base, min_arity, max_arity, abi,       \
                                  contract)                                    \
  TL_ASCEND_SEMANTIC_OP(base, "", min_arity, max_arity, abi)                   \
  TL_ASCEND_HELPER(name, kAlways, contract)

#define TL_ASCEND_FIXED_HELPER(name, base, arity, contract)                    \
  TL_ASCEND_SEMANTIC_HELPER(name, base, arity, arity, kArityOnly, contract)

#define TL_ASCEND_NEUTRAL_HELPER(name, base, arity)                            \
  TL_ASCEND_FIXED_HELPER(name, base, arity, kNeutral)

#define TL_ASCEND_COMPOSITE_HELPER(name, base, arity)                          \
  TL_ASCEND_FIXED_HELPER(name, base, arity, kCompositeNormalFullUnknown)

#define TL_ASCEND_NORMAL_FULL_HELPER(name, base, arity)                        \
  TL_ASCEND_FIXED_HELPER(name, base, arity, kSelfContainedNormalFull)

#define TL_ASCEND_NATURAL_BINARY(name, base, dtype_domain, intrinsic)          \
  TL_ASCEND_SEMANTIC_OP(base, "", 4, 4, kArityOnly)                            \
  TL_ASCEND_PHYSICAL(name##_raw, kNaturalNormal, dtype_domain, kSame012,       \
                     kMaskSpec, kRawBinary, intrinsic)

#define TL_ASCEND_NATURAL_UNARY(name, base, dtype_domain, intrinsic)           \
  TL_ASCEND_SEMANTIC_OP(base, "", 3, 3, kArityOnly)                            \
  TL_ASCEND_PHYSICAL(name##_raw, kNaturalNormal, dtype_domain, kSame01,        \
                     kMaskSpec, kRawUnary, intrinsic)

#define TL_ASCEND_NATURAL_SCALAR(name, base, max_arity, dtype_domain, emitter, \
                                 intrinsic)                                    \
  TL_ASCEND_SEMANTIC_OP(base, "", 4, max_arity, kArityOnly)                    \
  TL_ASCEND_PHYSICAL(name##_raw, kNaturalNormal, dtype_domain, kScalar,        \
                     kMaskSpec, emitter, intrinsic)

#define TL_ASCEND_NORMAL_MASK_OP(name, arity, emitter, intrinsic)              \
  TL_ASCEND_SEMANTIC_OP(name, "", arity, arity, kArityOnly)                    \
  TL_ASCEND_PHYSICAL(name##_raw_normal, kNormalMaskArg, kFloat16Float32,       \
                     kSame01, kMask, emitter, intrinsic)

#define TL_ASCEND_CLAMP_OP(name, arity, emitter, intrinsic)                    \
  TL_ASCEND_SEMANTIC_OP(name, "", arity, arity, kArityOnly)                    \
  TL_ASCEND_PHYSICAL(name##_raw_counter, kAlways, kFloat16Float32Int16Int32,   \
                     kSame12, kCount, emitter, intrinsic)

#define TL_ASCEND_ROW_EXPAND_OP(name, intrinsic)                               \
  TL_ASCEND_SEMANTIC_OP(name, "", 4, 5, kArityOnly)                            \
  TL_ASCEND_PHYSICAL(name##_raw_normal, kAlways, kFloat16Float32, kRowExpand,  \
                     kNone, kRawRowExpand, intrinsic)

TL_ASCEND_SEMANTIC_OP(builtin::call_extern, "copy_ub_to_ub", 3, 9, kCopyUb)
TL_ASCEND_PHYSICAL(ascend_copy_ub_to_ub_selected, kCopyUbSameType, kAny, kNone,
                   kNone, kHelper, "")

TL_ASCEND_NATURAL_BINARY(ascend_add, ascend_add, kFloat16Float32Int16Int32,
                         "AscendC::Add")
TL_ASCEND_NATURAL_BINARY(ascend_sub, ascend_sub, kFloat16Float32Int16Int32,
                         "AscendC::Sub")
TL_ASCEND_NATURAL_BINARY(ascend_mul, ascend_mul, kFloat16Float32Int16Int32,
                         "AscendC::Mul")
TL_ASCEND_NATURAL_BINARY(ascend_div, ascend_div, kFloat16Float32,
                         "AscendC::Div")
TL_ASCEND_NATURAL_BINARY(ascend_max, ascend_max, kFloat16Float32Int16Int32,
                         "AscendC::Max")
TL_ASCEND_NATURAL_BINARY(ascend_min, ascend_min, kFloat16Float32Int16Int32,
                         "AscendC::Min")
TL_ASCEND_NATURAL_BINARY(ascend_bitwise_and, ascend_bitwise_and, kInt16UInt16,
                         "AscendC::And")
TL_ASCEND_NATURAL_BINARY(ascend_bitwise_or, ascend_bitwise_or, kInt16UInt16,
                         "AscendC::Or")

TL_ASCEND_NATURAL_UNARY(ascend_exp, ascend_exp, kFloat16Float32, "AscendC::Exp")
TL_ASCEND_NATURAL_UNARY(ascend_ln, ascend_ln, kFloat16Float32, "AscendC::Ln")
TL_ASCEND_NATURAL_UNARY(ascend_abs, ascend_abs, kFloat16Float32, "AscendC::Abs")
TL_ASCEND_NATURAL_UNARY(ascend_reciprocal, ascend_reciprocal, kFloat16Float32,
                        "AscendC::Reciprocal")
TL_ASCEND_NATURAL_UNARY(ascend_sqrt, ascend_sqrt, kFloat16Float32,
                        "AscendC::Sqrt")
TL_ASCEND_NATURAL_UNARY(ascend_rsqrt, ascend_rsqrt, kFloat16Float32,
                        "AscendC::Rsqrt")
TL_ASCEND_NATURAL_UNARY(ascend_relu, ascend_relu, kFloat16Float32Int32,
                        "AscendC::Relu")
TL_ASCEND_NATURAL_UNARY(ascend_bitwise_not, ascend_bitwise_not, kInt16UInt16,
                        "AscendC::Not")

TL_ASCEND_NATURAL_SCALAR(ascend_adds, ascend_adds, 5, kFloat16Float32Int16Int32,
                         kRawScalar, "AscendC::Adds")
TL_ASCEND_NATURAL_SCALAR(ascend_subs, ascend_subs, 5, kFloat16Float32Int16Int32,
                         kRawSubs, "AscendC::Adds")
TL_ASCEND_NATURAL_SCALAR(ascend_muls, ascend_muls, 5, kFloat16Float32Int16Int32,
                         kRawScalar, "AscendC::Muls")
TL_ASCEND_NATURAL_SCALAR(ascend_divs, ascend_divs, 5, kFloat16Float32, kRawDivs,
                         "AscendC::Muls")
TL_ASCEND_NATURAL_SCALAR(ascend_maxs, ascend_maxs, 5, kFloat16Float32Int16Int32,
                         kRawScalar, "AscendC::Maxs")
TL_ASCEND_NATURAL_SCALAR(ascend_mins, ascend_mins, 5, kFloat16Float32Int16Int32,
                         kRawScalar, "AscendC::Mins")
TL_ASCEND_NATURAL_SCALAR(ascend_leaky_relu, ascend_leaky_relu, 4,
                         kFloat16Float32, kRawScalar, "AscendC::LeakyRelu")
TL_ASCEND_NATURAL_SCALAR(ascend_bitwise_lshift, ascend_bitwise_lshift, 4,
                         kShiftInteger, kRawShift, "AscendC::ShiftLeft")
TL_ASCEND_NATURAL_SCALAR(ascend_bitwise_rshift, ascend_bitwise_rshift, 4,
                         kShiftInteger, kRawShift, "AscendC::ShiftRight")

TL_ASCEND_SEMANTIC_OP(ascend_axpy, "", 4, 4, kArityOnly)
TL_ASCEND_PHYSICAL(ascend_axpy_raw, kNaturalNormal, kFloat16Float32, kAxpy,
                   kMaskSpec, kRawAxpy, "AscendC::Axpy")

TL_ASCEND_NORMAL_FULL_HELPER(ascend_silu_self_contained_normal_full,
                             ascend_silu, 3)
TL_ASCEND_NORMAL_FULL_HELPER(ascend_mul_add_dst_self_contained_normal_full,
                             ascend_mul_add_dst, 4)
TL_ASCEND_NEUTRAL_HELPER(ascend_compare_neutral, ascend_compare, 5)
TL_ASCEND_SEMANTIC_HELPER(ascend_compare_scalar_neutral, ascend_compare_scalar,
                          5, 6, kArityOnly, kNeutral)
TL_ASCEND_SEMANTIC_HELPER(ascend_select_self_contained_normal_full,
                          ascend_select, 7, 9, kSelect,
                          kSelfContainedNormalFull)
TL_ASCEND_NEUTRAL_HELPER(ascend_sort32_neutral, ascend_sort32, 4)
TL_ASCEND_FIXED_HELPER(ascend_createvecindex_unknown, ascend_createvecindex, 4,
                       kCreateVecIndex)
TL_ASCEND_FIXED_HELPER(ascend_arith_progression_composite,
                       ascend_arith_progression, 5,
                       kCompositeNormalFullToNormalUnknownPayload)
TL_ASCEND_COMPOSITE_HELPER(ascend_sin_composite, ascend_sin, 4)
TL_ASCEND_COMPOSITE_HELPER(ascend_cos_composite, ascend_cos, 4)
TL_ASCEND_NEUTRAL_HELPER(ascend_transpose_neutral, ascend_transpose, 2)
TL_ASCEND_FIXED_HELPER(ascend_gather_count, ascend_gather, 5, kGatherCount)

TL_ASCEND_SEMANTIC_OP(ascend_reduce, "", 4, 6, kReduce)
TL_ASCEND_PHYSICAL(ascend_reduce_narrow_raw_normal, kReduceNarrow,
                   kFloat16Float32, kReduce, kMask, kRawReduce,
                   "tl::ascend::reduce_narrow")
TL_ASCEND_PHYSICAL(ascend_reduce_half_sum_raw_normal, kReduceHalfSum, kFloat16,
                   kReduce, kMask, kRawReduce, "tl::ascend::reduce_sum_half")
TL_ASCEND_HELPER(ascend_reduce_advanced_composite, kReduceComposite,
                 kCompositeNormalFullUnknown)

TL_ASCEND_NORMAL_MASK_OP(ascend_block_reduce_max, 7, kRawBlockReduce,
                         "AscendC::BlockReduceMax")
TL_ASCEND_NORMAL_MASK_OP(ascend_block_reduce_min, 7, kRawBlockReduce,
                         "AscendC::BlockReduceMin")
TL_ASCEND_NORMAL_MASK_OP(ascend_block_reduce_sum, 7, kRawBlockReduce,
                         "AscendC::BlockReduceSum")
TL_ASCEND_NORMAL_MASK_OP(ascend_wholereducemax, 8, kRawWholeReduce,
                         "AscendC::WholeReduceMax")
TL_ASCEND_NORMAL_MASK_OP(ascend_wholereducemin, 8, kRawWholeReduce,
                         "AscendC::WholeReduceMin")
TL_ASCEND_NORMAL_MASK_OP(ascend_wholereducesum, 7, kRawWholeReduce,
                         "AscendC::WholeReduceSum")

TL_ASCEND_FIXED_HELPER(ascend_gatherb_payload_full, ascend_gatherb, 7,
                       kPayloadFull)
TL_ASCEND_SEMANTIC_OP(ascend_cast, "", 4, 4, kArityOnly)
TL_ASCEND_PHYSICAL(ascend_cast_raw_counter, kAlways, kAny, kCast, kCount,
                   kRawCast, "AscendC::Cast")
TL_ASCEND_COMPOSITE_HELPER(ascend_pow_composite, ascend_pow, 4)
TL_ASCEND_COMPOSITE_HELPER(ascend_bitwise_xor_composite, ascend_bitwise_xor, 4)

TL_ASCEND_SEMANTIC_OP(ascend_broadcast, "", 6, 9, kBroadcast)
TL_ASCEND_HELPER(ascend_broadcast_advanced_composite, kBroadcastComposite,
                 kCompositeNormalFullUnknown)
TL_ASCEND_PHYSICAL(ascend_broadcast_raw_counter, kBroadcastCounter, kAny,
                   kBroadcast, kCount, kRawBroadcast, "tl::ascend::Broadcast")

TL_ASCEND_SEMANTIC_OP(ascend_fill, "", 4, 4, kArityOnly)
TL_ASCEND_PHYSICAL(ascend_fill_raw_counter, kAlways, kDuplicate, kNone, kCount,
                   kRawFill, "AscendC::Duplicate")
TL_ASCEND_COMPOSITE_HELPER(ascend_sort_composite, ascend_sort, 6)
TL_ASCEND_SEMANTIC_HELPER(ascend_merge_sort_neutral, ascend_merge_sort, 7, 11,
                          kMergeSort, kNeutral)
TL_ASCEND_COMPOSITE_HELPER(ascend_topk_composite, ascend_topk, 8)

TL_ASCEND_SEMANTIC_OP(ascend_gather_mask, "", 4, 4, kArityOnly)
TL_ASCEND_PHYSICAL(ascend_gather_mask_selected, kGatherMaskFixed, kAny, kNone,
                   kNone, kHelper, "")

TL_ASCEND_FIXED_HELPER(ascend_init_sort_buf_composite, ascend_init_sort_buf, 4,
                       kCompositeNormalFullToNormalUnknownPayload)
TL_ASCEND_FIXED_HELPER(ascend_bilinear_interpolation_self_contained,
                       ascend_bilinear_interpolation, 11,
                       kSelfContainedNormalExplicit)
TL_ASCEND_NORMAL_FULL_HELPER(ascend_sigmoid_self_contained_normal_full,
                             ascend_sigmoid, 4)

TL_ASCEND_CLAMP_OP(ascend_clamp_max, 5, kRawClampMaxMin, "AscendC::Mins")
TL_ASCEND_CLAMP_OP(ascend_clamp_min, 5, kRawClampMaxMin, "AscendC::Maxs")
TL_ASCEND_CLAMP_OP(ascend_clamp, 6, kRawClamp, "AscendC::Maxs+Mins")

TL_ASCEND_SEMANTIC_OP(ascend_round, "", 3, 4, kRound)
TL_ASCEND_PHYSICAL(ascend_round_cast_raw_counter, kRoundCounter, kFloat32,
                   kSame01, kCount, kRawRound, "AscendC::Cast")
TL_ASCEND_HELPER(ascend_round_advanced_composite, kRoundComposite,
                 kCompositeNormalFullUnknown)

TL_ASCEND_NORMAL_FULL_HELPER(ascend_sub_experiment_self_contained_normal_full,
                             ascend_sub_experiment, 4)
TL_ASCEND_NORMAL_FULL_HELPER(ascend_abs_experiment_self_contained_normal_full,
                             ascend_abs_experiment, 3)
TL_ASCEND_NORMAL_FULL_HELPER(ascend_mins_experiment_self_contained_normal_full,
                             ascend_mins_experiment, 4)
TL_ASCEND_COMPOSITE_HELPER(ascend_reducesum_experiment_composite,
                           ascend_reducesum_experiment, 4)
TL_ASCEND_COMPOSITE_HELPER(ascend_reducesum_mask_experiment_composite,
                           ascend_reducesum_mask_experiment, 6)
TL_ASCEND_FIXED_HELPER(ascend_gather_mask_experiment_self_contained,
                       ascend_gather_mask_experiment, 11,
                       kSelfContainedNormalExplicit)
TL_ASCEND_FIXED_HELPER(ascend_fill_experiment_explicit_mask,
                       ascend_fill_experiment, 7, kNormalExplicitLowArg)
TL_ASCEND_COMPOSITE_HELPER(ascend_sum_experiment_composite,
                           ascend_sum_experiment, 6)
TL_ASCEND_NEUTRAL_HELPER(ascend_datacachecleanandinvalid_experiment_neutral,
                         ascend_datacachecleanandinvalid_experiment, 2)
TL_ASCEND_FIXED_HELPER(ascend_brcb_experiment_payload_full,
                       ascend_brcb_experiment, 6, kPayloadFull)

TL_ASCEND_ROW_EXPAND_OP(ascend_row_expand_mul_experiment, "mul_mask")
TL_ASCEND_ROW_EXPAND_OP(ascend_row_expand_sub_experiment, "sub_mask")
TL_ASCEND_ROW_EXPAND_OP(ascend_row_expand_div_experiment, "div_mask")
TL_ASCEND_SEMANTIC_OP(ascend_exp_experiment, "", 3, 3, kArityOnly)
TL_ASCEND_PHYSICAL(ascend_exp_experiment_raw_normal, kAlways, kFloat16Float32,
                   kSame12, kNone, kRawExpExperiment, "exp_mask")

TL_ASCEND_COMPOSITE_HELPER(ascend_tail_unary_composite, ascend_tail_unary, 6)
TL_ASCEND_COMPOSITE_HELPER(ascend_tail_binary_composite, ascend_tail_binary, 7)
TL_ASCEND_COMPOSITE_HELPER(ascend_tail_scalar_composite, ascend_tail_scalar, 7)
TL_ASCEND_COMPOSITE_HELPER(ascend_tail_reduce_composite, ascend_tail_reduce, 8)
TL_ASCEND_NEUTRAL_HELPER(ascend_tail_compare_neutral, ascend_tail_compare, 9)
TL_ASCEND_NEUTRAL_HELPER(ascend_tail_compare_scalar_neutral,
                         ascend_tail_compare_scalar, 9)
TL_ASCEND_NEUTRAL_HELPER(ascend_tail_select_neutral, ascend_tail_select, 13)
TL_ASCEND_SEMANTIC_HELPER(ascend_tail_broadcast_unknown, ascend_tail_broadcast,
                          12, 13, kArityOnly, kUnknown)

#undef TL_ASCEND_NATURAL_BINARY
#undef TL_ASCEND_CLAMP_OP
#undef TL_ASCEND_NORMAL_MASK_OP
#undef TL_ASCEND_NATURAL_SCALAR
#undef TL_ASCEND_NATURAL_UNARY
#undef TL_ASCEND_ROW_EXPAND_OP
#undef TL_ASCEND_NORMAL_FULL_HELPER
#undef TL_ASCEND_COMPOSITE_HELPER
#undef TL_ASCEND_NEUTRAL_HELPER
#undef TL_ASCEND_FIXED_HELPER
#undef TL_ASCEND_SEMANTIC_HELPER
