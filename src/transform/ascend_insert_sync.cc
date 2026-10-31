// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#include <tvm/node/structural_hash.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ascend_sync/legacy_adapter.h"
#include "ascend_sync/legacy_persistent_normalize.h"
#include "ascend_sync/legacy_persistent_pipeline.h"
#include "ascend_sync/legacy_pipeline_partition.h"
#include "ascend_sync/upstream/insert_sync.h"
#include "ascend_sync/z3_proof_service.h"

namespace tvm::tl {
namespace {

constexpr const char *kSyncProof = "tl.ascend_sync_proof";
constexpr const char *kSyncBackend = "tl.ascend_sync_backend";
constexpr const char *kSyncEngineVersion = "upstream-1c0f402e-legacy-v19";

String SyncInputHash(tir::PrimFunc function) {
  return std::to_string(
      StructuralHash()(WithoutAttr(std::move(function), kSyncProof)));
}

tir::PrimFunc SynchronizeLegacyGraph(
    tir::PrimFunc function, const Target &target, const String &platform,
    bool ordinary, ascend_sync::SyncTargetPolicy policy,
    const ascend_sync::LegacyPipelineCase *pipeline_case = nullptr) {
  auto imported = ascend_sync::ImportLegacySync(function, target, platform,
                                                ordinary, pipeline_case);
  policy.physical_conflict_oracle = imported.MakeOracle();
  std::vector<std::shared_ptr<ascend_sync::IRStructure>> synchronized;
  for (auto &graph : imported.local_graphs) {
    ascend_sync::RunSyncEngine(graph.roots, imported.multibuffer,
                               imported.manual_buffer_versions,
                               graph.reservations, policy);
    synchronized.insert(synchronized.end(), graph.roots.begin(),
                        graph.roots.end());
  }
  imported.roots = std::move(synchronized);
  return ascend_sync::ExportLegacySync(imported);
}

tir::PrimFunc InsertUpstreamSync(tir::PrimFunc function, const Target &target,
                                 const String &platform,
                                 const tvm::transform::PassContext &context) {
  bool ordinary =
      context->GetConfig<Bool>("tl.ascend_auto_sync", Bool(false)).value();
  bool vector_scalar =
      context->GetConfig<Bool>("tl.ascend_auto_sync_vs", Bool(false)).value();
  std::ostringstream signature;
  signature << kSyncEngineVersion << ':' << target << ':' << platform << ':'
            << ordinary << ':' << vector_scalar;
  auto proof = function->GetAttr<Map<String, String>>(kSyncProof);
  if (proof.defined()) {
    auto previous_signature = proof.value().Get("signature");
    auto previous_hash = proof.value().Get("input_hash");
    ICHECK(previous_signature.defined() && previous_hash.defined() &&
           previous_signature.value() == signature.str() &&
           previous_hash.value() == SyncInputHash(function))
        << "AscendInsertSync inputs changed after synchronization; rerun from "
           "pre-sync IR";
    return function;
  }
  if (!ordinary && !vector_scalar) {
    return function;
  }
  ICHECK(ascend_sync::Z3ProofService::Available())
      << "AscendInsertSync requires USE_ASCEND_SYNC_Z3=ON and a compatible "
         "Z3_ROOT";

  auto normalization =
      ascend_sync::NormalizeLegacyPersistentWithCertificates(function);
  function = normalization.function;

  ascend_sync::SyncTargetPolicy policy;
  policy.platform = platform;
  policy.ordinary_sync = ordinary;
  policy.vector_scalar_sync = vector_scalar;
  policy.Validate();
  tir::PrimFunc result;
  if (auto persistent = ascend_sync::SynchronizeLegacyPersistentPipeline(
          normalization, target, platform, ordinary, policy)) {
    result = persistent.value();
  } else if (ascend_sync::IsLegacyPipelineNoOp(function)) {
    result = function;
  } else if (auto partition =
                 ascend_sync::PartitionLegacyDynamicPipeline(function)) {
    std::vector<tir::PrimFunc> cases;
    for (const auto &pipeline_case : partition->cases) {
      cases.push_back(pipeline_case.zero_work
                          ? pipeline_case.function
                          : SynchronizeLegacyGraph(pipeline_case.function,
                                                   target, platform, ordinary,
                                                   policy, &pipeline_case));
    }
    result = ascend_sync::MergeLegacyPipelineCases(partition.value(), cases);
  } else {
    result =
        SynchronizeLegacyGraph(function, target, platform, ordinary, policy);
  }
  Map<String, String> marker{{"signature", signature.str()},
                             {"input_hash", SyncInputHash(result)}};
  return WithAttr(std::move(result), kSyncProof, std::move(marker));
}

} // namespace

tvm::transform::Pass AscendInsertSync(Target target, String platform) {
  auto pass = [target, platform](tir::PrimFunc function, IRModule,
                                 tvm::transform::PassContext context) {
    return InsertUpstreamSync(std::move(function), target, platform, context);
  };
  return tir::transform::CreatePrimFuncPass(pass, 0, "tl.AscendInsertSync", {});
}

TVM_REGISTER_PASS_CONFIG_OPTION(kSyncBackend, String);
TVM_REGISTER_GLOBAL("tl.transform.AscendInsertSync")
    .set_body_typed(AscendInsertSync);
// Cache hits bypass lowering and its proof marker. Python cache keys and
// restored-artifact manifests query this native ABI before trusting them.
TVM_REGISTER_GLOBAL("tl.ascend_sync.EngineCacheABI").set_body_typed([]() {
  return String(kSyncEngineVersion);
});

} // namespace tvm::tl
