// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <tvm/arith/int_set.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <unordered_map>

namespace tvm {
namespace tl {
using namespace tir;

Block MakeBlock(const Stmt &body,
                const Map<Var, Buffer> &buffer_data_to_buffer);

// These are the existing injector's stage/liveness rules, shared with the
// constraint planner rather than independently predicted buffer counts.
struct PipelineAnnotation {
  int stage;
  int order;
  bool async;
};
using PipelineInfo = std::unordered_map<Block, PipelineAnnotation,
                                        ObjectPtrHash, ObjectPtrEqual>;
struct BufferAccessInfo {
  int def = -1;
  int use = -1;
};

inline std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash,
                          ObjectPtrEqual>
GetLegacyPipelineBufferAccessInfo(const PipelineInfo &pipeline_info,
                                  int *max_stage) {
  std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual>
      infos;
  for (const auto &pair : pipeline_info) {
    const Block &block = pair.first;
    int stage = pair.second.stage;
    *max_stage = std::max(*max_stage, stage);

    for (const BufferRegion &write : block->writes) {
      if (!infos.count(write->buffer)) {
        infos.emplace(write->buffer, BufferAccessInfo{});
      }
      auto &info = infos.at(write->buffer);
      if (info.def == -1) {
        info.def = stage;
      } else {
        info.def = std::min(info.def, stage);
      }
    }

    for (const BufferRegion &read : block->reads) {
      if (!infos.count(read->buffer)) {
        infos.emplace(read->buffer, BufferAccessInfo{});
      }
      auto &info = infos.at(read->buffer);
      info.use = std::max(info.use, stage);
    }
  }
  return infos;
}

inline bool LegacyPipelineRegionsMayConflict(Region region1, Region region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

inline int
ComputeLegacyPipelineBufferVersions(const Buffer &buffer,
                                    const BufferAccessInfo &buffer_info,
                                    const PipelineInfo &pipeline_info) {
  if (buffer_info.def == -1) {
    // Keep the original number of versions as buffers defined outside the
    // software pipeline should not be mutated.
    return 1;
  }

  // `use - def + 1` is a upper bound of the needed versions
  // We optimize a few case where the number of versions can be smaller than
  // the upper bound
  int num_versions = buffer_info.use - buffer_info.def + 1;
  if (num_versions >= 2) {
    // A special case when `use - def + 1 == 2`. Double buffering is only
    // needed in this case when these exists a reader block_i and a writer
    // block_j such that order(block_i) < order(block_j) and stage(block_i) <
    // stage(block_j) and the access regions of block_i and block_j overlap.
    bool need_multi_version = false;
    for (const auto &pair1 : pipeline_info) {
      const Block &writer_block = pair1.first;
      const auto &writer_info = pair1.second;

      auto it1 =
          std::find_if(writer_block->writes.begin(), writer_block->writes.end(),
                       [&](const BufferRegion &buffer_region) {
                         return buffer_region->buffer.same_as(buffer);
                       });
      if (it1 == writer_block->writes.end()) {
        continue;
      }

      for (const auto &pair2 : pipeline_info) {
        const Block &reader_block = pair2.first;
        const auto &reader_info = pair2.second;
        auto it2 =
            std::find_if(reader_block->reads.begin(), reader_block->reads.end(),
                         [&](const BufferRegion &buffer_region) {
                           return buffer_region->buffer.same_as(buffer);
                         });
        if (it2 == reader_block->reads.end()) {
          continue;
        }
        if (writer_info.order < reader_info.order &&
            writer_info.stage < reader_info.stage &&
            LegacyPipelineRegionsMayConflict((*it1)->region, (*it2)->region)) {
          need_multi_version = true;
          break;
        }
      }
    }
    if (!need_multi_version) {
      num_versions--;
    }
  }
  return num_versions;
}

// The constraint bridge treats all dtype/offset views of a data Var as one
// physical storage. Legacy Buffer-identity liveness remains unchanged.
inline int
ComputeAscendStoragePipelineVersions(const Var &data,
                                     const PipelineInfo &pipeline_info) {
  int first = -1, last = -1;
  for (const auto &entry : pipeline_info) {
    for (const BufferRegion &write : entry.first->writes)
      if (write->buffer->data.same_as(data))
        first = first < 0 ? entry.second.stage
                          : std::min(first, entry.second.stage);
    for (const BufferRegion &read : entry.first->reads)
      if (read->buffer->data.same_as(data))
        last = std::max(last, entry.second.stage);
  }
  if (first < 0)
    return 1;
  int versions = std::max(1, last - first + 1);
  if (versions >= 2) {
    bool conflict = false;
    for (const auto &writer : pipeline_info)
      for (const BufferRegion &write : writer.first->writes) {
        if (!write->buffer->data.same_as(data))
          continue;
        for (const auto &reader : pipeline_info)
          for (const BufferRegion &read : reader.first->reads) {
            if (!read->buffer->data.same_as(data) ||
                writer.second.order >= reader.second.order ||
                writer.second.stage >= reader.second.stage)
              continue;
            // Different views may use different ranks, offsets and units. They
            // conservatively overlap unless they are literally the same view.
            conflict |=
                !write->buffer.same_as(read->buffer) ||
                LegacyPipelineRegionsMayConflict(write->region, read->region);
          }
      }
    if (!conflict)
      --versions;
  }
  return versions;
}

inline Buffer RewriteLegacyPipelineBuffer(const Buffer &buffer,
                                          int num_versions) {
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));

  // L1 buffers are physically zN-fractal, so one version of the buffer
  // occupies ceil(N/C0) * roundUp16(M) * C0 elements -- possibly MORE than
  // the logical M*N (e.g. zN(128, 200) spans 26624 vs 25600). Pad the
  // per-version shape up to that extent before prepending the version axis,
  // so the implicit version stride (product of the trailing dims) and the
  // flattened Allocate size both use the physical extent. Without this,
  // adjacent versions overlap: the previous version's fractal tail spills
  // into the next version's base and a K-tail Mmad reads it (issue #1341,
  // pipeline + T.Parallel variant). Only applies to constant-shape 2D
  // shared.l1 buffers; everything else keeps the original shape.
  const auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
  if (ptr_type != nullptr && ptr_type->storage_scope == "shared.l1" &&
      buffer->shape.size() == 2 && buffer->dtype.bits() > 0 &&
      buffer->dtype.lanes() == 1) {
    const auto *rows = buffer->shape[0].as<IntImmNode>();
    const auto *cols = buffer->shape[1].as<IntImmNode>();
    if (rows != nullptr && cols != nullptr) {
      const int64_t ele_per_c0 = 32 * 8 / buffer->dtype.bits();
      auto round_up = [](int64_t v, int64_t m) { return (v + m - 1) / m * m; };
      const int64_t padded_rows = round_up(rows->value, 16);
      const int64_t padded_cols = round_up(cols->value, ele_per_c0);
      const int64_t zn_extent = padded_rows * padded_cols;
      const int64_t logical = static_cast<int64_t>(rows->value) * cols->value;
      if (zn_extent > logical) {
        new_buffer->shape =
            Array<PrimExpr>{Integer(static_cast<int64_t>(padded_rows)),
                            Integer(static_cast<int64_t>(padded_cols))};
      }
    }
  }

  new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
  if (new_buffer->strides.size()) {
    ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
    PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
    new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
  }
  return Buffer(new_buffer);
}

} // namespace tl
} // namespace tvm
