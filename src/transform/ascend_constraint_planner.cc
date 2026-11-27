// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/arith/analyzer.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./common/ascend_constraint.h"
#include "./common/ascend_lifecycle.h"
#include "./common/ascend_pipeline_structure.h"
#include "./common/ascend_pipeline_versions.h"
#include "./common/ascend_schedule_hints.h"
#include "./common/ascend_storage_alias.h"

namespace tvm {
namespace tl {

TVM_REGISTER_PASS_CONFIG_OPTION(kAscendPipelinePlanner, String);
TVM_REGISTER_PASS_CONFIG_OPTION(kAscendPipelineCostModel, DictAttrs);

namespace {
using namespace tir;
using Record = Map<String, ObjectRef>;

int64_t IntegerField(const Record &record, const char *key) {
  const auto *value = record.at(key).as<IntImmNode>();
  ICHECK(value && value->dtype.is_int())
      << "Malformed constraint field: " << key;
  return value->value;
}

struct Candidate {
  int loop_id;
  Array<Integer> units;
  Array<ObjectRef> paths;
  Array<Integer> order, stages, starts, versions;
  Array<Buffer> buffers;
  Array<Block> blocks;
  For loop;
  int64_t bound{1}, ii{0};
  bool modulo{false}, offset{false};
  Map<Var, Integer> version_overrides;
};

void ValidateUBDeclaration(DataType dtype, const Array<PrimExpr> &extents) {
  ICHECK(dtype.is_float() && dtype.is_scalar() &&
         (dtype.bits() == 16 || dtype.bits() == 32))
      << "Constraint planner supports float16/float32 UB allocations only";
  const int64_t element_budget = kAscendA2A3UBBudget / dtype.bytes();
  int64_t elements = 1;
  for (const PrimExpr &extent : extents) {
    const auto *value = extent.as<IntImmNode>();
    // Validate each declaration dimension before multiplication. A capacity
    // already folded by TIR arithmetic may have wrapped back to a small value.
    ICHECK(value && !value->dtype.is_bool() && value->value > 0 &&
           value->value <= element_budget / elements)
        << "Constraint UB allocation extent/product exceeds the A3 UB budget "
           "or is not a positive static extent";
    elements *= value->value;
  }
}

void ValidateConstraintUBDeclarations(const PrimFunc &function) {
  PostOrderVisit(function->body, [&](const ObjectRef &node) {
    if (const auto *block = node.as<BlockNode>()) {
      for (const Buffer &buffer : block->alloc_buffers) {
        if (buffer.scope() == "shared.ub")
          ValidateUBDeclaration(buffer->dtype, buffer->shape);
      }
    } else if (const auto *allocation = node.as<AllocateNode>()) {
      const auto *pointer =
          allocation->buffer_var->type_annotation.as<PointerTypeNode>();
      if (pointer && pointer->storage_scope == "shared.ub")
        ValidateUBDeclaration(allocation->dtype, allocation->extents);
    }
  });
}

bool ProveCapacityByCongruence(arith::Analyzer *analyzer, const PrimExpr &start,
                               const PrimExpr &end, const PrimExpr &capacity) {
  PrimExpr span = analyzer->Simplify(end - start);
  const auto *bytes = span.as<IntImmNode>();
  const auto *limit = capacity.as<IntImmNode>();
  if (!bytes || !limit || bytes->value <= 0 || limit->value <= 0)
    return false;
  arith::ModularSet congruence = analyzer->modular_set(start);
  int64_t coefficient = congruence->coeff, base = congruence->base;
  if (coefficient <= 1 || base < 0 || base >= coefficient)
    return false;
  __int128 remaining = __int128(limit->value) - bytes->value - base;
  if (remaining < std::numeric_limits<int64_t>::min() ||
      remaining > std::numeric_limits<int64_t>::max())
    return false;
  PrimExpr divisor = IntImm(DataType::Int(64), coefficient);
  // Since start == coefficient*q + base, this is an exact integer comparison.
  // Factoring the byte stride lets the existing guard constrain q directly.
  PrimExpr quotient = analyzer->Simplify(
      floordiv(start - IntImm(DataType::Int(64), base), divisor));
  PrimExpr upper = floordiv(
      IntImm(DataType::Int(64), static_cast<int64_t>(remaining)), divisor);
  return analyzer->CanProve(quotient <= upper);
}

class ConstraintInputs : public StmtExprVisitor {
public:
  explicit ConstraintInputs(PrimFunc function) {
    size_t units = 0;
    ValidateConstraintUBDeclarations(function);
    PostOrderVisit(function->body, [&](const ObjectRef &node) {
      const auto *evaluate = node.as<EvaluateNode>();
      if (evaluate && evaluate->value.as<CallNode>())
        ICHECK_LE(++units, 4096U)
            << "Constraint hardware source-unit limit exceeded";
    });
    graph = AnalyzeAscendSchedule(function, "A3");
    tasks = Downcast<Array<ObjectRef>>(graph.at("tasks"));
    loops = Downcast<Array<ObjectRef>>(graph.at("loops"));
    storages = Downcast<Array<ObjectRef>>(graph.at("storages"));
    for (size_t i = 0; i < tasks.size(); ++i) {
      Record task = Downcast<Record>(tasks[i]);
      const Stmt &statement = Downcast<Stmt>(task.at("statement"));
      task_ids.emplace(statement.get(), i);
    }
    for (const ObjectRef &launch :
         Downcast<Array<ObjectRef>>(graph.at("launch_domains"))) {
      Record record = Downcast<Record>(launch);
      IterVar variable = Downcast<IterVar>(record.at("iter_var"));
      PrimExpr extent = Downcast<PrimExpr>(record.at("extent"));
      const auto *constant = extent.as<IntImmNode>();
      ICHECK(constant && constant->value > 0)
          << "Constraint planner requires static launch domains";
      analyzer.Bind(variable->var,
                    Range::FromMinExtent(make_zero(variable->var.dtype()),
                                         cast(variable->var.dtype(), extent)));
    }
    for (const auto &entry : function->buffer_map)
      buffers.Set(entry.second->data, entry.second);
    PostOrderVisit(function->body, [&](const ObjectRef &node) {
      if (const auto *block = node.as<BlockNode>())
        for (const Buffer &buffer : block->alloc_buffers)
          buffers.Set(buffer->data, buffer);
    });
    VisitStmt(function->body);
    ICHECK_EQ(next_loop_, loops.size()) << "Constraint loop identity mismatch";
    ICHECK(!candidates.empty()) << "Constraint planner requires at least one "
                                   "positive num_stages candidate";
    std::unordered_set<int> owned;
    for (const Candidate &candidate : candidates)
      for (Integer id : candidate.units)
        owned.insert(id->value);
    for (size_t id = 0; id < tasks.size(); ++id) {
      Record task = Downcast<Record>(tasks[id]);
      if (task.count("manual_stage") &&
          IntegerField(task, "manual_stage") > 0) {
        ICHECK(owned.count(id)) << "Nonzero manual Stage is outside a positive "
                                   "num_stages candidate";
      }
    }
    ValidateFootprints(function);
  }

  Record graph;
  Array<ObjectRef> tasks, loops, storages;
  std::unordered_map<const Object *, int> task_ids;
  std::vector<Candidate> candidates;
  std::unordered_set<int> empty_loops;
  Map<Var, Buffer> buffers;
  int64_t base_bytes{0};
  Record lifetime;

private:
  void VisitStmt_(const ForNode *loop) final {
    int loop_id = next_loop_++;
    ICHECK(loop->kind == ForKind::kSerial)
        << "Constraint planner supports static serial loops only";
    DataType dtype = loop->loop_var.dtype();
    ICHECK(dtype.is_int() && dtype.is_scalar() &&
           (dtype.bits() == 32 || dtype.bits() == 64))
        << "Constraint planner requires signed int32/int64 scalar loop "
           "variables";
    const auto *minimum = loop->min.as<IntImmNode>();
    const auto *extent = loop->extent.as<IntImmNode>();
    int64_t dtype_max = dtype.bits() == 32
                            ? std::numeric_limits<int32_t>::max()
                            : std::numeric_limits<int64_t>::max();
    int64_t dtype_min = dtype.bits() == 32
                            ? std::numeric_limits<int32_t>::min()
                            : std::numeric_limits<int64_t>::min();
    ICHECK(minimum && extent && extent->value >= 0 &&
           extent->value <= dtype_max && minimum->value >= dtype_min &&
           minimum->value <= dtype_max - extent->value)
        << "Constraint planner requires representable static loop domains";
    Record loop_record = Downcast<Record>(loops[loop_id]);
    ICHECK(Downcast<Var>(loop_record.at("var")).same_as(loop->loop_var))
        << "Constraint loop source order mismatch";
    if (bound_.insert(loop->loop_var.get()).second) {
      analyzer.Bind(loop->loop_var,
                    Range::FromMinExtent(loop->min, loop->extent));
    }
    for (const auto &annotation : loop->annotations) {
      std::string key = annotation.first;
      ICHECK(key != "tl_pipeline_order" && key != "tl_pipeline_stage" &&
             key != "tl_pipeline_sync" && key != "tl_pipeline_group" &&
             key != kAscendPipelineLoopId && key != kAscendPipelineVersions &&
             key.rfind("software_pipeline_", 0) != 0)
          << "Constraint planner rejects manual pipeline controls: " << key;
      if (key == "tl_cross_interval") {
        const auto *value = annotation.second.as<IntImmNode>();
        ICHECK(value && value->value == 1)
            << "Constraint planner rejects cross-core intervals";
      }
    }
    auto requested = loop->annotations.Get("num_stages");
    int64_t stage = 0;
    if (requested.defined()) {
      const auto *value = requested.value().as<IntImmNode>();
      ICHECK(value && value->dtype.is_int())
          << "Malformed constraint num_stages";
      stage = value->value;
      ICHECK(stage >= 0 && stage <= 3)
          << "Constraint planner supports num_stages=1/2/3 only";
    }
    std::vector<int> members = Members(loop->body);
    if (members.empty()) {
      empty_loops.insert(loop_id);
    } else if (stage > 0) {
      ICHECK_EQ(candidate_depth_, 0)
          << "Nested constraint candidates are unsupported";
      Stmt body = loop->body;
      Array<Buffer> allocations;
      if (const auto *realize = body.as<BlockRealizeNode>()) {
        ICHECK(is_one(realize->predicate))
            << "Guarded constraint block is unsupported";
        allocations = realize->block->alloc_buffers;
        body = realize->block->body;
      }
      if (const auto *guard = body.as<IfThenElseNode>()) {
        ICHECK(!guard->else_case.defined())
            << "Constraint candidate requires a common If-noelse guard";
        body = guard->then_case;
      }
      const auto *sequence = body.as<SeqStmtNode>();
      ICHECK(sequence)
          << "Constraint candidate body must be an injector SeqStmt";
      Candidate candidate;
      candidate.loop_id = loop_id;
      candidate.loop = GetRef<For>(loop);
      candidate.bound = stage;
      candidate.buffers = allocations;
      candidate.offset = stage > 1;
      if (auto offset = loop->annotations.Get("enable_offset")) {
        const auto *value = offset.value().as<IntImmNode>();
        ICHECK(value && value->dtype.is_bool())
            << "enable_offset must be boolean";
        candidate.offset = value->value;
        candidate.modulo = true;
      }
      candidate.modulo |= stage > 1;
      for (const auto &element :
           CollectAscendPipelineGroups(GetRef<SeqStmt>(sequence))) {
        if (element.allocation.defined()) {
          candidate.buffers.push_back(element.allocation);
          continue;
        }
        auto unit = Members(element.statement);
        ICHECK_EQ(unit.size(), 1U) << "Each constraint injector unit must "
                                      "contain exactly one hardware call";
        bool nested = false;
        PostOrderVisit(element.statement, [&](const ObjectRef &node) {
          nested |= node.as<ForNode>() != nullptr;
        });
        ICHECK(!nested) << "Constraint units cannot contain nested loops";
        candidate.units.push_back(Integer(unit[0]));
        candidate.paths.push_back(element.source_path);
        candidate.blocks.push_back(MakeBlock(element.statement, buffers));
        candidate.modulo |=
            Downcast<Record>(tasks[unit[0]]).count("manual_stage") != 0;
      }
      ICHECK_EQ(candidate.units.size(), members.size())
          << "Constraint unit coverage mismatch";
      PrimExpr guard = Downcast<PrimExpr>(
          Downcast<Record>(tasks[candidate.units[0]->value]).at("guard"));
      for (Integer id : candidate.units)
        ICHECK(StructuralEqual()(
            guard,
            Downcast<PrimExpr>(Downcast<Record>(tasks[id->value]).at("guard"))))
            << "Constraint candidate requires one common logical-iteration "
               "guard";
      candidates.push_back(candidate);
      ++candidate_depth_;
      StmtExprVisitor::VisitStmt_(loop);
      --candidate_depth_;
      return;
    }
    StmtExprVisitor::VisitStmt_(loop);
  }

  std::vector<int> Members(const Stmt &body) const {
    std::vector<int> members;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      auto found = task_ids.find(node.get());
      if (found != task_ids.end())
        members.push_back(found->second);
    });
    return members;
  }

  void ValidateFootprints(const PrimFunc &function) {
    struct Allocation {
      DataType dtype;
      Array<PrimExpr> shape;
    };
    std::unordered_map<const VarNode *, Allocation> declarations;
    PostOrderVisit(function->body, [&](const ObjectRef &node) {
      if (const auto *block = node.as<BlockNode>()) {
        for (const Buffer &buffer : block->alloc_buffers)
          declarations.emplace(buffer->data.get(),
                               Allocation{buffer->dtype, buffer->shape});
      } else if (const auto *allocation = node.as<AllocateNode>()) {
        declarations.emplace(
            allocation->buffer_var.get(),
            Allocation{allocation->dtype, allocation->extents});
      }
    });
    auto initial =
        function->GetAttr<Map<Var, Array<PrimExpr>>>("initial_buffer_shapes")
            .value_or(Map<Var, Array<PrimExpr>>());
    int64_t reserved_bytes = 0;
    for (const ObjectRef &object : storages) {
      Record storage = Downcast<Record>(object);
      if (!Downcast<Bool>(storage.at("physical_allocation"))->value)
        continue;
      ICHECK(Downcast<String>(storage.at("scope")) == "shared.ub")
          << "Constraint planner supports UB allocations only";
      int64_t bytes = IntegerField(storage, "capacity_bytes");
      ICHECK(bytes > 0 && bytes <= kAscendA2A3UBBudget)
          << "Constraint allocation exceeds the A3 UB budget";
      Var data = Downcast<Var>(storage.at("data"));
      ICHECK(declarations.count(data.get()))
          << "Missing constraint allocation declaration";
      const Allocation &allocation = declarations.at(data.get());
      ICHECK(allocation.dtype.is_float() && allocation.dtype.is_scalar() &&
             (allocation.dtype.bits() == 16 || allocation.dtype.bits() == 32))
          << "Constraint planner supports float16/float32 UB allocations only";
      Array<PrimExpr> shape =
          initial.count(data) ? initial.at(data) : allocation.shape;
      PrimExpr inner_expr = shape.empty() ? Integer(1) : shape.back();
      PrimExpr simplified_inner = analyzer.Simplify(inner_expr);
      const auto *inner_value = simplified_inner.as<IntImmNode>();
      ICHECK(inner_value && inner_value->value > 0)
          << "Constraint allocation layout must be static and positive";
      int64_t inner = inner_value->value;
      int64_t element_bytes = allocation.dtype.bytes();
      ICHECK(bytes % element_bytes == 0 &&
             inner <= kAscendA2A3UBBudget / element_bytes)
          << "Constraint allocation layout exceeds the A3 UB budget";
      int64_t elements = bytes / element_bytes;
      int64_t aligned_bytes;
      if (inner == 1) {
        // Flatten2DBuffer swaps this case to one contiguous UB row.
        aligned_bytes = (bytes + 31) / 32 * 32;
      } else {
        int64_t rows = elements / inner + (elements % inner != 0);
        int64_t row_bytes = (inner * element_bytes + 31) / 32 * 32;
        ICHECK(rows <= kAscendA2A3UBBudget / row_bytes)
            << "Constraint allocation layout exceeds the A3 UB budget";
        aligned_bytes = rows * row_bytes;
      }
      if (storage.count("explicit_address")) {
        int64_t address = IntegerField(storage, "explicit_address");
        ICHECK(address >= 0 && address <= kAscendA2A3UBBudget &&
               (address + 31) / 32 * 32 <= kAscendA2A3UBBudget - aligned_bytes)
            << "Constraint explicit address exceeds actual UB capacity";
      }
      ICHECK(reserved_bytes <= kAscendA2A3UBBudget - aligned_bytes)
          << "Constraint allocations exceed the A3 UB budget without reuse";
      reserved_bytes += aligned_bytes;
    }
    base_bytes = reserved_bytes;
    for (const ObjectRef &object : tasks) {
      Record task = Downcast<Record>(object);
      for (const char *kind : {"reads", "writes"}) {
        for (const ObjectRef &access_object :
             Downcast<Array<ObjectRef>>(task.at(kind))) {
          Record access = Downcast<Record>(access_object);
          Record storage =
              Downcast<Record>(storages[IntegerField(access, "storage_id")]);
          PrimExpr start = Downcast<PrimExpr>(access.at("start"));
          PrimExpr end = Downcast<PrimExpr>(access.at("end"));
          PrimExpr capacity = Downcast<PrimExpr>(storage.at("capacity_bytes"));
          With<arith::ConstraintContext> guard_context(
              &analyzer, Downcast<PrimExpr>(task.at("guard")));
          bool bounded =
              analyzer.CanProve(start >= 0 && end >= start && end <= capacity);
          if (!bounded && analyzer.CanProve(start >= 0 && end >= start))
            bounded =
                ProveCapacityByCongruence(&analyzer, start, end, capacity);
          ICHECK(bounded)
              << "Constraint planner cannot prove access bounds in the static "
                 "loop domain";
        }
      }
    }
  }

  arith::Analyzer analyzer;
  std::unordered_set<const VarNode *> bound_;
  size_t next_loop_{0};
  int candidate_depth_{0};
};

int64_t StaticInteger(const PrimExpr &expression, const char *description) {
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expression);
  const auto *value = simplified.as<IntImmNode>();
  ICHECK(value && value->dtype.is_int() && value->dtype.bits() <= 64)
      << "Constraint requires a static integer " << description;
  return value->value;
}

int64_t BufferBytes(const Buffer &buffer) {
  ValidateUBDeclaration(buffer->dtype, buffer->shape);
  ICHECK(buffer->strides.empty() && is_zero(buffer->elem_offset) &&
         (buffer->shape.size() == 1 ||
          (buffer->shape.size() == 2 && is_one(buffer->shape[0]))))
      << "Pipeline versioned UB requires one contiguous 1D/1xL logical row "
         "without "
         "views";
  int64_t bytes = buffer->dtype.bytes();
  for (const PrimExpr &extent : buffer->shape)
    bytes *= StaticInteger(extent, "UB extent");
  ICHECK(bytes % 32 == 0)
      << "Pipeline physical version stride must be 32-byte aligned";
  return bytes;
}

std::pair<int64_t, int64_t> TaskCost(const Record &task,
                                     const Record &profile) {
  const auto *evaluate =
      Downcast<Stmt>(task.at("statement")).as<EvaluateNode>();
  ICHECK(evaluate);
  const auto *call = evaluate->value.as<CallNode>();
  ICHECK(call);
  std::string operation = Downcast<String>(task.at("operation"));
  std::string kind;
  bool dma = false;
  if (operation.find("copy_gm_to_ub") != std::string::npos) {
    kind = "gm_to_ub";
    dma = true;
  } else if (operation.find("copy_ub_to_gm") != std::string::npos) {
    kind = "ub_to_gm";
    dma = true;
  } else if (operation == "tl.ascend_fill")
    kind = "fill";
  else if (operation == "tl.ascend_add")
    kind = "add";
  else
    LOG(FATAL) << "Unsupported pipeline cost operation: " << operation;
  const auto *pointer = call->args[1].as<CallNode>();
  ICHECK(pointer && pointer->args.size() == 5)
      << "Malformed pipeline cost pointer";
  DataType dtype = pointer->args[0].dtype();
  int64_t count = StaticInteger(call->args[dma ? 5 : 3], "operation count");
  ICHECK(count > 0 && count <= kAscendA2A3UBBudget / dtype.bytes())
      << "Invalid pipeline operation count";
  int64_t bytes = count * dtype.bytes();
  int matches = 0;
  int64_t latency = 0, interval = 0;
  for (const ObjectRef &item :
       Downcast<Array<ObjectRef>>(profile.at("records"))) {
    Record record = Downcast<Record>(item);
    if (Downcast<String>(record.at("kind")) != kind ||
        Downcast<String>(record.at("dtype")) !=
            (dtype.bits() == 16 ? "float16" : "float32") ||
        IntegerField(record, "elements") != count ||
        IntegerField(record, "bytes") != bytes ||
        IntegerField(record, "rows") != 1)
      continue;
    ++matches;
    latency = IntegerField(record, "latency");
    interval = IntegerField(record, "ii");
    ICHECK(latency > 0 && interval > 0 && latency <= 1000000000 &&
           interval <= 1000000000)
        << "Invalid pipeline calibrated latency/issue interval";
  }
  ICHECK_EQ(matches, 1)
      << "Pipeline cost profile requires exactly one matching "
         "operation/dtype/count/geometry record";
  return {latency, interval};
}

void VerifyModulo(const Record &problem, const Record &result) {
  const auto *verify = runtime::Registry::Get("tl.ascend.verify_schedule");
  ICHECK(verify) << "Pipeline requires the explicitly registered independent "
                    "schedule verifier";
  Record checked = (*verify)(problem, result);
  const auto *valid = checked.at("valid").as<IntImmNode>();
  ICHECK(valid && valid->dtype.is_bool() && valid->value)
      << "Invalid consumed modulo plan: " << checked.at("errors");
}

void SolveModuloCandidate(Candidate *candidate, const ConstraintInputs &inputs,
                          const Record &profile) {
  ICHECK_EQ(IntegerField(profile, "schema_version"), 1)
      << "Unsupported pipeline cost schema";
  ICHECK(Downcast<String>(profile.at("model")) == "a3-effective-op-v1" &&
         Downcast<String>(profile.at("unit")) == "ns" &&
         IntegerField(profile, "quantum_ns") > 0 &&
         Downcast<String>(
             Downcast<Record>(profile.at("identity")).at("platform")) == "A3")
      << "Pipeline requires an explicit A3 effective-nanosecond cost profile";
  const auto *solve = runtime::Registry::Get("tl.ascend.solve_schedule");
  ICHECK(solve)
      << "Pipeline requires the explicitly registered optional solver";
  const size_t count = candidate->units.size();
  Array<Integer> latencies, intervals, flags, manual;
  std::unordered_map<int, int> local;
  bool has_manual = false;
  int64_t max_manual = 0, upper = 0;
  for (size_t i = 0; i < count; ++i) {
    int id = candidate->units[i]->value;
    local[id] = i;
    Record task = Downcast<Record>(inputs.tasks[id]);
    auto [latency, issue] = TaskCost(task, profile);
    latencies.push_back(Integer(latency));
    intervals.push_back(Integer(issue));
    upper += latency + issue;
    String pipe = Downcast<String>(task.at("pipe"));
    flags.push_back(Integer(pipe == "PIPE_MTE2" ? 1
                            : pipe == "PIPE_V"  ? 2
                                                : 4));
    int64_t stage =
        task.count("manual_stage") ? IntegerField(task, "manual_stage") : 0;
    ICHECK(stage >= 0 && stage <= 64)
        << "Pipeline manual stage exceeds the supported staging horizon (64)";
    max_manual = std::max(max_manual, stage);
    has_manual |= task.count("manual_stage") != 0;
    manual.push_back(Integer(stage));
  }
  int64_t source_trip =
      StaticInteger(candidate->loop->extent, "candidate trip");
  ICHECK(source_trip <= 4096 / int64_t(count) &&
         source_trip + 2 * max_manual <= 65536 / int64_t(count))
      << "Pipeline requested staging window exceeds static expansion limits";
  ICHECK(candidate->offset || max_manual == 0)
      << "Nonzero manual Stage requires enable_offset=True";
  ICHECK(upper > 0 && upper <= std::numeric_limits<int32_t>::max())
      << "Pipeline cost search bound overflow";
  Array<Array<Integer>> dependencies, pipe_order, stage_order;
  std::set<std::pair<int, int>> edges;
  for (const ObjectRef &object :
       Downcast<Array<ObjectRef>>(inputs.graph.at("dependencies"))) {
    Record edge = Downcast<Record>(object);
    int u = IntegerField(edge, "src"), v = IntegerField(edge, "dst");
    if (local.count(u) && local.count(v))
      edges.emplace(local.at(u), local.at(v));
  }
  for (auto [u, v] : edges) {
    dependencies.push_back({Integer(u), Integer(v), Integer(0), latencies[u]});
    stage_order.push_back({Integer(u), Integer(v)});
  }
  for (size_t u = 0; u < count; ++u)
    for (size_t v = u + 1; v < count; ++v)
      if (flags[u]->value == flags[v]->value)
        pipe_order.push_back({Integer(u), Integer(v)});
  Array<Integer> sizes;
  std::unordered_map<const VarNode *, int> buffer_index;
  int64_t own_bytes = 0;
  for (const Buffer &buffer : candidate->buffers) {
    int64_t bytes = BufferBytes(buffer);
    ICHECK(!buffer_index.count(buffer->data.get()))
        << "Duplicate pipeline allocation ownership";
    buffer_index[buffer->data.get()] = sizes.size();
    sizes.push_back(Integer(bytes));
    own_bytes += bytes;
  }
  // Source readiness is proven before solving. Only private, owner-local
  // allocations can rotate. Every other touched storage remains one physical
  // instance and receives explicit cross-iteration hazard edges.
  std::vector<std::set<int>> accesses(sizes.size()), writers(sizes.size());
  std::unordered_map<int, std::vector<std::pair<int, bool>>> fixed_accesses;
  for (size_t i = 0; i < count; ++i) {
    Record task = Downcast<Record>(inputs.tasks[candidate->units[i]->value]);
    for (const char *effect : {"reads", "writes"}) {
      bool writing = std::string(effect) == "writes";
      for (const ObjectRef &item :
           Downcast<Array<ObjectRef>>(task.at(effect))) {
        Record access = Downcast<Record>(item);
        int storage_id = IntegerField(access, "storage_id");
        Record storage = Downcast<Record>(inputs.storages[storage_id]);
        if (Downcast<String>(storage.at("scope")) != "shared.ub")
          continue;
        Var data = Downcast<Var>(storage.at("data"));
        if (buffer_index.count(data.get())) {
          int b = buffer_index.at(data.get());
          accesses[b].insert(i);
          if (writing)
            writers[b].insert(i);
        } else {
          fixed_accesses[storage_id].push_back({int(i), writing});
        }
      }
    }
  }
  for (size_t b = 0; b < sizes.size(); ++b) {
    ICHECK(!writers[b].empty() && !accesses[b].empty())
        << "Rotating storage must be defined in its owner iteration";
    for (int access : accesses[b])
      for (int writer : writers[b])
        dependencies.push_back({Integer(access), Integer(writer),
                                Integer(-int(b) - 1), latencies[access]});
  }
  for (const auto &entry : fixed_accesses)
    for (auto [u, u_write] : entry.second)
      for (auto [v, v_write] : entry.second)
        if (u_write || v_write)
          dependencies.push_back(
              {Integer(u), Integer(v), Integer(1), latencies[u]});
  Array<Integer> memory{
      Integer(kAscendA2A3UBBudget - (inputs.base_bytes - own_bytes))};
  for (size_t b = 0; b < sizes.size(); ++b)
    memory.push_back(Integer(b));
  Record problem{{"schema_version", Integer(1)},
                 {"mode", String("modulo")},
                 {"latencies", latencies},
                 {"iis", intervals},
                 {"resource_flags", flags},
                 {"data_deps", dependencies},
                 {"pipe_order_deps", pipe_order},
                 {"stage_order_deps", stage_order},
                 {"num_stages", Integer(candidate->bound)},
                 {"buffer_sizes", sizes},
                 {"memory_groups", sizes.empty()
                                       ? Array<Array<Integer>>()
                                       : Array<Array<Integer>>{memory}},
                 {"ii_bounds", Array<Integer>{Integer(1), Integer(upper)}},
                 {"enable_offset", Bool(candidate->offset)}};
  if (has_manual || !candidate->offset)
    problem.Set("manual_stages", manual);
  Record result = (*solve)(problem);
  ICHECK_EQ(IntegerField(result, "schema_version"), 1)
      << "Malformed pipeline solver schema";
  ICHECK_EQ(Downcast<String>(result.at("status")), "SAT")
      << "Pipeline solver did not produce SAT: " << result;
  candidate->starts = Downcast<Array<Integer>>(result.at("start_times"));
  Array<Integer> execution = Downcast<Array<Integer>>(result.at("order"));
  candidate->versions = Downcast<Array<Integer>>(result.at("buffer_versions"));
  ICHECK(candidate->starts.size() == count && execution.size() == count &&
         candidate->versions.size() == sizes.size())
      << "Malformed pipeline solver field size";
  candidate->ii = IntegerField(result, "ii");
  ICHECK(candidate->ii > 0) << "Invalid pipeline II";
  std::vector<int> ranks(count, -1);
  for (size_t rank = 0; rank < count; ++rank) {
    ICHECK(execution[rank].dtype().is_int() &&
           candidate->starts[rank].dtype().is_int())
        << "Pipeline starts/order require signed integer values";
    int64_t id = execution[rank]->value;
    ICHECK(id >= 0 && id < int64_t(count) && ranks[id] < 0)
        << "Pipeline order must be a permutation";
    ranks[id] = rank;
  }
  for (size_t i = 0; i < count; ++i) {
    int64_t start = candidate->starts[i]->value;
    ICHECK(start >= 0 && start / candidate->ii <= 64)
        << "Pipeline stage exceeds the supported staging horizon (64)";
    candidate->stages.push_back(Integer(start / candidate->ii));
    candidate->order.push_back(Integer(ranks[i]));
  }
  VerifyModulo(problem, result);
  PipelineInfo info;
  for (size_t i = 0; i < count; ++i)
    info.emplace(
        candidate->blocks[i],
        PipelineAnnotation{int(candidate->stages[i]->value), ranks[i], false});
  int max_stage = 0;
  auto access = GetLegacyPipelineBufferAccessInfo(info, &max_stage);
  for (size_t b = 0; b < candidate->buffers.size(); ++b) {
    const Buffer &buffer = candidate->buffers[b];
    ICHECK(candidate->versions[b].dtype().is_int())
        << "Pipeline version count must be an integer";
    int64_t actual = std::max<int64_t>(
        candidate->versions[b]->value,
        ComputeAscendStoragePipelineVersions(buffer->data, info));
    ICHECK(actual >= 1 && actual <= candidate->bound)
        << "Pipeline actual injector versions exceed requested bound";
    candidate->versions.Set(b, Integer(actual));
    candidate->version_overrides.Set(buffer->data, Integer(actual));
  }
  result.Set("buffer_versions", candidate->versions);
  VerifyModulo(problem, result); // Revalidate the actual, possibly conservative
                                 // injector version counts.
  int64_t trip = StaticInteger(candidate->loop->extent, "candidate trip"),
          minimum = StaticInteger(candidate->loop->min, "candidate min");
  int64_t limit = candidate->loop->loop_var.dtype().bits() == 32
                      ? std::numeric_limits<int32_t>::max()
                      : std::numeric_limits<int64_t>::max();
  int64_t lower = candidate->loop->loop_var.dtype().bits() == 32
                      ? std::numeric_limits<int32_t>::min()
                      : std::numeric_limits<int64_t>::min();
  ICHECK(trip <= 4096 / int64_t(count) && max_stage <= 65535 - trip &&
         trip + 2 * int64_t(max_stage) <= 65536 / int64_t(count) &&
         minimum >= lower + max_stage && minimum <= limit - trip - max_stage)
      << "Expanded pipeline exceeds hardware/control or loop-update limits";
}

void SolveCandidate(Candidate *candidate, const ConstraintInputs &inputs) {
  const runtime::PackedFunc *solve =
      runtime::Registry::Get("tl.ascend.solve_schedule");
  ICHECK(solve)
      << "Constraint planner requires the optional ascend-scheduler solver; "
         "use the Python PipelinePlanning wrapper to register it";
  size_t count = candidate->units.size();
  Array<Integer> costs, flags;
  std::unordered_map<int, int> local;
  for (size_t i = 0; i < count; ++i) {
    local[candidate->units[i]->value] = i;
    costs.push_back(Integer(1));
    Record task = Downcast<Record>(inputs.tasks[candidate->units[i]->value]);
    String pipe = Downcast<String>(task.at("pipe"));
    ICHECK(pipe == "PIPE_MTE2" || pipe == "PIPE_V" || pipe == "PIPE_MTE3")
        << "Unsupported constraint pipe";
    flags.push_back(Integer(pipe == "PIPE_MTE2" ? 1
                            : pipe == "PIPE_V"  ? 2
                                                : 4));
  }
  std::set<std::pair<int, int>> edges;
  for (const ObjectRef &object :
       Downcast<Array<ObjectRef>>(inputs.graph.at("dependencies"))) {
    Record dependency = Downcast<Record>(object);
    int src = IntegerField(dependency, "src"),
        dst = IntegerField(dependency, "dst");
    if (local.count(src) && local.count(dst))
      edges.emplace(local.at(src), local.at(dst));
  }
  Array<Array<Integer>> dependencies;
  for (const auto &edge : edges)
    dependencies.push_back(
        {Integer(edge.first), Integer(edge.second), Integer(0), Integer(1)});
  Record problem{{"schema_version", Integer(1)}, {"mode", String("straight")},
                 {"latencies", costs},           {"iis", costs},
                 {"resource_flags", flags},      {"data_deps", dependencies}};
  Record result = (*solve)(problem);
  ICHECK_EQ(IntegerField(result, "schema_version"), 1)
      << "Malformed constraint solver schema";
  String status = Downcast<String>(result.at("status"));
  ICHECK_EQ(status, "SAT") << "Constraint solver did not produce SAT: "
                           << status << ", "
                           << result.Get("reason").value_or(String(""));
  Array<Integer> execution = Downcast<Array<Integer>>(result.at("order"));
  ICHECK_EQ(execution.size(), count)
      << "Malformed constraint solver order size";
  std::vector<int> ranks(count, -1);
  for (size_t rank = 0; rank < count; ++rank) {
    ICHECK(execution[rank].dtype().is_int())
        << "Constraint solver order must contain integer task IDs, not bool";
    int64_t unit = execution[rank]->value;
    ICHECK(unit >= 0 && unit < static_cast<int64_t>(count) && ranks[unit] < 0)
        << "Constraint solver order must be a permutation";
    ranks[unit] = rank;
  }
  for (const auto &edge : edges)
    ICHECK_LT(ranks[edge.first], ranks[edge.second])
        << "Constraint solver order reverses an original logical dependency";
  // Only order is consumed by the single-stage bridge. Numeric start times
  // remain the numeric callback's independently verified abstract scheduling
  // model.
  for (int rank : ranks) {
    candidate->order.push_back(Integer(rank));
    candidate->stages.push_back(Integer(0));
  }
}

class AnnotationEmitter : public StmtExprMutator {
public:
  explicit AnnotationEmitter(const ConstraintInputs &inputs) : inputs_(inputs) {
    for (const Candidate &candidate : inputs.candidates)
      candidates_.emplace(candidate.loop_id, &candidate);
  }

private:
  Stmt VisitStmt_(const ForNode *loop) final {
    int id = next_loop_++;
    For result = Downcast<For>(StmtExprMutator::VisitStmt_(loop));
    if (inputs_.empty_loops.count(id)) {
      result.CopyOnWrite()->annotations.erase("num_stages");
    } else if (candidates_.count(id)) {
      const Candidate &candidate = *candidates_.at(id);
      if (is_zero(result->extent))
        return Evaluate(0);
      Map<String, ObjectRef> annotations = result->annotations;
      annotations.erase("num_stages");
      annotations.erase("enable_offset");
      annotations.erase(kAscendPersistentGuard);
      annotations.Set(tir::attr::software_pipeline_stage, candidate.stages);
      if (candidate.modulo) {
        annotations.Set(kAscendPipelineLoopId, IntImm(DataType::Int(64), id));
        annotations.Set(kAscendPipelineVersions, candidate.version_overrides);
      }
      annotations.Set(tir::attr::software_pipeline_order, candidate.order);
      result.CopyOnWrite()->annotations = annotations;
    }
    return result;
  }
  const ConstraintInputs &inputs_;
  std::unordered_map<int, const Candidate *> candidates_;
  int next_loop_{0};
};

} // namespace

PrimFunc PlanAscendConstraintPipeline(PrimFunc function,
                                      tvm::transform::PassContext context) {
  auto target = function->GetAttr<Target>(tvm::attr::kTarget);
  auto platform = function->GetAttr<String>("npu_platform");
  ICHECK(target.defined() && platform.defined() && platform.value() == "A3")
      << "Constraint pipeline planner supports explicitly identified A3 only";
  String model = target.value()->GetAttr<String>("model", String("")).value();
  ICHECK(model == "ascendc" || model == "auto")
      << "Constraint pipeline planner requires AscendC/auto";
  ICHECK(
      context->GetConfig<Bool>("tl.ascend_auto_sync", Bool(false))
          .value()
          ->value &&
      context->GetConfig<Bool>("tl.ascend_memory_planning", Bool(false))
          .value()
          ->value &&
      !context->GetConfig<Bool>("tl.ascend_auto_cross_core_sync", Bool(false))
           .value()
           ->value &&
      !context->GetConfig<Bool>("tl.ascend_auto_sync_vs", Bool(false))
           .value()
           ->value)
      << "Constraint planner requires auto_sync=true, memory_planning=true, "
         "auto_cross_core_sync=false and auto_sync_vs=false";
  ICHECK(!function->GetAttr<ObjectRef>(kAscendConstraintPlan).defined())
      << "Constraint pipeline is already planned";
  ValidateConstraintUBDeclarations(function);
  function = CanonicalizeAscendStorageAliases(function);
  ConstraintInputs inputs(function);
  if (auto facts = function->GetAttr<Array<ObjectRef>>(
          kAscendPipelineDegenerateCandidates)) {
    for (const ObjectRef &item : facts.value()) {
      Record fact = Downcast<Record>(item);
      Var variable = Downcast<Var>(fact.at("var"));
      bool retained = false;
      for (const Candidate &candidate : inputs.candidates)
        retained |=
            candidate.loop->loop_var.same_as(variable) &&
            StructuralEqual()(candidate.loop->extent, fact.at("extent"));
      ICHECK(retained)
          << "Constraint degenerate candidate was removed before planning";
    }
  }
  Optional<Record> profile;
  if (auto configured = context->GetConfig<DictAttrs>(kAscendPipelineCostModel))
    profile = configured.value()->dict;
  Record catalog = BuildAscendStorageCatalog(function, inputs.graph);
  PrimFunc source = MarkAscendScheduleSource(function, inputs.graph);
  inputs.lifetime = BuildAscendLifetimeContract(source, inputs.graph, catalog);
  bool modulo = false;
  int64_t reserved = inputs.base_bytes;
  for (Candidate &candidate : inputs.candidates) {
    candidate.modulo |= profile.defined();
    if (candidate.modulo) {
      ICHECK(profile.defined())
          << "Pipeline multi-stage/offset/manual Stage requires an explicit "
             "calibrated cost profile";
      SolveModuloCandidate(&candidate, inputs, profile.value());
      for (size_t b = 0; b < candidate.buffers.size(); ++b) {
        int64_t bytes = BufferBytes(candidate.buffers[b]);
        int64_t extra = candidate.versions[b]->value - 1;
        ICHECK(extra <= (kAscendA2A3UBBudget - reserved) / bytes)
            << "Pipeline actual versions exceed whole-kernel UB budget";
        reserved += extra * bytes;
        Var data = candidate.buffers[b]->data;
        for (const ObjectRef &item : inputs.storages) {
          Record storage = Downcast<Record>(item);
          if (!Downcast<Var>(storage.at("data")).same_as(data) ||
              !storage.count("explicit_address"))
            continue;
          int64_t address = IntegerField(storage, "explicit_address");
          int64_t physical_bytes = (extra + 1) * bytes;
          ICHECK((address + 31) / 32 * 32 <=
                 kAscendA2A3UBBudget - physical_bytes)
              << "Constraint explicit address plus actual versions exceeds UB "
                 "capacity";
        }
      }
      modulo = true;
    } else
      SolveCandidate(&candidate, inputs);
  }
  Array<ObjectRef> candidates;
  for (size_t i = 0; i < inputs.candidates.size(); ++i) {
    const Candidate &candidate = inputs.candidates[i];
    candidates.push_back(Record{
        {"id", Integer(i)},
        {"loop_id", Integer(candidate.loop_id)},
        {"unit_ids", candidate.units},
        {"source_paths", candidate.paths},
        {"order", candidate.order},
        {"stages", candidate.stages},
        {"start_times", candidate.starts},
        {"ii", Integer(candidate.ii)},
        {"buffer_versions", candidate.versions},
        {"execution", String(is_zero(candidate.loop->extent) ? "zero_trip_noop"
                                                             : "injected")}});
  }
  Record manifest{{"schema_version", Integer(3)},
                  {"lifetime_contract", inputs.lifetime},
                  {"source_graph", inputs.graph},
                  {"source_function", source},
                  {"candidates", candidates}};
  if (modulo)
    manifest.Set("cost_model", profile.value());
  PrimFunc planned = source;
  planned.CopyOnWrite()->body = AnnotationEmitter(inputs)(source->body);
  planned = WithAttr(planned, kAscendPipelineBufferRemap, Array<ObjectRef>());
  return WithAttr(planned, kAscendConstraintPlan, manifest);
}

// Keep an early record of degenerate requests. Current lowering preserves these
// For nodes; losing one later is an error rather than a silent scheduling skip.
tir::transform::Pass AscendPrepareConstraintPipeline(Target target) {
  auto pass_func = [=](PrimFunc function, IRModule,
                       tvm::transform::PassContext context) {
    Array<ObjectRef> facts;
    PostOrderVisit(function->body, [&](const ObjectRef &node) {
      const auto *loop = node.as<ForNode>();
      if (!loop)
        return;
      auto requested = loop->annotations.Get("num_stages");
      if (!requested.defined())
        return;
      const auto *stages = requested.value().as<IntImmNode>();
      ICHECK(stages && stages->dtype.is_int() && stages->value >= 0 &&
             stages->value <= 3)
          << "Constraint planner supports num_stages=1/2/3 only";
      const auto *extent = loop->extent.as<IntImmNode>();
      if (stages->value > 0 && extent &&
          (extent->value == 0 || extent->value == 1))
        facts.push_back(Record{{"var", loop->loop_var},
                               {"extent", loop->extent},
                               {"num_stages", requested.value()}});
    });
    Array<Var> parameters;
    for (const Var &parameter : function->params)
      if (function->buffer_map.count(parameter))
        parameters.push_back(function->buffer_map[parameter]->data);
    function = WithAttr(function, kAscendStorageOriginalParameters, parameters);
    if (!facts.empty())
      function = WithAttr(function, kAscendPipelineDegenerateCandidates, facts);
    return function;
  };
  return tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.AscendPrepareConstraintPipeline", {});
}
TVM_REGISTER_GLOBAL("tl.transform.AscendPrepareConstraintPipeline")
    .set_body_typed(AscendPrepareConstraintPipeline);

} // namespace tl
} // namespace tvm
