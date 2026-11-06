// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

#include "common/ascend_constraint.h"
#include "common/ascend_lifecycle.h"
#include "common/ascend_resource.h"
#include "common/ascend_sync_a3.h"
#include "common/ascend_sync_handoff.h"

namespace tvm {
namespace tl {
bool HasAscendSyncHandoff(const tir::PrimFunc &function) {
  return function->GetAttr<tir::PrimFunc>(kAscendHandoffSource).defined();
}
namespace {
using namespace tir;
using Record = Map<String, ObjectRef>;
using Key = std::vector<int64_t>;
constexpr size_t kTasks = 4096;
constexpr size_t kControls = 65536;

int64_t Checked(__int128 value) {
  ICHECK(value >= std::numeric_limits<int64_t>::min() &&
         value <= std::numeric_limits<int64_t>::max())
      << "A3 handoff byte/control arithmetic overflow";
  return static_cast<int64_t>(value);
}
int64_t Constant(const PrimExpr &value, const char *what) {
  const auto *integer = value.as<IntImmNode>();
  ICHECK(integer && integer->dtype.is_int() && integer->dtype.lanes() == 1)
      << "A3 handoff requires a static signed " << what;
  return integer->value;
}
Stmt Sequence(Array<Stmt> statements) {
  if (statements.empty())
    return Evaluate(0);
  return statements.size() == 1 ? statements[0] : SeqStmt(statements);
}
bool Enabled(const Target &target, const String &platform) {
  auto context = tvm::transform::PassContext::Current();
  String strategy =
      context->GetConfig<String>(kAscendSyncStrategy, String("legacy")).value();
  ICHECK(strategy == "legacy" || strategy == "a3")
      << "Unknown Ascend sync strategy";
  if (strategy == "legacy" ||
      context->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
              .value() != "legacy")
    return false;
  ICHECK(platform == "A3" &&
         (target->GetAttr<String>("model").value_or("") == "ascendc" ||
          target->GetAttr<String>("model").value_or("") == "auto"))
      << "A3 handoff requires A3 AscendC";
  ICHECK(context->GetConfig<Bool>("tl.ascend_auto_sync", Bool(false))
             .value()
             ->value &&
         context->GetConfig<Bool>("tl.ascend_memory_planning", Bool(false))
             .value()
             ->value)
      << "A3 handoff requires legacy local synchronization and memory planning";
  return true;
}
std::string Callee(const CallNode *call) {
  if (call->op.same_as(builtin::call_extern())) {
    ICHECK(!call->args.empty() && call->args[0].as<StringImmNode>())
        << "Malformed handoff extern";
    return NormalizeFunctionName(Downcast<StringImm>(call->args[0])->value);
  }
  const auto *operation = call->op.as<OpNode>();
  return operation ? operation->name : "opaque";
}
bool Cross(const CallNode *call) {
  return call->op.same_as(ascend_set_cross_flag()) ||
         call->op.same_as(ascend_wait_cross_flag()) ||
         call->op.same_as(ascend_auto_set_cross_flag()) ||
         call->op.same_as(ascend_auto_wait_cross_flag());
}
bool LocalDomain(int actor, const std::string &src, const std::string &dst) {
  // CANN9.1 basic_api/kernel_event.h EventToIndexAic/EventToIndexAiv.
  static const std::set<std::string> cube{
      "MTE2_MTE1", "MTE1_MTE2", "MTE1_M",   "M_MTE1", "MTE3_MTE1",
      "MTE1_MTE3", "MTE2_M",    "M_MTE2",   "M_FIX",  "FIX_M",
      "MTE3_MTE2", "MTE2_MTE3", "S_MTE2",   "MTE2_S", "S_MTE3",
      "MTE3_S",    "MTE2_FIX",  "FIX_MTE2", "FIX_S",  "M_S",
      "FIX_MTE3",  "MTE1_FIX",  "FIX_MTE1", "FIX_FIX"};
  static const std::set<std::string> vector{
      "MTE2_V",    "V_MTE2",    "MTE3_V", "V_MTE3", "V_V",
      "MTE3_MTE2", "MTE2_MTE3", "S_V",    "V_S",    "S_MTE2",
      "MTE2_S",    "S_MTE3",    "MTE3_S"};
  return (actor == 0 ? cube : vector).count(src + "_" + dst);
}
struct Geometry {
  Var data;
  DataType dtype;
  PrimExpr offset, extent, stride, rows, columns;
  bool write;
  std::string pipe, name;
};
std::optional<Geometry> GMCall(const CallNode *call) {
  PostOrderVisit(GetRef<Call>(call), [&](const ObjectRef &object) {
    const auto *load = object.as<BufferLoadNode>();
    ICHECK(!load || load->buffer.scope() != "global")
        << "Unsupported hidden GM scalar read in handoff operation";
    if (const auto *nested = object.as<CallNode>())
      ICHECK(nested == call || nested->op.same_as(builtin::tvm_access_ptr()) ||
             nested->op.same_as(Op::Get("tir.type_annotation")) ||
             nested->op.same_as(builtin::if_then_else()))
          << "Unsupported nested opaque call in handoff operand";
  });
  std::string name = Callee(call);
  bool load = name == "copy_gm_to_ub" || name == "copy_gm_to_l1";
  bool store = name == "copy_ub_to_gm" || name == "copy_l0c_to_gm";
  if (!load && !store) {
    PostOrderVisit(GetRef<Call>(call), [](const ObjectRef &object) {
      const auto *pointer = object.as<CallNode>();
      if (pointer && pointer->op.same_as(builtin::tvm_access_ptr()))
        ICHECK(StorageScopeForAccessPtr(GetRef<Call>(pointer)) != "global")
            << "Unsupported opaque/compound GM handoff helper";
    });
    return std::nullopt;
  }
  ICHECK(call->op.same_as(builtin::call_extern()) && call->args.size() >= 7)
      << "Malformed GM handoff copy";
  const auto *pointer = call->args[store ? 2 : 1].as<CallNode>();
  ICHECK(pointer && pointer->op.same_as(builtin::tvm_access_ptr()) &&
         pointer->args.size() == 5)
      << "Malformed GM handoff pointer";
  const auto *annotation = pointer->args[0].as<CallNode>();
  const auto *data = pointer->args[1].as<VarNode>();
  const auto *mask = pointer->args[4].as<IntImmNode>();
  ICHECK(annotation && annotation->op.same_as(Op::Get("tir.type_annotation")) &&
         annotation->args.empty() && data && mask &&
         (mask->value & (store ? 2 : 1)))
      << "Malformed GM handoff access ABI";
  DataType dtype = pointer->args[0].dtype();
  ICHECK(dtype.is_scalar() &&
         (dtype == DataType::Float(16) || dtype == DataType::Float(32)))
      << "A3 handoff supports f16/f32 GM copies only";
  ICHECK(GetPtrStorageScope(GetRef<Var>(data)) == "global")
      << "Handoff GM scope mismatch";
  return Geometry{GetRef<Var>(data),
                  dtype,
                  pointer->args[2],
                  pointer->args[3],
                  call->args[3],
                  call->args[4],
                  call->args[5],
                  store,
                  name == "copy_l0c_to_gm" ? "FIX"
                  : store                  ? "MTE3"
                                           : "MTE2",
                  name};
}

class SourceMarker : public StmtExprMutator {
public:
  int sites{0};

private:
  Stmt VisitStmt_(const ForNode *loop) final {
    ICHECK(loop->kind == ForKind::kSerial)
        << "Handoff requires static serial loops";
    if (auto stages = loop->annotations.Get("num_stages"))
      ICHECK_EQ(Constant(Downcast<PrimExpr>(stages.value()), "num_stages"), 0)
          << "Positive mixed pipeline is unsupported by A3 handoff";
    int id = next_loop_++;
    loops_.push_back({id, loop->loop_var});
    Stmt result = StmtExprMutator::VisitStmt_(loop);
    loops_.pop_back();
    return result;
  }
  Stmt VisitStmt_(const EvaluateNode *evaluate) final {
    if (is_zero(evaluate->value))
      return GetRef<Stmt>(evaluate);
    const auto *call = evaluate->value.as<CallNode>();
    ICHECK(call) << "Unsupported scalar statement in handoff";
    ICHECK(!Cross(call)) << "Explicit cross-core IDs/protocols are unsupported "
                            "in managed handoff";
    if (!GMCall(call))
      return GetRef<Stmt>(evaluate);
    Stmt result = GetRef<Stmt>(evaluate);
    for (auto it = loops_.rbegin(); it != loops_.rend(); ++it)
      result = AttrStmt(IntImm(DataType::Int(64), it->first),
                        kAscendHandoffIteration,
                        cast(DataType::Int(64), it->second), result);
    return AttrStmt(IntImm(DataType::Int(64), sites++), kAscendHandoffTaskId,
                    Integer(1), result);
  }
  int next_loop_{0};
  std::vector<std::pair<int, Var>> loops_;
};

struct Access {
  Key key, round;
  int actor, site;
  Var data;
  DataType dtype;
  int64_t begin, end;
  bool write;
  std::string pipe;
};
struct Event {
  enum Kind { kTask, kSet, kWait, kLocalSet, kLocalWait, kBarrier } kind;
  int task{-1}, id{-1};
  std::string src, dst;
};
struct Trace {
  std::vector<Access> accesses;
  std::array<std::vector<Event>, 3> actors;
  std::map<Key, int> by_key;
  std::map<int, std::vector<Event>> site_events;
};
// Source traverses lexical C/V order. The final trace additionally records the
// three independent instruction streams; no traversal order supplies cross HB.
class TraceBuilder {
public:
  TraceBuilder(PrimFunc function, int64_t group, bool final)
      : function_(function), group_(group), final_(final) {
    for (const auto &entry : function_->buffer_map) {
      Buffer buffer = entry.second;
      if (buffer.scope() != "global")
        continue;
      int64_t bytes = buffer->dtype.bytes();
      ICHECK(buffer->strides.empty() && is_zero(buffer->elem_offset))
          << "Handoff root GM buffer must be dense with zero base offset";
      for (const PrimExpr &extent : buffer->shape) {
        int64_t count = Constant(extent, "GM shape");
        ICHECK_GT(count, 0) << "Invalid GM handoff capacity";
        bytes = Checked(__int128(bytes) * count);
      }
      capacities_[buffer->data.get()] = bytes;
    }
    Walk(function_->body);
  }
  Trace trace;

private:
  int64_t Value(const PrimExpr &value) {
    return EvaluateAscendLifetimeInteger(value, bindings_);
  }
  void Emit(Event event) {
    ICHECK(actor_ >= 0 && actor_ < 3) << "Handoff requires explicit C/V scope";
    trace.actors[actor_].push_back(event);
    if (site_ >= 0)
      trace.site_events[site_].push_back(event);
  }
  bool Synchronization(const CallNode *call) {
    if (Cross(call)) {
      ICHECK(final_) << "Explicit cross-core IDs/protocols are unsupported in "
                        "managed handoff";
      bool set = call->op.same_as(ascend_auto_set_cross_flag());
      ICHECK(set || call->op.same_as(ascend_auto_wait_cross_flag()))
          << "Unmanaged cross-core synchronization in handoff";
      ICHECK_EQ(call->args.size(), set ? 3U : 2U)
          << "Malformed cross-core event ABI";
      int64_t id = Value(call->args[set ? 2 : 0]);
      ICHECK(id >= 0 && id < kAscendA3HandoffUserEvents)
          << "Cross-core ID outside A3 user pool 0..10";
      std::string pipe = set ? Downcast<StringImm>(call->args[1])->value : "";
      if (set)
        ICHECK_EQ(Value(call->args[0]), 2) << "A3 handoff requires mode2";
      Emit({set ? Event::kSet : Event::kWait, -1, int(id),
            NormalizePipeName(pipe), ""});
      return true;
    }
    bool set = call->op.same_as(ascend_auto_set_flag()) ||
               call->op.same_as(ascend_set_flag());
    bool wait = call->op.same_as(ascend_auto_wait_flag()) ||
                call->op.same_as(ascend_wait_flag());
    bool barrier = call->op.same_as(ascend_auto_barrier()) ||
                   call->op.same_as(ascend_pipe_barrier());
    if (!set && !wait && !barrier)
      return false;
    if (barrier) {
      Emit({Event::kBarrier, -1, -1,
            NormalizePipeName(Downcast<StringImm>(call->args[0])->value), ""});
    } else {
      bool automatic = call->op.same_as(ascend_auto_set_flag()) ||
                       call->op.same_as(ascend_auto_wait_flag());
      std::string src = Downcast<StringImm>(call->args[0])->value, dst;
      if (automatic) {
        auto separator = src.find('_');
        ICHECK(separator != std::string::npos)
            << "Malformed local event in handoff";
        dst = src.substr(separator + 1);
        src = src.substr(0, separator);
      } else {
        dst = Downcast<StringImm>(call->args[1])->value;
      }
      int64_t id = Value(call->args[automatic ? 1 : 2]);
      ICHECK(id >= 0 && id < 8) << "Invalid local event ID in handoff";
      ICHECK(
          LocalDomain(actor_, NormalizePipeName(src), NormalizePipeName(dst)))
          << "Local HardEvent is not supported on this handoff core kind";
      Emit({set ? Event::kLocalSet : Event::kLocalWait, -1, int(id),
            NormalizePipeName(src), NormalizePipeName(dst)});
    }
    return true;
  }
  void CallStatement(const Call &call) {
    if (Synchronization(call.get()))
      return;
    ICHECK_GE(actor_, 0) << "Handoff operations require explicit C/V scopes";
    std::string operation;
    AscendResource owner = ResourceForCall(call.get(), &operation);
    ICHECK(owner != AscendResource::kCommon &&
           (owner == AscendResource::kExplicit ||
            owner == (actor_ == 0 ? AscendResource::kCube
                                  : AscendResource::kVector)))
        << "Unsupported/conflicting handoff resource owner";
    auto geometry = GMCall(call.get());
    if (!geometry) {
      std::string name = Callee(call.get());
      const std::set<std::string> audited_local{
          "copy_l1_to_l0a",    "copy_l1_to_l0b", "copy_ub_to_ub", "gemm_v0",
          "tl.ascend_gemm_v0", "tl.ascend_add",  "tl.ascend_fill"};
      ICHECK(audited_local.count(name))
          << "Unsupported opaque/implicit-state helper in handoff: " << name;
      return;
    }
    const Geometry &copy = *geometry;
    ICHECK(site_ >= 0) << "Unmarked or injected GM call in handoff";
    ICHECK((copy.name == "copy_gm_to_l1" || copy.name == "copy_l0c_to_gm") ==
           (actor_ == 0))
        << "GM handoff call is on the wrong core kind";
    ValidateAscendLifetimeCallIndices(call, bindings_);
    int64_t rows = Value(copy.rows), columns = Value(copy.columns),
            stride = Value(copy.stride);
    ICHECK(rows > 0 && rows <= 65535 && columns > 0 && stride == columns)
        << "Handoff supports full dense static GM rectangles only";
    // DAV_C220 DataCopyPad has a 12-bit blockCount. Keep the Vector subset
    // conservative even when UB->GM could flatten a particular small slab.
    if (copy.name == "copy_gm_to_ub" || copy.name == "copy_ub_to_gm")
      ICHECK_LE(rows, 4095)
          << "A3 handoff Vector DataCopyPad rows must be <=4095";
    std::string spelling = Downcast<StringImm>(call->args[0])->value;
    spelling.erase(std::remove_if(spelling.begin(), spelling.end(), ::isspace),
                   spelling.end());
    if (spelling.rfind("tl::ascend::", 0) == 0)
      spelling.erase(0, 12);
    std::string dtype = copy.dtype.bits() == 16 ? "half" : "float";
    std::string expected;
    if (copy.name == "copy_gm_to_l1") {
      expected = copy.name + "<" + dtype + "," + std::to_string(rows) + "," +
                 std::to_string(columns) + ">";
      ICHECK(rows % 16 == 0 && (columns * copy.dtype.bytes()) % 32 == 0)
          << "Handoff L1 copy must not require fractal padding";
    } else if (copy.name == "copy_l0c_to_gm") {
      ICHECK(call->args.size() == 10 && Value(call->args[8]) == 0 &&
             Value(call->args[9]) == 0)
          << "Handoff does not support fused relu/unitFlag FIX helpers";
      const auto *source = call->args[1].as<CallNode>();
      ICHECK(source && source->args[0].dtype() == DataType::Float(32))
          << "Handoff FIX source must be f32 L0C";
      expected = copy.name + "<float," + dtype + ",layout::RowMajor," +
                 std::to_string(rows) + "," + std::to_string(columns) + ",0>";
    } else {
      expected = copy.name + "<" + dtype + "," + std::to_string(columns);
      if (rows != 1 || spelling != expected + ">")
        expected += "," + std::to_string(rows);
      expected += ">";
    }
    ICHECK_EQ(spelling, expected)
        << "Handoff DMA template/actual footprint mismatch";
    int64_t elements = Checked(__int128(rows) * columns),
            offset = Value(copy.offset);
    ICHECK(offset >= 0 && elements <= Value(copy.extent))
        << "Handoff pointer extent exceeded";
    int64_t begin = Checked(__int128(offset) * copy.dtype.bytes());
    int64_t end =
        Checked(__int128(begin) + __int128(elements) * copy.dtype.bytes());
    ICHECK(begin % 32 == 0 && (end - begin) % 32 == 0 &&
           capacities_.count(copy.data.get()) &&
           end <= capacities_.at(copy.data.get()))
        << "Unaligned or out-of-bounds handoff GM footprint";
    Access access{{site_, actor_}, {},    actor_, site_,      copy.data,
                  copy.dtype,      begin, end,    copy.write, copy.pipe};
    for (const auto &entry : iterations_) {
      access.key.push_back(entry.first);
      access.key.push_back(entry.second);
      access.round.push_back(entry.first);
      access.round.push_back(entry.second);
    }
    int id = trace.accesses.size();
    ICHECK_LT(size_t(id), kTasks)
        << "Handoff task instance budget exceeded (4096)";
    ICHECK(trace.by_key.emplace(access.key, id).second)
        << "Duplicate handoff task activation";
    trace.accesses.push_back(access);
    Emit({Event::kTask, id});
  }
  void Walk(const Stmt &statement) {
    ICHECK_LT(visits_++, kControls)
        << "Handoff control visit budget exceeded (65536)";
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        Walk(child);
    } else if (const auto *attribute = statement.as<AttrStmtNode>()) {
      if (attribute->attr_key == "thread_extent") {
        IterVar axis = Downcast<IterVar>(attribute->node);
        int64_t extent = Value(attribute->value);
        auto saved = bindings_;
        if (axis->thread_tag == "blockIdx.x") {
          ICHECK(group_ >= 0 && group_ < extent) << "Invalid handoff group";
          bindings_.Set(axis->var, IntImm(axis->var.dtype(), group_));
        } else {
          ICHECK((axis->thread_tag == "threadIdx.x" ||
                  axis->thread_tag == "blockIdx.y") &&
                 extent == 2)
              << "Handoff requires exactly two AIV participants";
          vector_var_ = axis->var;
          bindings_.Set(axis->var,
                        IntImm(axis->var.dtype(), actor_ > 0 ? actor_ - 1 : 0));
        }
        Walk(attribute->body);
        bindings_ = saved;
      } else if (attribute->attr_key == "resource_scope") {
        int64_t scope = Value(attribute->value);
        ICHECK(scope == 0 || scope == 1) << "Malformed handoff resource_scope";
        if (actor_ >= 0) {
          ICHECK((actor_ == 0) == (scope == 0))
              << "Conflicting nested handoff resource scopes";
          Walk(attribute->body);
          return;
        }
        auto saved = bindings_;
        for (int actor = scope == 0 ? 0 : 1; actor <= (scope == 0 ? 0 : 2);
             ++actor) {
          actor_ = actor;
          if (vector_var_.defined())
            bindings_.Set(vector_var_, IntImm(vector_var_.dtype(),
                                              actor == 0 ? 0 : actor - 1));
          Walk(attribute->body);
        }
        actor_ = -1;
        bindings_ = saved;
      } else if (attribute->attr_key == kAscendHandoffTaskId) {
        int previous = site_;
        site_ =
            Constant(Downcast<PrimExpr>(attribute->node), "handoff task ID");
        ICHECK_EQ(Value(attribute->value), 1)
            << "Malformed handoff task marker";
        Walk(attribute->body);
        site_ = previous;
      } else if (attribute->attr_key == kAscendHandoffIteration) {
        int64_t id =
            Constant(Downcast<PrimExpr>(attribute->node), "handoff loop ID");
        ICHECK(!iterations_.count(id)) << "Duplicate handoff iteration marker";
        iterations_[id] = Value(attribute->value);
        Walk(attribute->body);
        iterations_.erase(id);
      } else {
        ICHECK(attribute->attr_key != "virtual_thread")
            << "Unsupported handoff virtual thread";
        Walk(attribute->body);
      }
    } else if (const auto *loop = statement.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial)
          << "Handoff requires static serial loops";
      if (auto stages = loop->annotations.Get("num_stages"))
        ICHECK_EQ(Value(Downcast<PrimExpr>(stages.value())), 0)
            << "Positive mixed pipeline unsupported";
      int64_t minimum = Value(loop->min), count = Value(loop->extent);
      ICHECK(count >= 0 && count <= int64_t(kControls))
          << "Handoff loop bound exceeds control budget";
      int64_t end = Checked(__int128(minimum) + count);
      DataType dtype = loop->loop_var.dtype();
      ICHECK(dtype == DataType::Int(32) || dtype == DataType::Int(64))
          << "Handoff loop requires int32/int64 induction";
      int64_t lower = dtype.bits() == 32 ? std::numeric_limits<int32_t>::min()
                                         : std::numeric_limits<int64_t>::min();
      int64_t upper = dtype.bits() == 32 ? std::numeric_limits<int32_t>::max()
                                         : std::numeric_limits<int64_t>::max();
      ICHECK(minimum >= lower && end <= upper)
          << "Handoff loop endpoint/update exceeds its actual integer dtype";
      auto saved = bindings_;
      for (int64_t i = 0; i < count; ++i) {
        bindings_.Set(loop->loop_var, IntImm(loop->loop_var.dtype(),
                                             Checked(__int128(minimum) + i)));
        Walk(loop->body);
      }
      bindings_ = saved;
    } else if (const auto *condition = statement.as<IfThenElseNode>()) {
      PostOrderVisit(condition->condition, [&](const ObjectRef &node) {
        ICHECK(!vector_var_.defined() || !node.same_as(vector_var_))
            << "Nonuniform AIV guard is unsupported by mode2 handoff";
      });
      if (Value(condition->condition))
        Walk(condition->then_case);
      else if (condition->else_case.defined())
        Walk(condition->else_case.value());
    } else if (const auto *binding = statement.as<LetStmtNode>()) {
      ICHECK(binding->var.dtype().is_int() || binding->var.dtype().is_uint())
          << "Unsupported pointer/data-dependent handoff scalar binding";
      auto saved = bindings_;
      bindings_.Set(binding->var,
                    IntImm(binding->var.dtype(),
                           Value(Cast(binding->var.dtype(), binding->value))));
      Walk(binding->body);
      bindings_ = saved;
    } else if (const auto *realize = statement.as<BlockRealizeNode>()) {
      ICHECK(realize->iter_values.empty()) << "Unsupported handoff block axes";
      if (!Value(realize->predicate))
        return;
      ICHECK(realize->block->match_buffers.empty() &&
             !realize->block->init.defined())
          << "Unsupported handoff block alias/init";
      Walk(realize->block->body);
    } else if (const auto *allocation = statement.as<AllocateNode>()) {
      ICHECK_EQ(Value(allocation->condition), 1)
          << "Conditional handoff allocation unsupported";
      Walk(allocation->body);
    } else if (const auto *declaration = statement.as<DeclBufferNode>()) {
      Walk(declaration->body);
    } else if (const auto *evaluate = statement.as<EvaluateNode>()) {
      if (is_zero(evaluate->value))
        return;
      ICHECK(evaluate->value.as<CallNode>())
          << "Unsupported scalar side effect in handoff";
      CallStatement(Downcast<Call>(evaluate->value));
    } else {
      LOG(FATAL) << "Unsupported handoff control/effect: "
                 << statement->GetTypeKey();
    }
  }
  PrimFunc function_;
  int64_t group_;
  bool final_;
  int actor_{-1}, site_{-1};
  size_t visits_{0};
  Var vector_var_{ObjectPtr<Object>()};
  Map<Var, PrimExpr> bindings_;
  std::map<int64_t, int64_t> iterations_;
  std::map<const VarNode *, int64_t> capacities_;
};

struct Role {
  bool producer;
  std::string pipe;
};
struct GroupContract {
  Trace source;
  // Each round has one C endpoint and the two participating V endpoints.
  std::vector<std::array<int, 3>> rounds;
};
struct Contract {
  std::vector<GroupContract> groups;
  std::map<int, Role> roles;
  Var storage{ObjectPtr<Object>()};
  bool cube_producer{false};
};
int64_t Groups(const PrimFunc &function) {
  auto ratio = function->GetAttr<StringImm>("npu_cv_ratio");
  ICHECK(ratio.defined() && ratio.value()->value == "cv_1_2")
      << "A3 handoff supports the real cv_1_2 mapping only";
  int64_t result = -1;
  int vector_axes = 0;
  PostOrderVisit(function->body, [&](const ObjectRef &object) {
    const auto *attribute = object.as<AttrStmtNode>();
    if (!attribute || attribute->attr_key != "thread_extent")
      return;
    IterVar axis = Downcast<IterVar>(attribute->node);
    if (axis->thread_tag == "blockIdx.x") {
      ICHECK(result < 0) << "Multiple handoff launch axes";
      result = Constant(attribute->value, "group count");
    } else {
      ICHECK((axis->thread_tag == "blockIdx.y" ||
              axis->thread_tag == "threadIdx.x") &&
             Constant(attribute->value, "AIV extent") == 2)
          << "A3 handoff requires the actual two-AIV launch axis";
      ++vector_axes;
    }
  });
  ICHECK_EQ(vector_axes, 1)
      << "A3 handoff requires exactly one two-AIV launch axis";
  ICHECK(result > 0 && result <= 4096) << "A3 handoff group budget exceeded";
  return result;
}
bool Overlap(const Access &a, const Access &b) {
  return a.data.same_as(b.data) && a.begin < b.end && b.begin < a.end;
}
using Intervals = std::vector<std::pair<int64_t, int64_t>>;
void Merge(Intervals *intervals) {
  std::sort(intervals->begin(), intervals->end());
  Intervals merged;
  for (auto interval : *intervals) {
    if (!merged.empty() && interval.first <= merged.back().second)
      merged.back().second = std::max(merged.back().second, interval.second);
    else
      merged.push_back(interval);
  }
  *intervals = std::move(merged);
}
bool Intersects(const Intervals &left, const Intervals &right) {
  size_t i = 0, j = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i].first < right[j].second && right[j].first < left[i].second)
      return true;
    if (left[i].second <= right[j].second)
      ++i;
    else
      ++j;
  }
  return false;
}
void CheckOtherGMHazards(const std::vector<Trace> &groups, const Var &channel) {
  using Effects = std::map<const VarNode *, std::array<Intervals, 2>>;
  std::vector<std::array<Effects, 3>> footprints(groups.size());
  for (size_t group = 0; group < groups.size(); ++group) {
    for (const Access &access : groups[group].accesses)
      footprints[group][access.actor][access.data.get()][access.write]
          .push_back({access.begin, access.end});
    for (auto &actor : footprints[group])
      for (auto &owner : actor)
        for (auto &intervals : owner.second)
          Merge(&intervals);
  }
  for (size_t a = 0; a < footprints.size() * 3; ++a)
    for (size_t b = a + 1; b < footprints.size() * 3; ++b) {
      size_t group_a = a / 3, group_b = b / 3;
      const auto &left = footprints[group_a][a % 3],
                 &right = footprints[group_b][b % 3];
      for (const auto &owner : left) {
        auto found = right.find(owner.first);
        if (found == right.end())
          continue; // Distinct GM roots use the required tir.noalias contract.
        if (group_a == group_b && channel.defined() &&
            owner.first == channel.get())
          continue; // Exact partitioning and typed mode2 prove this channel.
        const auto &x = owner.second, &y = found->second;
        ICHECK(!Intersects(x[1], y[0]) && !Intersects(x[0], y[1]) &&
               !Intersects(x[1], y[1]))
            << "Uncovered non-channel or cross-group GM RAW/WAR/WAW in A3 "
               "handoff";
      }
    }
}
Contract BuildContract(const PrimFunc &source) {
  Contract result;
  int64_t groups = Groups(source);
  std::map<const VarNode *, std::pair<bool, bool>> writers, readers;
  for (int64_t group = 0; group < groups; ++group) {
    Trace trace = TraceBuilder(source, group, false).trace;
    for (const Access &access : trace.accesses) {
      auto &sides = access.write ? writers[access.data.get()]
                                 : readers[access.data.get()];
      (access.actor == 0 ? sides.first : sides.second) = true;
    }
    result.groups.push_back({trace, {}});
  }
  for (const auto &entry : writers) {
    auto found = readers.find(entry.first);
    if (found == readers.end())
      continue;
    bool c2v = entry.second.first && found->second.second;
    bool v2c = entry.second.second && found->second.first;
    if (!c2v && !v2c)
      continue;
    ICHECK(!result.storage.defined())
        << "A3 handoff supports one cross-core GM channel per kernel";
    ICHECK(c2v != v2c && !(entry.second.first && entry.second.second))
        << "Ambiguous handoff producer/consumer ownership";
    result.storage = GetRef<Var>(entry.first);
    result.cube_producer = c2v;
  }
  size_t total = 0;
  for (const GroupContract &group : result.groups)
    total += group.source.accesses.size();
  if (!total)
    return result; // A legitimately eliminated zero-trip program is a no-op.
  ICHECK(result.storage.defined()) << "No audited C/V GM handoff channel";
  ICHECK(source->HasNonzeroAttr(tir::attr::kNoAlias))
      << "A3 handoff requires tir.noalias for distinct GM parameters";
  if (auto hidden = source->GetAttr<Array<Integer>>("auto_gm_indices")) {
    for (const Integer &index : hidden.value()) {
      ICHECK(index->value >= 0 && size_t(index->value) < source->params.size())
          << "Invalid internal GM workspace parameter metadata";
      Var parameter = source->params[index->value];
      ICHECK(source->buffer_map.count(parameter))
          << "Unknown internal GM workspace parameter";
      Var data = source->buffer_map[parameter]->data;
      for (const GroupContract &group : result.groups)
        for (const Access &access : group.source.accesses)
          if (access.data.same_as(data) && !access.write)
            ICHECK(data.same_as(result.storage))
                << "Internal GM workspace has no audited complete handoff "
                   "producer; caller entry is not initialization";
    }
  }
  std::vector<Access> group_spans;
  for (GroupContract &group : result.groups) {
    std::array<std::vector<int>, 3> endpoints;
    for (size_t i = 0; i < group.source.accesses.size(); ++i) {
      const Access &access = group.source.accesses[i];
      if (!access.data.same_as(result.storage))
        continue;
      bool producer = (access.actor == 0) == result.cube_producer;
      ICHECK(access.write == producer)
          << "Handoff storage has an additional same-side effect";
      endpoints[access.actor].push_back(i);
      auto previous = result.roles.find(access.site);
      ICHECK(previous == result.roles.end() ||
             (previous->second.producer == producer &&
              previous->second.pipe == access.pipe))
          << "Handoff site has conflicting roles";
      result.roles[access.site] = {producer, access.pipe};
    }
    ICHECK(endpoints[0].size() == endpoints[1].size() &&
           endpoints[0].size() == endpoints[2].size())
        << "Handoff rounds/guard participation differ between C and both V "
           "cores";
    for (size_t round = 0; round < endpoints[0].size(); ++round) {
      std::array<int, 3> ids{endpoints[0][round], endpoints[1][round],
                             endpoints[2][round]};
      const Access &cube = group.source.accesses[ids[0]];
      const Access &v0 = group.source.accesses[ids[1]],
                   &v1 = group.source.accesses[ids[2]];
      auto epoch = [](const Key &round) {
        Key coordinates;
        for (size_t i = 1; i < round.size(); i += 2)
          coordinates.push_back(round[i]);
        return coordinates;
      };
      ICHECK(epoch(cube.round) == epoch(v0.round) &&
             epoch(cube.round) == epoch(v1.round))
          << "Handoff uniform guard/epoch coordinates differ between "
             "participants";
      ICHECK(cube.dtype == v0.dtype && cube.dtype == v1.dtype &&
             v0.begin == cube.begin && v0.end == v1.begin && v1.end == cube.end)
          << "Handoff requires the two V regions to cover the complete C "
             "footprint exactly";
      if (round) {
        const auto &first = group.rounds[0];
        for (int actor = 0; actor < 3; ++actor) {
          const Access &original = group.source.accesses[first[actor]],
                       &current = group.source.accesses[ids[actor]];
          ICHECK(original.site == current.site &&
                 original.begin == current.begin && original.end == current.end)
              << "Handoff requires one stable N1 physical interval/site per "
                 "participant";
        }
      }
      group.rounds.push_back(ids);
    }
    if (!group.rounds.empty())
      group_spans.push_back(group.source.accesses[group.rounds[0][0]]);
  }
  for (size_t i = 0; i < group_spans.size(); ++i)
    for (size_t j = i + 1; j < group_spans.size(); ++j)
      ICHECK(!Overlap(group_spans[i], group_spans[j]))
          << "Handoff workspace aliases different core groups; mode2 cannot "
             "order them";
  std::vector<Trace> traces;
  for (const GroupContract &group : result.groups)
    traces.push_back(group.source);
  CheckOtherGMHazards(traces, result.storage);
  return result;
}

Stmt SetCross(int id, const std::string &pipe) {
  return Evaluate(Call(DataType::Handle(), ascend_auto_set_cross_flag(),
                       {Integer(2), StringImm(pipe), Integer(id)}));
}
Stmt WaitCross(int id, const std::string &pipe) {
  return Evaluate(Call(DataType::Handle(), ascend_auto_wait_cross_flag(),
                       {Integer(id), StringImm(pipe)}));
}
class Inserter : public StmtExprMutator {
public:
  explicit Inserter(const std::map<int, Role> &roles) : roles_(roles) {}

private:
  Stmt VisitStmt_(const AttrStmtNode *attribute) final {
    if (attribute->attr_key != kAscendHandoffTaskId)
      return StmtExprMutator::VisitStmt_(attribute);
    int id = Constant(Downcast<PrimExpr>(attribute->node), "handoff task ID");
    auto role = roles_.find(id);
    if (role == roles_.end())
      return StmtExprMutator::VisitStmt_(attribute);
    ICHECK(!active_) << "Nested handoff task markers";
    active_ = role->second;
    found_ = false;
    Stmt body = VisitStmt(attribute->body);
    ICHECK(found_) << "Missing handoff GM operation at marked site";
    active_.reset();
    AttrStmt result = GetRef<AttrStmt>(attribute);
    result.CopyOnWrite()->body = body;
    return result;
  }
  Stmt VisitStmt_(const EvaluateNode *evaluate) final {
    const auto *call = evaluate->value.as<CallNode>();
    if (!active_ || !call || !GMCall(call))
      return GetRef<Stmt>(evaluate);
    ICHECK(!found_) << "Multiple GM operations at one handoff site";
    found_ = true;
    Role role = active_.value();
    Stmt operation = GetRef<Stmt>(evaluate);
    // Conservative N1: the producer drains release before it can issue any
    // following instruction, including a later round's overwrite.
    if (role.producer)
      return Sequence(
          {operation, SetCross(0, role.pipe), WaitCross(1, role.pipe)});
    return Sequence(
        {WaitCross(0, role.pipe), operation, SetCross(1, role.pipe)});
  }
  const std::map<int, Role> &roles_;
  std::optional<Role> active_;
  bool found_{false};
};

using Completion = std::set<int>;
struct ProtocolState {
  std::array<size_t, 3> pc{};
  std::array<Completion, 3> known;
  std::array<std::map<std::string, Completion>, 3> issued, local_known;
  std::array<std::map<std::tuple<std::string, std::string, int>, Completion>, 3>
      local_tokens;
  std::array<std::array<std::deque<Completion>, kAscendA3HandoffUserEvents>, 3>
      tokens;
  std::array<std::array<std::deque<Completion>, kAscendA3HandoffUserEvents>, 2>
      vector_sets;
};
void Include(Completion *into, const Completion &from) {
  into->insert(from.begin(), from.end());
}
Completion Snapshot(const ProtocolState &state, int actor,
                    const std::string &pipe) {
  Completion result = state.known[actor];
  auto issued = state.issued[actor].find(pipe),
       known = state.local_known[actor].find(pipe);
  if (issued != state.issued[actor].end())
    Include(&result, issued->second);
  if (known != state.local_known[actor].end())
    Include(&result, known->second);
  return result;
}
void CheckProtocolShape(const Trace &actual, const Contract &contract) {
  // Reconstruct the protocol from actual instructions, not from a stored plan.
  // Dedicated IDs and an immediate release wait imply outstanding depth <=1.
  for (int actor = 0; actor < 3; ++actor) {
    const auto &events = actual.actors[actor];
    std::set<size_t> owned;
    for (size_t i = 0; i < events.size(); ++i) {
      if (events[i].kind != Event::kTask)
        continue;
      const Access &task = actual.accesses[events[i].task];
      auto role = contract.roles.find(task.site);
      if (role == contract.roles.end())
        continue;
      if (role->second.producer) {
        ICHECK(i + 2 < events.size()) << "Missing handoff ready/release drain";
        const Event &set = events[i + 1], &wait = events[i + 2];
        ICHECK(set.kind == Event::kSet && set.id == 0 && set.src == task.pipe &&
               wait.kind == Event::kWait && wait.id == 1)
            << "Wrong producer ready PIPE/ID or missing immediate release "
               "drain";
        owned.insert(i + 1);
        owned.insert(i + 2);
      } else {
        ICHECK(i > 0 && i + 1 < events.size())
            << "Missing consumer ready/release";
        const Event &wait = events[i - 1], &set = events[i + 1];
        ICHECK(wait.kind == Event::kWait && wait.id == 0 &&
               set.kind == Event::kSet && set.id == 1 && set.src == task.pipe)
            << "Wrong consumer release PIPE/ID or missing ready wait";
        owned.insert(i - 1);
        owned.insert(i + 1);
      }
    }
    for (size_t i = 0; i < events.size(); ++i)
      if (events[i].kind == Event::kSet || events[i].kind == Event::kWait)
        ICHECK(owned.count(i)) << "Unowned/extra cross-core event in handoff";
  }
}

void VerifyGroup(const GroupContract &expected, const Trace &actual,
                 const Contract &contract) {
  ICHECK_EQ(expected.source.accesses.size(), actual.accesses.size())
      << "Handoff source/final GM task coverage changed";
  std::vector<int> correspondence(expected.source.accesses.size());
  for (size_t i = 0; i < expected.source.accesses.size(); ++i) {
    const Access &source = expected.source.accesses[i];
    auto found = actual.by_key.find(source.key);
    ICHECK(found != actual.by_key.end())
        << "Missing stable handoff task/round/participant";
    const Access &final = actual.accesses[found->second];
    ICHECK(source.data.same_as(final.data) && source.dtype == final.dtype &&
           source.begin == final.begin && source.end == final.end &&
           source.write == final.write && source.pipe == final.pipe)
        << "Handoff physical GM owner/footprint/payload differs from source";
    correspondence[i] = found->second;
  }
  for (int actor = 0; actor < 3; ++actor) {
    std::vector<Key> source_order, final_order;
    for (const Event &event : expected.source.actors[actor])
      if (event.kind == Event::kTask)
        source_order.push_back(expected.source.accesses[event.task].key);
    for (const Event &event : actual.actors[actor])
      if (event.kind == Event::kTask)
        final_order.push_back(actual.accesses[event.task].key);
    ICHECK(source_order == final_order)
        << "Handoff changed a participant's source issue order";
  }
  CheckProtocolShape(actual, contract);
  std::vector<Completion> requirements(actual.accesses.size());
  for (size_t round = 0; round < expected.rounds.size(); ++round) {
    const auto &endpoints = expected.rounds[round];
    for (int producer = 0; producer < 3; ++producer) {
      if ((producer == 0) != contract.cube_producer)
        continue;
      int writer = correspondence[endpoints[producer]];
      for (int consumer = 0; consumer < 3; ++consumer) {
        if ((consumer == 0) == contract.cube_producer)
          continue;
        int reader = correspondence[endpoints[consumer]];
        if (Overlap(actual.accesses[writer], actual.accesses[reader]))
          requirements[reader].insert(writer);
        if (round + 1 < expected.rounds.size()) {
          int next_writer =
              correspondence[expected.rounds[round + 1][producer]];
          if (Overlap(actual.accesses[next_writer], actual.accesses[reader]))
            requirements[next_writer].insert(reader);
        }
      }
    }
  }
  ProtocolState state;
  size_t remaining = 0;
  for (const auto &stream : actual.actors)
    remaining += stream.size();
  while (remaining) {
    bool progress = false;
    for (int actor = 0; actor < 3; ++actor) {
      if (state.pc[actor] == actual.actors[actor].size())
        continue;
      const Event &event = actual.actors[actor][state.pc[actor]];
      if (event.kind == Event::kWait && state.tokens[actor][event.id].empty())
        continue;
      if (event.kind == Event::kTask) {
        const Access &access = actual.accesses[event.task];
        Completion known = state.known[actor];
        Include(&known, state.local_known[actor][access.pipe]);
        for (int dependency : requirements[event.task])
          ICHECK(known.count(dependency))
              << "Uncovered joint handoff production/consumption/reuse "
                 "dependency";
        state.issued[actor][access.pipe].insert(event.task);
      } else if (event.kind == Event::kSet) {
        Completion completion = Snapshot(state, actor, event.src);
        if (actor == 0) {
          for (int vector = 1; vector < 3; ++vector)
            state.tokens[vector][event.id].push_back(completion);
        } else {
          state.vector_sets[actor - 1][event.id].push_back(completion);
          auto &first = state.vector_sets[0][event.id],
               &second = state.vector_sets[1][event.id];
          while (!first.empty() && !second.empty()) {
            Completion joined = first.front();
            Include(&joined, second.front());
            state.tokens[0][event.id].push_back(joined);
            first.pop_front();
            second.pop_front();
          }
        }
        for (int recipient = 0; recipient < 3; ++recipient)
          ICHECK_LE(state.tokens[recipient][event.id].size(),
                    size_t(kAscendA3CrossCounterDepth))
              << "A3 cross-core counter depth exceeded";
      } else if (event.kind == Event::kWait) {
        Include(&state.known[actor], state.tokens[actor][event.id].front());
        state.tokens[actor][event.id].pop_front();
      } else if (event.kind == Event::kLocalSet) {
        auto key = std::make_tuple(event.src, event.dst, event.id);
        ICHECK(!state.local_tokens[actor].count(key))
            << "Duplicate active local event in handoff";
        state.local_tokens[actor][key] = Snapshot(state, actor, event.src);
      } else if (event.kind == Event::kLocalWait) {
        auto key = std::make_tuple(event.src, event.dst, event.id);
        auto token = state.local_tokens[actor].find(key);
        ICHECK(token != state.local_tokens[actor].end())
            << "Local handoff wait without set";
        Include(&state.local_known[actor][event.dst], token->second);
        state.local_tokens[actor].erase(token);
      } else if (event.kind == Event::kBarrier) {
        if (event.src == "ALL") {
          for (const auto &issued : state.issued[actor])
            Include(&state.known[actor], issued.second);
          for (const auto &known : state.local_known[actor])
            Include(&state.known[actor], known.second);
        } else {
          Include(&state.local_known[actor][event.src],
                  state.issued[actor][event.src]);
        }
      }
      ++state.pc[actor];
      --remaining;
      progress = true;
    }
    ICHECK(progress)
        << "Joint C/V handoff deadlock: no participant can progress";
  }
  for (int actor = 0; actor < 3; ++actor) {
    ICHECK(state.local_tokens[actor].empty())
        << "Undrained local handoff event at exit";
    for (int id = 0; id < kAscendA3HandoffUserEvents; ++id)
      ICHECK(state.tokens[actor][id].empty())
          << "Undrained cross-core token at exit";
  }
  for (const auto &vector : state.vector_sets)
    for (const auto &pending : vector)
      ICHECK(pending.empty())
          << "Incomplete two-V cross-core aggregation at exit";
}

PrimFunc Prepare(PrimFunc function) {
  ICHECK(!HasAscendSyncHandoff(function))
      << "Repeated or stale handoff preparation";
  if (!function->GetAttr<StringImm>("npu_cv_ratio").defined()) {
    // Expert KernelLaunch (ir.cc) uses blockIdx.y=2 without an explicit ratio.
    // Codegen's non-cv_1_1 branch selects MIX_AIC_1_2 and divides AIV cid by 2.
    // Materialize that proven default, rather than relying on codegen state.
    int expert_axes = 0;
    bool other_vector_axis = false;
    PostOrderVisit(function->body, [&](const ObjectRef &object) {
      const auto *attribute = object.as<AttrStmtNode>();
      if (!attribute || attribute->attr_key != "thread_extent")
        return;
      IterVar axis = Downcast<IterVar>(attribute->node);
      if (axis->thread_tag == "blockIdx.y") {
        ICHECK_EQ(Constant(attribute->value, "Expert AIV extent"), 2)
            << "Unknown Expert handoff core mapping";
        ++expert_axes;
      } else if (axis->thread_tag != "blockIdx.x") {
        other_vector_axis = true;
      }
    });
    ICHECK(expert_axes == 1 && !other_vector_axis)
        << "Missing handoff cv ratio is supported only for the proven Expert "
           "blockIdx.y=2 default";
    function = WithAttr(function, "npu_cv_ratio", StringImm("cv_1_2"));
  }
  SourceMarker marker;
  PrimFunc source = function;
  source.CopyOnWrite()->body = marker(function->body);
  BuildContract(source);
  PrimFunc result = source;
  return WithAttr(result, kAscendHandoffSource, source);
}
PrimFunc Insert(PrimFunc function) {
  auto source = function->GetAttr<PrimFunc>(kAscendHandoffSource);
  ICHECK(source.defined()) << "Missing prepared handoff source";
  ICHECK(!function->GetAttr<Record>(kAscendHandoffPlan).defined())
      << "Repeated handoff generation";
  Contract contract = BuildContract(source.value());
  PrimFunc result = function;
  result.CopyOnWrite()->body = Inserter(contract.roles)(function->body);
  Record plan{{"schema_version", Integer(1)},
              {"mode", Integer(2)},
              {"ready_id", Integer(0)},
              {"release_id", Integer(1)},
              {"round_overlap", Bool(false)},
              {"user_event_count", Integer(11)},
              {"counter_depth", Integer(15)},
              {"channel_count", Integer(contract.storage.defined() ? 1 : 0)}};
  return WithAttr(result, kAscendHandoffPlan, plan);
}
PrimFunc Verify(PrimFunc function) {
  auto source = function->GetAttr<PrimFunc>(kAscendHandoffSource);
  ICHECK(source.defined() &&
         function->GetAttr<Record>(kAscendHandoffPlan).defined())
      << "Missing actual handoff source/generation contract";
  ICHECK(!HasAscendSyncHandoff(source.value()))
      << "Recursive handoff source contract";
  Contract contract = BuildContract(source.value());
  Record metadata = function->GetAttr<Record>(kAscendHandoffPlan).value();
  for (auto field : std::vector<std::pair<const char *, int64_t>>{
           {"schema_version", 1},
           {"mode", 2},
           {"ready_id", 0},
           {"release_id", 1},
           {"user_event_count", 11},
           {"counter_depth", 15},
           {"channel_count", contract.storage.defined() ? 1 : 0}}) {
    ICHECK(metadata.count(field.first) &&
           Constant(Downcast<PrimExpr>(metadata[field.first]), field.first) ==
               field.second)
        << "Stale/inconsistent handoff metadata: " << field.first;
  }
  ICHECK(metadata.count("round_overlap") &&
         is_zero(Downcast<PrimExpr>(metadata["round_overlap"])))
      << "Handoff metadata incorrectly claims round overlap";
  ICHECK_EQ(Groups(function), contract.groups.size())
      << "Handoff launch group mapping changed";
  std::vector<Trace> actual;
  for (size_t group = 0; group < contract.groups.size(); ++group) {
    actual.push_back(TraceBuilder(function, group, true).trace);
    VerifyGroup(contract.groups[group], actual.back(), contract);
  }
  CheckOtherGMHazards(actual, contract.storage);
  return function;
}
} // namespace
using namespace tir::transform;
using namespace tir;
tvm::transform::Pass AscendPrepareSyncHandoff(Target target, String platform) {
  auto pass = [target, platform](PrimFunc f, IRModule,
                                 tvm::transform::PassContext) {
    return Enabled(target, platform) ? Prepare(f) : f;
  };
  return CreatePrimFuncPass(pass, 0, "tl.AscendPrepareSyncHandoff", {});
}
tvm::transform::Pass AscendInsertSyncHandoff(Target target, String platform) {
  auto pass = [target, platform](PrimFunc f, IRModule,
                                 tvm::transform::PassContext) {
    return Enabled(target, platform) ? Insert(f) : f;
  };
  return CreatePrimFuncPass(pass, 0, "tl.AscendInsertSyncHandoff", {});
}
tvm::transform::Pass AscendVerifySyncHandoff(Target target, String platform) {
  auto pass = [target, platform](PrimFunc f, IRModule,
                                 tvm::transform::PassContext) {
    return Enabled(target, platform) ? Verify(f) : f;
  };
  return CreatePrimFuncPass(pass, 0, "tl.AscendVerifySyncHandoff", {});
}
TVM_REGISTER_GLOBAL("tl.transform.AscendPrepareSyncHandoff")
    .set_body_typed(AscendPrepareSyncHandoff);
TVM_REGISTER_GLOBAL("tl.transform.AscendInsertSyncHandoff")
    .set_body_typed(AscendInsertSyncHandoff);
TVM_REGISTER_GLOBAL("tl.transform.AscendVerifySyncHandoff")
    .set_body_typed(AscendVerifySyncHandoff);
} // namespace tl
} // namespace tvm
