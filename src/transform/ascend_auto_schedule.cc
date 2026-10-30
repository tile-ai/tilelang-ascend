// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include "../tl_templates/ascend/reduce_2d_v2.h"
#include "common/ascend_auto_schedule.h"
#include "common/ascend_constraint.h"
#include "common/ascend_lifecycle.h"
#include "common/ascend_resource.h"
#include "common/ascend_storage_alias.h"

namespace tvm {
namespace tl {
namespace {
using namespace tir;
using R = AutoRecord;
int64_t I(const PrimExpr &x) { return EvaluateAscendLifetimeInteger(x, {}); }
int64_t Num(const ObjectRef &x) { return I(Downcast<PrimExpr>(x)); }
int64_t Safe(__int128 value) {
  ICHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max())
      << "Auto schedule byte/index overflow";
  return static_cast<int64_t>(value);
}
Integer Int(int64_t x) { return Integer(IntImm(DataType::Int(64), x)); }
std::vector<std::string> Params(std::string text) {
  auto a = text.find('<'), b = text.rfind('>');
  ICHECK(a != std::string::npos && b == text.size() - 1)
      << "Unsupported auto helper template: " << text;
  std::stringstream stream(text.substr(a + 1, b - a - 1));
  std::vector<std::string> result;
  while (std::getline(stream, text, ',')) {
    text.erase(std::remove_if(text.begin(), text.end(), ::isspace), text.end());
    result.push_back(text);
  }
  return result;
}
int64_t Decimal(const std::string &text) {
  ICHECK(!text.empty() && std::all_of(text.begin(), text.end(), ::isdigit))
      << "Auto helper requires a positive literal template integer";
  size_t used = 0;
  int64_t value = std::stoll(text, &used);
  ICHECK(used == text.size() && value > 0);
  return value;
}
R Ptr(PrimExpr expr, int64_t count, int access) {
  const auto *p = expr.as<CallNode>();
  ICHECK(p && p->op.same_as(builtin::tvm_access_ptr()) && p->args.size() == 5)
      << "Auto operation requires a declared access_ptr";
  const auto *type = p->args[0].as<CallNode>();
  const auto *var = p->args[1].as<VarNode>();
  ICHECK(type && type->op.same_as(Op::Get("tir.type_annotation")) &&
         type->args.empty() && var)
      << "Malformed auto access_ptr type/data";
  DataType dtype = p->args[0].dtype();
  ICHECK(dtype.is_scalar() &&
         (dtype == DataType::Float(16) || dtype == DataType::Float(32) ||
          dtype == DataType::UInt(8)))
      << "Unsupported auto pointer dtype";
  ICHECK(p->args[2].dtype().is_int() && p->args[3].dtype().is_int() &&
         p->args[2].dtype().is_scalar() && p->args[3].dtype().is_scalar())
      << "Auto pointer offset/extent must be signed scalar integers";
  int64_t offset = I(p->args[2]), extent = I(p->args[3]);
  const auto *mask = p->args[4].as<IntImmNode>();
  ICHECK(mask && mask->dtype.is_int() && mask->value >= 1 && mask->value <= 3 &&
         (mask->value & access) == access && offset >= 0 && count >= 0 &&
         count <= extent)
      << "Unsupported auto pointer interval/access mask";
  return R{{"data", GetRef<Var>(var)},
           {"scope", String(GetPtrStorageScope(GetRef<Var>(var)))},
           {"dtype", String(runtime::DLDataType2String(dtype))},
           {"begin", Int(Safe(__int128(offset) * dtype.bytes()))},
           {"end", Int(Safe((__int128(offset) + count) * dtype.bytes()))}};
}
void Context(Target target, String platform) {
  auto ctx = tvm::transform::PassContext::Current();
  ICHECK(ctx->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
             .value() == "auto")
      << "Auto backend requires the explicit auto planner";
  ICHECK(platform == "A3" &&
         (target->GetAttr<String>("model").value_or("") == "ascendc" ||
          target->GetAttr<String>("model").value_or("") == "auto"));
  auto flag = [&](const char *key, bool fallback) {
    return bool(ctx->GetConfig<Bool>(key, Bool(fallback)).value()->value);
  };
  ICHECK(ctx->GetConfig<String>("tl.ascend_sync_strategy", String("legacy"))
                 .value() == "a3" &&
         flag("tl.ascend_auto_sync", false) &&
         flag("tl.ascend_memory_planning", false) &&
         flag("tl.ascend_auto_cv_combine", true) &&
         !flag("tl.ascend_auto_sync_vs", false) &&
         !flag("tl.ascend_auto_cross_core_sync", false) &&
         !ctx->GetConfig<DictAttrs>(kAscendPipelineCostModel).defined())
      << "Auto requires a3 sync and memory/CV ownership; legacy cross/VS sync "
         "and constraint cost models are not supported";
}

class Source {
public:
  explicit Source(PrimFunc input)
      : function(CanonicalizeAscendStorageAliases(input)) {
    ICHECK(!function->GetAttr<ObjectRef>(kAscendAutoSchedule).defined())
        << "Auto source already carries an owned schedule";
    for (const Var &param : function->params)
      if (function->buffer_map.count(param))
        Register(function->buffer_map[param], -1, {});
    std::vector<int64_t> path;
    Catalog(function->body, path);
    if (vector_extent == 2) {
      auto ratio = function->GetAttr<StringImm>("npu_cv_ratio");
      ICHECK((ratio.defined() && ratio.value()->value == "cv_1_2") ||
             (!ratio.defined() && expert_vector_axis))
          << "Auto two-AIV launch requires the proven cv_1_2 mapping";
      if (!ratio.defined())
        function = WithAttr(function, "npu_cv_ratio", StringImm("cv_1_2"));
    }
    Array<Var> original;
    if (auto declared =
            function->GetAttr<Array<Var>>(kAscendStorageOriginalParameters))
      original = declared.value();
    else
      for (const Var &param : function->params)
        if (function->buffer_map.count(param))
          original.push_back(function->buffer_map[param]->data);
    std::set<const VarNode *> hidden;
    if (auto indices = function->GetAttr<Array<Integer>>("auto_gm_indices"))
      for (Integer index : indices.value()) {
        ICHECK(index->value >= 0 &&
               index->value < int64_t(function->params.size()));
        Var param = function->params[index->value];
        ICHECK(function->buffer_map.count(param));
        hidden.insert(function->buffer_map[param]->data.get());
      }
    for (R &storage : storages) {
      Var data = Downcast<Var>(storage.at("data"));
      bool external =
          Num(storage.at("allocation_site_id")) < 0 &&
          !hidden.count(data.get()) &&
          std::any_of(original.begin(), original.end(),
                      [&](const Var &v) { return v.same_as(data); });
      storage.Set("origin",
                  String(external ? "external_parameter" : "allocation"));
    }
    for (int64_t group = 0; group < groups; ++group) {
      for (int actor = 0; actor < (vector_extent == 2 ? 3 : 2); ++actor) {
        group_ = group;
        actor_ = actor;
        bindings = {};
        if (cid.defined())
          bindings.Set(cid, IntImm(cid.dtype(), group));
        if (vid.defined())
          bindings.Set(vid, IntImm(vid.dtype(), actor == 2 ? 1 : 0));
        Execute(function->body);
      }
    }
  }
  R Build() {
    Array<ObjectRef> storage_rows, loop_rows, task_rows;
    for (const R &r : storages)
      storage_rows.push_back(r);
    for (const R &r : loops)
      loop_rows.push_back(r);
    for (const R &r : tasks)
      task_rows.push_back(r);
    return R{{"schema_version", Int(1)},
             {"source_function", function},
             {"tasks", task_rows},
             {"storages", storage_rows},
             {"loops", loop_rows},
             {"groups", Int(groups)},
             {"vector_extent", Int(vector_extent)}};
  }

private:
  void Register(Buffer buffer, int64_t site,
                const std::vector<int64_t> &parents) {
    Var data = buffer->data;
    int64_t bytes = buffer->dtype.bytes();
    for (PrimExpr extent : buffer->shape) {
      int64_t size = I(extent);
      ICHECK(size > 0 && size <= std::numeric_limits<int64_t>::max() / bytes)
          << "Auto allocation extent/product overflow";
      bytes *= size;
    }
    int64_t offset = I(buffer->elem_offset);
    bytes = Safe(__int128(bytes) + __int128(offset) * buffer->dtype.bytes());
    if (storage_ids.count(data.get())) {
      R &old = storages[storage_ids.at(data.get())];
      ICHECK(site < 0 || Num(old.at("allocation_site_id")) < 0)
          << "Multiple allocation sites for one auto storage";
      if (site < 0)
        return;
      old.Set("buffer", buffer);
      old.Set("capacity_bytes", Int(bytes));
      old.Set("allocation_site_id", Int(site));
      return;
    }
    Array<Integer> parent;
    for (int64_t id : parents)
      parent.push_back(Int(id));
    storage_ids[data.get()] = storages.size();
    storages.push_back(
        R{{"id", Int(storages.size())},
          {"data", data},
          {"buffer", buffer},
          {"shape", buffer->shape},
          {"scope", String(buffer.scope())},
          {"dtype", String(runtime::DLDataType2String(buffer->dtype))},
          {"capacity_bytes", Int(bytes)},
          {"allocation_site_id", Int(site)},
          {"parent_loop_ids", parent}});
  }
  void Catalog(Stmt stmt, std::vector<int64_t> &parents) {
    if (const auto *n = stmt.as<SeqStmtNode>()) {
      for (Stmt s : n->seq)
        Catalog(s, parents);
    } else if (const auto *n = stmt.as<ForNode>()) {
      ICHECK(!loop_ids.count(n)) << "Auto requires distinct lexical loop nodes";
      int64_t id = loops.size();
      loop_ids[n] = id;
      int64_t minimum = I(n->min), extent = I(n->extent);
      ICHECK(extent >= 0 &&
             extent <= kAscendAutoScheduleControlExpansionLimit &&
             n->loop_var.dtype().is_int() &&
             (n->loop_var.dtype().bits() == 32 ||
              n->loop_var.dtype().bits() == 64))
          << "Auto requires bounded signed static loops";
      __int128 endpoint = __int128(minimum) + extent;
      int64_t limit = n->loop_var.dtype().bits() == 32 ? INT32_MAX : INT64_MAX;
      ICHECK(endpoint <= limit &&
             minimum >=
                 (n->loop_var.dtype().bits() == 32 ? INT32_MIN : INT64_MIN))
          << "Auto induction endpoint overflow";
      int64_t requested = 2;
      for (const auto &a : n->annotations) {
        std::string key = a.first;
        if (key == "num_stages") {
          requested = Num(a.second);
          ICHECK(requested >= 0 && requested <= 3);
        } else
          ICHECK(key.rfind("software_pipeline_", 0) != 0 &&
                 key.rfind("tl_pipeline_", 0) != 0 && key != "enable_offset")
              << "Unsupported auto manual loop control";
      }
      loops.push_back(R{{"id", Int(id)},
                        {"var", n->loop_var},
                        {"min", Int(minimum)},
                        {"extent", Int(extent)},
                        {"num_stages", Int(requested)}});
      parents.push_back(id);
      Catalog(n->body, parents);
      parents.pop_back();
    } else if (const auto *n = stmt.as<BlockRealizeNode>())
      Catalog(n->block, parents);
    else if (const auto *n = stmt.as<BlockNode>()) {
      for (Buffer b : n->alloc_buffers)
        Register(b, next_site++, parents);
      ICHECK(!n->init.defined()) << "Auto block init must be lowered";
      Catalog(n->body, parents);
    } else if (const auto *n = stmt.as<AllocateNode>()) {
      Buffer b = decl_buffer(n->extents, n->dtype, n->buffer_var->name_hint,
                             GetPtrStorageScope(n->buffer_var));
      b.CopyOnWrite()->data = n->buffer_var;
      Register(b, next_site++, parents);
      Catalog(n->body, parents);
    } else if (const auto *n = stmt.as<AttrStmtNode>()) {
      if (n->attr_key == "thread_extent") {
        IterVar axis = Downcast<IterVar>(n->node);
        int64_t extent = I(n->value);
        if (axis->thread_tag == "blockIdx.x") {
          cid = axis->var;
          groups = extent;
        } else if (axis->thread_tag == "blockIdx.y" ||
                   axis->thread_tag == "threadIdx.x") {
          ICHECK(!vid.defined() || vid.same_as(axis->var))
              << "Multiple auto Vector axes";
          vid = axis->var;
          vector_extent = extent;
          expert_vector_axis = axis->thread_tag == "blockIdx.y";
        } else
          LOG(FATAL) << "Unsupported auto launch axis";
        ICHECK(groups > 0 && groups <= 4 &&
               (vector_extent == 1 || vector_extent == 2));
      }
      Catalog(n->body, parents);
    } else if (const auto *n = stmt.as<LetStmtNode>())
      Catalog(n->body, parents);
    else if (const auto *n = stmt.as<DeclBufferNode>())
      Catalog(n->body, parents);
    else if (const auto *n = stmt.as<IfThenElseNode>()) {
      Catalog(n->then_case, parents);
      if (n->else_case.defined())
        Catalog(n->else_case.value(), parents);
    } else if (const auto *n = stmt.as<EvaluateNode>()) {
      if (!is_zero(n->value)) {
        ICHECK(!unit_ids.count(n));
        unit_ids[n] = next_unit++;
      }
    } else
      LOG(FATAL) << "Unsupported auto source syntax";
  }
  int64_t Value(PrimExpr expr) {
    return EvaluateAscendLifetimeInteger(expr, bindings);
  }
  void Execute(Stmt stmt) {
    ICHECK(++controls <= kAscendAutoScheduleControlExpansionLimit)
        << "Auto control expansion budget exceeded";
    if (const auto *n = stmt.as<SeqStmtNode>()) {
      for (Stmt s : n->seq)
        Execute(s);
    } else if (const auto *n = stmt.as<AttrStmtNode>()) {
      if (n->attr_key == "resource_scope") {
        int64_t scope = Value(n->value);
        ICHECK(scope == 0 || scope == 1);
        ICHECK(explicit_scope_ < 0 || explicit_scope_ == scope)
            << "Conflicting nested auto resource scopes";
        if ((scope == 0) == (actor_ == 0)) {
          int previous = explicit_scope_;
          explicit_scope_ = scope;
          Execute(n->body);
          explicit_scope_ = previous;
        }
      } else {
        ICHECK(n->attr_key == "thread_extent")
            << "Unsupported auto semantic attribute: " << n->attr_key;
        Execute(n->body);
      }
    } else if (const auto *n = stmt.as<ForNode>()) {
      int64_t id = loop_ids.at(n), minimum = Value(n->min),
              extent = Value(n->extent);
      for (int64_t x = 0; x < extent; ++x) {
        bindings.Set(n->loop_var, IntImm(n->loop_var.dtype(), minimum + x));
        coords.push_back(id);
        coords.push_back(minimum + x);
        epoch.push_back(minimum + x);
        Execute(n->body);
        coords.pop_back();
        coords.pop_back();
        epoch.pop_back();
      }
      bindings.erase(n->loop_var);
    } else if (const auto *n = stmt.as<LetStmtNode>()) {
      auto old = bindings.Get(n->var);
      bindings.Set(n->var, IntImm(n->var.dtype(), Value(n->value)));
      Execute(n->body);
      if (old.defined())
        bindings.Set(n->var, old.value());
      else
        bindings.erase(n->var);
    } else if (const auto *n = stmt.as<IfThenElseNode>()) {
      if (Value(n->condition))
        Execute(n->then_case);
      else if (n->else_case.defined())
        Execute(n->else_case.value());
    } else if (const auto *n = stmt.as<BlockRealizeNode>()) {
      if (Value(n->predicate))
        Execute(n->block);
    } else if (const auto *n = stmt.as<BlockNode>())
      Execute(n->body);
    else if (const auto *n = stmt.as<AllocateNode>()) {
      if (Value(n->condition))
        Execute(n->body);
    } else if (const auto *n = stmt.as<DeclBufferNode>())
      Execute(n->body);
    else if (const auto *n = stmt.as<EvaluateNode>()) {
      if (is_zero(n->value))
        return;
      const auto *raw = n->value.as<CallNode>();
      ICHECK(raw) << "Unsupported auto scalar evaluation";
      std::string operation;
      AscendResource resource = ResourceForCall(raw, &operation);
      ICHECK(resource == AscendResource::kCube ||
             resource == AscendResource::kVector)
          << "Unknown/ambiguous auto operation ownership: " << operation;
      ICHECK(explicit_scope_ < 0 ||
             (explicit_scope_ == 0) == (resource == AscendResource::kCube))
          << "Auto operation contradicts its explicit resource scope";
      if ((resource == AscendResource::kCube) != (actor_ == 0))
        return;
      ValidateAscendLifetimeCallIndices(GetRef<Call>(raw), bindings);
      Call call = Downcast<Call>(Substitute(n->value, bindings));
      R task = AuditAscendAutoCall(call, actor_);
      Array<Integer> coordinate, values,
          key{Int(group_), Int(actor_), Int(unit_ids.at(n))};
      for (int64_t x : coords) {
        coordinate.push_back(Int(x));
        key.push_back(Int(x));
      }
      for (int64_t x : epoch)
        values.push_back(Int(x));
      task.Set("id", Int(tasks.size()));
      task.Set("unit_id", Int(unit_ids.at(n)));
      task.Set("group", Int(group_));
      task.Set("actor", Int(actor_));
      task.Set("key", key);
      task.Set("loop_coords", coordinate);
      task.Set("epoch", values);
      task.Set("call", call);
      task.Set("statement", Evaluate(call));
      for (const char *field : {"reads", "writes", "scratch"})
        for (const ObjectRef &item :
             Downcast<Array<ObjectRef>>(task.at(field))) {
          R effect = Downcast<R>(item);
          Var data = Downcast<Var>(effect.at("data"));
          ICHECK(storage_ids.count(data.get())) << "Undeclared auto storage";
          R storage = storages.at(storage_ids.at(data.get()));
          ICHECK(Num(effect.at("begin")) >= 0 &&
                 Num(effect.at("end")) <= Num(storage.at("capacity_bytes")))
              << "Auto source footprint exceeds its real allocation";
        }
      ICHECK_LT(tasks.size(), kAscendAutoScheduleTaskLimit)
          << "Auto finite task budget exceeded";
      tasks.push_back(task);
    }
  }
  PrimFunc function;
  Var cid{ObjectPtr<Object>()}, vid{ObjectPtr<Object>()};
  int64_t groups{1}, vector_extent{1}, next_site{0}, next_unit{0}, controls{0},
      group_{0};
  int actor_{0};
  int explicit_scope_{-1};
  bool expert_vector_axis{false};
  std::vector<R> tasks, storages, loops;
  std::map<const VarNode *, size_t> storage_ids;
  std::map<const ForNode *, int64_t> loop_ids;
  std::map<const EvaluateNode *, int64_t> unit_ids;
  Map<Var, PrimExpr> bindings;
  std::vector<int64_t> coords, epoch;
};
} // namespace

int64_t AscendAutoMemoryBudget(const String &scope) {
  if (scope == "shared.ub")
    return kAscendA2A3UBBudget;
  if (scope == "shared.l1")
    return 524032;
  if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b")
    return 65536;
  if (scope == "wmma.accumulator")
    return 131072;
  LOG(FATAL) << "Unsupported auto compiler memory pool: " << scope;
  return 0;
}
void CheckAscendAutoScheduleContext(Target target, String platform) {
  Context(target, platform);
}

AutoRecord AuditAscendAutoCall(tir::Call call, int actor) {
  using namespace tir;
  Array<ObjectRef> reads, writes, defines, scratch, implicit;
  std::string name =
      call->op.as<OpNode>() ? call->op.as<OpNode>()->name : "opaque";
  std::string pipe;
  bool compound = false;
  int64_t latency = 2;
  auto add = [&](PrimExpr ptr, int64_t count, bool write, bool define = true) {
    R e = Ptr(ptr, count, write ? 2 : 1);
    (write ? writes : reads).push_back(e);
    if (write && define)
      defines.push_back(e);
  };
  PostOrderVisit(call, [&](const ObjectRef &node) {
    if (const auto *nested = node.as<CallNode>())
      ICHECK(nested == call.get() ||
             nested->op.same_as(builtin::tvm_access_ptr()) ||
             nested->op.same_as(Op::Get("tir.type_annotation")))
          << "Hidden opaque auto operand";
    ICHECK(!node.as<BufferLoadNode>())
        << "Hidden scalar memory effect in auto operand";
  });
  if (call->op.same_as(builtin::call_extern())) {
    ICHECK(!call->args.empty() && call->args[0].as<StringImmNode>());
    std::string tag = Downcast<StringImm>(call->args[0])->value;
    name = NormalizeFunctionName(tag);
    auto params = Params(tag);
    bool load = name == "copy_gm_to_ub" || name == "copy_gm_to_l1";
    bool store = name == "copy_ub_to_gm" || name == "copy_l0c_to_gm";
    ICHECK(load || store) << "Unsupported auto compound/extern: " << tag;
    ICHECK(call->args.size() >= 7);
    int64_t stride = I(call->args[3]), rows = I(call->args[4]),
            cols = I(call->args[5]);
    for (size_t i : {3U, 4U, 5U})
      ICHECK(call->args[i].dtype().is_int());
    ICHECK(rows > 0 && cols > 0 && stride >= cols && rows <= 65535);
    bool cube = name == "copy_gm_to_l1" || name == "copy_l0c_to_gm";
    ICHECK((actor == 0) == cube) << "Wrong auto DMA owner";
    if (!cube)
      ICHECK(rows <= 4095) << "Auto DataCopyPad row descriptor exceeds4095";
    if (name == "copy_l0c_to_gm") {
      ICHECK(params.size() == 6 && params[0] == "float" &&
             params[1] == "float" && params[2] == "layout::RowMajor" &&
             Decimal(params[3]) == rows && Decimal(params[4]) == cols &&
             params[5] == "0" && call->args.size() == 10 &&
             I(call->args[8]) == 0 && I(call->args[9]) == 0)
          << "Unsupported auto FIX conversion/layout/state";
    } else if (name == "copy_gm_to_l1") {
      ICHECK(params.size() == 3 && Decimal(params[1]) == rows &&
             Decimal(params[2]) == cols)
          << "Auto GM2L1 template/runtime footprint mismatch";
    } else {
      ICHECK((params.size() == 2 && rows == 1 && Decimal(params[1]) == cols) ||
             (params.size() == 3 && Decimal(params[1]) == cols &&
              Decimal(params[2]) == rows))
          << "Auto DMA template/runtime footprint mismatch";
    }
    PrimExpr gm = call->args[load ? 1 : 2], local = call->args[load ? 2 : 1];
    const auto *gmp = gm.as<CallNode>();
    const auto *lp = local.as<CallNode>();
    ICHECK(gmp && lp && gmp->args.size() == 5 && lp->args.size() == 5);
    DataType dtype = gmp->args[0].dtype();
    ICHECK(dtype == DataType::Float(16) || dtype == DataType::Float(32));
    ICHECK(GetPtrStorageScope(Downcast<Var>(gmp->args[1])) == "global" &&
           dtype == lp->args[0].dtype());
    ICHECK((cols * dtype.bytes()) % 32 == 0 && cols <= 65535 / dtype.bytes());
    R endpoint = Ptr(local, Safe(__int128(rows) * cols), load ? 2 : 1);
    std::string local_scope = Downcast<String>(endpoint.at("scope"));
    ICHECK(local_scope == (name == "copy_gm_to_l1"    ? "shared.l1"
                           : name == "copy_l0c_to_gm" ? "wmma.accumulator"
                                                      : "shared.ub"));
    if (cube)
      ICHECK(rows % 16 == 0 && cols % 16 == 0);
    for (int64_t row = 0; row < rows; ++row) {
      const auto *p = gm.as<CallNode>();
      ICHECK(p);
      Array<PrimExpr> args = p->args;
      args.Set(2, Int(Safe(__int128(I(p->args[2])) + __int128(row) * stride)));
      add(Call(p->dtype, p->op, args), cols, !load);
    }
    add(local, Safe(__int128(rows) * cols), load);
    pipe = load ? "MTE2" : cube ? "FIX" : "MTE3";
  } else if (call->op.same_as(ascend_fill()) ||
             call->op.same_as(ascend_add())) {
    ICHECK(actor != 0 && call->args.size() == 4);
    ICHECK(call->args[3].dtype().is_int());
    int64_t count = I(call->args[3]);
    ICHECK(count > 0);
    bool fill = call->op.same_as(ascend_fill());
    // Fill carries a helper name before its destination; Add starts with its
    // destination.
    add(call->args[fill ? 1 : 0], count, true);
    String dtype = Downcast<String>(Downcast<R>(writes[0]).at("dtype"));
    if (fill) {
      const auto *helper = call->args[0].as<StringImmNode>();
      ICHECK(helper && helper->value ==
                           (dtype == "float16" ? "Fill<half>" : "Fill<float>"))
          << "Auto fill helper does not match its destination dtype";
      DataType scalar_dtype = call->args[2].dtype();
      ICHECK(scalar_dtype.is_scalar() &&
             (scalar_dtype.is_float() || scalar_dtype.is_int() ||
              scalar_dtype.is_uint()))
          << "Auto fill requires an arithmetic scalar value";
      PostOrderVisit(call->args[2], [](const ObjectRef &node) {
        ICHECK(!node.as<BufferLoadNode>() && !node.as<CallNode>())
            << "Unsupported memory-dependent/opaque auto fill scalar";
      });
    } else {
      add(call->args[1], count, false);
      add(call->args[2], count, false);
    }
    for (const auto &items : {reads, writes})
      for (const ObjectRef &item : items) {
        R e = Downcast<R>(item);
        ICHECK(Downcast<String>(e.at("scope")) == "shared.ub" &&
               Downcast<String>(e.at("dtype")) == dtype && dtype != "uint8" &&
               (Num(e.at("end")) - Num(e.at("begin"))) % 32 == 0)
            << "Auto managed operation requires aligned homogeneous UB "
               "operands";
      }
    pipe = "V";
  } else if (call->op.same_as(ascend_gemm_v0())) {
    ICHECK(actor == 0 && call->args.size() == 6);
    auto p = Params(Downcast<StringImm>(call->args[0])->value);
    ICHECK(p.size() == 8 && p[0] == "half" && p[1] == "float" &&
           p[5] == "false" && p[6] == "false")
        << "Unsupported auto GEMM dtype/transpose";
    int64_t m = Decimal(p[2]), n = Decimal(p[3]), k = Decimal(p[4]),
            tile = Decimal(p[7]);
    ICHECK(m % 16 == 0 && n % 16 == 0 && k % 16 == 0 && tile % 16 == 0 &&
           I(call->args[4]) == 1 && I(call->args[5]) == n &&
           __int128(m) * tile * 4 <= 65536 && __int128(n) * tile * 4 <= 65536 &&
           __int128(m) * n * 4 <= 131072)
        << "Unsupported auto GEMM clear/geometry/scratch";
    add(call->args[1], Safe(__int128(m) * k), false);
    add(call->args[2], Safe(__int128(k) * n), false);
    add(call->args[3], Safe(__int128(m) * n), true);
    for (const ObjectRef &item : reads) {
      R e = Downcast<R>(item);
      ICHECK(Downcast<String>(e.at("scope")) == "shared.l1" &&
             Downcast<String>(e.at("dtype")) == "float16");
    }
    ICHECK(Downcast<String>(Downcast<R>(writes[0]).at("scope")) ==
               "wmma.accumulator" &&
           Downcast<String>(Downcast<R>(writes[0]).at("dtype")) == "float32");
    implicit.push_back(
        R{{"scope", String("wmma.matrix_a")}, {"bytes", Int(m * tile * 4)}});
    implicit.push_back(
        R{{"scope", String("wmma.matrix_b")}, {"bytes", Int(n * tile * 4)}});
    pipe = "M";
    compound = true;
    latency = 4;
  } else if (call->op.same_as(ascend_reduce()) ||
             call->op.same_as(ascend_reduce_fp32_v2())) {
    ICHECK(actor != 0 && (call->args.size() == 5 || call->args.size() == 6));
    std::string tag = Downcast<StringImm>(call->args[0])->value;
    auto p = Params(tag);
    ICHECK(p.size() == 4 && p[0] == "float" && p[3] == "-1" &&
           (tag.rfind("reduce_sum<", 0) == 0 ||
            tag.rfind("reduce_max<", 0) == 0 ||
            tag.rfind("reduce_min<", 0) == 0))
        << "Unsupported auto row reduction";
    int64_t m = Decimal(p[1]), n = Decimal(p[2]);
    ICHECK(n % 8 == 0 && I(call->args[4]) == 1 &&
           (call->args.size() == 5 || I(call->args[5]) == n))
        << "Auto reduction requires full-row clear=true";
    add(call->args[1], m, true);
    add(call->args[2], Safe(__int128(m) * n), false);
    for (const auto &items : {reads, writes})
      for (const ObjectRef &item : items) {
        R e = Downcast<R>(item);
        ICHECK(Downcast<String>(e.at("scope")) == "shared.ub" &&
               Downcast<String>(e.at("dtype")) == "float32");
      }
    const auto *tmp = call->args[3].as<CallNode>();
    ICHECK(tmp);
    R private_scratch = Ptr(call->args[3], I(tmp->args[3]), 2);
    ICHECK(Downcast<String>(private_scratch.at("scope")) == "shared.ub" &&
           Downcast<String>(private_scratch.at("dtype")) == "float32")
        << "Auto reduction scratch requires an fp32 UB view";
    for (const auto &accesses : {reads, writes})
      for (const ObjectRef &item : accesses) {
        R access = Downcast<R>(item);
        if (Downcast<Var>(private_scratch.at("data"))
                .same_as(Downcast<Var>(access.at("data"))))
          ICHECK(Num(private_scratch.at("end")) <= Num(access.at("begin")) ||
                 Num(access.at("end")) <= Num(private_scratch.at("begin")))
              << "Auto reduction scratch overlaps its input/output";
      }
    ICHECK(m <= reduce2d_v2::kUbElements && n <= reduce2d_v2::kUbElements &&
           __int128(m) * n < reduce2d_v2::kUbElements)
        << "Unsupported auto reduction scratch geometry";
    uint32_t required = reduce2d_v2::Reduce2DScratchElements(
        uint32_t(m), uint32_t(n), uint32_t(n), true);
    ICHECK(required > 0 &&
           Num(private_scratch.at("end")) - Num(private_scratch.at("begin")) >=
               int64_t(required) * 4)
        << "Auto reduction scratch extent is smaller than the audited required "
           "size";
    private_scratch.Set("required_bytes", Int(int64_t(required) * 4));
    private_scratch.Set("role", String("self_contained_internal_rw"));
    scratch.push_back(private_scratch);
    pipe = "V";
    compound = true;
    latency = 4;
  } else
    LOG(FATAL) << "Unsupported auto operation/state: " << name;
  Array<String> resources{String(pipe)};
  if (call->op.same_as(ascend_gemm_v0()))
    resources = {String("MTE1"), String("M")};
  return R{{"operation", String(name)},
           {"resources", resources},
           {"pipe", String(pipe)},
           {"compound", Bool(compound)},
           {"reads", reads},
           {"writes", writes},
           {"must_defines", defines},
           {"scratch", scratch},
           {"implicit_scratch", implicit},
           {"latency", Int(latency)},
           {"ii", Int(1)}};
}

AutoRecord AnalyzeAscendAutoSource(tir::PrimFunc function) {
  return Source(function).Build();
}

namespace {
bool Touch(const R &task, const Var &data) {
  for (const char *field : {"reads", "writes", "scratch"})
    for (const ObjectRef &item : Downcast<Array<ObjectRef>>(task.at(field)))
      if (Downcast<Var>(Downcast<R>(item).at("data")).same_as(data))
        return true;
  return false;
}
std::pair<int64_t, int64_t> Clock(const R &source, const Var &data) {
  int64_t clock = -1;
  for (const ObjectRef &item : Downcast<Array<ObjectRef>>(source.at("tasks"))) {
    R task = Downcast<R>(item);
    if (!Touch(task, data))
      continue;
    auto coords = Downcast<Array<Integer>>(task.at("loop_coords"));
    if (coords.empty())
      return {-1, 0};
    int64_t current = coords[coords.size() - 2]->value;
    if (clock >= 0 && clock != current)
      return {-1, 0};
    clock = current;
  }
  if (clock < 0)
    return {-1, 0};
  R loop = Downcast<R>(Downcast<Array<ObjectRef>>(source.at("loops"))[clock]);
  if (Num(loop.at("num_stages")) < 2)
    return {-1, 0};
  return {clock, Num(loop.at("min"))};
}
int64_t Slot(const R &task, int64_t clock, int64_t minimum, int64_t versions) {
  if (versions == 1)
    return 0;
  auto coords = Downcast<Array<Integer>>(task.at("loop_coords"));
  for (size_t i = 0; i + 1 < coords.size(); i += 2)
    if (coords[i]->value == clock) {
      __int128 n = __int128(coords[i + 1]->value) - minimum;
      return int64_t((n % versions + versions) % versions);
    }
  LOG(FATAL) << "Versioned auto access is outside its clock owner";
  return 0;
}
PrimFunc Schedule(PrimFunc f, Target target, String platform) {
  Context(target, platform);
  ICHECK(!f->GetAttr<ObjectRef>(kAscendAutoSchedule).defined())
      << "Repeated auto planning";
  R source = AnalyzeAscendAutoSource(f);
  f = Downcast<PrimFunc>(source.at("source_function"));
  R contract = BuildAscendAutoSourceContract(source);
  auto tasks = Downcast<Array<ObjectRef>>(source.at("tasks"));
  auto storages = Downcast<Array<ObjectRef>>(source.at("storages"));
  std::set<const VarNode *> pinned;
  for (Var data : Downcast<Array<Var>>(contract.at("pin_n1")))
    pinned.insert(data.get());
  std::vector<size_t> eligible;
  std::vector<int64_t> pitches, clocks, minima;
  for (size_t i = 0; i < storages.size(); ++i) {
    R storage = Downcast<R>(storages[i]);
    Var data = Downcast<Var>(storage.at("data"));
    std::string scope = Downcast<String>(storage.at("scope"));
    int64_t span = Num(storage.at("capacity_bytes"));
    int64_t pitch = Safe((__int128(span) + 31) / 32 * 32);
    auto [clock, minimum] = Clock(source, data);
    pitches.push_back(pitch);
    clocks.push_back(clock);
    minima.push_back(minimum);
    if ((scope == "shared.ub" || scope == "shared.l1") &&
        Num(storage.at("allocation_site_id")) >= 0 && clock >= 0 &&
        !pinned.count(data.get()))
      eligible.push_back(i);
  }
  ICHECK_LE(eligible.size(), 8U)
      << "Auto version candidate enumeration budget exceeded";
  const auto *solve = runtime::Registry::Get("tl.ascend.solve_schedule");
  const auto *verify = runtime::Registry::Get("tl.ascend.verify_schedule");
  ICHECK(solve && verify)
      << "Auto requires the explicitly registered numeric solver/verifier";
  int64_t best_cost = INT64_MAX, best_bytes = INT64_MAX;
  R best_problem, best_solution;
  Map<Var, Integer> best_versions;
  Array<ObjectRef> best_layout;
  for (size_t choice = 0; choice < (size_t(1) << eligible.size()); ++choice) {
    std::vector<int64_t> versions(storages.size(), 1);
    for (size_t b = 0; b < eligible.size(); ++b)
      if (choice & (size_t(1) << b))
        versions[eligible[b]] = 2;
    std::map<std::string, int64_t> occupied;
    bool fits = true;
    int64_t total = 0;
    Map<Var, Integer> selected;
    Array<ObjectRef> layout;
    std::map<const VarNode *, size_t> owner;
    for (size_t b = 0; b < storages.size(); ++b) {
      R storage = Downcast<R>(storages[b]);
      Var data = Downcast<Var>(storage.at("data"));
      owner[data.get()] = b;
      selected.Set(data, Int(versions[b]));
      std::string scope = Downcast<String>(storage.at("scope"));
      if (scope == "global")
        continue;
      int64_t bytes = Safe(__int128(pitches[b]) * versions[b]);
      int64_t budget = AscendAutoMemoryBudget(scope);
      if (bytes > budget || occupied[scope] > budget - bytes) {
        fits = false;
        break;
      }
      occupied[scope] += bytes;
      total += bytes;
      Array<ObjectRef> slots;
      for (const ObjectRef &task_item : tasks) {
        R task = Downcast<R>(task_item);
        if (Touch(task, data))
          slots.push_back(
              R{{"task", task.at("id")},
                {"slot", Int(Slot(task, clocks[b], minima[b], versions[b]))}});
      }
      layout.push_back(R{{"logical_data", data},
                         {"physical_data", data},
                         {"versions", Int(versions[b])},
                         {"pitch", Int(pitches[b])},
                         {"span", storage.at("capacity_bytes")},
                         {"total", Int(bytes)},
                         {"clock_loop", Int(clocks[b])},
                         {"clock_min", Int(minima[b])},
                         {"slots", slots}});
    }
    if (!fits)
      continue;
    for (const ObjectRef &task_item : tasks)
      for (const ObjectRef &scratch : Downcast<Array<ObjectRef>>(
               Downcast<R>(task_item).at("implicit_scratch"))) {
        R item = Downcast<R>(scratch);
        std::string scope = Downcast<String>(item.at("scope"));
        if (occupied[scope] + Num(item.at("bytes")) >
            AscendAutoMemoryBudget(scope))
          fits = false;
      }
    if (!fits)
      continue;
    Array<Integer> latencies, intervals, flags;
    std::map<std::string, int> domains;
    for (const ObjectRef &item : tasks) {
      R task = Downcast<R>(item);
      latencies.push_back(Downcast<Integer>(task.at("latency")));
      intervals.push_back(Int(1));
      int64_t mask = 0;
      for (String resource : Downcast<Array<String>>(task.at("resources"))) {
        std::string domain = std::to_string(Num(task.at("group"))) + ":" +
                             std::to_string(Num(task.at("actor"))) + ":" +
                             std::string(resource);
        auto inserted = domains.emplace(domain, domains.size());
        ICHECK_LT(inserted.first->second, 62);
        mask |= int64_t(1) << inserted.first->second;
      }
      flags.push_back(Int(mask));
    }
    std::set<std::pair<int64_t, int64_t>> edges;
    for (const ObjectRef &item :
         Downcast<Array<ObjectRef>>(contract.at("logical_deps"))) {
      R edge = Downcast<R>(item);
      edges.emplace(Num(edge.at("src")), Num(edge.at("dst")));
    }
    auto overlap = [&](R left, R right, R lt, R rt) {
      Var a = Downcast<Var>(left.at("data")),
          b = Downcast<Var>(right.at("data"));
      if (!a.same_as(b))
        return false;
      size_t id = owner.at(a.get());
      int64_t ls = Slot(lt, clocks[id], minima[id], versions[id]) * pitches[id];
      int64_t rs = Slot(rt, clocks[id], minima[id], versions[id]) * pitches[id];
      return Num(left.at("begin")) + ls < Num(right.at("end")) + rs &&
             Num(right.at("begin")) + rs < Num(left.at("end")) + ls;
    };
    for (size_t i = 0; i < tasks.size(); ++i)
      for (size_t j = i + 1; j < tasks.size(); ++j) {
        R a = Downcast<R>(tasks[i]), b = Downcast<R>(tasks[j]);
        if (Num(a.at("group")) != Num(b.at("group")) ||
            Num(a.at("actor")) != Num(b.at("actor")))
          continue;
        for (const auto &pair :
             {std::pair{"writes", "reads"}, std::pair{"reads", "writes"},
              std::pair{"writes", "writes"}})
          for (const ObjectRef &x :
               Downcast<Array<ObjectRef>>(a.at(pair.first)))
            for (const ObjectRef &y :
                 Downcast<Array<ObjectRef>>(b.at(pair.second)))
              if (overlap(Downcast<R>(x), Downcast<R>(y), a, b))
                edges.emplace(i, j);
      }
    Array<Array<Integer>> deps, exclusions;
    for (auto [u, v] : edges)
      deps.push_back({Int(u), Int(v), Int(0), latencies[u]});
    for (size_t i = 0; i < tasks.size(); ++i)
      for (size_t j = i + 1; j < tasks.size(); ++j) {
        R a = Downcast<R>(tasks[i]), b = Downcast<R>(tasks[j]);
        if (Num(a.at("group")) == Num(b.at("group")) &&
            Num(a.at("actor")) == Num(b.at("actor")) &&
            Downcast<String>(a.at("pipe")) == Downcast<String>(b.at("pipe")) &&
            (Num(a.at("compound")) || Num(b.at("compound"))))
          exclusions.push_back({Int(i), Int(j), latencies[i], latencies[j]});
      }
    R problem{{"schema_version", Int(1)},
              {"mode", String("straight")},
              {"latencies", latencies},
              {"iis", intervals},
              {"resource_flags", flags},
              {"data_deps", deps},
              {"owner_exclusion_deps", exclusions},
              {"timeout_ms", Int(5000)},
              {"seed", Int(0)}};
    R result;
    if (tasks.empty())
      result = R{{"schema_version", Int(1)},
                 {"status", String("SAT")},
                 {"start_times", Array<Integer>()},
                 {"order", Array<Integer>()},
                 {"ii", Int(0)},
                 {"buffer_versions", Array<Integer>()}};
    else {
      result = (*solve)(problem);
      ICHECK(Downcast<String>(result.at("status")) == "SAT")
          << "Auto solver did not produce a verified SAT plan: " << result;
      R checked = (*verify)(problem, result);
      ICHECK(Num(checked.at("valid")))
          << "Auto numeric verifier rejected result: " << checked;
    }
    auto starts = Downcast<Array<Integer>>(result.at("start_times"));
    int64_t cost = 0;
    for (size_t i = 0; i < starts.size(); ++i)
      cost = std::max(cost,
                      Safe(__int128(starts[i]->value) + latencies[i]->value));
    if (cost < best_cost || (cost == best_cost && total < best_bytes)) {
      best_cost = cost;
      best_bytes = total;
      best_problem = problem;
      best_solution = result;
      best_versions = selected;
      best_layout = layout;
    }
  }
  ICHECK(best_cost != INT64_MAX) << "No in-budget auto version candidate";
  R manifest{{"schema_version", Int(1)},
             {"model", String("abstract_issue_step_v1")},
             {"source_function", f},
             {"source_analysis", source},
             {"source_contract", contract},
             {"problem", best_problem},
             {"solution", best_solution},
             {"selected_versions", best_versions},
             {"materialized", Bool(false)}};
  f = WithAttr(f, kAscendAutoScheduleBufferRemap, best_layout);
  return WithAttr(f, kAscendAutoSchedule, manifest);
}
} // namespace

tir::transform::Pass AscendAutoSchedule(Target target, String platform) {
  auto pass = [=](tir::PrimFunc f, IRModule, tvm::transform::PassContext) {
    return Schedule(f, target, platform);
  };
  return tir::transform::CreatePrimFuncPass(pass, 0, "tl.AscendAutoSchedule",
                                            {});
}

tir::transform::Pass AscendPrepareAutoSchedule(Target target) {
  auto pass = [target](tir::PrimFunc f, IRModule, tvm::transform::PassContext) {
    Context(target, "A3");
    ICHECK(!f->GetAttr<ObjectRef>(kAscendAutoSchedulePrepared).defined())
        << "Stale auto preparation";
    Array<tir::Var> parameters;
    for (const tir::Var &p : f->params)
      if (f->buffer_map.count(p))
        parameters.push_back(f->buffer_map[p]->data);
    f = WithAttr(f, kAscendStorageOriginalParameters, parameters);
    f = CanonicalizeAscendStorageAliases(f);
    return WithAttr(f, kAscendAutoSchedulePrepared, Bool(true));
  };
  return tir::transform::CreatePrimFuncPass(pass, 0,
                                            "tl.AscendPrepareAutoSchedule", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AnalyzeAscendAutoSchedule")
    .set_body_typed(AnalyzeAscendAutoSource);
TVM_REGISTER_GLOBAL("tl.transform.AscendPrepareAutoSchedule")
    .set_body_typed(AscendPrepareAutoSchedule);
TVM_REGISTER_GLOBAL("tl.transform.AscendAutoSchedule")
    .set_body_typed(AscendAutoSchedule);
} // namespace tl
} // namespace tvm
