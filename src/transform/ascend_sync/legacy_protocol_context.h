// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/tir/expr.h>

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace tvm::tl::ascend_sync {

using LegacyProtocolConstants =
    std::unordered_map<tir::Var, int64_t, ObjectPtrHash, ObjectPtrEqual>;

// Pure, concrete protocol evaluation. Every original-width intermediate is
// checked; unknown values stay unknown. This does not add execution facts.
std::optional<int64_t>
EvaluateLegacyProtocolInteger(const PrimExpr &value,
                              const LegacyProtocolConstants &constants);

enum class LegacyProtocolOwner { kUnassigned = -1, kCube = 0, kVector = 1 };

// Shared lexical environment, not a control-flow or token proof strategy.
// Callers retain their own branch/loop rules and choose all initial facts.
class LegacyProtocolContext {
public:
  LegacyProtocolContext() = default;
  LegacyProtocolContext(const LegacyProtocolContext &) = delete;
  LegacyProtocolContext &operator=(const LegacyProtocolContext &) = delete;

  class Binding {
  public:
    ~Binding();
    Binding(const Binding &) = delete;
    Binding &operator=(const Binding &) = delete;
    void Set(std::optional<int64_t> value);

  private:
    friend class LegacyProtocolContext;
    Binding(LegacyProtocolContext *context, const tir::Var &variable,
            std::optional<int64_t> value);
    LegacyProtocolContext *context_;
    tir::Var variable_;
    std::optional<int64_t> previous_;
  };

  class OwnerScope {
  public:
    ~OwnerScope();
    OwnerScope(const OwnerScope &) = delete;
    OwnerScope &operator=(const OwnerScope &) = delete;

  private:
    friend class LegacyProtocolContext;
    OwnerScope(LegacyProtocolContext *context, LegacyProtocolOwner owner);
    LegacyProtocolContext *context_;
    LegacyProtocolOwner previous_;
  };

  std::optional<int64_t> Evaluate(const PrimExpr &value) const;
  [[nodiscard]] Binding Bind(const tir::Var &variable,
                             std::optional<int64_t> value);
  [[nodiscard]] OwnerScope EnterOwner(LegacyProtocolOwner owner);
  LegacyProtocolOwner Owner() const { return owner_; }

private:
  LegacyProtocolConstants constants_;
  LegacyProtocolOwner owner_{LegacyProtocolOwner::kUnassigned};
};

} // namespace tvm::tl::ascend_sync
