// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "constant.h"
#include "support/check.h"
#include "target/source/codegen_params.h"

#include <tvm/runtime/tensor.h>
#include <tvm/tirx/expr.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <vector>

namespace tvm::codegen::a2a3 {
using namespace ffi;
using namespace tirx;

std::string ConstantDeclarations::Emit(
    const Call &call, std::ostream &decl_stream,
    const std::function<void(DataType, std::ostream &)> &print_type,
    const std::function<std::string(const std::string &)> &reserve_name) {
  ICHECK_GE(call->args.size(), 5U) << "Malformed A2/A3 constant data";
  auto field = [&](size_t index) -> std::string {
    const auto *text = call->args[index].as<StringImmNode>();
    ICHECK(text) << "A2/A3 constant metadata must contain literal strings";
    return text->value;
  };
  std::string dtype_string = field(0), bytes_hex = field(1);
  std::string id = field(2), name = field(3);
  ICHECK(!id.empty() && !name.empty())
      << "A2/A3 constant data requires frontend normalization before codegen";
  DataType dtype(ffi::StringToDLDataType(dtype_string));
  ICHECK_EQ(dtype.lanes(), 1) << "A2/A3 constant arrays need scalar elements";
  ICHECK((dtype.is_int() || dtype.is_uint()) &&
             (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32 ||
              dtype.bits() == 64) ||
         dtype.is_float() &&
             (dtype.bits() == 16 || dtype.bits() == 32 || dtype.bits() == 64) ||
         dtype.is_bfloat16())
      << "A2/A3 constant dtype is not supported by the source literal "
         "formatter: "
      << dtype;
  ICHECK_GT(dtype.bits(), 0)
      << "A2/A3 constant dtype must have positive bit width";
  ICHECK_EQ(dtype.bits() % 8, 0) << "A2/A3 constant dtype must be byte aligned";
  std::vector<int64_t> shape;
  size_t count = 1;
  std::string fingerprint =
      dtype_string + ":" + bytes_hex + ":" + name + ":" + field(4);
  for (size_t i = 5; i < call->args.size(); ++i) {
    const auto *extent = call->args[i].as<IntImmNode>();
    ICHECK(extent && extent->value >= 0)
        << "A2/A3 constant payload shape must be nonnegative and static";
    if (count != 0) {
      ICHECK_LE(static_cast<uint64_t>(extent->value),
                static_cast<uint64_t>(INT_MAX) / count)
          << "A2/A3 constant array exceeds the source literal formatter "
             "capacity";
    }
    count *= extent->value;
    shape.push_back(extent->value);
    fingerprint += ":" + std::to_string(extent->value);
  }
  size_t element_bytes = dtype.bits() / 8;
  ICHECK_LE(count, std::numeric_limits<size_t>::max() / element_bytes / 2);
  ICHECK_EQ(bytes_hex.size(), count * element_bytes * 2)
      << "A2/A3 constant literal data size does not match shape/dtype";
  auto existing = declarations_.find(id);
  if (existing != declarations_.end()) {
    ICHECK_EQ(existing->second, fingerprint)
        << "A2/A3 C/V constant clones have inconsistent literal metadata";
  } else {
    ICHECK(!names_.count(name))
        << "A2/A3 constant name is ambiguous in raw source: " << name;
    ICHECK_EQ(reserve_name(name), name)
        << "A2/A3 constant name conflicts with another source symbol: " << name
        << "; rename the declaration and its raw source together";
    names_[name] = id;
    declarations_[id] = fingerprint;
    std::vector<uint8_t> bytes(count * element_bytes);
    auto nibble = [](char c) -> unsigned {
      ICHECK((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
          << "Invalid A2/A3 constant hex payload";
      return c <= '9' ? c - '0' : c - 'a' + 10;
    };
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = (nibble(bytes_hex[2 * i]) << 4) | nibble(bytes_hex[2 * i + 1]);
    uint16_t byte_order = 1;
    if (*reinterpret_cast<uint8_t *>(&byte_order) != 1) {
      for (size_t i = 0; i < count; ++i)
        std::reverse(bytes.begin() + i * element_bytes,
                     bytes.begin() + (i + 1) * element_bytes);
    }
    runtime::Tensor data =
        runtime::Tensor::Empty(ffi::Shape(shape), dtype, DLDevice{kDLCPU, 0});
    if (!bytes.empty()) {
      data.CopyFromBytes(bytes.data(), bytes.size());
    }
    decl_stream << "#ifdef __cplusplus\nextern \"C\" {\n#endif\nstatic const ";
    print_type(dtype, decl_stream);
    decl_stream << " __attribute__((section(\".rodata.tvm\"), aligned(16))) "
                << name << "[" << count << "] = {\n";
    TensorDataToC(data, 4, decl_stream);
    decl_stream << "};\n#ifdef __cplusplus\n}\n#endif\n";
  }
  return name;
}

} // namespace tvm::codegen::a2a3
