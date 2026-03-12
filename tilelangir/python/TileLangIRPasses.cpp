// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// Pybind11 module: TileLangIR create_pass_runner and pass registration.
// create_pass_runner(pass_name, pass_spec) returns a callable with pass_name
// for dump. All pass execution goes through create_pass_runner.

#include "tilelangir/InitAllDialects.h"
#include "tilelangir/InitAllPasses.h"

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/pybind11.h>

#include <string>
#include <utility>

namespace tilelangir {
namespace python {

static bool tryBuiltinModuleRoot(llvm::StringRef pipelineStr, llvm::StringRef* inner) {
  pipelineStr = pipelineStr.trim();
  if (!pipelineStr.consume_front("builtin.module(") || !pipelineStr.consume_back(")"))
    return false;
  *inner = pipelineStr.trim();
  return true;
}

// Returns (true, result_mlir_str) on success, (false, error_message) on failure.
static std::pair<bool, std::string> runPassPipeline(llvm::StringRef mlirStr,
                                                    llvm::StringRef pipelineStr) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  bishengir::registerAllDialects(registry);
  bishengir::registerAllExtensions(registry);
  ::tilelangir::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();

  mlir::ParserConfig config(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlirStr, config, "input.mlir");
  if (!module) {
    return {false, "Failed to parse MLIR string"};
  }

  llvm::StringRef innerPipeline;
  if (tryBuiltinModuleRoot(pipelineStr, &innerPipeline)) {
    mlir::PassManager pm(&context, "builtin.module");
    if (failed(mlir::parsePassPipeline(innerPipeline, pm, llvm::errs()))) {
      return {false, "Failed to parse inner pass pipeline: " + innerPipeline.str()};
    }
    if (failed(pm.run(*module))) {
      return {false, "Pass pipeline run failed"};
    }
  } else {
    mlir::PassManager pm(&context);
    if (failed(mlir::parsePassPipeline(pipelineStr, pm, llvm::errs()))) {
      return {false, "Failed to parse pass pipeline: " + pipelineStr.str() +
                         " (pass may not be registered; check .so and registerAllPasses)"};
    }
    if (failed(pm.run(*module))) {
      return {false, "Pass pipeline run failed"};
    }
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os, mlir::OpPrintingFlags().useLocalScope());
  os.flush();
  return {true, result};
}

}  // namespace python
}  // namespace tilelangir

// Register passes when the .so is loaded so create_pass_runner can resolve pass names.
namespace {
struct RegisterPassesOnLoad {
  RegisterPassesOnLoad() {
    mlir::registerAllPasses();
    bishengir::registerAllPasses();
    tilelangir::registerAllPasses();
  }
} registerPassesOnLoad;
}  // namespace

/// Creates a pass runner callable with pass_name attached for dump/debug.
/// Returns a Python callable: fn(mlir_str) -> result_str.
struct PassRunner {
  std::string pass_name;
  std::string pass_spec;

  PassRunner(std::string name, std::string spec)
      : pass_name(std::move(name)), pass_spec(std::move(spec)) {}

  std::string operator()(const std::string& mlir_str) const {
    auto [ok, out] = tilelangir::python::runPassPipeline(mlir_str, pass_spec);
    if (!ok) {
      throw std::runtime_error(out);
    }
    return out;
  }
};

PYBIND11_MODULE(tilelangir, m) {
  m.doc() = "TileLangIR: create_pass_runner(pass_name, pass_spec)";

  pybind11::class_<PassRunner>(m, "PassRunner",
                               "Callable pass runner with pass_name for dump.")
      .def(pybind11::init<std::string, std::string>(),
           pybind11::arg("pass_name"), pybind11::arg("pass_spec"))
      .def_readonly("pass_name", &PassRunner::pass_name)
      .def("__call__", &PassRunner::operator(), pybind11::arg("mlir_str"));

  m.def(
      "create_pass_runner",
      [](const std::string& pass_name, const std::string& pass_spec) {
        return pybind11::cast(PassRunner(pass_name, pass_spec));
      },
      pybind11::arg("pass_name"), pybind11::arg("pass_spec"),
      "Create a pass runner callable. The returned callable has a pass_name attribute for dump.");
}
