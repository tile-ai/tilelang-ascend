// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

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
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

namespace tilelangir {
namespace python {

static void populateDialectRegistry(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  bishengir::registerAllDialects(registry);
  bishengir::registerAllExtensions(registry);
  ::tilelangir::registerAllDialects(registry);
}

static std::pair<bool, std::string>
runPassPipeline(llvm::StringRef mlirStr, llvm::StringRef innerPipeline) {
  mlir::DialectRegistry registry;
  populateDialectRegistry(registry);

  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();

  mlir::ParserConfig config(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlirStr, config, "input.mlir");
  if (!module) {
    return {false, "Failed to parse MLIR string"};
  }

  mlir::PassManager pm(&context, "builtin.module");
  if (failed(mlir::parsePassPipeline(innerPipeline, pm, llvm::errs()))) {
    return {
        false,
        "Failed to parse pass pipeline: " + innerPipeline.str() +
            " (pass may not be registered; check .so and registerAllPasses)"};
  }
  if (failed(pm.run(*module))) {
    return {false, "Pass pipeline run failed"};
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os, mlir::OpPrintingFlags().useLocalScope());
  os.flush();
  return {true, result};
}

} // namespace python
} // namespace tilelangir

namespace {
struct RegisterPassesOnLoad {
  RegisterPassesOnLoad() {
    mlir::registerAllPasses();
    bishengir::registerAllPasses();
    tilelangir::registerAllPasses();
  }
} registerPassesOnLoad;
} // namespace

struct PassPipeline {
  std::vector<std::string> elements;
  bool irPrinting = false;
  std::string irPrintingFileTreeDir;

  void add(const std::string &pipeline_text) {
    elements.push_back(pipeline_text);
  }

  void enableIRPrinting() { irPrinting = true; }

  void enableIRPrintingToFileTree(const std::string &dir) {
    irPrintingFileTreeDir = dir;
  }

  std::string str() const {
    std::string result;
    for (size_t i = 0; i < elements.size(); ++i) {
      if (i > 0)
        result += ",";
      result += elements[i];
    }
    return result;
  }

  std::string run(const std::string &mlir_str) const {
    mlir::DialectRegistry registry;
    tilelangir::python::populateDialectRegistry(registry);

    mlir::MLIRContext context(registry);
    context.allowUnregisteredDialects();
    if (irPrinting || !irPrintingFileTreeDir.empty()) {
      context.disableMultithreading();
    }

    mlir::ParserConfig config(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_str, config, "input.mlir");
    if (!module) {
      throw std::runtime_error("Failed to parse MLIR string");
    }

    mlir::PassManager pm(&context, "builtin.module");

    for (const auto &elem : elements) {
      if (failed(mlir::parsePassPipeline(elem, pm, llvm::errs()))) {
        throw std::runtime_error(
            "Failed to parse pass pipeline element: " + elem +
            " (pass may not be registered; check .so and registerAllPasses)");
      }
    }

    if (irPrinting) {
      pm.enableIRPrinting(
          /*shouldPrintBeforePass=*/[](mlir::Pass *,
                                       mlir::Operation *) { return false; },
          /*shouldPrintAfterPass=*/
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false,
          /*out=*/llvm::errs());
    }

    if (!irPrintingFileTreeDir.empty()) {
      pm.enableIRPrintingToFileTree(
          /*shouldPrintBeforePass=*/[](mlir::Pass *,
                                       mlir::Operation *) { return false; },
          /*shouldPrintAfterPass=*/
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false,
          /*printTreeDir=*/irPrintingFileTreeDir);
    }

    if (failed(pm.run(*module))) {
      throw std::runtime_error("Pass pipeline run failed");
    }

    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os, mlir::OpPrintingFlags().useLocalScope());
    os.flush();
    return result;
  }
};

PYBIND11_MODULE(tilelangir, m) {
  m.doc() = "TileLangIR pass execution: PassPipeline";

  pybind11::class_<PassPipeline>(
      m, "PassPipeline",
      "Batched pass pipeline. add() elements using MLIR textual pipeline "
      "format, then run() once for a single parse/serialize cycle.")
      .def(pybind11::init<>())
      .def("add", &PassPipeline::add, pybind11::arg("pipeline_text"),
           "Add a textual pipeline element.")
      .def("enable_ir_printing", &PassPipeline::enableIRPrinting,
           "Enable per-pass IR printing to stderr after each pass.")
      .def("enable_ir_printing_to_file_tree",
           &PassPipeline::enableIRPrintingToFileTree,
           pybind11::arg("dir") = ".pass_manager_output",
           "Enable per-pass IR printing to a directory tree.")
      .def("run", &PassPipeline::run, pybind11::arg("mlir_str"),
           "Execute all passes on the MLIR string. Returns result MLIR string.")
      .def("__str__", &PassPipeline::str)
      .def("__repr__", [](const PassPipeline &self) {
        return "PassPipeline(\"" + self.str() + "\")";
      });
}
