//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "hcl/Bindings/Python/HCLModule.h"
#include "hcl-c/Dialect/Dialects.h"
#include "hcl-c/Dialect/HCLAttributes.h"
#include "hcl-c/Dialect/HCLTypes.h"
#include "hcl-c/Dialect/Registration.h"
#include "hcl-c/Translation/EmitIntelHLS.h"
#include "hcl-c/Translation/EmitVivadoHLS.h"
#include "hcl/Conversion/HCLToLLVM.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Transforms/Passes.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "taskflow/taskflow.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "mlir-c/Conversion.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

using namespace mlir;
using namespace mlir::python;
using namespace hcl;

//===----------------------------------------------------------------------===//
// Customized Python classes
//===----------------------------------------------------------------------===//

// PybindUtils.h
class PyFileAccumulator {
public:
  PyFileAccumulator(const pybind11::object &fileObject, bool binary)
      : pyWriteFunction(fileObject.attr("write")), binary(binary) {}

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      pybind11::gil_scoped_acquire acquire;
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      if (accum->binary) {
        // Note: Still has to copy and not avoidable with this API.
        pybind11::bytes pyBytes(part.data, part.length);
        accum->pyWriteFunction(pyBytes);
      } else {
        pybind11::str pyStr(part.data,
                            part.length); // Decodes as UTF-8 by default.
        accum->pyWriteFunction(pyStr);
      }
    };
  }

private:
  pybind11::object pyWriteFunction;
  bool binary;
};

//===----------------------------------------------------------------------===//
// Loop transform APIs
//===----------------------------------------------------------------------===//

static bool loopTransformation(MlirModule &mlir_mod) {
  py::gil_scoped_release();
  auto mod = unwrap(mlir_mod);
  return applyLoopTransformation(mod);
}

//===----------------------------------------------------------------------===//
// Emission APIs
//===----------------------------------------------------------------------===//

static bool emitVivadoHls(MlirModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitVivadoHls(mod, accum.getCallback(), accum.getUserData()));
}

static bool emitIntelHls(MlirModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitIntelHls(mod, accum.getCallback(), accum.getUserData()));
}

//===----------------------------------------------------------------------===//
// Lowering APIs
//===----------------------------------------------------------------------===//

static bool lowerHCLToLLVM(MlirModule &mlir_mod, MlirContext &mlir_ctx) {
  auto mod = unwrap(mlir_mod);
  auto ctx = unwrap(mlir_ctx);
  return applyHCLToLLVMLoweringPass(mod, *ctx);
}

static bool lowerFixedPointToInteger(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyFixedPointToInteger(mod);
}

static bool lowerAnyWidthInteger(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyAnyWidthInteger(mod);
}

static bool moveReturnToInput(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyMoveReturnToInput(mod);
}

static bool lowerCompositeType(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerCompositeType(mod);
}

static bool lowerBitOps(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerBitOps(mod);
}

static bool legalizeCast(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLegalizeCast(mod);
}

static bool removeStrideMap(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyRemoveStrideMap(mod);
}

//===----------------------------------------------------------------------===//
// taskFlow Executor engine APIs
//===----------------------------------------------------------------------===//
static void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

void lowerModuleToLLVM(MlirContext ctx, MlirModule module) {
  MlirPassManager pm = mlirPassManagerCreate(ctx);
  MlirOpPassManager opm = mlirPassManagerGetNestedUnder(
      pm, mlirStringRefCreateFromCString("func.func"));

  mlirPassManagerAddOwnedPass(pm, mlirCreateConversionSCFToControlFlow());
  mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertMemRefToLLVM());
  mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVM());
  mlirPassManagerAddOwnedPass(pm, mlirCreateConversionReconcileUnrealizedCasts());

  mlirOpPassManagerAddOwnedPass(opm,
                                mlirCreateConversionConvertAffineToStandard());
  mlirOpPassManagerAddOwnedPass(opm,
                                mlirCreateConversionConvertArithmeticToLLVM());

  MlirLogicalResult status = mlirPassManagerRun(pm, module);

  if (mlirLogicalResultIsFailure(status)) {
    fprintf(stderr, "Unexpected failure running pass pipeline\n");
    exit(2);
  }
  mlirPassManagerDestroy(pm);
}

static bool runTaskFlowExecutor(
    std::map<std::string, std::string> &modules,
    std::map<std::string, std::vector<py::array_t<float>>> argsMap) {
      
      tf::Executor executor;
      tf::Taskflow taskflow;
      std::map<std::string, tf::Task> taskMap;

      for (auto& [stage, mlir_mod_str]: modules ) {
        std::cout << mlir_mod_str << "\n";
        /*
        auto mod = unwrap(mlir_mod);
        mlir::registerLLVMDialectTranslation(*mod->getContext());

        auto maybeEngine = mlir::ExecutionEngine::create(mod);
        if (!maybeEngine)
          throw std::runtime_error("maybeEngine failed");

        // auto &engine = maybeEngine.get();
        auto engine = std::move(*maybeEngine);
        auto entryPoint = StringRef("top");
        auto expectedFPtr = engine->lookupPacked(entryPoint);
        if (!expectedFPtr)
          throw std::runtime_error("not found entryPoint top");
        */

        auto modArgs = argsMap[stage];
        void** args = (void **) malloc(sizeof(void *) * modArgs.size());;

        size_t index = 0;
        for (auto tensor: modArgs) {
          py::buffer_info buf = tensor.request();
          std::cout << stage << ".buffer #" << index << ": num of items " 
            << buf.size << ", ndim=" << buf.ndim << std::endl;
          args[index] = buf.ptr;
          index++;
        }
        
        MlirContext ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);

        // MlirModule mlir_mod = mlirModuleCreateParse(
        //   ctx, mlirStringRefCreateFromCString(const_cast<char*>(mlir_mod_str.c_str())));
        
        MlirModule mlir_mod = mlirModuleCreateParse(
          ctx, mlirStringRefCreateFromCString(
"module {                                                                    \n"
"  func.func @top(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {     \n"
"    %res = arith.addi %arg0, %arg0 : i32                                        \n"
"    return %res : i32                                                           \n"
"  }                                                                             \n"
"}"));

        lowerModuleToLLVM(ctx, mlir_mod);  
        // lowerHCLToLLVM(mlir_mod, ctx);
        mlirRegisterAllLLVMTranslations(ctx);  

        MlirExecutionEngine jit = mlirExecutionEngineCreate(
          mlir_mod, /*optLevel=*/2, /*numPaths=*/0, /*sharedLibPaths=*/NULL);  
  
        if (mlirExecutionEngineIsNull(jit)) {
          fprintf(stderr, "Execution engine creation failed");
          exit(2);
        }

        // void (*fptr)(void **) = *expectedFPtr;
        // (*fptr)((void**)args);
        // llvm::Error error = engine->invokePacked(entryPoint, 
        //     llvm::MutableArrayRef<void *>{args, (size_t)0});
        // if (error)
        //   return false;

        if (mlirLogicalResultIsFailure(mlirExecutionEngineInvokePacked(
                jit, mlirStringRefCreateFromCString("top"), args))) {
          fprintf(stderr, "Execution engine creation failed");
          abort();
        }

        // Separate JIT execution engine for submodule
        auto task = taskflow.emplace([&]() {
            std::cout << "stage " << stage;
        });
        taskMap[stage] = task;
      }

      executor.run(taskflow).wait(); 


      std::cout << "===================";

      MlirContext ctx = mlirContextCreate();
      registerAllUpstreamDialects(ctx);
      MlirModule module = mlirModuleCreateParse(
        ctx, mlirStringRefCreateFromCString(
"module {                                                                    \n"
"  func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {     \n"
"    %res = arith.addi %arg0, %arg0 : i32                                        \n"
"    return %res : i32                                                           \n"
"  }                                                                             \n"
"}"));
  lowerModuleToLLVM(ctx, module);  
  mlirRegisterAllLLVMTranslations(ctx);  
  MlirExecutionEngine jit = mlirExecutionEngineCreate(
    module, /*optLevel=*/2, /*numPaths=*/0, /*sharedLibPaths=*/NULL);  
  
  if (mlirExecutionEngineIsNull(jit)) {
    fprintf(stderr, "Execution engine creation failed");
    exit(2);
  }

  int input = 42;
  int result = -1;
  void *args[2] = {&input, &result};
  if (mlirLogicalResultIsFailure(mlirExecutionEngineInvokePacked(
          jit, mlirStringRefCreateFromCString("add"), args))) {
    fprintf(stderr, "Execution engine creation failed");
    abort();
  }
  // Input: 42 Result: 84
  // printf("Input: %d Result: %d\n", input, result);
  mlirExecutionEngineDestroy(jit);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return true;
}

//===----------------------------------------------------------------------===//
// HCL Python module definition
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_hcl, m) {
  m.doc() = "HCL Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  // register passes
  hclMlirRegisterAllPasses();

  auto hcl_m = m.def_submodule("hcl");

  // register dialects
  hcl_m.def(
      "register_dialect",
      [](MlirContext context) {
        MlirDialectHandle hcl = mlirGetDialectHandle__hcl__();
        mlirDialectHandleRegisterDialect(hcl, context);
        mlirDialectHandleLoadDialect(hcl, context);
      },
      py::arg("context") = py::none());

  // Declare customized types and attributes
  populateHCLIRTypes(hcl_m);
  populateHCLAttributes(hcl_m);

  // Loop transform APIs.
  hcl_m.def("loop_transformation", &loopTransformation);

  // Codegen APIs.
  hcl_m.def("emit_vhls", &emitVivadoHls);
  hcl_m.def("emit_ihls", &emitIntelHls);

  // LLVM backend APIs.
  hcl_m.def("lower_hcl_to_llvm", &lowerHCLToLLVM);
  hcl_m.def("lower_fixed_to_int", &lowerFixedPointToInteger);
  hcl_m.def("lower_anywidth_int", &lowerAnyWidthInteger);
  hcl_m.def("move_return_to_input", &moveReturnToInput);

  // TaskFlow backend APIs
  hcl_m.def("tf_execute_tasks", &runTaskFlowExecutor);

  // Lowering APIs.
  hcl_m.def("lower_composite_type", &lowerCompositeType);
  hcl_m.def("lower_bit_ops", &lowerBitOps);
  hcl_m.def("legalize_cast", &legalizeCast);
  hcl_m.def("remove_stride_map", &removeStrideMap);
}
