set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRExecutionEngine
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        MLIRIR
        MLIROptLib
        MLIRHeteroCL
        MLIRHCLConversion
        MLIRHCLPasses
        )
add_llvm_executable(hcl-opt hcl-opt.cpp)

llvm_update_compile_flags(hcl-opt)
target_link_libraries(hcl-opt PRIVATE ${LIBS})

if (CUDA_BACKEND)
  target_compile_definitions(hcl-opt PRIVATE CUDA_BACKEND_ENABLED=1)
endif()

mlir_check_all_link_libraries(hcl-opt)
