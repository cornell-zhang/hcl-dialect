set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(hcl-translate
  hcl-translate.cpp
  )
llvm_update_compile_flags(hcl-translate)
target_link_libraries(hcl-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  MLIRHCLEmitHLSCpp
  MLIRHCLEmitOpenSCoP
  MLIRHeteroCL
  gmp
  )

mlir_check_link_libraries(hcl-translate)
