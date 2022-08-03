file(GLOB globbed *.cpp)

add_mlir_library(MLIRHCLEmitHLSCpp
  ${globbed}

  LINK_LIBS PUBLIC
  MLIRHeteroCL
  MLIRHCLSupport
  MLIRMemRefDialect
  MLIRAnalysis
)