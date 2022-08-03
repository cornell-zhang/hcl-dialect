file(GLOB globbed *.cpp)

add_mlir_library(MLIRHCLSupport
  ${globbed}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRHeteroCL
  MLIRMemRefDialect
  MLIRAnalysis
  MLIRAffineAnalysis
  )
