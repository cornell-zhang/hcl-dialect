file(GLOB globbed *.cpp)

add_mlir_library(MLIRHCLSupport
  ${globbed}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRHeteroCL
  MLIRMemRef
  MLIRAnalysis
  # Needed for the OpenSCoPLib linkup.
  # Can link up other Polyhedral tools as needed
  libosl
  MLIRAffineAnalysis
  )
