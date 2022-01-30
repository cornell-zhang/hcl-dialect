file(GLOB globbed *.cpp)

add_mlir_library(MLIRHCLSupport
  ${globbed}

  DEPENDS
  mlir-headers

  ADDIOTIONAL_HEADERS
  ${HCL_MAIN_INCLUDE_DIR}/hcl/Support

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
  )
