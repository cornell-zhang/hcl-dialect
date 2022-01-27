add_mlir_public_c_api_library(MLIRHCLCAPIHLSCpp
  Dialects.cpp
  Registration.cpp
  HCLTypes.cpp
  HCLAttributes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSupport
  MLIRHeteroCL
  MLIRHCLConversion
  )
