# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)

add_mlir_library(MLIRHCLEmitOpenSCoP
  ${globbed}
    
  ADDITIONAL_HEADER_DIRS
  "${OSL_SOURCE_DIR}/include"

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRIR
  MLIRAffine
  MLIRAffineUtils
  MLIRTranslation
  MLIRParser
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
