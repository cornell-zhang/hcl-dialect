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
  libosl
)
