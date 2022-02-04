file(GLOB globbed *.cpp)

add_mlir_library(MLIRHCLEmitOpenSCoP
  ${globbed}
    
  LINK_LIBS PUBLIC
  MLIRPass
  MLIRIR
  MLIRAffine
  MLIRAffineUtils
  MLIRTranslation
)
