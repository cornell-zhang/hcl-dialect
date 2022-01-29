file(GLOB globbed *.cpp)

add_mlir_library(MLIRHCLEmitOpenSCoP
  ${globbed}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffine
  MLIRAffineUtils
  MLIRTranslation
)
