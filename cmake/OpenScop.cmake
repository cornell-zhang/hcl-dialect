# Install OpenScop as an external project.
#===----------------------------------------------------------------------===//
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# Modified from the Polymer project [https://github.com/kumasento/polymer]
#
#===----------------------------------------------------------------------===//


include(ExternalProject)

ExternalProject_Add(
  osl 
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/openscop"
  CONFIGURE_COMMAND "${CMAKE_SOURCE_DIR}/openscop/autogen.sh" && "${CMAKE_SOURCE_DIR}/openscop/configure" --prefix=${CMAKE_BINARY_DIR}/openscop
  PREFIX ${CMAKE_BINARY_DIR}/openscop
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS "${CMAKE_BINARY_DIR}/openscop/lib/libosl.a"
)

set(OSL_INCLUDE_DIR "${CMAKE_BINARY_DIR}/openscop/include")
set(OSL_LIB_DIR "${CMAKE_BINARY_DIR}/openscop/lib")

add_library(libosl SHARED IMPORTED)
set_target_properties(libosl PROPERTIES IMPORTED_LOCATION "${OSL_LIB_DIR}/libosl.a")
add_dependencies(libosl osl)
