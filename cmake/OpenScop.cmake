# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from the Polymer project: https://github.com/kumasento/polymer

include(ExternalProject)

set(OSL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/openscop")
set(OSL_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/openscop/include")
set(OSL_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/openscop/lib")
set(OSL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/openscop")

ExternalProject_Add(
  osl 
  PREFIX ${OSL_BINARY_DIR}
  SOURCE_DIR ${OSL_SOURCE_DIR}
  CONFIGURE_COMMAND "${OSL_SOURCE_DIR}/autogen.sh" && "${OSL_SOURCE_DIR}/configure" --prefix=${OSL_BINARY_DIR}
  BUILD_COMMAND make -j 4
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS "${OSL_LIB_DIR}/libosl.a"
)

add_library(libosl SHARED IMPORTED)
set_target_properties(libosl PROPERTIES IMPORTED_LOCATION "${OSL_LIB_DIR}/libosl.so")
add_dependencies(libosl osl)
