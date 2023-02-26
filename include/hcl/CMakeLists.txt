# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

if (PYTHON_BINDING)
add_subdirectory(Bindings)
endif()
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
