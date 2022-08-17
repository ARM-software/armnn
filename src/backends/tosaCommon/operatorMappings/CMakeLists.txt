#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnTosaBackendOperators_sources
        AdditionOperator.hpp
        TosaOperatorUtils.hpp
    )

add_library(armnnTosaBackendOperators OBJECT ${armnnTosaBackendOperators_sources})
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
