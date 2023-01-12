#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnTosaBackendOperators_sources
        AvgPool2DIgnoreValueOperator.hpp
        AvgPool2DIgnoreValueOperator.cpp
        ConcatOperator.hpp
        ConcatOperator.cpp
        ConstantOperator.hpp
        ConstantOperator.cpp
        Conv2dOperator.hpp
        Conv2dOperator.cpp
        ElementwiseBinaryOperator.hpp
        ElementwiseBinaryOperator.cpp
        ElementwiseUnaryOperator.cpp
        ElementwiseUnaryOperator.hpp
        Pooling2DOperator.hpp
        Pooling2DOperator.cpp
        ReshapeOperator.hpp
        ReshapeOperator.cpp
        SliceOperator.hpp
        SliceOperator.cpp
        TosaOperatorUtils.hpp
        TransposeConv2dOperator.hpp
        TransposeConv2dOperator.cpp
        TransposeOperator.hpp
        TransposeOperator.cpp
    )

add_library(armnnTosaBackendOperators OBJECT ${armnnTosaBackendOperators_sources})
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
