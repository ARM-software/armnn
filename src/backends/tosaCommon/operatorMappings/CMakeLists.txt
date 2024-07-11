#
# Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnTosaBackendOperators_sources
        LeakyReluOperator.hpp
        LeakyReluOperator.cpp
        ReluOperator.hpp
        ReluOperator.cpp
        AvgPool2DIgnoreValueOperator.hpp
        AvgPool2DIgnoreValueOperator.cpp
        BatchMatMulOperator.hpp
        BatchMatMulOperator.cpp
        ConcatOperator.hpp
        ConcatOperator.cpp
        ConstantOperator.hpp
        ConstantOperator.cpp
        Conv2dOperator.hpp
        Conv2dOperator.cpp
        DepthwiseConv2dOperator.hpp
        DepthwiseConv2dOperator.cpp
        ElementwiseBinaryOperator.hpp
        ElementwiseBinaryOperator.cpp
        ElementwiseUnaryOperator.cpp
        ElementwiseUnaryOperator.hpp
        FullyConnectedOperator.cpp
        FullyConnectedOperator.hpp
        Pooling2DOperator.hpp
        Pooling2DOperator.cpp
        QuantizeOperator.hpp
        QuantizeOperator.cpp
        ReshapeOperator.hpp
        ReshapeOperator.cpp
        ResizeOperator.hpp
        ResizeOperator.cpp
        SliceOperator.hpp
        SliceOperator.cpp
        SplitOperator.hpp
        SplitOperator.cpp
        TosaOperatorUtils.hpp
        TosaRescaleOperatorUtils.hpp
        TransposeConv2dOperator.hpp
        TransposeConv2dOperator.cpp
        TransposeOperator.hpp
        TransposeOperator.cpp
    )

add_library(armnnTosaBackendOperators OBJECT ${armnnTosaBackendOperators_sources})
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
