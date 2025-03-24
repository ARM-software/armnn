#
# Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnTosaBackendOperators_sources
        LeakyReluOperator.hpp
        LeakyReluOperator.cpp
        ReluOperator.hpp
        ReluOperator.cpp
        GeluOperator.hpp
        GeluOperator.cpp
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
        Conv3dOperator.hpp
        Conv3dOperator.cpp
        DepthToSpaceOperator.hpp
        DepthToSpaceOperator.cpp
        DepthwiseConv2dOperator.hpp
        DepthwiseConv2dOperator.cpp
        DequantizeOperator.hpp
        DequantizeOperator.cpp
        ElementwiseBinaryOperator.hpp
        ElementwiseBinaryOperator.cpp
        FullyConnectedOperator.cpp
        FullyConnectedOperator.hpp
        GatherOperator.hpp
        GatherOperator.cpp
        HardSwishOperator.hpp
        HardSwishOperator.cpp
        RsqrtOperator.cpp
        RsqrtOperator.hpp
        ExpOperator.cpp
        ExpOperator.hpp
        LogOperator.cpp
        LogOperator.hpp
        PadOperator.cpp
        PadOperator.hpp
        Pooling2DOperator.hpp
        Pooling2DOperator.cpp
        PReluOperator.hpp
        PReluOperator.cpp
        QuantizeOperator.hpp
        QuantizeOperator.cpp
        ReduceOperator.hpp
        ReduceOperator.cpp
        ReshapeOperator.hpp
        ReshapeOperator.cpp
        ResizeOperator.hpp
        ResizeOperator.cpp
        SigmoidOperator.hpp
        SigmoidOperator.cpp
        SliceOperator.hpp
        SliceOperator.cpp
        SoftmaxOperator.hpp
        SoftmaxOperator.cpp
        StridedSliceOperator.hpp
        StridedSliceOperator.cpp
        SplitOperator.hpp
        SplitOperator.cpp
        StackOperator.hpp
        StackOperator.cpp
        TanhOperator.hpp
        TanhOperator.cpp
        TosaOperatorUtils.hpp
        TosaRescaleOperatorUtils.hpp
        TosaSoftmaxOperatorUtils.hpp
        TransposeConv2dOperator.hpp
        TransposeConv2dOperator.cpp
        TransposeOperator.hpp
        TransposeOperator.cpp
    )

add_library(armnnTosaBackendOperators OBJECT ${armnnTosaBackendOperators_sources})
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnTosaBackendOperators PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
