//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/test/EndToEndTestImpl.hpp"

#include "backendsCommon/test/AdditionEndToEndTestImpl.hpp"
#include "backendsCommon/test/Convolution2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/ConcatEndToEndTestImpl.hpp"
#include "backendsCommon/test/ElementwiseBinaryEndToEndTestImpl.hpp"
#include "backendsCommon/test/ElementwiseUnaryEndToEndTestImpl.hpp"
#include "backendsCommon/test/MultiplicationEndToEndTestImpl.hpp"
#include "backendsCommon/test/Pooling2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/QuantizationEndToEndTestImpl.hpp"
#include "backendsCommon/test/ReshapeEndToEndTestImpl.hpp"
#include "backendsCommon/test/ResizeEndToEndTestImpl.hpp"
#include "backendsCommon/test/SliceEndToEndTestImpl.hpp"
#include "backendsCommon/test/SplitterEndToEndTestImpl.hpp"
#include "backendsCommon/test/SubtractionEndToEndTestImpl.hpp"
#include "backendsCommon/test/TransposeConvolution2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/TransposeEndToEndTestImpl.hpp"

#include <doctest/doctest.h>

TEST_SUITE("TosaRefEndToEnd")
{
std::vector<BackendId> tosaDefaultBackends = { "TosaRef" };

// Addition
TEST_CASE("TosaRefAdditionEndtoEndTestFloat32")
{
    AdditionEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefAdditionEndtoEndTestInt32")
{
    AdditionEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefAdditionEndtoEndTestFloat16")
{
    AdditionEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

// Concat
TEST_CASE("TosaRefConcatEndToEndDim0TestFloat32")
{
    ConcatDim0EndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim0TestInt32")
{
    ConcatDim0EndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim1TestFloat32")
{
    ConcatDim1EndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim1TestInt32")
{
    ConcatDim1EndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim2TestFloat32")
{
    ConcatDim2EndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim2TestInt32")
{
    ConcatDim2EndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim3TestFloat32")
{
    ConcatDim3EndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim3TestInt32")
{
    ConcatDim3EndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

// Conv2d
TEST_CASE("TosaRefConv2dEndtoEndTestFloat32")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("TosaRefConv2dWithoutBiasEndtoEndTestFloat32")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC, false);
}

// Average Pool 2D
TEST_CASE("TosaRefAvgPool2DEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefAvgPool2DEndtoEndTestFloat16")
{
    AvgPool2dEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefAvgPool2DIgnoreValueEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(tosaDefaultBackends, PaddingMethod::IgnoreValue);
}

// Maximum
TEST_CASE("TosaRefMaximumEndtoEndTestInt8")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::Signed32>(tosaDefaultBackends,
                                                                 armnn::BinaryOperation::Maximum);
}

// Max Pool 2D
TEST_CASE("TosaRefMaxPool2DEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMaxPool2DEndtoEndTestFloat16")
{
    MaxPool2dEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMaxPool2DIgnoreValueEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(tosaDefaultBackends, PaddingMethod::IgnoreValue);
}

TEST_CASE("TosaRefMaxPool2DTwoLayerEndtoEndTestFloat32")
{
    MaxPool2dTwoLayerEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMaxPool2DThreeLayerEndtoEndTestFloat32")
{
    MaxPool2dThreeLayerEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

// Quantization
TEST_CASE("TosaRefQuantizeFromFloat32ToInt8")
{
    QuantizationEndToEndFloat32<DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat32ToInt16")
{
    QuantizationEndToEndFloat32<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat32ToInt32")
{
    QuantizationEndToEndFloat32<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat16ToInt8")
{
    QuantizationEndToEndFloat16<DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat16ToInt16")
{
    QuantizationEndToEndFloat16<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat16ToInt32")
{
    QuantizationEndToEndFloat16<DataType::Signed32>(tosaDefaultBackends);
}

// Reshape
TEST_CASE("TosaRefReshapeEndtoEndTestFloat32")
{
    ReshapeEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefReshapeEndtoEndTestInt32")
{
    ReshapeEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefReshapeEndtoEndTestFloat16")
{
    ReshapeEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefRsqrtEndtoEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends,
                                                             UnaryOperation::Rsqrt);
}

// Resize
TEST_CASE("TosaRefResizeNearestNeighborEndToEndFloat32AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndFloat32HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndFloat16AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float16>(tosaDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndFloat16HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float16>(tosaDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndInt8AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndInt8HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndInt16AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndInt16HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

// Slice
TEST_CASE("TosaRefSliceEndtoEndTestFloat32")
{
    SliceEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSliceEndtoEndTestInt32")
{
    SliceEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSliceEndtoEndTestFloat16")
{
    SliceEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

// Split
TEST_CASE("TosaRefSplit1dEndtoEndTestBoolean")
{
    Splitter1dEndToEnd<DataType::Boolean>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndtoEndTestInt8")
{
    Splitter1dEndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndtoEndTestSigned16")
{
    Splitter1dEndToEnd<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndtoEndTestInt32")
{
    Splitter1dEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndtoEndTestFloat16")
{
    Splitter1dEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndToEndFloat32")
{
    Splitter1dEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit2dDim0EndtoEndTestFloat32")
{
    Splitter2dDim0EndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit2dDim1EndtoEndTestFloat32")
{
    Splitter2dDim1EndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim0EndtoEndTestFloat32")
{
    Splitter3dDim0EndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestFloat32")
{
    Splitter3dDim1EndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestFloat16")
{
    Splitter3dDim1EndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestBoolean")
{
    Splitter3dDim1EndToEnd<DataType::Boolean>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestInt8")
{
    Splitter3dDim1EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestSigned16")
{
    Splitter3dDim1EndToEnd<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestInt32")
{
    Splitter3dDim1EndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim2EndtoEndTestInt8")
{
    Splitter3dDim2EndToEnd<DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim0EndtoEndTestInt8")
{
    Splitter4dDim0EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim1EndtoEndTestInt8")
{
    Splitter4dDim1EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestBoolean")
{
    Splitter4dDim2EndToEnd<DataType::Boolean>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestInt8")
{
    Splitter4dDim2EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestInt16")
{
    Splitter4dDim2EndToEnd<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestInt32")
{
    Splitter4dDim2EndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestFloat16")
{
    Splitter4dDim2EndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim3EndtoEndTestInt8")
{
    Splitter4dDim3EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

// Subtraction
TEST_CASE("TosaRefSubtractionEndtoEndTestFloat32")
{
    SubtractionEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSubtractionEndtoEndTestInt32")
{
    SubtractionEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSubtractionEndtoEndTestFloat16")
{
    SubtractionEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMultiplicationEndtoEndTestFloat32")
{
    MultiplicationEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMultiplicationEndtoEndTestInt32")
{
    MultiplicationEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMultiplicationEndtoEndTestFloat16")
{
    MultiplicationEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

// TransposeConvolution2d
TEST_CASE("TosaRefTransposeConvolution2dEndToEndFloatNhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        tosaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("TosaRefSimpleTransposeConvolution2dEndToEndFloatNhwcTest")
{
    SimpleTransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        tosaDefaultBackends, armnn::DataLayout::NHWC);
}

// Transpose
TEST_CASE("TosaRefTransposeEndtoEndTestFloat32")
{
    TransposeEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

}