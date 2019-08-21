//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>

#include <Half.hpp>
#include "TensorCopyUtils.hpp"
#include "WorkloadTestUtils.hpp"
#include "TensorUtils.hpp"
#include "Permute.hpp"

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <reference/workloads/Decoders.hpp>
#include <reference/workloads/Encoders.hpp>
#include <test/TensorHelpers.hpp>

#include <boost/multi_array.hpp>
#include <boost/assert.hpp>

#include <array>

// Layer callables.

namespace armnn
{
class IWorkloadFactory;
}

template <std::size_t n>
boost::array<unsigned int, n> GetTensorShapeAsArray(const armnn::TensorInfo& tensorInfo)
{
    BOOST_ASSERT_MSG(n == tensorInfo.GetNumDimensions(),
        "Attempting to construct a shape array of mismatching size");

    boost::array<unsigned int, n> shape;
    for (unsigned int i = 0; i < n; i++)
    {
        shape[i] = tensorInfo.GetShape()[i];
    }
    return shape;
}

template <typename T, std::size_t n>
struct LayerTestResult
{
    LayerTestResult(const armnn::TensorInfo& outputInfo)
    {
        auto shape( GetTensorShapeAsArray<n>(outputInfo) );
        output.resize(shape);
        outputExpected.resize(shape);
        supported = true;
        compareBoolean = false;
    }

    boost::multi_array<T, n> output;
    boost::multi_array<T, n> outputExpected;
    bool supported;
    bool compareBoolean;
};

LayerTestResult<float, 4> SimpleConvolution2d3x5Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleConvolution2d3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleConvolution2d3x3Stride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleConvolution2d3x3NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled);

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout layout);

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout layout);

LayerTestResult<float,   4> Convolution1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled);

LayerTestResult<uint8_t, 4> Convolution1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled);

LayerTestResult<float, 4> DepthwiseConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2d3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2d2x3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2d3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2d2x3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dMult4Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dMult2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> DepthwiseConvolution2dDepthNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled);

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul64Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DepthwiseConvolution2dAsymmetricTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> CompareDepthwiseConvolution2dFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> CompareDepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::DataLayout layout);

LayerTestResult<float,   4> SimpleMaxPooling2dSize2x2Stride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding);

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dSize2x2Stride2x2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding);

LayerTestResult<int16_t, 4> SimpleMaxPooling2dSize2x2Stride2x2Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding);

LayerTestResult<float,   4> SimpleMaxPooling2dSize3x3Stride2x4Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding);

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dSize3x3Stride2x4Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding );

LayerTestResult<int16_t, 4> SimpleMaxPooling2dSize3x3Stride2x4Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding );

LayerTestResult<float,   4> SimpleMaxPooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<uint8_t, 4> SimpleMaxPooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<int16_t, 4> SimpleMaxPooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<float,   4> IgnorePaddingSimpleMaxPooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleMaxPooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> IgnorePaddingSimpleMaxPooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> IgnorePaddingMaxPooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> IgnorePaddingMaxPooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> IgnorePaddingMaxPooling2dSize3Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> SimpleAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<uint8_t, 4> SimpleAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<int16_t, 4> SimpleAveragePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<float,   4> LargeTensorsAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> LargeTensorsAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> LargeTensorsAveragePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> IgnorePaddingAveragePooling2dSize3x2Stride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool forceNoPadding);

LayerTestResult<float,   4> IgnorePaddingSimpleAveragePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleAveragePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> IgnorePaddingSimpleAveragePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4>   IgnorePaddingSimpleAveragePooling2dNoPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> IgnorePaddingAveragePooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> IgnorePaddingAveragePooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> IgnorePaddingAveragePooling2dSize3Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> SimpleL2Pooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<uint8_t, 4> SimpleL2Pooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<int16_t, 4> SimpleL2Pooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout);

LayerTestResult<float,   4> L2Pooling2dSize3Stride1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> L2Pooling2dSize3Stride1Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> L2Pooling2dSize3Stride3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> L2Pooling2dSize3Stride3Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> L2Pooling2dSize3Stride4Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride4Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> L2Pooling2dSize3Stride4Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> L2Pooling2dSize7Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> L2Pooling2dSize7Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> L2Pooling2dSize7Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> L2Pooling2dSize9Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> L2Pooling2dSize9Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> L2Pooling2dSize9Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> IgnorePaddingSimpleL2Pooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> IgnorePaddingSimpleL2Pooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> IgnorePaddingSimpleL2Pooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> IgnorePaddingL2Pooling2dSize3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> IgnorePaddingL2Pooling2dSize3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> IgnorePaddingL2Pooling2dSize3Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,   4> AsymmetricNonSquarePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> AsymmetricNonSquarePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> AsymmetricNonSquarePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> ComparePooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm  poolingType);

LayerTestResult<uint8_t, 4> ComparePooling2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm  poolingType);

LayerTestResult<int16_t, 4> ComparePooling2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::PoolingAlgorithm  poolingType);

LayerTestResult<float, 4> ConstantLinearActivationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SimpleNormalizationAcrossTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SimpleNormalizationWithinTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float,4> SimpleNormalizationAcrossNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> SimpleSoftmaxTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta);

LayerTestResult<float, 2> SimpleAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta,
        int axis);

LayerTestResult<float, 3> Simple3dSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta);

LayerTestResult<float, 3> Simple3dAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta,
        int axis);

LayerTestResult<float, 4> Simple4dSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta);

LayerTestResult<float, 4> Simple4dAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta,
        int axis);

LayerTestResult<uint8_t, 2> SimpleSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta);

LayerTestResult<uint8_t,3> Simple3dSoftmaxUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta);

LayerTestResult<uint8_t,4> Simple4dSoftmaxUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta);

LayerTestResult<int16_t,2> SimpleSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta);

LayerTestResult<int16_t,3> Simple3dSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta);

LayerTestResult<int16_t,4> Simple4dSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta);

LayerTestResult<float, 4> SimpleSigmoidTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleReshapeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> Reshape5dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SimpleFloorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 1> Concatenation1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> Concatenation2dDim0Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> Concatenation2dDim1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> Concatenation2dDim0DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> Concatenation2dDim1DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Concatenation3dDim0Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Concatenation3dDim1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Concatenation3dDim2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor);

LayerTestResult<float, 3> Concatenation3dDim0DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Concatenation3dDim1DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Concatenation3dDim2DiffInputDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor);

LayerTestResult<float, 4> Concatenation4dDim0Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> Concatenation4dDim1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> Concatenation4dDim2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> Concatenation4dDim3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor);

LayerTestResult<float, 4> Concatenation4dDiffShapeDim0Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> Concatenation4dDiffShapeDim1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> Concatenation4dDiffShapeDim2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> Concatenation4dDiffShapeDim3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor);

LayerTestResult<uint8_t, 4> Concatenation4dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Concatenation4dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Concatenation4dDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Concatenation4dDim3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor);

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Concatenation4dDiffShapeDim3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor);

LayerTestResult<uint8_t, 4> SimpleSigmoidUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SimpleSigmoidInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> CompareConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory);

template<typename T>
LayerTestResult<T, 4> CompareDepthwiseConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> CompareNormalizationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::NormalizationAlgorithmChannel normChannel,
    armnn::NormalizationAlgorithmMethod normMethod);

LayerTestResult<float, 2> CompareSoftmaxTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    float beta);

LayerTestResult<float, 2> FullyConnectedFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    bool transposeWeights);

std::vector<LayerTestResult<float, 3>> SplitterTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> CopyViaSplitterTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> ConcatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> AdditionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 5> Addition5dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> AdditionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> AdditionBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> CompareAdditionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory);

LayerTestResult<float, 4> SubtractionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SubtractionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SubtractionBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> CompareActivationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::ActivationFunction f,
    unsigned int batchSize);

LayerTestResult<float, 4> DivisionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DivisionByZeroTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DivisionBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DivisionBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> MultiplicationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 5> Multiplication5dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> MultiplicationBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> MultiplicationBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> CompareMultiplicationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory);

LayerTestResult<float, 4> BatchNormTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchNormNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> CompareBatchNormTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory);

LayerTestResult<float, 4> BoundedReLuUpperAndLowerBoundTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BoundedReLuUint8UpperAndLowerBoundTest(
    armnn::IWorkloadFactory& workloadFactor,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManagery);

LayerTestResult<float, 4> BoundedReLuUpperBoundOnlyTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BoundedReLuUint8UpperBoundOnlyTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> CompareBoundedReLuTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    float upperBound,
    float lowerBound);

LayerTestResult<float, 4> ReLuTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> ReLuUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> ReLuInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> BoundedReLuInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SoftReLuTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SoftReLuUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SoftReLuInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> LeakyReLuTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> LeakyReLuUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> LeakyReLuInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> AbsTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> AbsUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> AbsInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SqrtTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SqrtUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SqrtInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SquareTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SquareUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SquareInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> TanhTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> TanhUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> TanhInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);


// Tests that the output should be identical to the input when the output dimensions match the input ones.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeBilinearNopTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout);

// Tests the behaviour of the resize bilinear operation when rescaling a 2x2 image into a 1x1 image.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleResizeBilinearTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout);

// Tests the resize bilinear for minification of a square input matrix (also: input dimensions are a
// multiple of output dimensions).
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeBilinearSqMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout);

// Tests the resize bilinear for minification (output dimensions smaller than input dimensions).
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeBilinearMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout);

// Tests the resize bilinear for magnification (output dimensions bigger than input dimensions).
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeBilinearMagTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout  dataLayout);

// Tests that the output should be identical to the input when the output dimensions match the input ones.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeNearestNeighborNopTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout);

// Tests the behaviour of the resize NearestNeighbor operation when rescaling a 2x2 image into a 1x1 image.
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleResizeNearestNeighborTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout);

// Tests the resize NearestNeighbor for minification of a square input matrix (also: input dimensions are a
// multiple of output dimensions).
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeNearestNeighborSqMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout);

// Tests the resize NearestNeighbor for minification (output dimensions smaller than input dimensions).
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeNearestNeighborMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout);

// Tests the resize NearestNeighbor for magnification (output dimensions bigger than input dimensions).
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeNearestNeighborMagTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout  dataLayout,
        float inQuantScale,
        int32_t inQuantOffset,
        float outQuantScale,
        int32_t outQuantOffset);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Rsqrt2dTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::TensorInfo inputTensorInfo,
        const armnn::TensorInfo outputTensorInfo,
        const std::vector<float>& inputValues,
        const std::vector<float>& expectedOutputValues);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Rsqrt2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Rsqrt3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> RsqrtZeroTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> RsqrtNegativeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchNormTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchNormNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> FakeQuantizationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> L2NormalizationDefaultEpsilonTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout layout);

LayerTestResult<float, 4> L2NormalizationNonDefaultEpsilonTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout layout);

LayerTestResult<float, 4> L2Normalization1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> L2Normalization1dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> L2Normalization1dUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout layout);

LayerTestResult<float, 4> L2Normalization2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> L2Normalization2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> L2Normalization2dUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout layout);

LayerTestResult<float, 2> L2Normalization2dShapeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> L2Normalization3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> L2Normalization3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> L2Normalization3dUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout layout);

LayerTestResult<float, 4> L2Normalization4dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> L2Normalization4dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> L2Normalization4dUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout layout);

LayerTestResult<float, 4> ConstantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> ConstantUint8SimpleQuantizationScaleNoOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> ConstantInt16SimpleQuantizationScaleNoOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BoundedReLuUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float upperBound);

LayerTestResult<uint8_t, 4> BoundedReLuUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float upperBound,
    float lowerBound);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> FullyConnectedTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled);

std::vector<LayerTestResult<uint8_t, 3>> SplitterUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

std::vector<LayerTestResult<int16_t, 3>> SplitterInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> CopyViaSplitterUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 3> CopyViaSplitterInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> ConcatUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint16_t, 3> ConcatUint16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> ConcatUint8DifferentQParamsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> AdditionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> AdditionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> AdditionBroadcastUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> AdditionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> AdditionBroadcastInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> AdditionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SubtractionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SubtractionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SubtractionBroadcastUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SubtractionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SubtractionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SubtractionBroadcastInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> CompareActivationUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    armnn::ActivationFunction f);

LayerTestResult<int16_t, 4> CompareActivationInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        armnn::IWorkloadFactory& refWorkloadFactory,
        armnn::ActivationFunction f);

LayerTestResult<uint8_t, 2> CompareSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    float beta);

LayerTestResult<uint8_t, 4> MultiplicationUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> MultiplicationBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> MultiplicationBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> MultiplicationInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> MultiplicationBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> MultiplicationBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> DivisionUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> DivisionBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> DivisionBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> DivisionInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> DivisionBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> DivisionBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x5Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> SimpleConvolution2d3x5QSymm16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> SimpleConvolution2d3x3QSymm16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool biasEnabled,
        const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dDepthMul1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> DepthwiseConvolution2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> DepthwiseConvolution2dDepthMul1Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> ConstantLinearActivationUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> ConstantLinearActivationInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchNormUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchNormUint8NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> BatchNormInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> BatchNormInt16NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> ConstantUint8CustomQuantizationScaleAndOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> ConstantInt16CustomQuantizationScaleAndOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 1> Concatenation1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> Concatenation2dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> Concatenation2dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> Concatenation2dDim0DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> Concatenation2dDim1DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Concatenation3dDim0Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Concatenation3dDim1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Concatenation3dDim2Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor);

LayerTestResult<uint8_t, 3> Concatenation3dDim0DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Concatenation3dDim1DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Concatenation3dDim2DiffInputDimsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool useSubtensor);

LayerTestResult<uint8_t, 4> EqualSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> EqualBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> EqualBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> EqualUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> EqualBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> EqualBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> GreaterSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> GreaterBroadcast1DVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> GreaterUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> GreaterBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> FullyConnectedLargeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool transposeWeights);

LayerTestResult<uint8_t, 2> PadUint82dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> PadUint82dCustomPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> PadUint83dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> PadUint84dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> PadFloat322dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> PadFloat322dCustomPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> PadFloat323dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> PadFloat324dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Pad2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset,
    const float customPaddingValue = 0.0f);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Pad3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Pad4dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset);

void LstmUtilsZeroVectorTest();
void LstmUtilsMeanStddevNormalizationNoneZeroInputTest();
void LstmUtilsMeanStddevNormalizationAllZeroInputTest();
void LstmUtilsMeanStddevNormalizationMixedZeroInputTest();
void LstmUtilsVectorBatchVectorCwiseProductTest();
void LstmUtilsVectorBatchVectorAddTest();

LayerTestResult<float, 2> LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgNoPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> LstmLayerInt16WithCifgWithPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgWithPeepholeWithProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16ConstantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

// QuantizedLstm
LayerTestResult<uint8_t, 2> QuantizedLstmTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SimpleConvertFp16ToFp32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<armnn::Half, 4> SimpleConvertFp32ToFp16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> MaximumSimpleTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> MaximumBroadcast1ElementTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> MaximumBroadcast1DVectorTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t , 4> MaximumUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> MaximumBroadcast1ElementUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> MaximumBroadcast1DVectorUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t , 4> MaximumInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> MaximumBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> MaximumBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> MeanSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> MeanSimpleAxisTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> MeanKeepDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> MeanMultipleDimsTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> MeanVts1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> MeanVts2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> MeanVts3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> MinimumBroadcast1ElementTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> MinimumBroadcast1ElementTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> MinimumBroadcast1DVectorUint8Test(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager);

LayerTestResult<int16_t , 4> MinimumInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> MinimumBroadcast1ElementInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> MinimumBroadcast1DVectorInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> AdditionAfterMaxPoolTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToBatchNdSimpleFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToBatchNdMultiChannelsFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToBatchNdMultiBlockFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToBatchNdPaddingFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToBatchNdSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiChannelsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiBlockUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToBatchNdPaddingUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToBatchNdSimpleNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToBatchNdMultiChannelsNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToBatchNdMultiBlockNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToBatchNdPaddingNHWCFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToBatchNdSimpleNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiChannelsNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToBatchNdMultiBlockNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToBatchNdPaddingNHWCUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToBatchNdSimpleUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToBatchNdMultiChannelsUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToBatchNdMultiBlockUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToBatchNdPaddingUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToBatchNdSimpleNHWCUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToBatchNdMultiChannelsNHWCUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToBatchNdMultiBlockNHWCUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToBatchNdPaddingNHWCUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchToSpaceNdNhwcTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchToSpaceNdNhwcTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchToSpaceNdNhwcTest3(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchToSpaceNdNhwcTest4(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchToSpaceNdNchwTest1(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchToSpaceNdNchwTest2(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> BatchToSpaceNdNchwTest3(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchToSpaceNdNhwcTest5(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchToSpaceNdNhwcTest6(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchToSpaceNdNhwcTest7(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchToSpaceNdNchwTest4(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchToSpaceNdNchwTest5(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchToSpaceNdNchwTest6(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> BatchToSpaceNdNchwTest7(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> StridedSlice4DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> StridedSlice4DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> StridedSliceSimpleStrideFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> StridedSliceSimpleRangeMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> StridedSlice3DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> StridedSlice3DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> StridedSlice2DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> StridedSlice2DReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> StridedSlice4DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> StridedSlice4DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> StridedSliceSimpleStrideUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> StridedSliceSimpleRangeMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> StridedSlice3DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> StridedSlice3DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> StridedSlice2DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> StridedSlice2DReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> StridedSlice4DInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> StridedSlice4DReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> StridedSliceSimpleStrideInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> StridedSliceSimpleRangeMaskInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> StridedSliceShrinkAxisMaskInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 3> StridedSlice3DInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 3> StridedSlice3DReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> StridedSlice2DInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> StridedSlice2DReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> Debug4DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Debug3DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> Debug2DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 1> Debug1DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Debug4DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Debug3DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> Debug2DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 1> Debug1DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> PreCompiledConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> PreCompiledConvolution2dStride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> PreCompiledDepthwiseConvolution2dTest(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager);

LayerTestResult<uint8_t, 4> PreCompiledDepthwiseConvolution2dStride2x2Test(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager);

LayerTestResult<uint8_t, 4> PreCompiledMaxPooling2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> Debug4DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Debug3DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> Debug2DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 1> Debug1DFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Debug4DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Debug3DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> Debug2DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 1> Debug1DUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 1> Gather1DParamsFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 1> Gather1DParamsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 1> Gather1DParamsInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> GatherMultiDimParamsFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> GatherMultiDimParamsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> GatherMultiDimParamsInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> GatherMultiDimParamsMultiDimIndicesFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> GatherMultiDimParamsMultiDimIndicesUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> GatherMultiDimParamsMultiDimIndicesInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DequantizeSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DequantizeOffsetUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DequantizeSimpleInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToDepthNCHWAsymmQ8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> SpaceToDepthNHWCAsymmQ8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToDepthNHWCFloat32Test1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToDepthNCHWFloat32Test1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToDepthNHWCFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> SpaceToDepthNCHWFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToDepthNHWCQSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> SpaceToDepthNCHWQSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> QuantizeSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> QuantizeClampUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> QuantizeClampInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> ConvertToDataType(const std::vector<float>& input,
                                 const armnn::TensorInfo& inputTensorInfo)
{
    std::vector<T> output(input.size());
    auto outputTensorInfo = inputTensorInfo;
    outputTensorInfo.SetDataType(ArmnnType);

    std::unique_ptr<armnn::Encoder<float>> pOutputEncoder = armnn::MakeEncoder<float>(outputTensorInfo, output.data());
    armnn::Encoder<float>& rOutputEncoder = *pOutputEncoder;

    for (auto it = input.begin(); it != input.end(); ++it)
    {
        rOutputEncoder.Set(*it);
        ++rOutputEncoder;
    }
    return output;
}

// Utility method to convert a single value to the correct type
template <typename T>
T ConvertToDataType(const float& value,
                    const armnn::TensorInfo& tensorInfo)
{
    std::vector<T> output(1);
    std::unique_ptr<armnn::Encoder<float>> pEncoder = armnn::MakeEncoder<float>(tensorInfo, output.data());
    armnn::Encoder<float>& rEncoder = *pEncoder;
    rEncoder.Set(value);
    return output[0];
}

template<typename T, typename B>
LayerTestResult<T, 2> SimpleFullyConnectedTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        armnn::TensorInfo inputTensorInfo,
        armnn::TensorInfo outputTensorInfo,
        armnn::TensorInfo weightsDesc,
        armnn::TensorInfo biasesDesc,
        boost::multi_array<T, 2>& weights,
        boost::multi_array<B, 1>& bias,
        boost::multi_array<T, 4>& input,
        bool biasEnabled,
        bool transposeWeights)
{
    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::FullyConnectedQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle weightsTensor(weightsDesc);
    armnn::ScopedCpuTensorHandle biasTensor(biasesDesc);

    AllocateAndCopyDataToITensorHandle(&weightsTensor, &weights[0][0]);
    AllocateAndCopyDataToITensorHandle(&biasTensor, &bias[0]);

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_TransposeWeightMatrix = transposeWeights;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateFullyConnected(data, info);
    LayerTestResult<T, 2> result(outputTensorInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(&result.output[0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> FullyConnectedTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool biasEnabled)
{
    constexpr static unsigned int inputWidth = 3u;
    constexpr static unsigned int inputHeight = 2u;
    constexpr static unsigned int inputChannels = 1u;

    constexpr static unsigned int inputSize = inputWidth * inputHeight * inputChannels;

    constexpr static unsigned int outputChannels = 2u;

    armnn::TensorInfo inputTensorInfo({ 1, inputChannels, inputHeight, inputWidth }, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(63);

    armnn::TensorInfo outputTensorInfo({ 1, outputChannels }, ArmnnType);
    outputTensorInfo.SetQuantizationScale(5.f);
    outputTensorInfo.SetQuantizationOffset(biasEnabled ? -50 : 10);

    armnn::TensorInfo weightsDesc({ outputChannels, inputSize }, ArmnnType);
    weightsDesc.SetQuantizationScale(0.2f);
    weightsDesc.SetQuantizationOffset(93);

    armnn::TensorInfo biasesDesc({ outputChannels }, GetBiasTypeFromWeightsType(weightsDesc.GetDataType()).value());
    biasesDesc.SetQuantizationScale(inputTensorInfo.GetQuantizationScale() * weightsDesc.GetQuantizationScale());
    biasesDesc.SetQuantizationOffset(0);

    LayerTestResult<T, 2> result(outputTensorInfo);

    auto input = MakeTensor<T, 4>(inputTensorInfo, ConvertToDataType<ArmnnType>(
        {
            -1.2f, 6.1f, -3.5f,
            18.8f, -5.5f, 2.9f
        },
        inputTensorInfo));

    auto weights = MakeTensor<T, 2>(weightsDesc, ConvertToDataType<ArmnnType>(
        {
            -8.4f, 20.0f, -10.4f, -8, 16.4f, -11.8f,
            23.4f, 10.4f, -14.0f, -3.8f, -11.8f, 11.4f
        },
        weightsDesc));

    auto bias = MakeTensor<int32_t, 1>(biasesDesc, std::vector<int32_t>{9250, 67500});

    result = SimpleFullyConnectedTestImpl<T>(
            workloadFactory,
            memoryManager,
            inputTensorInfo, outputTensorInfo,
            weightsDesc, biasesDesc,
            weights, bias, input,
            biasEnabled, true
    );

    if (biasEnabled)
    {
        result.outputExpected = MakeTensor<T, 2>(outputTensorInfo,
                                                 ConvertToDataType<ArmnnType>({80.f, 1460.f}, outputTensorInfo));
    }
    else
    {
        result.outputExpected = MakeTensor<T, 2>(outputTensorInfo,
                                                 ConvertToDataType<ArmnnType>({-107.04f, 110.f}, outputTensorInfo));
    }

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Rsqrt2dTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::TensorInfo inputTensorInfo,
        const armnn::TensorInfo outputTensorInfo,
        const std::vector<float>& inputValues,
        const std::vector<float>& expectedOutputValues)
{
    auto inputTensor = MakeTensor<T, 2>(inputTensorInfo, ConvertToDataType<ArmnnType>(inputValues,inputTensorInfo));

    LayerTestResult<T, 2> result(outputTensorInfo);

    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo,
                                             ConvertToDataType<ArmnnType>(expectedOutputValues,outputTensorInfo));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::RsqrtQueueDescriptor descriptor;

    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateRsqrt(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Rsqrt2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const armnn::TensorShape inputShape{ 2, 2 };
    const armnn::TensorShape outputShape{ 2, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(0);

    armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);
    outputTensorInfo.SetQuantizationOffset(0);

    std::vector<float> inputValues
    {
        1.f, 4.f,
        16.f, 25.f
    };

    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f,
        0.25f, 0.2f
    };

    return Rsqrt2dTestCommon<ArmnnType>(workloadFactory, memoryManager,
                                inputTensorInfo, outputTensorInfo,
                                inputValues, expectedOutputValues);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Rsqrt3dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const armnn::TensorShape inputShape{ 3, 1, 2 };
    const armnn::TensorShape outputShape{ 3, 1, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(0);

    armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);
    outputTensorInfo.SetQuantizationOffset(0);

    std::vector<float> inputValues
    {
        1.f, 4.f, 16.f,
        25.f, 64.f, 100.f
    };

    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f, 0.25f,
        0.2f, 0.125f, 0.1f
    };

    auto inputTensor = MakeTensor<T, 3>(inputTensorInfo, ConvertToDataType<ArmnnType>(inputValues,inputTensorInfo));

    LayerTestResult<T, 3> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo,
                                             ConvertToDataType<ArmnnType>(expectedOutputValues,outputTensorInfo));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::RsqrtQueueDescriptor descriptor;

    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateRsqrt(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> RsqrtZeroTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const armnn::TensorShape inputShape{ 1, 2 };
    const armnn::TensorShape outputShape{ 1, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);

    armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);

    std::vector<float> inputValues
    {
        0.f, -0.f
    };

    std::vector<float> expectedOutputValues
    {
        INFINITY, -INFINITY
    };

    return Rsqrt2dTestCommon<ArmnnType>(workloadFactory, memoryManager,
                                inputTensorInfo, outputTensorInfo,
                                inputValues, expectedOutputValues);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> RsqrtNegativeTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const armnn::TensorShape inputShape{ 1, 2 };
    const armnn::TensorShape outputShape{ 1, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(0);

    armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);
    outputTensorInfo.SetQuantizationOffset(0);

    std::vector<float> inputValues
    {
        -25.f, -16.f
    };

    std::vector<float> expectedOutputValues
    {
        -NAN, -NAN
    };

    return Rsqrt2dTestCommon<ArmnnType>(workloadFactory, memoryManager,
                                inputTensorInfo, outputTensorInfo,
                                inputValues, expectedOutputValues);
}

template<typename T, size_t NumDims>
LayerTestResult<T, NumDims> SimpleReshapeTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::TensorInfo inputTensorInfo,
    armnn::TensorInfo outputTensorInfo,
    const std::vector<T>& inputData,
    const std::vector<T>& outputExpectedData)
{
    auto input = MakeTensor<T, NumDims>(inputTensorInfo, inputData);

    LayerTestResult<T, NumDims> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, NumDims>(outputTensorInfo, outputExpectedData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ReshapeQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateReshape(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.origin());

    workload->Execute();

    CopyDataFromITensorHandle(ret.output.origin(), outputHandle.get());

    return ret;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> SimpleReshapeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 2, 2, 3, 3 };
    unsigned int outputShape[] = { 2, 2, 9, 1 };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(1.0f);

    auto input = ConvertToDataType<ArmnnType>(
        {
            0.0f, 1.0f, 2.0f,
            3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f,

            9.0f, 10.0f, 11.0f,
            12.0f, 13.0f, 14.0f,
            15.0f, 16.0f, 17.0f,

            18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f,

            27.0f, 28.0f, 29.0f,
            30.0f, 31.0f, 32.0f,
            33.0f, 34.0f, 35.0f,
        },
        inputTensorInfo);

    auto outputExpected = ConvertToDataType<ArmnnType>(
        {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,

            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,

            18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f,

            27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
        },
        outputTensorInfo);

    return SimpleReshapeTestImpl<T, 4>(
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 5> Reshape5dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 2, 2, 8, 1, 1 };
    unsigned int outputShape[] = { 2, 2, 2, 2, 2 };

    inputTensorInfo = armnn::TensorInfo(5, inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(5, outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(1.0f);

    auto input = ConvertToDataType<ArmnnType>(
        {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,

            16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
        },
        inputTensorInfo);

    auto outputExpected = ConvertToDataType<ArmnnType>(
        {
            0.0f, 1.0f,
            2.0f, 3.0f,

            4.0f, 5.0f,
            6.0f, 7.0f,


            8.0f, 9.0f,
            10.0f, 11.0f,

            12.0f, 13.0f,
            14.0f, 15.0f,



            16.0f, 17.0f,
            18.0f, 19.0f,

            20.0f, 21.0f,
            22.0f, 23.0f,


            24.0f, 25.0f,
            26.0f, 27.0f,

            28.0f, 29.0f,
            30.0f, 31.0f,
        },
        outputTensorInfo);

    return SimpleReshapeTestImpl<T, 5>(
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleFloorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo({1, 3, 2, 3}, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);

    armnn::TensorInfo outputTensorInfo(inputTensorInfo);
    outputTensorInfo.SetQuantizationScale(0.1f);

    auto input = MakeTensor<T, 4>(inputTensorInfo, ConvertToDataType<ArmnnType>(
        { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f,
        1.0f, 0.4f, 0.5f, 1.3f, 1.5f, 2.0f, 8.76f, 15.2f, 37.5f },
        inputTensorInfo));

    LayerTestResult<T, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, ConvertToDataType<ArmnnType>(
        { -38.0f, -16.0f, -9.0f, -2.0f, -2.0f, -2.0f, -1.0f, -1.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 8.0f, 15.0f, 37.0f },
        outputTensorInfo));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::FloorQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateFloor(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}


template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeBilinearNopTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.5f);
        inputTensorInfo.SetQuantizationOffset(-3);
        outputTensorInfo.SetQuantizationScale(1.5f);
        outputTensorInfo.SetQuantizationOffset(-3);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                     {
                                             1, 2, 3, 4,
                                             2, 3, 4, 5,
                                             3, 4, 5, 6,
                                             4, 5, 6, 7
                                     }
                                   : std::initializer_list<float>
                                     {
                                             1.0f, 2.0f, 3.0f, 4.0f,
                                             2.0f, 3.0f, 4.0f, 5.0f,
                                             3.0f, 4.0f, 5.0f, 6.0f,
                                             4.0f, 5.0f, 6.0f, 7.0f,

                                             1.0f, 2.0f, 3.0f, 4.0f,
                                             2.0f, 3.0f, 4.0f, 5.0f,
                                             3.0f, 4.0f, 5.0f, 6.0f,
                                             4.0f, 5.0f, 6.0f, 7.0f
                                     };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = input;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> SimpleResizeBilinearTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 2, 2, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 1, 1, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 1, 1, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(0.1567f);
        inputTensorInfo.SetQuantizationOffset(1);
        outputTensorInfo.SetQuantizationScale(0.1567f);
        outputTensorInfo.SetQuantizationOffset(1);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                     {
                                             1, 255,
                                             200, 250
                                     }
                                   : std::initializer_list<float>
                                     {
                                             1.0f, 255.0f,
                                             200.0f, 250.0f,

                                             250.0f, 200.0f,
                                             250.0f,   1.0f
                                     };

    // The 'resize bilinear' operation projects the top-left corner of output texels into the input image,
    // then figures out the interpolants and weights. Note this is different to projecting the centre of the
    // output texel. Thus, for a input matrix of 2x2, we'll expect the output 1x1 matrix to contain, as
    // its single element, the value that was at position (0,0) of the input matrix (rather than an average,
    // which we would expect if projecting the centre).

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
                                    ? std::initializer_list<float>
                                      {
                                              1
                                      }
                                    : std::initializer_list<float>
                                      {
                                              1.0f,

                                              250.0f
                                      };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeBilinearSqMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 2, 2, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(3.141592f);
        inputTensorInfo.SetQuantizationOffset(3);
        outputTensorInfo.SetQuantizationScale(3.141592f);
        outputTensorInfo.SetQuantizationOffset(3);
    }

        std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                       ? std::initializer_list<float>
                                         {
                                                1, 2, 3, 4,
                                                2, 3, 4, 5,
                                                3, 4, 5, 6,
                                                4, 5, 6, 7
                                         }
                                       : std::initializer_list<float>
                                         {
                                                1.0f, 2.0f, 3.0f, 4.0f,
                                                2.0f, 3.0f, 4.0f, 5.0f,
                                                3.0f, 4.0f, 5.0f, 6.0f,
                                                4.0f, 5.0f, 6.0f, 7.0f,

                                                7.0f, 6.0f, 5.0f, 4.0f,
                                                6.0f, 5.0f, 4.0f, 3.0f,
                                                5.0f, 4.0f, 3.0f, 2.0f,
                                                4.0f, 3.0f, 2.0f, 1.0f
                                         };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
                                    ? std::initializer_list<float>
                                      {
                                              1, 3,
                                              3, 5
                                      }
                                    : std::initializer_list<float>
                                      {
                                              1.0f, 3.0f,
                                              3.0f, 5.0f,

                                              7.0f, 5.0f,
                                              5.0f, 3.0f
                                      };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeBilinearMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 2, 3, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 3, 5, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 1, 2, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 2, 3, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.5f);
        inputTensorInfo.SetQuantizationOffset(-1);
        outputTensorInfo.SetQuantizationScale(1.5f);
        outputTensorInfo.SetQuantizationOffset(-1);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                     {
                                             3.0f, 4.5f, 6.0f, // 1,  2,  3, : Expected quantised values
                                             9.0f, 13.5f, 21.0f // 5,  8, 13
                                     }
                                   : std::initializer_list<float>
                                     {
                                             1.0f, 2.0f, 3.0f, 5.0f, 8.0f,
                                             13.0f, 21.0f, 34.0f, 55.0f, 89.0f,
                                             144.0f, 233.0f, 377.0f, 610.0f, 987.0f,

                                             987.0f, 610.0f, 377.0f, 233.0f, 144.0f,
                                             89.0f, 55.0f, 34.0f, 21.0f, 13.0f,
                                             8.0f, 5.0f, 3.0f, 2.0f, 1.0f
                                     };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
                                    ? std::initializer_list<float>
                                      {
                                              3.0f, 5.25f // 1, 3
                                      }
                                    : std::initializer_list<float>
                                      {
                                              1.0f,   2.6666f,   6.00f,
                                              78.5f, 179.3333f, 401.00f,

                                              987.0f, 454.6670f, 203.33f,
                                              48.5f,  22.3333f,  10.00f
                                      };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeBilinearMagTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 3, 2, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 3, 2, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 3, 5, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 3, 5, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(0.010765f);
        inputTensorInfo.SetQuantizationOffset(7);
        outputTensorInfo.SetQuantizationScale(0.010132f);
        outputTensorInfo.SetQuantizationOffset(-18);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                     {
                                             0.183005f, 2.379065f, // 24, 228, : Expected quantised values
                                             1.05497f, 1.302565f, // 105, 128,
                                             2.400595f, 0.68896f // 230, 71
                                     }
                                   : std::initializer_list<float>
                                     {
                                             1.0f,   2.0f,
                                             13.0f,  21.0f,
                                             144.0f, 233.0f,

                                             233.0f, 144.0f,
                                             21.0f,  13.0f,
                                             2.0f,   1.0f
                                     };
    std::vector<float> outputData = armnn::IsQuantizedType<T>()
                                    ? std::initializer_list<float>
                                      {
                                              0.18300501f, 1.06142902f, 1.93985295f, 2.37906504f, 2.37906504f,
                                              1.05497003f, 1.15400803f, 1.25304604f, 1.30256498f, 1.30256498f,
                                              2.40059495f, 1.71594095f, 1.03128707f, 0.68896002f, 0.68896002f
                                              // 0, 87, 173, 217, 217, : Expected quantised values
                                              // 86, 96, 106, 111, 111,
                                              // 219, 151, 84, 50, 50
                                      }
                                    : std::initializer_list<float>
                                      {
                                              1.0f,   1.4f,   1.8f,   2.0f,   2.0f,
                                              13.0f,  16.2f,  19.4f,  21.0f,  21.0f,
                                              144.0f, 179.6f, 215.2f, 233.0f, 233.0f,

                                              233.0f, 197.4f, 161.8f, 144.0f, 144.0f,
                                              21.0f,  17.8f,  14.6f,  13.0f,  13.0f,
                                              2.0f,   1.6f,   1.2f,   1.0f,   1.0f
                                      };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}


template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeNearestNeighborNopTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.5f);
        inputTensorInfo.SetQuantizationOffset(-3);
        outputTensorInfo.SetQuantizationScale(1.5f);
        outputTensorInfo.SetQuantizationOffset(-3);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                           {
                                                   1, 2, 3, 4,
                                                   2, 3, 4, 5,
                                                   3, 4, 5, 6,
                                                   4, 5, 6, 7
                                           }
                                   : std::initializer_list<float>
                                           {
                                                   1.0f, 2.0f, 3.0f, 4.0f,
                                                   2.0f, 3.0f, 4.0f, 5.0f,
                                                   3.0f, 4.0f, 5.0f, 6.0f,
                                                   4.0f, 5.0f, 6.0f, 7.0f,

                                                   1.0f, 2.0f, 3.0f, 4.0f,
                                                   2.0f, 3.0f, 4.0f, 5.0f,
                                                   3.0f, 4.0f, 5.0f, 6.0f,
                                                   4.0f, 5.0f, 6.0f, 7.0f
                                           };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = input;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method = armnn::ResizeMethod::NearestNeighbor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> SimpleResizeNearestNeighborTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 2, 2, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 1, 1, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 1, 1, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(0.1567f);
        inputTensorInfo.SetQuantizationOffset(1);
        outputTensorInfo.SetQuantizationScale(0.1567f);
        outputTensorInfo.SetQuantizationOffset(1);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                           {
                                                   1, 255,
                                                   200, 250
                                           }
                                   : std::initializer_list<float>
                                           {
                                                   1.0f, 255.0f,
                                                   200.0f, 250.0f,

                                                   250.0f, 200.0f,
                                                   250.0f,   1.0f
                                           };

    // The 'resize' operation projects the top-left corner of output texels into the input image,
    // then figures out the interpolants and weights. Note this is different to projecting the centre of the
    // output texel. Thus, for a input matrix of 2x2, we'll expect the output 1x1 matrix to contain, as
    // its single element, the value that was at position (0,0) of the input matrix (rather than an average,
    // which we would expect if projecting the centre).

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
                                    ? std::initializer_list<float>
                                            {
                                                    1
                                            }
                                    : std::initializer_list<float>
                                            {
                                                    1.0f,

                                                    250.0f
                                            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    descriptor.m_Parameters.m_Method = armnn::ResizeMethod::NearestNeighbor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeNearestNeighborSqMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 2, 2, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(3.141592f);
        inputTensorInfo.SetQuantizationOffset(3);
        outputTensorInfo.SetQuantizationScale(3.141592f);
        outputTensorInfo.SetQuantizationOffset(3);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                           {
                                                  1, 2, 3, 4,
                                                  2, 3, 4, 5,
                                                  3, 4, 5, 6,
                                                  4, 5, 6, 7
                                           }
                                   : std::initializer_list<float>
                                           {
                                                   1.0f, 2.0f, 3.0f, 4.0f,
                                                   2.0f, 3.0f, 4.0f, 5.0f,
                                                   3.0f, 4.0f, 5.0f, 6.0f,
                                                   4.0f, 5.0f, 6.0f, 7.0f,

                                                   7.0f, 6.0f, 5.0f, 4.0f,
                                                   6.0f, 5.0f, 4.0f, 3.0f,
                                                   5.0f, 4.0f, 3.0f, 2.0f,
                                                   4.0f, 3.0f, 2.0f, 1.0f
                                           };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
                                    ? std::initializer_list<float>
                                            {
                                                    1, 3,
                                                    3, 5
                                            }
                                    : std::initializer_list<float>
                                            {
                                                    1.0f, 3.0f,
                                                    3.0f, 5.0f,

                                                    7.0f, 5.0f,
                                                    5.0f, 3.0f
                                            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    descriptor.m_Parameters.m_Method = armnn::ResizeMethod::NearestNeighbor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeNearestNeighborMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 2, 3, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 3, 5, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 1, 2, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 2, 3, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.5f);
        inputTensorInfo.SetQuantizationOffset(-1);
        outputTensorInfo.SetQuantizationScale(1.5f);
        outputTensorInfo.SetQuantizationOffset(-1);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                           {
                                                   3.0f, 4.5f, 6.0f, // 1,  2,  3, : Expected quantised values
                                                   9.0f, 13.5f, 21.0f // 5,  8, 13
                                           }
                                   : std::initializer_list<float>
                                           {
                                                   1.0f, 2.0f, 3.0f, 5.0f, 8.0f,
                                                   13.0f, 21.0f, 34.0f, 55.0f, 89.0f,
                                                   144.0f, 233.0f, 377.0f, 610.0f, 987.0f,

                                                   987.0f, 610.0f, 377.0f, 233.0f, 144.0f,
                                                   89.0f, 55.0f, 34.0f, 21.0f, 13.0f,
                                                   8.0f, 5.0f, 3.0f, 2.0f, 1.0f
                                           };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
                                    ? std::initializer_list<float>
                                            {
                                                    3.0f, 4.5f // 1, 3
                                            }
                                    : std::initializer_list<float>
                                            {
                                                    1.f,   2.f,   5.f,
                                                   13.f,  21.f,  55.f,

                                                  987.f, 610.f, 233.f,
                                                   89.f,  55.f,  21.f
                                            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    descriptor.m_Parameters.m_Method = armnn::ResizeMethod::NearestNeighbor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ResizeNearestNeighborMagTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout,
        float inQuantScale,
        int32_t inQuantOffset,
        float outQuantScale,
        int32_t outQuantOffset)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
                                        ?  armnnUtils::GetTensorInfo(1, 1, 3, 2, dataLayout, ArmnnType)
                                        :  armnnUtils::GetTensorInfo(1, 2, 3, 2, dataLayout, ArmnnType);
    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
                                         ?  armnnUtils::GetTensorInfo(1, 1, 3, 5, dataLayout, ArmnnType)
                                         :  armnnUtils::GetTensorInfo(1, 2, 3, 5, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(inQuantScale);
        inputTensorInfo.SetQuantizationOffset(inQuantOffset);
        outputTensorInfo.SetQuantizationScale(outQuantScale);
        outputTensorInfo.SetQuantizationOffset(outQuantOffset);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
                                   ? std::initializer_list<float>
                                        {
                                            0.183005f, 2.379065f, //  24, 228, : expected quantised values
                                            1.054970f, 1.302565f, // 105, 128,
                                            2.400595f, 0.688960f  // 230, 71
                                        }
                                   : std::initializer_list<float>
                                        {
                                               1.0f,   2.0f,
                                              13.0f,  21.0f,
                                            144.0f, 233.0f,

                                            233.0f, 144.0f,
                                             21.0f,  13.0f,
                                              2.0f,   1.0f
                                        };
    std::vector<float> outputData = armnn::IsQuantizedType<T>()
                                    ? std::initializer_list<float>
                                        {
                                            0.183005f, 0.183005f, 0.183005f, 2.379065f, 2.379065f,
                                            1.054970f, 1.054970f, 1.054970f, 1.302565f, 1.302565f,
                                            2.400595f, 2.400595f, 2.400595f, 0.688960f, 0.688960f
                                        }
                                    : std::initializer_list<float>
                                        {
                                              1.f,   1.f,   1.f,   2.f,   2.f,
                                             13.f,  13.f,  13.f,  21.f,  21.f,
                                            144.f, 144.f, 144.f, 233.f, 233.f,

                                            233.f, 233.f, 233.f, 144.f, 144.f,
                                             21.f,  21.f,  21.f,  13.f,  13.f,
                                              2.f,   2.f,   2.f,   1.f,   1.f
                                        };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    descriptor.m_Parameters.m_Method = armnn::ResizeMethod::NearestNeighbor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T, std::size_t InputDim, std::size_t OutputDim>
LayerTestResult<T, OutputDim> MeanTestHelper(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const unsigned int* inputShape,
        const std::vector<float>& inputData,
        const std::vector<unsigned int>& axis,
        bool keepDims,
        const unsigned int* outputShape,
        const std::vector<float>& outputData,
        float scale = 1.0f,
        int32_t offset = 0)
{
    armnn::TensorInfo inputTensorInfo(InputDim, inputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(OutputDim, outputShape, ArmnnType);

    inputTensorInfo.SetQuantizationScale(scale);
    inputTensorInfo.SetQuantizationOffset(offset);

    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);

    auto input = MakeTensor<T, InputDim>(inputTensorInfo, ConvertToDataType<ArmnnType>(inputData, inputTensorInfo));

    LayerTestResult<T, OutputDim> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, OutputDim>(
            outputTensorInfo, ConvertToDataType<ArmnnType>(outputData, outputTensorInfo));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::MeanQueueDescriptor data;
    data.m_Parameters.m_Axis = axis;
    data.m_Parameters.m_KeepDims = keepDims;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateMean(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.origin());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(result.output.origin(), outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 1> MeanSimpleTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 3, 2 };
    const unsigned int outputShape[] = { 1 };

    std::vector<float> input({ 1.5f, 1.5f, 2.5f, 2.5f, 3.5f, 3.5f });
    std::vector<float> output({ 2.5f });

    return MeanTestHelper<ArmnnType, T, 2, 1>(
            workloadFactory, memoryManager, inputShape, input, {}, false, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> MeanSimpleAxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 2, 3, 1, 2 };
    const unsigned int outputShape[] = { 3, 1, 2 };

    std::vector<float> input({ 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f });
    std::vector<float> output({ 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f });

    return MeanTestHelper<ArmnnType, T, 4, 3>(
            workloadFactory, memoryManager, inputShape, input, { 0 }, false, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> MeanKeepDimsTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 1, 1, 3, 2 };
    const unsigned int outputShape[] = { 1, 1, 1, 2 };

    std::vector<float> input({ 1.5f, 1.5f, 2.5f, 2.5f, 3.5f, 3.5f });
    std::vector<float> output({ 2.5f, 2.5f });

    return MeanTestHelper<ArmnnType, T, 4, 4>(
            workloadFactory, memoryManager, inputShape, input, { 2 }, true, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> MeanMultipleDimsTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 2, 3, 1, 2 };
    const unsigned int outputShape[] = { 1, 3, 1, 1 };

    std::vector<float> input({ 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5 });
    std::vector<float> output({ 2.0f, 4.0f, 6.0f });

    return MeanTestHelper<ArmnnType, T, 4, 4>(
            workloadFactory, memoryManager, inputShape, input, { 0, 3 }, true, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 1> MeanVts1Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 4, 3, 2 };
    const unsigned int outputShape[] = { 2 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                               15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f });
    std::vector<float> output({ 12.0f, 13.0f });

    return MeanTestHelper<ArmnnType, T, 3, 1>(
            workloadFactory, memoryManager, inputShape, input, { 0, 1 }, false, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> MeanVts2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 4, 3, 2 };
    const unsigned int outputShape[] = { 1, 3, 1 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                               15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f });
    std::vector<float> output({ 10.5f, 12.5f, 14.5f });

    return MeanTestHelper<ArmnnType, T, 3, 3>(
            workloadFactory, memoryManager, inputShape, input, { 0, 2 }, true, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> MeanVts3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = { 1, 2, 2, 1 };
    const unsigned int outputShape[] = { 1, 2, 1 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f });
    std::vector<float> output({ 1.5f, 3.5f });

    return MeanTestHelper<ArmnnType, T, 4, 3>(
            workloadFactory, memoryManager, inputShape, input, { 2 }, false, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> ConcatDifferentInputOutputQParamTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        bool useSubtensor)
{
    // Defines the tensor descriptors.
    armnn::TensorInfo outputTensorInfo({ 3, 6, 3 }, ArmnnType);
    armnn::TensorInfo inputTensorInfo1({ 3, 6, 2 }, ArmnnType);
    armnn::TensorInfo inputTensorInfo2({ 3, 6, 1 }, ArmnnType);

    std::vector<armnn::TensorShape> inputTensorShapes({inputTensorInfo1.GetShape(), inputTensorInfo2.GetShape()});

    // Quantized input1 tensor.
    const float inputScale1 = 0.5f;
    const int32_t inputOffset1 = 5;

    auto input1 = MakeTensor<T, 3>(inputTensorInfo1, std::vector<T>(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        28, 29, 30,
        31, 32, 33,
        34, 35, 36
    }));

    // Quatized input2 tensor.
    const float inputScale2 = 0.2f;
    const int32_t inputOffset2 = 10;

    auto input2 = MakeTensor<T, 3>(inputTensorInfo2, std::vector<T>(
    {
        37, 38, 39,
        40, 41, 42,
        43, 44, 45,
        46, 47, 48,
        49, 50, 51,
        52, 53, 54
    }));

    // Quantized output tensor.
    const float outputScale = 0.1f;
    const int32_t outputOffset = 20;

    LayerTestResult<T, 3> ret(outputTensorInfo);

    ret.outputExpected = MakeTensor<T, 3>(outputTensorInfo, std::vector<T>(
    {
        0,   5,  74,
        10,  15,  76,
        20,  25,  78,
        30,  35,  80,
        40,  45,  82,
        50,  55,  84,

        60,  65,  86,
        70,  75,  88,
        80,  85,  90,
        90,  95,  92,
        100, 105,  94,
        110, 115,  96,

        120, 125,  98,
        130, 135, 100,
        140, 145, 102,
        150, 155, 104,
        160, 165, 106,
        170, 175, 108
    }));

    outputTensorInfo.SetQuantizationScale(outputScale);
    outputTensorInfo.SetQuantizationOffset(outputOffset);
    inputTensorInfo1.SetQuantizationScale(inputScale1);
    inputTensorInfo1.SetQuantizationOffset(inputOffset1);
    inputTensorInfo2.SetQuantizationScale(inputScale2);
    inputTensorInfo2.SetQuantizationOffset(inputOffset2);

    std::vector<unsigned int> wOrigin1 = { 0, 0, 0 }; //Extent of the window is defined by size of input[0].
    armnn::ConcatQueueDescriptor::ViewOrigin window1(wOrigin1);

    std::vector<unsigned int> wOrigin2 = { 0, 0, 2 }; //Extent of the window is defined by size of input[1].
    armnn::ConcatQueueDescriptor::ViewOrigin window2(wOrigin2);

    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    bool subTensorsSupported = useSubtensor && workloadFactory.SupportsSubTensors();

    std::unique_ptr<armnn::ITensorHandle> inputHandle1 =
            subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo1.GetShape(), wOrigin1.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo1);

    std::unique_ptr<armnn::ITensorHandle> inputHandle2 =
            subTensorsSupported ?
            workloadFactory.CreateSubTensorHandle(*outputHandle, inputTensorInfo2.GetShape(), wOrigin2.data()) :
            workloadFactory.CreateTensorHandle(inputTensorInfo2);

    armnn::ConcatQueueDescriptor data;
    armnn::OriginsDescriptor desc = armnn::CreateDescriptorForConcatenation(
            inputTensorShapes.begin(),inputTensorShapes.end(), 2);
    data.m_Parameters = desc;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo1, inputHandle1.get());
    AddInputToWorkload(data, info, inputTensorInfo2, inputHandle2.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_ViewOrigins.push_back(window1);
    data.m_ViewOrigins.push_back(window2);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConcat(data, info);

    inputHandle1->Allocate();
    inputHandle2->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle1.get(), &input1[0][0][0]);
    CopyDataToITensorHandle(inputHandle2.get(), &input2[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0], outputHandle.get());

    return ret;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> PreluTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 1, 2, 2, 3 }, ArmnnType);
    armnn::TensorInfo alphaTensorInfo ({ 1, 1, 1, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 2, 2, 3 }, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(0.25f);
        inputTensorInfo.SetQuantizationOffset(128);
        alphaTensorInfo.SetQuantizationScale(0.25f);
        alphaTensorInfo.SetQuantizationOffset(50);
        outputTensorInfo.SetQuantizationScale(0.5f);
        outputTensorInfo.SetQuantizationOffset(120);
    }

    std::vector<float> inputData
    {
        // Expected quantized values:
        // 128, 128, 128, 132, 132, 132, 124, 124, 124, 120, 120, 120
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -2.0f, -2.0f, -2.0f
    };
    std::vector<float> alphaData
    {
        // Expected quantized values:
        // 50, 54, 58
        0.0f, 1.0f, 2.0f
    };
    std::vector<float> outputExpectedData =
    {
        // Expected quantized values:
        // 20, 120, 120, 122, 122, 122, 120, 118, 116, 120, 116, 112
       0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, -1.0f, -2.0f, 0.0f, -2.0f, -4.0f
    };

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));
    auto alpha = MakeTensor<T, 4>(alphaTensorInfo, QuantizedVector<T>(alphaTensorInfo.GetQuantizationScale(),
                                                                      alphaTensorInfo.GetQuantizationOffset(),
                                                                      alphaData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputExpectedData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> alphaHandle  = workloadFactory.CreateTensorHandle(alphaTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PreluQueueDescriptor descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload (descriptor, info, inputTensorInfo,  inputHandle.get());
    AddInputToWorkload (descriptor, info, alphaTensorInfo,  alphaHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePrelu(descriptor, info);

    inputHandle->Allocate();
    alphaHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);
    CopyDataToITensorHandle(alphaHandle.get(), &alpha[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType,
        std::size_t InputDim,
        std::size_t OutputDim,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, OutputDim> BatchToSpaceNdHelper(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout& dataLayout,
        const unsigned int *inputShape,
        const std::vector<float> &inputData,
        const std::vector<unsigned int> &blockShape,
        const std::vector<std::pair<unsigned int, unsigned int>> &crops,
        const unsigned int *outputShape,
        const std::vector<float> &outputData,
        float scale = 1.0f,
        int32_t offset = 0)
{
    armnn::TensorInfo inputTensorInfo(InputDim, inputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(OutputDim, outputShape, ArmnnType);

    inputTensorInfo.SetQuantizationScale(scale);
    inputTensorInfo.SetQuantizationOffset(offset);

    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);

    auto input = MakeTensor<T, InputDim>(inputTensorInfo, ConvertToDataType<ArmnnType>(inputData, inputTensorInfo));

    LayerTestResult<T, OutputDim> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, OutputDim>(outputTensorInfo,
                                                     ConvertToDataType<ArmnnType>(outputData, outputTensorInfo));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::BatchToSpaceNdQueueDescriptor data;
    data.m_Parameters.m_DataLayout = dataLayout;
    data.m_Parameters.m_BlockShape = blockShape;
    data.m_Parameters.m_Crops = crops;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateBatchToSpaceNd(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.origin());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest1(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 2, 2, 1};
    const unsigned int outputShape[] = {1, 4, 4, 1};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1.0f, 3.0f,
                                     // Batch 0, Height 1, Width (2) x Channel (1)
                                     9.0f, 11.0f,


                                     // Batch 1, Height 0, Width (2) x Channel (1)
                                     2.0f, 4.0f,
                                     // Batch 1, Height 1, Width (2) x Channel (1)
                                     10.0f, 12.0f,


                                     // Batch 2, Height 0, Width (2) x Channel (1)
                                     5.0f, 7.0f,
                                     // Batch 2, Height 1, Width (2) x Channel (1)
                                     13.0f, 15.0f,

                                     // Batch 3, Height 0, Width (2) x Channel (3)
                                     6.0f, 8.0f,
                                     // Batch 3, Height 1, Width (2) x Channel (1)
                                     14.0f, 16.0f
                             });

    std::vector<float> expectedOutput({
                                              1.0f,   2.0f,  3.0f,  4.0f,
                                              5.0f,   6.0f,  7.0f,  8.0f,
                                              9.0f,  10.0f, 11.0f,  12.0f,
                                              13.0f, 14.0f, 15.0f,  16.0f
                                      });

    std::vector<unsigned int> blockShape {2, 2};
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                                armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest2(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 2, 2, 1};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1.0f, 2.0f, 3.0f, 4.0f
                             });

    std::vector<float> expectedOutput({1.0f, 2.0f, 3.0f, 4.0f});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                                armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest3(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    std::vector<float> input({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    std::vector<float> expectedOutput({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                                armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest4(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {8, 1, 3, 1};
    const unsigned int outputShape[] = {2, 2, 4, 1};

    std::vector<float> input({
                                     0.0f, 1.0f, 3.0f,
                                     0.0f, 9.0f, 11.0f,
                                     0.0f, 2.0f, 4.0f,
                                     0.0f, 10.0f, 12.0f,
                                     0.0f, 5.0f, 7.0f,
                                     0.0f, 13.0f, 15.0f,
                                     0.0f, 6.0f, 8.0f,
                                     0.0f, 14.0f, 16.0f
                             });

    std::vector<float> expectedOutput({
                                              1.0f, 2.0f, 3.0f, 4.0f,
                                              5.0f, 6.0f, 7.0f, 8.0f,
                                              9.0f, 10.0f, 11.0f, 12.0f,
                                              13.0f, 14.0f, 15.0f, 16.0f
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {2, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                                armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest5(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 2, 2, 1};
    const unsigned int outputShape[] = {1, 4, 4, 1};

    std::vector<float> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    std::vector<float> expectedOutput({1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager, armnn::DataLayout::NHWC, inputShape,
                                                 input, blockShape, crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest6(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 2, 2, 1};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1, 2, 3, 4
                             });

    std::vector<float> expectedOutput({1, 2, 3, 4});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                 armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNhwcTest7(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    std::vector<float> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<float> expectedOutput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                 armnn::DataLayout::NHWC, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest1(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    std::vector<float> expectedOutput({
                                              // Batch 0, Channel 0, Height (2) x Width (2)
                                              1.0f,  4.0f,
                                              7.0f, 10.0f,

                                              // Batch 0, Channel 1, Height (2) x Width (2)
                                              2.0f,  5.0f,
                                              8.0f, 11.0f,

                                              // Batch 0, Channel 2, Height (2) x Width (2)
                                              3.0f,  6.0f,
                                              9.0f, 12.0f,
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                                armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest2(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 1, 2, 2};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1.0f, 2.0f, 3.0f, 4.0f
                             });

    std::vector<float> expectedOutput({1.0f, 2.0f, 3.0f, 4.0f});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                                armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest3(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f});

    std::vector<float> expectedOutput({
                                              // Batch 0, Channel 0, Height (2) x Width (2)
                                              1.0f,  7.0f,
                                              2.0f,  8.0f,

                                              // Batch 0, Channel 1, Height (2) x Width (2)
                                              3.0f,  9.0f,
                                              4.0f, 10.0f,

                                              // Batch 0, Channel 2, Height (2) x Width (2)
                                              5.0f, 11.0f,
                                              6.0f, 12.0f,
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                                armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                                crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest4(
        armnn::IWorkloadFactory &workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<float> expectedOutput({
                                              // Batch 0, Channel 0, Height (2) x Width (2)
                                              1,  4,
                                              7, 10,

                                              // Batch 0, Channel 1, Height (2) x Width (2)
                                              2,  5,
                                              8, 11,

                                              // Batch 0, Channel 2, Height (2) x Width (2)
                                              3,  6,
                                              9, 12,
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                 armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest5(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 1, 1, 1};
    const unsigned int outputShape[] = {1, 1, 2, 2};

    std::vector<float> input({
                                     // Batch 0, Height 0, Width (2) x Channel (1)
                                     1, 2, 3, 4
                             });

    std::vector<float> expectedOutput({1, 2, 3, 4});

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                 armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest6(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {4, 3, 1, 1};
    const unsigned int outputShape[] = {1, 3, 2, 2};

    std::vector<float> input({1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12});

    std::vector<float> expectedOutput({
                                              // Batch 0, Channel 0, Height (2) x Width (2)
                                              1,  7,
                                              2,  8,

                                              // Batch 0, Channel 1, Height (2) x Width (2)
                                              3,  9,
                                              4, 10,

                                              // Batch 0, Channel 2, Height (2) x Width (2)
                                              5, 11,
                                              6, 12,
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                 armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchToSpaceNdNchwTest7(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputShape[] = {8, 1, 1, 3};
    const unsigned int outputShape[] = {2, 1, 2, 4};

    std::vector<float> input({
                                     0, 1, 3, 0,  9, 11,
                                     0, 2, 4, 0, 10, 12,
                                     0, 5, 7, 0, 13, 15,
                                     0, 6, 8, 0, 14, 16
                             });

    std::vector<float> expectedOutput({
                                              1,  2,  3,  4,
                                              5,  6,  7,  8,
                                              9, 10, 11, 12,
                                              13, 14, 15, 16
                                      });

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {2, 0}};

    return BatchToSpaceNdHelper<ArmnnType, 4, 4>(workloadFactory, memoryManager,
                                                 armnn::DataLayout::NCHW, inputShape, input, blockShape,
                                                 crops, outputShape, expectedOutput);
}

template LayerTestResult<typename armnn::ResolveType<armnn::DataType::Float32>, 4>
PreluTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<typename armnn::ResolveType<armnn::DataType::QuantisedAsymm8>, 4>
PreluTest<armnn::DataType::QuantisedAsymm8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<typename armnn::ResolveType<armnn::DataType::QuantisedSymm16>, 4>
PreluTest<armnn::DataType::QuantisedSymm16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T, std::size_t outputDimLength>
LayerTestResult<T, outputDimLength> StackTestHelper(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::TensorInfo& inputTensorInfo,
        const armnn::TensorInfo& outputTensorInfo,
        unsigned int axis,
        const std::vector<std::vector<T>>& inputData,
        const std::vector<T>& outputExpectedData)
{
    unsigned int numInputs = static_cast<unsigned int>(inputData.size());
    std::vector<boost::multi_array<T, outputDimLength-1>> inputs;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        inputs.push_back(MakeTensor<T, outputDimLength-1>(inputTensorInfo, inputData[i]));
    }

    LayerTestResult<T, outputDimLength> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, outputDimLength>(outputTensorInfo, outputExpectedData);

    std::vector<std::unique_ptr<armnn::ITensorHandle>> inputHandles;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        inputHandles.push_back(workloadFactory.CreateTensorHandle(inputTensorInfo));
    }
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::StackQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Axis = axis;
    descriptor.m_Parameters.m_InputShape = inputTensorInfo.GetShape();
    descriptor.m_Parameters.m_NumInputs = numInputs;

    armnn::WorkloadInfo info;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        std::unique_ptr<armnn::ITensorHandle>& inputHandle = inputHandles[i];
        AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
        inputHandle->Allocate();
        CopyDataToITensorHandle(inputHandle.get(), inputs[i].origin());
    }

    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());
    outputHandle->Allocate();

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateStack(descriptor, info);

    workload->Execute();

    CopyDataFromITensorHandle(result.output.origin(), outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Stack0AxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 2, 3, 2, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18,


        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    };

    return StackTestHelper<ArmnnType, T, 4>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        0U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Stack4dOutput1AxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 3, 2, 2, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        4, 5, 6,

        19, 20, 21,
        22, 23, 24,


        7, 8, 9,
        10, 11, 12,

        25, 26, 27,
        28, 29, 30,


        13, 14, 15,
        16, 17, 18,

        31, 32, 33,
        34, 35, 36
    };

    return StackTestHelper<ArmnnType, T, 4>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Stack4dOutput2AxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 3, 2, 2, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        19, 20, 21,

        4, 5, 6,
        22, 23, 24,


        7, 8, 9,
        25, 26, 27,

        10, 11, 12,
        28, 29, 30,

        13, 14, 15,
        31, 32, 33,

        16, 17, 18,
        34, 35, 36
    };

    return StackTestHelper<ArmnnType, T, 4>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        2U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Stack4dOutput3AxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 3, 2, 3, 2 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> outputExpectedData =
    {
        1, 19,
        2, 20,
        3, 21,

        4, 22,
        5, 23,
        6, 24,


        7, 25,
        8, 26,
        9, 27,

        10, 28,
        11, 29,
        12, 30,


        13, 31,
        14, 32,
        15, 33,

        16, 34,
        17, 35,
        18, 36
    };

    return StackTestHelper<ArmnnType, T, 4>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        3U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Stack3dOutput1Axis3InputTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 3, 3, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    inputData.push_back(
    {
        10, 11, 12,
        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,
        25, 26, 27
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        10, 11, 12,
        19, 20, 21,

        4, 5, 6,
        13, 14, 15,
        22, 23, 24,

        7, 8, 9,
        16, 17, 18,
        25, 26, 27
    };

    return StackTestHelper<ArmnnType, T, 3>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> Stack5dOutputTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 2, 2, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 2, 2, 2, 2, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,


        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24
    });

    inputData.push_back(
    {
        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36,


        37, 38, 39,
        40, 41, 42,

        43, 44, 45,
        46, 47, 48
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,


        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36,



        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,


        37, 38, 39,
        40, 41, 42,

        43, 44, 45,
        46, 47, 48

    };

    return StackTestHelper<ArmnnType, T, 5>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}
