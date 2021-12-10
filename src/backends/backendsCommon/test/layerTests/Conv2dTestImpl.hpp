//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <ResolveType.hpp>

#include <armnn/Types.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

//
// Convolution2d
//

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2d3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2d2x3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test(
    armnn::IWorkloadFactory &workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr &memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleConvolution2d3x5Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleConvolution2d3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleConvolution2d3x3Stride2x2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleConvolution2d3x3NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled);

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x5Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x3Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> SimpleConvolution2d3x5QSymm16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> SimpleConvolution2d3x3QSymm16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout layout);

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout layout);

LayerTestResult<float, 4> Convolution2dAsymmetricPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout layout);

LayerTestResult<float,   4> Convolution1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled);

LayerTestResult<uint8_t, 4> Convolution1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled);

LayerTestResult<float, 4> CompareConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory);

LayerTestResult<uint8_t, 4> Convolution2dPerAxisQuantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> Convolution2d3x3Stride2x2BFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout& dataLayout);

LayerTestResult<float, 4> Convolution2d3x3Stride2x2BFloat16SmallValueTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        const armnn::DataLayout& dataLayout);

//
// DepthwiseConvolution2d
//

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2d3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2d2x3x3Dilation3x3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dMult4Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> DepthwiseConvolution2dMult2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template<typename T>
LayerTestResult<T, 4> CompareDepthwiseConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> DepthwiseConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> DepthwiseConvolution2dDepthNhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled);

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul64Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 4> DepthwiseConvolution2dAsymmetricTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dDepthMul1Uint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> DepthwiseConvolution2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<int16_t, 4> DepthwiseConvolution2dDepthMul1Int16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dPerAxisQuantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout);

LayerTestResult<float, 4> CompareDepthwiseConvolution2dFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    const armnn::DataLayout layout);

LayerTestResult<uint8_t, 4> CompareDepthwiseConvolution2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    const armnn::DataLayout layout);
