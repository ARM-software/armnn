//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <Half.hpp>

#include <ResolveType.hpp>

#include <armnn/Types.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

//
// Convolution3d
//

LayerTestResult<float, 5> SimpleConvolution3d3x3x3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<int8_t , 5> SimpleConvolution3d3x3x3Int8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<uint8_t, 5> SimpleConvolution3d3x3x3Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<int16_t, 5> SimpleConvolution3d3x3x3Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<float, 5> Convolution3d2x2x2Strides3x5x5Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<int8_t , 5> Convolution3d2x2x2Strides3x5x5Int8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<uint8_t, 5> Convolution3d2x2x2Strides3x5x5Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<int16_t, 5> Convolution3d2x2x2Strides3x5x5Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<float, 5> Convolution3d2x2x2Dilation2x2x2Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<int8_t , 5> Convolution3d2x2x2Dilation2x2x2Int8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<uint8_t, 5> Convolution3d2x2x2Dilation2x2x2Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<int16_t, 5> Convolution3d2x2x2Dilation2x2x2Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<float, 5> Convolution3dPaddingSame3x3x3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<int8_t , 5> Convolution3dPaddingSame3x3x3Int8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<uint8_t, 5> Convolution3dPaddingSame3x3x3Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<int16_t, 5> Convolution3dPaddingSame3x3x3Int16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<float, 5> Convolution3dStrideDilationPadding3x3x3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<float, 5> Convolution3d2x2x2Stride3x3x3SmallFloat32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<armnn::Half, 5> Convolution3d2x3x3Float16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);

LayerTestResult<armnn::Half, 5> Convolution3d2x2x2SmallFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        armnn::DataLayout dataLayout);
