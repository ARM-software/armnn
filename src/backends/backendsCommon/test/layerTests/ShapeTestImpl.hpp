//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

template<typename T , std::size_t n>
LayerTestResult<int32_t, 1> ShapeTest(
        armnn::TensorInfo inputTensorInfo,
        std::vector<T> input,
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 1> ShapeDimSize1Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 1> ShapeDimSize2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 1> ShapeDimSize3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 1> ShapeDimSize4Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);
