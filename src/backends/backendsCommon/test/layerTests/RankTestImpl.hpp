//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

template<typename T , std::size_t n>
LayerTestResult<int32_t, 1> RankTest(
        armnn::TensorInfo inputTensorInfo,
        std::vector<T> input,
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 1> RankDimSize1Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 1> RankDimSize2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 1> RankDimSize3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 1> RankDimSize4Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);
