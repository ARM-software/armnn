//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMaxSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMinSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMinChannelTest(armnn::IWorkloadFactory& workloadFactory,
                                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                              const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMaxChannelTest(armnn::IWorkloadFactory& workloadFactory,
                                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                              const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMaxHeightTest(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMinWidthTest(armnn::IWorkloadFactory& workloadFactory,
                                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                            const armnn::ITensorHandleFactory& tensorHandleFactory);