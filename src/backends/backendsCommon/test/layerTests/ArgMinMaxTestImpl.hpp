//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <ResolveType.hpp>

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMaxSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMinSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMinChannel4dTest(armnn::IWorkloadFactory& workloadFactory,
                                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMaxChannel4dTest(armnn::IWorkloadFactory& workloadFactory,
                                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);