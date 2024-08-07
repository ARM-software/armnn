//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include "ResolveType.hpp"

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Tile4dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                 const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Tile3dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                 const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Tile2dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                 const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> Tile1dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                 const armnn::ITensorHandleFactory& tensorHandleFactory);