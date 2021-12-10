//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

// Float32 tests
LayerTestResult<float, 4> Slice4dFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Slice3dFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> Slice2dFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 1> Slice1dFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

// Uint8 tests
LayerTestResult<uint8_t, 4> Slice4dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Slice3dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> Slice2dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 1> Slice1dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

// Int16 tests
LayerTestResult<int16_t, 4> Slice4dInt16Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 3> Slice3dInt16Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> Slice2dInt16Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 1> Slice1dInt16Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);
