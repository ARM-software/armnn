//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <Half.hpp>

LayerTestResult<float, 4> DequantizeSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DequantizeOffsetUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DequantizeSimpleAsymmInt8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DequantizeOffsetAsymmInt8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DequantizeSimpleInt8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 4> DequantizeSimpleInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<armnn::Half, 4> DequantizeSimpleUint8ToFp16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<armnn::Half, 4> DequantizeSimpleInt8ToFp16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<armnn::Half, 4> DequantizeSimpleInt16ToFp16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);
