//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#if defined(ARMNNREF_ENABLED)
void LstmUtilsZeroVectorTest();
void LstmUtilsMeanStddevNormalizationNoneZeroInputTest();
void LstmUtilsMeanStddevNormalizationAllZeroInputTest();
void LstmUtilsMeanStddevNormalizationMixedZeroInputTest();
void LstmUtilsVectorBatchVectorCwiseProductTest();
void LstmUtilsVectorBatchVectorAddTest();
#endif

LayerTestResult<float, 2> LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 2> LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 2> LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 2> LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgNoPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<int16_t, 2> LstmLayerInt16WithCifgWithPeepholeNoProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgWithPeepholeWithProjectionTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<int16_t, 2> LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16ConstantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

//
// QuantizedLstm
//

LayerTestResult<uint8_t, 2> QuantizedLstmTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

//
// QLstm
//

LayerTestResult<int8_t, 2> QLstmTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<int8_t, 2> QLstmTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<int8_t, 2> QLstmTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);