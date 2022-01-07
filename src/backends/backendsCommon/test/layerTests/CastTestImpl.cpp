//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CastTestImpl.hpp"
#include "ElementwiseUnaryTestImpl.hpp"


template<armnn::DataType inputDataType, armnn::DataType outputDataType, typename TInput, typename TOutput>
LayerTestResult<TOutput, 4> CastTest(armnn::IWorkloadFactory& workloadFactory,
                                     const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                     const armnn::ITensorHandleFactory& tensorHandleFactory,
                                     const std::vector<TInput>& inputValues,
                                     const std::vector<TOutput>& outputValues)
{
    IgnoreUnused(memoryManager);
    armnn::TensorInfo inputTensorInfo({1, 3, 2, 3}, inputDataType);
    armnn::TensorInfo outputTensorInfo({1, 3, 2, 3}, outputDataType);
    float quantizationScale = 1.0f;
    int32_t quantizationOffset = 0;

    if(armnn::IsQuantizedType<TInput>())
    {
        inputTensorInfo.SetQuantizationScale(quantizationScale);
        inputTensorInfo.SetQuantizationOffset(quantizationOffset);
    }
    if(armnn::IsQuantizedType<TOutput>())
    {
        outputTensorInfo.SetQuantizationScale(quantizationScale);
        outputTensorInfo.SetQuantizationOffset(quantizationOffset);
    }

    std::vector<TOutput> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::CastQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Cast, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputValues.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<TOutput, 4>(actualOutput,
                                       outputValues,
                                       outputHandle->GetShape(),
                                       outputTensorInfo.GetShape());
}

LayerTestResult<float, 4> CastInt32ToFloat2dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                 const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<int32_t> inputValues = { -1, -3, -1, -3, -1, -3, -1, -3, 1,
                                         3, 1, 3, 1, 2, 1, 3, 1, 3 };
    std::vector<float> outputValues  = { -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, 1.0f,
                                        3.0f, 1.0f, 3.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };
    return CastTest<armnn::DataType::Signed32, armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                         outputValues);
}

LayerTestResult<float, 4> CastInt16ToFloat2dTest(armnn::IWorkloadFactory& workloadFactory,
                                               const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                               const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<int16_t> inputValues = { -1, -3, -1, -3, -1, -3, -1, -3, 1,
                                         3, 1, 3, 1, 2, 1, 3, 1, 3 };
    std::vector<float> outputValues  = { -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, 1.0f,
                                         3.0f, 1.0f, 3.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };
    return CastTest<armnn::DataType::QSymmS16, armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                         outputValues);
}

LayerTestResult<float, 4> CastInt8ToFloat2dTest(armnn::IWorkloadFactory& workloadFactory,
                                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                                 const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<int8_t> inputValues = { -1, -3, -1, -3, -1, -3, -1, -3, 1,
                                         3, 1, 3, 1, 2, 1, 3, 1, 3 };
    std::vector<float> outputValues  = { -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, 1.0f,
                                         3.0f, 1.0f, 3.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };
    return CastTest<armnn::DataType::QSymmS8, armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                        outputValues);
}

LayerTestResult<float, 4> CastInt8AsymmToFloat2dTest(armnn::IWorkloadFactory& workloadFactory,
                                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                                const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<int8_t> inputValues = { -1, -3, -1, -3, -1, -3, -1, -3, 1,
                                        3, 1, 3, 1, 2, 1, 3, 1, 3 };
    std::vector<float> outputValues  = { -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, 1.0f,
                                         3.0f, 1.0f, 3.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };
    return CastTest<armnn::DataType::QAsymmS8, armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                                        tensorHandleFactory, inputValues, outputValues);
}

LayerTestResult<float, 4> CastUInt8ToFloat2dTest(armnn::IWorkloadFactory& workloadFactory,
                                               const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                               const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<uint8_t> inputValues = { 1, 3, 1, 3, 1, 3, 1, 3, 1,
                                         3, 1, 3, 1, 2, 1, 3, 1, 3 };
    std::vector<float> outputValues  = { 1.0f, 3.0f, 1.0f, 3.0f, 1.0f, 3.0f, 1.0f, 3.0f, 1.0f,
                                         3.0f, 1.0f, 3.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };
    return CastTest<armnn::DataType::QAsymmU8, armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                         outputValues);
}

LayerTestResult<uint8_t, 4> CastInt8ToUInt82dTest(armnn::IWorkloadFactory& workloadFactory,
                                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                                  const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<int8_t> inputValues  = { -1, -3, -1, -3, -1, -3, -1, -3, -1,
                                         3, 1, 3, 1, 2, 1, 3, 1, 3 };
    std::vector<uint8_t> outputValues = { 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          3, 1, 3, 1, 2, 1, 3, 1, 3 };
    return CastTest<armnn::DataType::QSymmS8, armnn::DataType::QAsymmU8>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                         outputValues);
}

LayerTestResult<uint8_t, 4> CastInt8AsymmToUInt82dTest(armnn::IWorkloadFactory& workloadFactory,
                                               const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                               const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<int8_t> inputValues  = { -1, -3, -1, -3, -1, -3, -1, -3, -1,
                                         3, 1, 3, 1, 2, 1, 3, 1, 3 };
    std::vector<uint8_t> outputValues = { 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          3, 1, 3, 1, 2, 1, 3, 1, 3 };
    return CastTest<armnn::DataType::QAsymmS8, armnn::DataType::QAsymmU8>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                          outputValues);
}

LayerTestResult<float, 4> CastFloat16ToFloat322dTest(armnn::IWorkloadFactory& workloadFactory,
                                               const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                               const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    std::vector<armnn::Half> inputValues = { -1.10_h, -3._h, -1.30_h, -3._h, -1._h, -3._h, -1._h, -3._h, 1._h,
                                         3.10_h, 1._h, 3.30_h, 1._h, 2._h, 1._h, 3._h, 1._h, 3._h };
    std::vector<float> outputValues  = { -1.1f, -3.0f, -1.3f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, 1.0f,
                                       3.1f, 1.0f, 3.3f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };
    return CastTest<armnn::DataType::Float16, armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                                        tensorHandleFactory, inputValues,
                                                                        outputValues);
}

LayerTestResult<float, 4> CastBFloat16ToFloat322dTest(armnn::IWorkloadFactory& workloadFactory,
                                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                              const armnn::ITensorHandleFactory& tensorHandleFactory)
{

    std::vector<armnn::BFloat16> inputValues = armnnUtils::QuantizedVector<armnn::BFloat16>(
            {
                    -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f,
                    1.0f, 0.4f, 0.5f, 1.3f, 1.5f, 2.0f, 8.76f, 15.2f, 37.5f
            },
            1.0f, 0);


    std::vector<float> outputValues = { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f,
                                                1.0f, 0.4f, 0.5f, 1.3f, 1.5f, 2.0f, 8.76f, 15.2f, 37.5f };

    return CastTest<armnn::DataType::BFloat16, armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                                        tensorHandleFactory, inputValues, outputValues);
}

LayerTestResult<armnn::Half, 4> CastFloat32ToFloat162dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    std::vector<float> inputValues = { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f,
                                       0.00000004f, 3.4E38f, 300.0f, 0.5f, 1.3f, 1.5f, 2.1E4f, 8.76f, 15.2f, 37.5f };
    std::vector<armnn::Half> outputValues = {-37.50_h, -15.20_h, -8.76_h, -2._h, -1.50_h, -1.30_h, -0.50_h, -0.40_h,
                                     0._h, 6.55E4_h, 300._h, 0.50_h, 1.30_h, 1.50_h, 2.1E4_h, 8.76_h, 15.20_h, 37.50_h};

    return CastTest<armnn::DataType::Float32, armnn::DataType::Float16>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                         outputValues);
}

LayerTestResult<int8_t , 4> CastFloat32ToInt82dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<float> inputValues  = { -1.0f, -3.5f, -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, 1.0f,
                                         3.1f, 1.5f, 3.9f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };
    std::vector<int8_t> outputValues = { -1, -3, -1, -3, -1, -3, -1, -3, 1,
                                        3, 1, 3, 1, 2, 1, 3, 1, 3 };
    return CastTest<armnn::DataType::Float32, armnn::DataType::QAsymmS8>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                         outputValues);
}

LayerTestResult<uint8_t , 4> CastFloat32ToUInt82dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    std::vector<float> inputValues  = { -1.0f, -3.5f, -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, 1.0f,
                                        3.1f, 1.5f, 3.9f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };
    std::vector<uint8_t> outputValues = { 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                         3, 1, 3, 1, 2, 1, 3, 1, 3 };
    return CastTest<armnn::DataType::Float32, armnn::DataType::QAsymmU8>(workloadFactory, memoryManager,
                                                                         tensorHandleFactory, inputValues,
                                                                         outputValues);
}
