//
// Copyright © 2020 Samsung Electronics Co Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReduceSumTestImpl.hpp"

#include <DataTypeUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<float, 4> ReduceTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::TensorInfo inputTensorInfo,
        const armnn::TensorInfo outputTensorInfo,
        const std::vector<float>& inputData,
        const std::vector<float>& outputData,
        const std::vector<int32_t> vAxis,
        const armnn::ReduceOperation reduceOperation,
        bool keepDims = false)
{
    IgnoreUnused(memoryManager);
    auto inputTensor = ConvertToDataType<ArmnnType>(inputData, inputTensorInfo);

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ReduceQueueDescriptor descriptor;
    std::vector<uint32_t> updated_idx;
    uint32_t resolvedAxis = 0;
    for (uint32_t i = 0; i < vAxis.size(); ++i)
    {
        if (vAxis[i] <  0)
        {
            resolvedAxis = inputTensorInfo.GetNumDimensions() + static_cast<uint32_t>(vAxis[i]);
        } else
        {
            resolvedAxis = static_cast<uint32_t>(vAxis[i]);
        }

        updated_idx.push_back(resolvedAxis);
    }

    descriptor.m_Parameters.m_vAxis = updated_idx;
    descriptor.m_Parameters.m_ReduceOperation = reduceOperation;
    descriptor.m_Parameters.m_KeepDims = keepDims;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Reduce,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputTensor.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<float, 4>(actualOutput,
                                     outputData,
                                     outputHandle->GetShape(),
                                     outputTensorInfo.GetShape());
}

} // namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceSumSimpleTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 1, 1, 5 };
    const armnn::TensorShape outputShape{ 1, 1, 1, 1};

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues({ 5.0f, 2.0f, 8.0f, 10.0f, 9.0f });
    std::vector<float> outputValues({ 34.0f });

    return ReduceTestCommon<ArmnnType>(workloadFactory,
                                       memoryManager,
                                       tensorHandleFactory,
                                       inputTensorInfo,
                                       outputTensorInfo,
                                       inputValues,
                                       outputValues,
                                       { -1 },
                                       armnn::ReduceOperation::Sum);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceSumSingleAxisTest1(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4 };
    const armnn::TensorShape outputShape{ 1, 1, 2, 4};

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues({  1.0f,   2.0f,   3.0f,   4.0f,
                                      5.0f,   6.0f,   7.0f,   8.0f,

                                     10.0f,  20.0f,  30.0f,  40.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,

                                    100.0f, 200.0f, 300.0f, 400.0f,
                                    500.0f, 600.0f, 700.0f, 800.0f });
    std::vector<float> outputValues({ 111.0f, 222.0f, 333.0f, 444.0f,
                                      555.0f, 666.0f, 777.0f, 888.0f });

    return ReduceTestCommon<ArmnnType>(workloadFactory,
                                       memoryManager,
                                       tensorHandleFactory,
                                       inputTensorInfo,
                                       outputTensorInfo,
                                       inputValues,
                                       outputValues,
                                       { 1 },
                                       armnn::ReduceOperation::Sum);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceSumSingleAxisTest2(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 6, 3, 4 };
    const armnn::TensorShape outputShape{ 1, 1, 3, 4};

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues( {7, 8, 6, 1,
                                     1, 1, 8, 7,
                                     3, 7, 7, 7,

                                     6, 8, 4, 7,
                                     3, 8, 7, 3,
                                     5, 8, 8, 8,


                                     7, 8, 2, 7,
                                     3, 8, 5, 6,
                                     8, 4, 2, 7,

                                     1, 6, 7, 2,
                                     8, 3, 3, 1,
                                     7, 6, 2, 6,


                                     5, 3, 4, 8,
                                     7, 8, 2, 4,
                                     6, 6, 2, 8,

                                     2, 2, 7, 2,
                                     5, 3, 6, 3,
                                     6, 1, 8, 8});
    std::vector<float> outputValues({  28.0f, 35.0f, 30.0f, 27.0f,
                                       27.0f, 31.0f, 31.0f, 24.0f,
                                       35.0f, 32.0f, 29.0f, 44.0f});

    return ReduceTestCommon<ArmnnType>(workloadFactory,
                                       memoryManager,
                                       tensorHandleFactory,
                                       inputTensorInfo,
                                       outputTensorInfo,
                                       inputValues,
                                       outputValues,
                                       { 1 },
                                       armnn::ReduceOperation::Sum);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceSumSingleAxisTest3(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 6, 3, 4 };
    const armnn::TensorShape outputShape{ 1, 6, 3, 1};

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues( {7, 8, 6, 1,
                                     1, 1, 8, 7,
                                     3, 7, 7, 7,

                                     6, 8, 4, 7,
                                     3, 8, 7, 3,
                                     5, 8, 8, 8,


                                     7, 8, 2, 7,
                                     3, 8, 5, 6,
                                     8, 4, 2, 7,

                                     1, 6, 7, 2,
                                     8, 3, 3, 1,
                                     7, 6, 2, 6,


                                     5, 3, 4, 8,
                                     7, 8, 2, 4,
                                     6, 6, 2, 8,

                                     2, 2, 7, 2,
                                     5, 3, 6, 3,
                                     6, 1, 8, 8});
    std::vector<float> outputValues({  22.0f, 17.0f, 24.0f,
                                       25.0f, 21.0f, 29.0f,

                                       24.0f, 22.0f, 21.0f,
                                       16.0f, 15.0f, 21.0f,

                                       20.0f, 21.0f, 22.0f,
                                       13.0f, 17.0f, 23.0f});

    return ReduceTestCommon<ArmnnType>(workloadFactory,
                                       memoryManager,
                                       tensorHandleFactory,
                                       inputTensorInfo,
                                       outputTensorInfo,
                                       inputValues,
                                       outputValues,
                                       { 3 },
                                       armnn::ReduceOperation::Sum,
                                       true);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceSumMultipleAxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4 };
    const armnn::TensorShape outputShape{ 1, 1, 1, 4};

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues({  1.0f,   2.0f,   3.0f,   4.0f,
                                      5.0f,   6.0f,   7.0f,   8.0f,

                                     10.0f,  20.0f,  30.0f,  40.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,

                                    100.0f, 200.0f, 300.0f, 400.0f,
                                    500.0f, 600.0f, 700.0f, 800.0f });
    std::vector<float> outputValues({ 666.0f, 888.0f, 1110.0f, 1332.0f });

    return ReduceTestCommon<ArmnnType>(workloadFactory,
                                       memoryManager,
                                       tensorHandleFactory,
                                       inputTensorInfo,
                                       outputTensorInfo,
                                       inputValues,
                                       outputValues,
                                       { 1, 2 },
                                       armnn::ReduceOperation::Sum);
}

// Explicit template specializations

template LayerTestResult<float, 4>
ReduceSumSimpleTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<float, 4>
ReduceSumSingleAxisTest1<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<float, 4>
ReduceSumSingleAxisTest2<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<float, 4>
ReduceSumSingleAxisTest3<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<float, 4>
ReduceSumMultipleAxisTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);
