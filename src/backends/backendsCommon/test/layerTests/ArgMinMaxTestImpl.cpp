//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArgMinMaxTestImpl.hpp"


#include <DataTypeUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<int32_t, 3> ArgMinMaxTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        armnn::ArgMinMaxFunction argMinMaxFunction,
        const armnn::TensorInfo inputTensorInfo,
        const armnn::TensorInfo outputTensorInfo,
        const std::vector<float>& inputData,
        const std::vector<int32_t>& outputData,
        int axis = 3)
{
    std::vector<T> inputTensor = ConvertToDataType<ArmnnType>(inputData, inputTensorInfo);
    std::vector<int32_t> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ArgMinMaxQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Function = argMinMaxFunction;
    descriptor.m_Parameters.m_Axis = axis;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::ArgMinMax,
                                                                                descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputTensor.data());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<int32_t, 3>(actualOutput,
                                       outputData,
                                       outputHandle->GetShape(),
                                       outputTensorInfo.GetShape());
}

} // namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 3> ArgMaxSimpleTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 1, 1, 5 };
    const armnn::TensorShape outputShape{ 1, 1, 1 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Signed32);

    std::vector<float> inputValues({ 5.0f, 2.0f, 8.0f, 10.0f, 9.0f });
    std::vector<int32_t> outputValues({ 3 });

    return ArgMinMaxTestCommon<ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory,
                                          armnn::ArgMinMaxFunction::Max,
                                          inputTensorInfo, outputTensorInfo,
                                          inputValues, outputValues, -1); // axis -1 === 3
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 3> ArgMinSimpleTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 1, 1, 5 };
    const armnn::TensorShape outputShape{ 1, 1, 1 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Signed32);

    std::vector<float> inputValues({ 5.0f, 2.0f, 8.0f, 10.0f, 9.0f });
    std::vector<int32_t> outputValues({ 1 });

    return ArgMinMaxTestCommon<ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory,
                                          armnn::ArgMinMaxFunction::Min,
                                          inputTensorInfo, outputTensorInfo,
                                          inputValues, outputValues, 3);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 3> ArgMinChannelTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4};
    const armnn::TensorShape outputShape{ 1, 2, 4 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Signed32);

    std::vector<float> inputValues({   1.0f,   2.0f,   3.0f,   4.0f,
                                       5.0f,   6.0f,   7.0f,   8.0f,

                                      10.0f,  20.0f,  30.0f,  40.0f,
                                      50.0f,  60.0f,  70.0f,  80.0f,

                                     100.0f, 200.0f, 300.0f, 400.0f,
                                     500.0f, 600.0f, 700.0f, 800.0f });
    std::vector<int32_t> outputValues({ 0, 0, 0, 0,
                                        0, 0, 0, 0 });

    return ArgMinMaxTestCommon<ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory,
                                          armnn::ArgMinMaxFunction::Min,
                                          inputTensorInfo, outputTensorInfo,
                                          inputValues, outputValues, 1);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 3> ArgMaxChannelTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4};
    const armnn::TensorShape outputShape{ 1, 2, 4 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Signed32);

    std::vector<float> inputValues({  1.0f,   2.0f,   3.0f,   4.0f,
                                      5.0f,   6.0f,   7.0f,   8.0f,

                                     10.0f,  20.0f,  30.0f,  40.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,

                                    100.0f, 200.0f, 300.0f, 400.0f,
                                    500.0f, 600.0f, 700.0f, 800.0f });
    std::vector<int32_t> outputValues({ 2, 2, 2, 2,
                                        2, 2, 2, 2 });

    return ArgMinMaxTestCommon<ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory,
                                          armnn::ArgMinMaxFunction::Max,
                                          inputTensorInfo, outputTensorInfo,
                                          inputValues, outputValues, 1);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 3> ArgMaxHeightTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4};
    const armnn::TensorShape outputShape{ 1, 3, 4 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Signed32);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    std::vector<float> inputValues({  1.0f,   2.0f,   3.0f,   4.0f,
                                      5.0f,   6.0f,   7.0f,   8.0f,

                                     10.0f,  20.0f,  30.0f,  40.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,

                                    100.0f, 200.0f, 300.0f, 400.0f,
                                    500.0f, 600.0f, 700.0f, 800.0f });
    std::vector<int32_t> outputValues({ 1, 1, 1, 1,
                                        1, 1, 1, 1,
                                        1, 1, 1, 1 });

    return ArgMinMaxTestCommon<ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory,
                                          armnn::ArgMinMaxFunction::Max,
                                          inputTensorInfo, outputTensorInfo,
                                          inputValues, outputValues, 2);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 3> ArgMinWidthTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 3, 2, 4};
    const armnn::TensorShape outputShape{ 1, 3, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Signed32);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    std::vector<float> inputValues({  1.0f,   2.0f,   3.0f,   4.0f,
                                      5.0f,   6.0f,   7.0f,   8.0f,

                                     10.0f,  20.0f,  30.0f,  40.0f,
                                     50.0f,  60.0f,  70.0f,  80.0f,

                                    100.0f, 200.0f, 300.0f, 400.0f,
                                    500.0f, 600.0f, 700.0f, 800.0f });
    std::vector<int32_t> outputValues({ 0, 0,
                                        0, 0,
                                        0, 0 });

    return ArgMinMaxTestCommon<ArmnnType>(workloadFactory, memoryManager, tensorHandleFactory,
                                          armnn::ArgMinMaxFunction::Min,
                                          inputTensorInfo, outputTensorInfo,
                                          inputValues, outputValues, 3);
}


// Explicit template specializations

template LayerTestResult<int32_t, 3>
ArgMaxSimpleTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxSimpleTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxSimpleTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxSimpleTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxSimpleTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxSimpleTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinSimpleTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinSimpleTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinSimpleTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinSimpleTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinSimpleTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinSimpleTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinChannelTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinChannelTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinChannelTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinChannelTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinChannelTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinChannelTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxChannelTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxChannelTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxChannelTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxChannelTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxChannelTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxChannelTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxHeightTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxHeightTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxHeightTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxHeightTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMaxHeightTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinWidthTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinWidthTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinWidthTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinWidthTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 3>
ArgMinWidthTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);
