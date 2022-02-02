//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SoftmaxTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>


#include <armnn/backends/TensorHandle.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <algorithm>

namespace
{

struct Simple3dSoftmaxOutputData
{
    const std::vector<float> outputData =
    {
        0.0964599f, 0.26220518f, 0.0964599f, 0.0964599f,
        0.15903549f, 0.0964599f, 0.0964599f, 0.0964599f
    };

    const armnn::TensorShape inputShape{ 1, 8, 1 };

    const std::vector<float> inputData =
    {
        0.0f, 1.0f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.0f, 0.0f,
    };
};

struct Simple4dSoftmaxData
{
    const armnn::TensorShape inputShape{ 1, 8, 1, 1 };

    const std::vector<float> outputData =
    {
        0.0964599f, 0.26220518f, 0.0964599f, 0.0964599f,
        0.15903549f, 0.0964599f, 0.0964599f, 0.0964599f
    };

    const std::vector<float> inputData =
    {
         0.0f, 1.0f, 0.0f, 0.0f,
         0.5f, 0.0f, 0.0f, 0.0f
    };
};

template<armnn::DataType ArmnnType, std::size_t n, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, n> SimpleSoftmaxBaseTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float beta,
    const armnn::TensorShape& inputShape,
    const std::vector<float>& outputData,
    const std::vector<float>& inputData,
    int axis = -1)
{
    using std::exp;

    const float qScale = 1.f / 256.f;
    const int qOffset = 0;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    inputTensorInfo = armnn::TensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);

    outputTensorInfo = armnn::TensorInfo(inputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);

    // Each row is independently softmax'd.
    std::vector<T> input = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(outputData, qScale, qOffset);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::SoftmaxQueueDescriptor data;
    data.m_Parameters.m_Beta = beta;
    data.m_Parameters.m_Axis = axis;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Softmax, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ARMNN_ASSERT(workload);

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, n>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> SimpleSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float beta)
{
    using std::exp;
    const armnn::TensorShape inputShape{ 2, 4 };

    float x0[4] = { exp((0.f - 1.0f) * beta), exp((1.0f - 1.0f) * beta),
                    exp((0.0f - 1.0f) * beta), exp((0.0f - 1.0f) * beta) };
    float sum0 = x0[0] + x0[1] + x0[2] + x0[3];
    float x1[4] = { exp((0.5f - 0.5f) * beta), exp((0.0f - 0.5f) * beta),
                    exp((0.0f - 0.5f) * beta), exp((0.0f - 0.5f) * beta) };
    float sum1 = x1[0] + x1[1] + x1[2] + x1[3];

    const std::vector<float> outputData = { x0[0] / sum0, x0[1] / sum0, x0[2] / sum0, x0[3] / sum0,
                                            x1[0] / sum1, x1[1] / sum1, x1[2] / sum1, x1[3] / sum1 };

    const std::vector<float> inputData =
            {
                0.f, 1.f, 0.f, 0.f,
                .5f, 0.f, 0.f, 0.f,
            };

    return SimpleSoftmaxBaseTestImpl<ArmnnType, 2>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                   inputShape, outputData, inputData);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> SimpleSoftmaxTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta,
        int axis)
{
    armnn::TensorShape inputShape;
    std::vector<float> inputData;
    std::vector<float> outputData;
    switch (axis)
    {
    case -2:
    case 0:
        {
        inputShape = {5, 2};

        inputData =
                {
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f
                };

        outputData =
                {
                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f
                };
        break;
        }
    case -1:
    case 1:
        {
        inputShape = {2, 5};

        inputData =
                {
                        17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f
                };

        outputData =
                {
                        0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                        7.246299848982885e-08f,
                        0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                        7.246299848982885e-08f
                };
        break;
        }
    }
    return SimpleSoftmaxBaseTestImpl<ArmnnType, 2>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                   inputShape, outputData, inputData, axis);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Simple3dSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float beta,
    const armnn::TensorShape& inputShape,
    const std::vector<float>& outputData,
    const std::vector<float>& inputData,
    int axis = 1)
{
    return SimpleSoftmaxBaseTestImpl<ArmnnType, 3>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                   inputShape, outputData, inputData, axis);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Simple4dSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float beta,
    const armnn::TensorShape& inputShape,
    const std::vector<float>& outputData,
    const std::vector<float>& inputData,
    int axis = 1)
{

    return SimpleSoftmaxBaseTestImpl<ArmnnType, 4>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                   inputShape, outputData, inputData, axis);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> CompareSoftmaxTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        armnn::IWorkloadFactory& refWorkloadFactory,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::ITensorHandleFactory& refTensorHandleFactory,
        float beta)
{
    const int batchSize = 20;
    const int channels = 30;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { batchSize, channels };

    inputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    float qScale = 1.f / 256.f;
    int qOffset = 0;
    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);

    auto input = MakeRandomTensor<T>(inputTensorInfo, 0xF00D, 0.0f, 1.0f);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> expectedOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::SoftmaxQueueDescriptor data;
    data.m_Parameters.m_Beta = beta;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef =
        refTensorHandleFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef  =
        refTensorHandleFactory.CreateTensorHandle(inputTensorInfo);

    armnn::SoftmaxQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Softmax, data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateWorkload(armnn::LayerType::Softmax,
                                                                                      refData,
                                                                                      refInfo);

    outputHandleRef->Allocate();
    inputHandleRef->Allocate();

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());
    CopyDataToITensorHandle(inputHandleRef.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    workloadRef->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());
    CopyDataFromITensorHandle(expectedOutput.data(), outputHandleRef.get());

    return LayerTestResult<T, 2>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

} // anonymous namespace

LayerTestResult<float,2> SimpleSoftmaxTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float beta)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, beta);
}

LayerTestResult<float,2> SimpleAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta,
        int axis)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                           tensorHandleFactory, beta, axis);
}

LayerTestResult<float,3> Simple3dSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    Simple3dSoftmaxOutputData data;
    return Simple3dSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                             data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<float,3> Simple3dAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta,
        int axis)
{
    armnn::TensorShape inputShape;
    std::vector<float> inputData;
    std::vector<float> outputData;
    switch (axis)
    {
    case -3:
    case 0:
        {
            inputShape = {5, 2, 2};

            inputData =
                    {
                            17.0f, -1.0f, 17.0f, -1.0f, 16.0f, -2.0f, 16.0f, -2.0f, 15.0f, -3.0f,

                            15.0f, -3.0f, 14.0f, -4.0f, 14.0f, -4.0f, 1.0f, -17.0f, 1.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.643914213228014f,
                            0.236882800924671f,
                            0.236882800924671f, 0.236882800924671f, 0.236882800924671f, 0.087144312427294f,
                            0.087144312427294f,

                            0.087144312427294f, 0.087144312427294f, 0.032058600957022f, 0.032058600957022f,
                            0.032058600957022f,
                            0.032058600957022f, 7.246299848982885e-08f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f
                    };
            break;
        }
    case -2:
    case 1:
        {
            inputShape = {2, 5, 2};

            inputData =
                    {
                            17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f,

                            17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                            0.087144312427294f,
                            0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                            7.246299848982885e-08f,

                            0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                            0.087144312427294f,
                            0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                            7.246299848982885e-08f
                    };
        break;
        }
    case -1:
    case 2:
        {
            inputShape = {2, 2, 5};

            inputData =
                    {
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f,
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,

                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f
                    };
            break;
        }
    }

    return Simple3dSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                             inputShape, outputData, inputData, axis);
}

LayerTestResult<float,4> Simple4dSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    Simple4dSoftmaxData data;
    return Simple4dSoftmaxTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory,
                                                             beta, data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<float,4> Simple4dAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta,
        int axis)
{
    armnn::TensorShape inputShape;
    std::vector<float> inputData;
    std::vector<float> outputData;
    switch (axis)
    {
    case -4:
    case 0:
        {
            inputShape = {5, 2, 2, 2};

            inputData =
                    {
                            17.0f, -1.0f, 17.0f, -1.0f, 17.0f, -1.0f, 17.0f, -1.0f, 16.0f, -2.0f,
                            16.0f, -2.0f, 16.0f, -2.0f, 16.0f, -2.0f, 15.0f, -3.0f, 15.0f, -3.0f,
                            15.0f, -3.0f, 15.0f, -3.0f, 14.0f, -4.0f, 14.0f, -4.0f, 14.0f, -4.0f,
                            14.0f, -4.0f, 1.0f, -17.0f, 1.0f, -17.0f, 1.0f, -17.0f, 1.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.643914213228014f,
                            0.643914213228014f,
                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.236882800924671f,
                            0.236882800924671f,
                            0.236882800924671f, 0.236882800924671f, 0.236882800924671f, 0.236882800924671f,
                            0.236882800924671f,
                            0.236882800924671f, 0.087144312427294f, 0.087144312427294f, 0.087144312427294f,
                            0.087144312427294f,

                            0.087144312427294f, 0.087144312427294f, 0.087144312427294f, 0.087144312427294f,
                            0.032058600957022f,
                            0.032058600957022f, 0.032058600957022f, 0.032058600957022f, 0.032058600957022f,
                            0.032058600957022f,
                            0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f,
                            7.246299848982885e-08f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f, 7.246299848982885e-08f
                    };
            break;
        }
    case -3:
    case 1:
        {
            inputShape = {2, 5, 2, 2};

            inputData =
                    {
                            17.0f, -1.0f, 17.0f, -1.0f, 16.0f, -2.0f, 16.0f, -2.0f, 15.0f, -3.0f,
                            15.0f, -3.0f, 14.0f, -4.0f, 14.0f, -4.0f, 1.0f, -17.0f, 1.0f, -17.0f,
                            17.0f, -1.0f, 17.0f, -1.0f, 16.0f, -2.0f, 16.0f, -2.0f, 15.0f, -3.0f,
                            15.0f, -3.0f, 14.0f, -4.0f, 14.0f, -4.0f, 1.0f, -17.0f, 1.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.643914213228014f,
                            0.236882800924671f,
                            0.236882800924671f, 0.236882800924671f, 0.236882800924671f, 0.087144312427294f,
                            0.087144312427294f,
                            0.087144312427294f, 0.087144312427294f, 0.032058600957022f, 0.032058600957022f,
                            0.032058600957022f,
                            0.032058600957022f, 7.246299848982885e-08f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f,


                            0.643914213228014f, 0.643914213228014f, 0.643914213228014f, 0.643914213228014f,
                            0.236882800924671f,
                            0.236882800924671f, 0.236882800924671f, 0.236882800924671f, 0.087144312427294f,
                            0.087144312427294f,
                            0.087144312427294f, 0.087144312427294f, 0.032058600957022f, 0.032058600957022f,
                            0.032058600957022f,
                            0.032058600957022f, 7.246299848982885e-08f, 7.246299848982885e-08f, 7.246299848982885e-08f,
                            7.246299848982885e-08f
                    };
            break;
        }
    case -2:
    case 2:
        {
        inputShape = {2, 2, 5, 2};

        inputData =
                {
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f,
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f,
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f,
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f
                };

        outputData =
                {
                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f,
                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f,

                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f,
                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f
                };
        break;
        }
    case -1:
    case 3:
        {
            inputShape = {2, 2, 2, 5};

            inputData =
                    {
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f,
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f,
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f,
                            17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f
                    };

            outputData =
                    {
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,

                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f,
                            0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                            7.246299848982885e-08f
                    };
            break;
        }
    }

    return Simple4dSoftmaxTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        beta,
        inputShape,
        outputData,
        inputData,
        axis);
}

LayerTestResult<uint8_t,2> SimpleSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float beta)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, beta);
}

LayerTestResult<uint8_t,3> Simple3dSoftmaxUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    Simple3dSoftmaxOutputData data;
    return Simple3dSoftmaxTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        beta,
        data.inputShape,
        data.outputData,
        data.inputData);
}

LayerTestResult<uint8_t,4> Simple4dSoftmaxUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    Simple4dSoftmaxData data;

    return Simple4dSoftmaxTestImpl<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                                     data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<armnn::Half,2> SimpleSoftmaxFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::Float16>(workloadFactory, memoryManager, tensorHandleFactory, beta);
}

LayerTestResult<armnn::Half,3> Simple3dSoftmaxFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    Simple3dSoftmaxOutputData data;
    return Simple3dSoftmaxTestImpl<armnn::DataType::Float16>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                             data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<armnn::Half,4> Simple4dSoftmaxFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    Simple4dSoftmaxData data;
    return Simple4dSoftmaxTestImpl<armnn::DataType::Float16>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                             data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<int16_t,2> SimpleSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    return SimpleSoftmaxTestImpl<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, beta);
}

LayerTestResult<int16_t,3> Simple3dSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    Simple3dSoftmaxOutputData data;
    return Simple3dSoftmaxTestImpl<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                                     data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<int16_t,4> Simple4dSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta)
{
    Simple4dSoftmaxData data;

    return Simple4dSoftmaxTestImpl<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, beta,
                                                                     data.inputShape, data.outputData, data.inputData);
}

LayerTestResult<float,2> CompareSoftmaxTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    float beta)
{
    return CompareSoftmaxTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory, beta);
}

LayerTestResult<uint8_t,2> CompareSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    float beta)
{
    return CompareSoftmaxTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory, refTensorHandleFactory, beta);
}
