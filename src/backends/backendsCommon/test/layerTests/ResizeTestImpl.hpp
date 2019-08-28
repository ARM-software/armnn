//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <Permute.hpp>
#include <ResolveType.hpp>
#include <TensorUtils.hpp>

#include <armnn/ArmNN.hpp>

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

//
// ResizeBilinear
//

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeBilinearNopTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.5f);
        inputTensorInfo.SetQuantizationOffset(-3);
        outputTensorInfo.SetQuantizationScale(1.5f);
        outputTensorInfo.SetQuantizationOffset(-3);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                1, 2, 3, 4,
                2, 3, 4, 5,
                3, 4, 5, 6,
                4, 5, 6, 7
            }
        : std::initializer_list<float>
            {
                1.0f, 2.0f, 3.0f, 4.0f,
                2.0f, 3.0f, 4.0f, 5.0f,
                3.0f, 4.0f, 5.0f, 6.0f,
                4.0f, 5.0f, 6.0f, 7.0f,

                1.0f, 2.0f, 3.0f, 4.0f,
                2.0f, 3.0f, 4.0f, 5.0f,
                3.0f, 4.0f, 5.0f, 6.0f,
                4.0f, 5.0f, 6.0f, 7.0f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = input;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleResizeBilinearTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 2, 2, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 1, 1, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 1, 1, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(0.1567f);
        inputTensorInfo.SetQuantizationOffset(1);
        outputTensorInfo.SetQuantizationScale(0.1567f);
        outputTensorInfo.SetQuantizationOffset(1);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                  1, 255,
                200, 250
            }
        : std::initializer_list<float>
            {
                  1.0f, 255.0f,
                200.0f, 250.0f,

                250.0f, 200.0f,
                250.0f,   1.0f
            };

    // The 'resize bilinear' operation projects the top-left corner of output texels into the input image,
    // then figures out the interpolants and weights. Note this is different to projecting the centre of the
    // output texel. Thus, for a input matrix of 2x2, we'll expect the output 1x1 matrix to contain, as
    // its single element, the value that was at position (0,0) of the input matrix (rather than an average,
    // which we would expect if projecting the centre).

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                1
            }
        : std::initializer_list<float>
            {
                  1.0f,

                250.0f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeBilinearSqMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 2, 2, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(3.141592f);
        inputTensorInfo.SetQuantizationOffset(3);
        outputTensorInfo.SetQuantizationScale(3.141592f);
        outputTensorInfo.SetQuantizationOffset(3);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                1, 2, 3, 4,
                2, 3, 4, 5,
                3, 4, 5, 6,
                4, 5, 6, 7
            }
        : std::initializer_list<float>
            {
                1.0f, 2.0f, 3.0f, 4.0f,
                2.0f, 3.0f, 4.0f, 5.0f,
                3.0f, 4.0f, 5.0f, 6.0f,
                4.0f, 5.0f, 6.0f, 7.0f,

                7.0f, 6.0f, 5.0f, 4.0f,
                6.0f, 5.0f, 4.0f, 3.0f,
                5.0f, 4.0f, 3.0f, 2.0f,
                4.0f, 3.0f, 2.0f, 1.0f
            };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                1, 3,
                3, 5
            }
        : std::initializer_list<float>
            {
                1.0f, 3.0f,
                3.0f, 5.0f,

                7.0f, 5.0f,
                5.0f, 3.0f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeBilinearMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 2, 3, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 3, 5, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 1, 2, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 2, 3, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.5f);
        inputTensorInfo.SetQuantizationOffset(-1);
        outputTensorInfo.SetQuantizationScale(1.5f);
        outputTensorInfo.SetQuantizationOffset(-1);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                3.0f,  4.5f,  6.0f, // 1,  2,  3, : Expected quantised values
                9.0f, 13.5f, 21.0f  // 5,  8, 13
            }
        : std::initializer_list<float>
            {
                  1.0f,   2.0f,   3.0f,   5.0f,   8.0f,
                 13.0f,  21.0f,  34.0f,  55.0f,  89.0f,
                144.0f, 233.0f, 377.0f, 610.0f, 987.0f,

                987.0f, 610.0f, 377.0f, 233.0f, 144.0f,
                 89.0f,  55.0f,  34.0f,  21.0f,  13.0f,
                   8.0f,  5.0f,   3.0f,   2.0f,   1.0f
            };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                3.0f, 5.25f // 1, 3
            }
        : std::initializer_list<float>
            {
                 1.0f,   2.6666f,   6.00f,
                78.5f, 179.3333f, 401.00f,

                987.0f, 454.6670f, 203.33f,
                 48.5f,  22.3333f,  10.00f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeBilinearMagTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 3, 2, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 3, 2, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 3, 5, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 3, 5, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(0.010765f);
        inputTensorInfo.SetQuantizationOffset(7);
        outputTensorInfo.SetQuantizationScale(0.010132f);
        outputTensorInfo.SetQuantizationOffset(-18);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                0.183005f, 2.379065f, // 24, 228, : Expected quantised values
                1.054970f, 1.302565f, // 105, 128,
                2.400595f, 0.688960f  // 230, 71
            }
        : std::initializer_list<float>
            {
                  1.0f,   2.0f,
                 13.0f,  21.0f,
                144.0f, 233.0f,

                233.0f, 144.0f,
                 21.0f,  13.0f,
                  2.0f,   1.0f
            };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                0.18300501f, 1.06142902f, 1.93985295f, 2.37906504f, 2.37906504f,
                1.05497003f, 1.15400803f, 1.25304604f, 1.30256498f, 1.30256498f,
                2.40059495f, 1.71594095f, 1.03128707f, 0.68896002f, 0.68896002f
                // 0, 87, 173, 217, 217, : Expected quantised values
                // 86, 96, 106, 111, 111,
                // 219, 151, 84, 50, 50
            }
        : std::initializer_list<float>
            {
                  1.0f,   1.4f,   1.8f,   2.0f,   2.0f,
                 13.0f,  16.2f,  19.4f,  21.0f,  21.0f,
                144.0f, 179.6f, 215.2f, 233.0f, 233.0f,

                233.0f, 197.4f, 161.8f, 144.0f, 144.0f,
                 21.0f,  17.8f,  14.6f,  13.0f,  13.0f,
                  2.0f,   1.6f,   1.2f,   1.0f,   1.0f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::Bilinear;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

//
// ResizeNearestNeighbor
//

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeNearestNeighborNopTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.5f);
        inputTensorInfo.SetQuantizationOffset(-3);
        outputTensorInfo.SetQuantizationScale(1.5f);
        outputTensorInfo.SetQuantizationOffset(-3);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                1, 2, 3, 4,
                2, 3, 4, 5,
                3, 4, 5, 6,
                4, 5, 6, 7
            }
        : std::initializer_list<float>
            {
                1.0f, 2.0f, 3.0f, 4.0f,
                2.0f, 3.0f, 4.0f, 5.0f,
                3.0f, 4.0f, 5.0f, 6.0f,
                4.0f, 5.0f, 6.0f, 7.0f,

                1.0f, 2.0f, 3.0f, 4.0f,
                2.0f, 3.0f, 4.0f, 5.0f,
                3.0f, 4.0f, 5.0f, 6.0f,
                4.0f, 5.0f, 6.0f, 7.0f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = input;

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Method = armnn::ResizeMethod::NearestNeighbor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleResizeNearestNeighborTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 2, 2, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 1, 1, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 1, 1, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(0.1567f);
        inputTensorInfo.SetQuantizationOffset(1);
        outputTensorInfo.SetQuantizationScale(0.1567f);
        outputTensorInfo.SetQuantizationOffset(1);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                  1, 255,
                200, 250
            }
        : std::initializer_list<float>
            {
                  1.0f, 255.0f,
                200.0f, 250.0f,

                250.0f, 200.0f,
                250.0f,   1.0f
            };

    // The 'resize' operation projects the top-left corner of output texels into the input image,
    // then figures out the interpolants and weights. Note this is different to projecting the centre of the
    // output texel. Thus, for a input matrix of 2x2, we'll expect the output 1x1 matrix to contain, as
    // its single element, the value that was at position (0,0) of the input matrix (rather than an average,
    // which we would expect if projecting the centre).

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                1
            }
        : std::initializer_list<float>
            {
                  1.0f,

                250.0f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::NearestNeighbor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeNearestNeighborSqMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 4, 4, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 4, 4, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 2, 2, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 2, 2, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(3.141592f);
        inputTensorInfo.SetQuantizationOffset(3);
        outputTensorInfo.SetQuantizationScale(3.141592f);
        outputTensorInfo.SetQuantizationOffset(3);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                1, 2, 3, 4,
                2, 3, 4, 5,
                3, 4, 5, 6,
                4, 5, 6, 7
            }
        : std::initializer_list<float>
            {
                1.0f, 2.0f, 3.0f, 4.0f,
                2.0f, 3.0f, 4.0f, 5.0f,
                3.0f, 4.0f, 5.0f, 6.0f,
                4.0f, 5.0f, 6.0f, 7.0f,

                7.0f, 6.0f, 5.0f, 4.0f,
                6.0f, 5.0f, 4.0f, 3.0f,
                5.0f, 4.0f, 3.0f, 2.0f,
                4.0f, 3.0f, 2.0f, 1.0f
            };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                1, 3,
                3, 5
            }
        : std::initializer_list<float>
            {
                1.0f, 3.0f,
                3.0f, 5.0f,

                7.0f, 5.0f,
                5.0f, 3.0f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    descriptor.m_Parameters.m_Method     = armnn::ResizeMethod::NearestNeighbor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeNearestNeighborMinTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 2, 3, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 3, 5, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 1, 2, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 2, 3, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.5f);
        inputTensorInfo.SetQuantizationOffset(-1);
        outputTensorInfo.SetQuantizationScale(1.5f);
        outputTensorInfo.SetQuantizationOffset(-1);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                3.0f,  4.5f,  6.0f, // 1,  2,  3, : Expected quantised values
                9.0f, 13.5f, 21.0f  // 5,  8, 13
            }
        : std::initializer_list<float>
            {
                  1.0f,   2.0f,   3.0f,   5.0f,   8.0f,
                 13.0f,  21.0f,  34.0f,  55.0f,  89.0f,
                144.0f, 233.0f, 377.0f, 610.0f, 987.0f,

                987.0f, 610.0f, 377.0f, 233.0f, 144.0f,
                 89.0f,  55.0f,  34.0f,  21.0f,  13.0f,
                  8.0f,   5.0f,   3.0f,   2.0f,   1.0f
            };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                3.0f, 4.5f // 1, 3
            }
        : std::initializer_list<float>
            {
                  1.f,   2.f,   5.f,
                 13.f,  21.f,  55.f,

                987.f, 610.f, 233.f,
                 89.f,  55.f,  21.f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    descriptor.m_Parameters.m_Method = armnn::ResizeMethod::NearestNeighbor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ResizeNearestNeighborMagTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::DataLayout dataLayout,
        float inQuantScale,
        int32_t inQuantOffset,
        float outQuantScale,
        int32_t outQuantOffset)
{
    armnn::TensorInfo inputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 3, 2, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 3, 2, dataLayout, ArmnnType);

    armnn::TensorInfo outputTensorInfo = armnn::IsQuantizedType<T>()
        ?  armnnUtils::GetTensorInfo(1, 1, 3, 5, dataLayout, ArmnnType)
        :  armnnUtils::GetTensorInfo(1, 2, 3, 5, dataLayout, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(inQuantScale);
        inputTensorInfo.SetQuantizationOffset(inQuantOffset);
        outputTensorInfo.SetQuantizationScale(outQuantScale);
        outputTensorInfo.SetQuantizationOffset(outQuantOffset);
    }

    std::vector<float> inputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                0.183005f, 2.379065f, //  24, 228, : expected quantised values
                1.054970f, 1.302565f, // 105, 128,
                2.400595f, 0.688960f  // 230, 71
            }
        : std::initializer_list<float>
            {
                  1.0f,   2.0f,
                 13.0f,  21.0f,
                144.0f, 233.0f,

                233.0f, 144.0f,
                 21.0f,  13.0f,
                  2.0f,   1.0f
            };

    std::vector<float> outputData = armnn::IsQuantizedType<T>()
        ? std::initializer_list<float>
            {
                0.183005f, 0.183005f, 0.183005f, 2.379065f, 2.379065f,
                1.054970f, 1.054970f, 1.054970f, 1.302565f, 1.302565f,
                2.400595f, 2.400595f, 2.400595f, 0.688960f, 0.688960f
            }
        : std::initializer_list<float>
            {
                  1.f,   1.f,   1.f,   2.f,   2.f,
                 13.f,  13.f,  13.f,  21.f,  21.f,
                144.f, 144.f, 144.f, 233.f, 233.f,

                233.f, 233.f, 233.f, 144.f, 144.f,
                 21.f,  21.f,  21.f,  13.f,  13.f,
                  2.f,   2.f,   2.f,   1.f,   1.f
            };

    const armnn::PermutationVector NCHWToNHWC = { 0, 3, 1, 2 };
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NCHWToNHWC, inputData.data(), tmp.data(), sizeof(float));
        inputData = tmp;

        std::vector<float> tmp1(outputData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NCHWToNHWC, outputData.data(), tmp1.data(), sizeof(float));
        outputData = tmp1;
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(inputTensorInfo.GetQuantizationScale(),
                                                                      inputTensorInfo.GetQuantizationOffset(),
                                                                      inputData));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
                                             QuantizedVector<T>(outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset(),
                                                                outputData));

    std::unique_ptr <armnn::ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ResizeQueueDescriptor descriptor;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    descriptor.m_Parameters.m_Method = armnn::ResizeMethod::NearestNeighbor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateResize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());
    return result;
}
