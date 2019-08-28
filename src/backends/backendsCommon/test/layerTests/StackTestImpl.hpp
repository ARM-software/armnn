//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <ResolveType.hpp>

#include <armnn/ArmNN.hpp>

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

namespace
{

template<armnn::DataType ArmnnType, typename T, std::size_t outputDimLength>
LayerTestResult<T, outputDimLength> StackTestHelper(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::TensorInfo& inputTensorInfo,
        const armnn::TensorInfo& outputTensorInfo,
        unsigned int axis,
        const std::vector<std::vector<T>>& inputData,
        const std::vector<T>& outputExpectedData)
{
    unsigned int numInputs = static_cast<unsigned int>(inputData.size());
    std::vector<boost::multi_array<T, outputDimLength-1>> inputs;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        inputs.push_back(MakeTensor<T, outputDimLength-1>(inputTensorInfo, inputData[i]));
    }

    LayerTestResult<T, outputDimLength> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, outputDimLength>(outputTensorInfo, outputExpectedData);

    std::vector<std::unique_ptr<armnn::ITensorHandle>> inputHandles;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        inputHandles.push_back(workloadFactory.CreateTensorHandle(inputTensorInfo));
    }
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::StackQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Axis = axis;
    descriptor.m_Parameters.m_InputShape = inputTensorInfo.GetShape();
    descriptor.m_Parameters.m_NumInputs = numInputs;

    armnn::WorkloadInfo info;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        std::unique_ptr<armnn::ITensorHandle>& inputHandle = inputHandles[i];
        AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
        inputHandle->Allocate();
        CopyDataToITensorHandle(inputHandle.get(), inputs[i].origin());
    }

    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());
    outputHandle->Allocate();

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateStack(descriptor, info);

    workload->Execute();

    CopyDataFromITensorHandle(result.output.origin(), outputHandle.get());

    return result;
}

} // anonymous namespace

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Stack0AxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 2, 3, 2, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18,


        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    };

    return StackTestHelper<ArmnnType, T, 4>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        0U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Stack4dOutput1AxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 3, 2, 2, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        4, 5, 6,

        19, 20, 21,
        22, 23, 24,


        7, 8, 9,
        10, 11, 12,

        25, 26, 27,
        28, 29, 30,


        13, 14, 15,
        16, 17, 18,

        31, 32, 33,
        34, 35, 36
    };

    return StackTestHelper<ArmnnType, T, 4>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Stack4dOutput2AxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 3, 2, 2, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        19, 20, 21,

        4, 5, 6,
        22, 23, 24,

        7, 8, 9,
        25, 26, 27,

        10, 11, 12,
        28, 29, 30,

        13, 14, 15,
        31, 32, 33,

        16, 17, 18,
        34, 35, 36
    };

    return StackTestHelper<ArmnnType, T, 4>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        2U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Stack4dOutput3AxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 3, 2, 3, 2 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> outputExpectedData =
    {
        1, 19,
        2, 20,
        3, 21,

        4, 22,
        5, 23,
        6, 24,


        7, 25,
        8, 26,
        9, 27,

        10, 28,
        11, 29,
        12, 30,


        13, 31,
        14, 32,
        15, 33,

        16, 34,
        17, 35,
        18, 36
    };

    return StackTestHelper<ArmnnType, T, 4>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        3U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Stack3dOutput1Axis3InputTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 3, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 3, 3, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    inputData.push_back(
    {
        10, 11, 12,
        13, 14, 15,
        16, 17, 18
    });

    inputData.push_back(
    {
        19, 20, 21,
        22, 23, 24,
        25, 26, 27
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        10, 11, 12,
        19, 20, 21,

        4, 5, 6,
        13, 14, 15,
        22, 23, 24,

        7, 8, 9,
        16, 17, 18,
        25, 26, 27
    };

    return StackTestHelper<ArmnnType, T, 3>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> Stack5dOutputTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo inputTensorInfo ({ 2, 2, 2, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 2, 2, 2, 2, 3 }, ArmnnType);

    std::vector<std::vector<T>> inputData;

    inputData.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,


        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24
    });

    inputData.push_back(
    {
        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36,


        37, 38, 39,
        40, 41, 42,

        43, 44, 45,
        46, 47, 48
    });

    std::vector<T> outputExpectedData =
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,


        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36,



        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,


        37, 38, 39,
        40, 41, 42,

        43, 44, 45,
        46, 47, 48

    };

    return StackTestHelper<ArmnnType, T, 5>(
        workloadFactory,
        memoryManager,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}
