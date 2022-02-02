//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StackTestImpl.hpp"
#include <armnnTestUtils/LayerTestResult.hpp>

#include <ResolveType.hpp>


#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<armnn::DataType ArmnnType, typename T, std::size_t outputDimLength>
LayerTestResult<T, outputDimLength> StackTestHelper(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::TensorInfo& inputTensorInfo,
        const armnn::TensorInfo& outputTensorInfo,
        unsigned int axis,
        const std::vector<std::vector<T>>& inputData,
        const std::vector<T>& outputExpectedData)
{
    IgnoreUnused(memoryManager);
    unsigned int numInputs = static_cast<unsigned int>(inputData.size());
    std::vector<std::vector<T>> inputs;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        inputs.emplace_back(inputData[i]);
    }

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::vector<std::unique_ptr<armnn::ITensorHandle>> inputHandles;
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        inputHandles.push_back(tensorHandleFactory.CreateTensorHandle(inputTensorInfo));
    }
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

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
        CopyDataToITensorHandle(inputHandle.get(), inputs[i].data());
    }

    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());
    outputHandle->Allocate();

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Stack,
                                                                                descriptor,
                                                                                info);

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, outputDimLength>(actualOutput,
                                               outputExpectedData,
                                               outputHandle->GetShape(),
                                               outputTensorInfo.GetShape());
}

} // anonymous namespace

//
// Implementation templates
//

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StackAxis0TestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        0U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StackOutput4DAxis1TestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StackOutput4DAxis2TestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        2U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StackOutput4DAxis3TestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        3U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StackOutput3DInputs3TestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 5> StackOutput5DTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        1U,
        inputData,
        outputExpectedData
    );
}

//
// Implementation functions
//

LayerTestResult<float, 4> StackAxis0Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StackAxis0TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<float, 4> StackOutput4DAxis1Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StackOutput4DAxis1TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<float, 4> StackOutput4DAxis2Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StackOutput4DAxis2TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<float, 4> StackOutput4DAxis3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StackOutput4DAxis3TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<float, 3> StackOutput3DInputs3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StackOutput3DInputs3TestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<float, 5> StackOutput5DFloat32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StackOutput5DTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> StackFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    armnn::TensorInfo inputTensorInfo ({ 3, 2, 3 }, armnn::DataType::Float16);
    armnn::TensorInfo outputTensorInfo({ 3, 2, 2, 3 }, armnn::DataType::Float16);

    std::vector<std::vector<armnn::Half>> inputData;

    inputData.push_back(
    {
        1.0_h,  2.0_h,  3.0_h,
        4.0_h,  5.0_h,  6.0_h,

        7.0_h,  8.0_h,  9.0_h,
        10.0_h, 11.0_h, 12.0_h,

        13.0_h, 14.0_h, 15.0_h,
        16.0_h, 17.0_h, 18.0_h
    });

    inputData.push_back(
    {
        19.0_h, 20.0_h, 21.0_h,
        22.0_h, 23.0_h, 24.0_h,

        25.0_h, 26.0_h, 27.0_h,
        28.0_h, 29.0_h, 30.0_h,

        31.0_h, 32.0_h, 33.0_h,
        34.0_h, 35.0_h, 36.0_h
    });

    std::vector<armnn::Half> outputExpectedData =
    {
        1.0_h,  2.0_h,  3.0_h,
        19.0_h, 20.0_h, 21.0_h,

        4.0_h,  5.0_h,  6.0_h,
        22.0_h, 23.0_h, 24.0_h,

        7.0_h,  8.0_h,  9.0_h,
        25.0_h, 26.0_h, 27.0_h,

        10.0_h, 11.0_h, 12.0_h,
        28.0_h, 29.0_h, 30.0_h,

        13.0_h, 14.0_h, 15.0_h,
        31.0_h, 32.0_h, 33.0_h,

        16.0_h, 17.0_h, 18.0_h,
        34.0_h, 35.0_h, 36.0_h
    };

    return StackTestHelper<armnn::DataType::Float16, armnn::Half, 4>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        2U,
        inputData,
        outputExpectedData
    );
}

LayerTestResult<int32_t, 4> StackInt32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StackAxis0TestImpl<armnn::DataType::Signed32>(workloadFactory, memoryManager, tensorHandleFactory);
}