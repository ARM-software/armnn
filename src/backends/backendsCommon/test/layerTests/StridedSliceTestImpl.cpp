//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StridedSliceTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>


#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<typename T, std::size_t InDim, std::size_t OutDim>
LayerTestResult<T, OutDim> StridedSliceTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::TensorInfo& inputTensorInfo,
    armnn::TensorInfo& outputTensorInfo,
    std::vector<float>& inputData,
    std::vector<float>& outputExpectedData,
    armnn::StridedSliceQueueDescriptor descriptor,
    const float qScale = 1.0f,
    const int32_t qOffset = 0)
{
    IgnoreUnused(memoryManager);
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);

        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(outputExpectedData, qScale, qOffset);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle =
        tensorHandleFactory.CreateTensorHandle(inputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> outputHandle =
        tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::StridedSlice,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, OutDim>(actualOutput,
                                      expectedOutput,
                                      outputHandle->GetShape(),
                                      outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSlice4dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {1, 2, 3, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin  = {1, 0, 0, 0};
    desc.m_Parameters.m_End    = {2, 2, 3, 1};
    desc.m_Parameters.m_Stride = {1, 1, 1, 1};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

        3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

        5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f
    });

    return StridedSliceTestImpl<T, 4, 4>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSlice4dReverseTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {1, 2, 3, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin  = {1, -1, 0, 0};
    desc.m_Parameters.m_End    = {2, -3, 3, 1};
    desc.m_Parameters.m_Stride = {1, -1, 1, 1};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

        3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

        5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f
    });

    return StridedSliceTestImpl<T, 4, 4>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSliceSimpleStrideTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {2, 1, 2, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin  = {0, 0, 0, 0};
    desc.m_Parameters.m_End    = {3, 2, 3, 1};
    desc.m_Parameters.m_Stride = {2, 2, 2, 1};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

        3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

        5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        1.0f, 1.0f,

        5.0f, 5.0f
    });

    return StridedSliceTestImpl<T, 4, 4>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSliceSimpleRangeMaskTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {3, 2, 3, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin     = {1, 1, 1, 1};
    desc.m_Parameters.m_End       = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride    = {1, 1, 1, 1};
    desc.m_Parameters.m_BeginMask = (1 << 4) - 1;
    desc.m_Parameters.m_EndMask   = (1 << 4) - 1;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

        3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

        5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

        3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

        5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
    });

    return StridedSliceTestImpl<T, 4, 4>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSliceShrinkAxisMaskTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {3, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 1, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 1) | (1 << 2);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

        13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        2.0f, 8.0f, 14.0f
    });

    return StridedSliceTestImpl<T, 4, 2>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StridedSliceShrinkAxisMaskBitPosition0Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {2, 3, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 0);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(3, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
            });

    return StridedSliceTestImpl<T, 4, 3>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StridedSliceShrinkAxisMaskBitPosition1Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {3, 3, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 1);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(3, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f, 13.0f, 14.0f, 15.0f
            });

    return StridedSliceTestImpl<T, 4, 3>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StridedSliceShrinkAxisMaskBitPosition2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {3, 2, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 2);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(3, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 4.0f, 7.0f, 10.0f, 13.0f, 16.0f
            });

    return StridedSliceTestImpl<T, 4, 3>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StridedSliceShrinkAxisMaskBitPosition3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {3, 2, 3};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 3);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(3, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    return StridedSliceTestImpl<T, 4, 3>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSliceShrinkAxisMaskBitPosition0And1Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {3, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 0) | (1 << 1);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f
            });

    return StridedSliceTestImpl<T, 4, 2>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSliceShrinkAxisMaskBitPosition0Dim3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {2, 3, 1};
    unsigned int outputShape[] = {3, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0};
    desc.m_Parameters.m_End            = {0, 0, 0};
    desc.m_Parameters.m_Stride         = {1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 0);

    inputTensorInfo = armnn::TensorInfo(3, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f
            });

    return StridedSliceTestImpl<T, 3, 2>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

void FillVector(std::vector<float>& inputArray, float start, float step)
{
    for (uint32_t i = 0; i < inputArray.size(); ++i)
    {
        inputArray[i] = start;
        start += step;
    }
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSliceShrinkAxisMaskCTSTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {1, 1, 8, 942};
    unsigned int outputShape[] = {1, 1, 1, 279};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 1, 229};
    desc.m_Parameters.m_End            = {1, 1, 2, 787};
    desc.m_Parameters.m_Stride         = {2, 3, 3, 2};
    desc.m_Parameters.m_BeginMask      = 2;
    desc.m_Parameters.m_EndMask        = 0;
    desc.m_Parameters.m_ShrinkAxisMask = 0;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    // Array from 1 to 7535
    std::vector<float> input(7536);
    FillVector(input, 1.0f, 1.0f);

    // Array from 1171 to 1727 in steps of 2
    std::vector<float> outputExpected(279);
    FillVector(outputExpected, 1171.0, 2.0f);

    return StridedSliceTestImpl<T, 4, 4>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSliceShrinkAxisMaskBitPosition0And2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {2, 1};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 0) | (1 << 2);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 4.0f
            });

    return StridedSliceTestImpl<T, 4, 2>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSliceShrinkAxisMaskBitPosition0And3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {2, 3};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 0) | (1 << 3);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
            });

    return StridedSliceTestImpl<T, 4, 2>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> StridedSliceShrinkAxisMaskBitPosition0And1And3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 2, 3, 1};
    unsigned int outputShape[] = {3};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin          = {0, 0, 0, 0};
    desc.m_Parameters.m_End            = {1, 1, 1, 1};
    desc.m_Parameters.m_Stride         = {1, 1, 1, 1};
    desc.m_Parameters.m_EndMask        = (1 << 4) - 1;
    desc.m_Parameters.m_ShrinkAxisMask = (1 << 0) | (1 << 1) | (1 << 3);

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(1, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,

                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 2.0f, 3.0f
            });

    return StridedSliceTestImpl<T, 4, 1>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StridedSlice3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 3, 3};
    unsigned int outputShape[] = {2, 2, 2};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin   = {0, 0, 0};
    desc.m_Parameters.m_End     = {1, 1, 1};
    desc.m_Parameters.m_Stride  = {2, 2, 2};
    desc.m_Parameters.m_EndMask = (1 << 3) - 1;

    inputTensorInfo = armnn::TensorInfo(3, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(3, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,

        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,

        19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        1.0f, 3.0f, 7.0f, 9.0f,

        19.0f, 21.0f, 25.0f, 27.0f
    });

    return StridedSliceTestImpl<T, 3, 3>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StridedSlice3dReverseTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 3, 3};
    unsigned int outputShape[] = {2, 2, 2};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin  = {-1, -1, -1};
    desc.m_Parameters.m_End    = {-4, -4, -4};
    desc.m_Parameters.m_Stride = {-2, -2, -2};

    inputTensorInfo = armnn::TensorInfo(3, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(3, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,

        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,

        19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        27.0f, 25.0f, 21.0f, 19.0f,

        9.0f, 7.0f, 3.0f, 1.0f
    });

    return StridedSliceTestImpl<T, 3, 3>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSlice2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 3};
    unsigned int outputShape[] = {2, 2};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin   = {0, 0};
    desc.m_Parameters.m_End     = {1, 1};
    desc.m_Parameters.m_Stride  = {2, 2};
    desc.m_Parameters.m_EndMask = (1 << 2) - 1;

    inputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f,

        4.0f, 5.0f, 6.0f,

        7.0f, 8.0f, 9.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        1.0f, 3.0f,

        7.0f, 9.0f
    });

    return StridedSliceTestImpl<T, 2, 2>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSlice2dReverseTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = {3, 3};
    unsigned int outputShape[] = {2, 2};

    armnn::StridedSliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin     = {0, 0};
    desc.m_Parameters.m_End       = {1, 1};
    desc.m_Parameters.m_Stride    = {-2, -2};
    desc.m_Parameters.m_BeginMask = (1 << 2) - 1;
    desc.m_Parameters.m_EndMask   = (1 << 2) - 1;

    inputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f,

        4.0f, 5.0f, 6.0f,

        7.0f, 8.0f, 9.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        9.0f, 7.0f,

        3.0f, 1.0f
    });

    return StridedSliceTestImpl<T, 2, 2>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

} // anonymous namespace

LayerTestResult<float, 4> StridedSlice4dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice4dTest<armnn::DataType::Float32>(workloadFactory,
                                                        memoryManager,
                                                        tensorHandleFactory);
}

LayerTestResult<float, 4> StridedSlice4dReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice4dReverseTest<armnn::DataType::Float32>(workloadFactory,
                                                               memoryManager,
                                                               tensorHandleFactory);
}

LayerTestResult<float, 4> StridedSliceSimpleStrideFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::Float32>(workloadFactory,
                                                                  memoryManager,
                                                                  tensorHandleFactory);
}

LayerTestResult<float, 4> StridedSliceSimpleRangeMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::Float32>(workloadFactory,
                                                                     memoryManager,
                                                                     tensorHandleFactory);
}

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::Float32>(workloadFactory,
                                                                    memoryManager,
                                                                    tensorHandleFactory);
}

LayerTestResult<float, 4> StridedSliceShrinkAxisMaskCTSFloat32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskCTSTest<armnn::DataType::Float32>(workloadFactory,
                                                                       memoryManager,
                                                                       tensorHandleFactory);
}

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskBitPosition0Dim3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0Dim3Test<armnn::DataType::Float32>(workloadFactory,
                                                                                    memoryManager,
                                                                                    tensorHandleFactory);
}

LayerTestResult<float, 3> StridedSliceShrinkAxisMaskBitPosition0Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0Test<armnn::DataType::Float32>(workloadFactory,
                                                                                memoryManager,
                                                                                tensorHandleFactory);
}

LayerTestResult<float, 3> StridedSliceShrinkAxisMaskBitPosition1Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition1Test<armnn::DataType::Float32>(workloadFactory,
                                                                                memoryManager,
                                                                                tensorHandleFactory);
}

LayerTestResult<float, 3> StridedSliceShrinkAxisMaskBitPosition2Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition2Test<armnn::DataType::Float32>(workloadFactory,
                                                                                memoryManager,
                                                                                tensorHandleFactory);
}

LayerTestResult<float, 3> StridedSliceShrinkAxisMaskBitPosition3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition3Test<armnn::DataType::Float32>(workloadFactory,
                                                                                memoryManager,
                                                                                tensorHandleFactory);
}

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskBitPosition0And1Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0And1Test<armnn::DataType::Float32>(workloadFactory,
                                                                                    memoryManager,
                                                                                    tensorHandleFactory);
}

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskBitPosition0And2Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0And2Test<armnn::DataType::Float32>(workloadFactory,
                                                                                    memoryManager,
                                                                                    tensorHandleFactory);
}

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskBitPosition0And3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0And3Test<armnn::DataType::Float32>(workloadFactory,
                                                                                    memoryManager,
                                                                                    tensorHandleFactory);
}

LayerTestResult<float, 1> StridedSliceShrinkAxisMaskBitPosition0And1And3Float32Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0And1And3Test<armnn::DataType::Float32>(workloadFactory,
                                                                                        memoryManager,
                                                                                        tensorHandleFactory);
}

LayerTestResult<float, 3> StridedSlice3dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice3dTest<armnn::DataType::Float32>(workloadFactory,
                                                        memoryManager,
                                                        tensorHandleFactory);
}

LayerTestResult<float, 3> StridedSlice3dReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice3dReverseTest<armnn::DataType::Float32>(workloadFactory,
                                                               memoryManager,
                                                               tensorHandleFactory);
}

LayerTestResult<float, 2> StridedSlice2dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice2dTest<armnn::DataType::Float32>(workloadFactory,
                                                        memoryManager,
                                                        tensorHandleFactory);
}

LayerTestResult<float, 2> StridedSlice2dReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice2dReverseTest<armnn::DataType::Float32>(workloadFactory,
                                                               memoryManager,
                                                               tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> StridedSlice4dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice4dTest<armnn::DataType::QAsymmU8>(workloadFactory,
                                                         memoryManager,
                                                         tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> StridedSlice4dReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice4dReverseTest<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                memoryManager,
                                                                tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> StridedSliceSimpleStrideUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                   memoryManager,
                                                                   tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> StridedSliceSimpleRangeMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                      memoryManager,
                                                                      tensorHandleFactory);
}

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                     memoryManager,
                                                                     tensorHandleFactory);
}

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskBitPosition0Dim3Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0Dim3Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                     memoryManager,
                                                                                     tensorHandleFactory);
}

LayerTestResult<uint8_t, 3> StridedSliceShrinkAxisMaskBitPosition0Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                 memoryManager,
                                                                                 tensorHandleFactory);
}

LayerTestResult<uint8_t, 3> StridedSliceShrinkAxisMaskBitPosition1Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition1Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                 memoryManager,
                                                                                 tensorHandleFactory);
}

LayerTestResult<uint8_t, 3> StridedSliceShrinkAxisMaskBitPosition2Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition2Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                 memoryManager,
                                                                                 tensorHandleFactory);
}

LayerTestResult<uint8_t, 3> StridedSliceShrinkAxisMaskBitPosition3Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition3Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                 memoryManager,
                                                                                 tensorHandleFactory);
}

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskBitPosition0And1Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0And1Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                     memoryManager,
                                                                                     tensorHandleFactory);
}

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskBitPosition0And2Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0And2Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                     memoryManager,
                                                                                     tensorHandleFactory);
}

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskBitPosition0And3Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0And3Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                     memoryManager,
                                                                                     tensorHandleFactory);
}

LayerTestResult<uint8_t, 1> StridedSliceShrinkAxisMaskBitPosition0And1And3Uint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskBitPosition0And1And3Test<armnn::DataType::QAsymmU8>(workloadFactory,
                                                                                         memoryManager,
                                                                                         tensorHandleFactory);
}

LayerTestResult<uint8_t, 3> StridedSlice3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice3dTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<uint8_t, 3> StridedSlice3dReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice3dReverseTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<uint8_t, 2> StridedSlice2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice2dTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<uint8_t, 2> StridedSlice2dReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice2dReverseTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 4> StridedSlice4dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice4dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 4> StridedSlice4dReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice4dReverseTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 4> StridedSliceSimpleStrideInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::QSymmS16>(workloadFactory,
                                                                   memoryManager,
                                                                   tensorHandleFactory);
}

LayerTestResult<int16_t, 4> StridedSliceSimpleRangeMaskInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::QSymmS16>(workloadFactory,
                                                                      memoryManager,
                                                                      tensorHandleFactory);
}

LayerTestResult<int16_t, 2> StridedSliceShrinkAxisMaskInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::QSymmS16>(workloadFactory,
                                                                     memoryManager,
                                                                     tensorHandleFactory);
}

LayerTestResult<int16_t, 3> StridedSlice3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice3dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 3> StridedSlice3dReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice3dReverseTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 2> StridedSlice2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice2dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 2> StridedSlice2dReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return StridedSlice2dReverseTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory);
}
