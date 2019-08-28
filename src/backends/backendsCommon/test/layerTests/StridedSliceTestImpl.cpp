//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StridedSliceTestImpl.hpp"

#include <ResolveType.hpp>

#include <armnn/ArmNN.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

namespace
{

template<typename T, std::size_t InDim, std::size_t OutDim>
LayerTestResult<T, OutDim> StridedSliceTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::TensorInfo& inputTensorInfo,
    armnn::TensorInfo& outputTensorInfo,
    std::vector<float>& inputData,
    std::vector<float>& outputExpectedData,
    armnn::StridedSliceQueueDescriptor descriptor,
    const float qScale = 1.0f,
    const int32_t qOffset = 0)
{
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);

        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    boost::multi_array<T, InDim> input =
        MakeTensor<T, InDim>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, inputData));

    LayerTestResult<T, OutDim> ret(outputTensorInfo);
    ret.outputExpected =
        MakeTensor<T, OutDim>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, outputExpectedData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle =
        workloadFactory.CreateTensorHandle(inputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> outputHandle =
        workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateStridedSlice(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(ret.output.data(), outputHandle.get());

    return ret;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSlice4dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSlice4dReverseTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSliceSimpleStrideTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedSliceSimpleRangeMaskTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSliceShrinkAxisMaskTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StridedSlice3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> StridedSlice3dReverseTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSlice2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> StridedSlice2dReverseTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

} // anonymous namespace

LayerTestResult<float, 4> StridedSlice4dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4dTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSlice4dReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4dReverseTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSliceSimpleStrideFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> StridedSliceSimpleRangeMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSliceShrinkAxisMaskFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> StridedSlice3dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3dTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> StridedSlice3dReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3dReverseTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSlice2dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2dTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> StridedSlice2dReverseFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2dReverseTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSlice4dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4dTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSlice4dReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4dReverseTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSliceSimpleStrideUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> StridedSliceSimpleRangeMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSliceShrinkAxisMaskUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> StridedSlice3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3dTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> StridedSlice3dReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3dReverseTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSlice2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2dTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> StridedSlice2dReverseUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2dReverseTest<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> StridedSlice4dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4dTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> StridedSlice4dReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice4dReverseTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> StridedSliceSimpleStrideInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleStrideTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> StridedSliceSimpleRangeMaskInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceSimpleRangeMaskTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> StridedSliceShrinkAxisMaskInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSliceShrinkAxisMaskTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 3> StridedSlice3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3dTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 3> StridedSlice3dReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice3dReverseTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> StridedSlice2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2dTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> StridedSlice2dReverseInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return StridedSlice2dReverseTest<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}
