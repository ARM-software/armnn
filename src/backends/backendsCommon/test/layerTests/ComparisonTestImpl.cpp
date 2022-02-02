//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ComparisonTestImpl.hpp"

#include <armnn/utility/Assert.hpp>
#include <Half.hpp>
#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template <std::size_t NumDims,
          armnn::DataType ArmnnInType,
          typename InType = armnn::ResolveType<ArmnnInType>>
LayerTestResult<uint8_t, NumDims> ComparisonTestImpl(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ComparisonDescriptor& descriptor,
    const armnn::TensorShape& shape0,
    std::vector<InType> values0,
    float quantScale0,
    int quantOffset0,
    const armnn::TensorShape& shape1,
    std::vector<InType> values1,
    float quantScale1,
    int quantOffset1,
    const armnn::TensorShape& outShape,
    std::vector<uint8_t> outValues,
    float outQuantScale,
    int outQuantOffset)
{
    IgnoreUnused(memoryManager);
    ARMNN_ASSERT(shape0.GetNumDimensions() == NumDims);
    armnn::TensorInfo inputTensorInfo0(shape0, ArmnnInType, quantScale0, quantOffset0);

    ARMNN_ASSERT(shape1.GetNumDimensions() == NumDims);
    armnn::TensorInfo inputTensorInfo1(shape1, ArmnnInType, quantScale1, quantOffset1);

    ARMNN_ASSERT(outShape.GetNumDimensions() == NumDims);
    armnn::TensorInfo outputTensorInfo(outShape, armnn::DataType::Boolean, outQuantScale, outQuantOffset);

    std::vector<uint8_t> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ComparisonQueueDescriptor qDescriptor;
    qDescriptor.m_Parameters = descriptor;

    armnn::WorkloadInfo info;
    AddInputToWorkload(qDescriptor, info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(qDescriptor, info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(qDescriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::Comparison, qDescriptor, info);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), values0.data());
    CopyDataToITensorHandle(inputHandle1.get(), values1.data());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<uint8_t, NumDims>(actualOutput,
                                             outValues,
                                             outputHandle->GetShape(),
                                             outputTensorInfo.GetShape(),
                                             true);
}

template <std::size_t NumDims,
          armnn::DataType ArmnnInType,
          typename InType = armnn::ResolveType<ArmnnInType>>
LayerTestResult<uint8_t, NumDims> ComparisonTestImpl(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ComparisonDescriptor& descriptor,
    const armnn::TensorShape& shape0,
    std::vector<InType> values0,
    const armnn::TensorShape& shape1,
    std::vector<InType> values1,
    const armnn::TensorShape outShape,
    std::vector<uint8_t> outValues,
    float quantScale = 1.f,
    int quantOffset = 0)
{
    return ComparisonTestImpl<NumDims, ArmnnInType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        descriptor,
        shape0,
        values0,
        quantScale,
        quantOffset,
        shape1,
        values1,
        quantScale,
        quantOffset,
        outShape,
        outValues,
        quantScale,
        quantOffset);
}

template<typename TestData>
std::vector<uint8_t> GetExpectedOutputData(const TestData& testData, armnn::ComparisonOperation operation)
{
    switch (operation)
    {
        case armnn::ComparisonOperation::Equal:
            return testData.m_OutputEqual;
        case armnn::ComparisonOperation::Greater:
            return testData.m_OutputGreater;
        case armnn::ComparisonOperation::GreaterOrEqual:
            return testData.m_OutputGreaterOrEqual;
        case armnn::ComparisonOperation::Less:
            return testData.m_OutputLess;
        case armnn::ComparisonOperation::LessOrEqual:
            return testData.m_OutputLessOrEqual;
        case armnn::ComparisonOperation::NotEqual:
        default:
            return testData.m_OutputNotEqual;
    }
}

template<armnn::DataType ArmnnInType, typename TestData>
LayerTestResult<uint8_t, 4> ComparisonTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                               const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                               const armnn::ITensorHandleFactory& tensorHandleFactory,
                                               const TestData& testData,
                                               armnn::ComparisonOperation operation,
                                               float quantScale = 1.f,
                                               int quantOffset = 0)
{
    using T = armnn::ResolveType<ArmnnInType>;

    std::vector<T> inputData0 = armnnUtils::QuantizedVector<T>(testData.m_InputData0, quantScale, quantOffset);
    std::vector<T> inputData1 = armnnUtils::QuantizedVector<T>(testData.m_InputData1, quantScale, quantOffset);

    return ComparisonTestImpl<4, ArmnnInType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        armnn::ComparisonDescriptor(operation),
        testData.m_InputShape0,
        inputData0,
        testData.m_InputShape1,
        inputData1,
        testData.m_OutputShape,
        GetExpectedOutputData(testData, operation),
        quantScale,
        quantOffset);
}

class ComparisonTestData
{
public:
    ComparisonTestData()          = default;
    virtual ~ComparisonTestData() = default;

    armnn::TensorShape m_InputShape0;
    armnn::TensorShape m_InputShape1;
    armnn::TensorShape m_OutputShape;

    std::vector<float> m_InputData0;
    std::vector<float> m_InputData1;

    std::vector<uint8_t> m_OutputEqual;
    std::vector<uint8_t> m_OutputGreater;
    std::vector<uint8_t> m_OutputGreaterOrEqual;
    std::vector<uint8_t> m_OutputLess;
    std::vector<uint8_t> m_OutputLessOrEqual;
    std::vector<uint8_t> m_OutputNotEqual;
};

class SimpleTestData : public ComparisonTestData
{
public:
    SimpleTestData() : ComparisonTestData()
    {
        m_InputShape0 = { 2, 2, 2, 2 };

        m_InputShape1 = m_InputShape0;
        m_OutputShape = m_InputShape0;

        m_InputData0 =
        {
            1.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f,
            3.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f, 4.f
        };

        m_InputData1 =
        {
            1.f, 1.f, 1.f, 1.f, 3.f, 3.f, 3.f, 3.f,
            5.f, 5.f, 5.f, 5.f, 4.f, 4.f, 4.f, 4.f
        };

        m_OutputEqual =
        {
            1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1
        };

        m_OutputGreater =
        {
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0
        };

        m_OutputGreaterOrEqual =
        {
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1
        };

        m_OutputLess =
        {
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 0, 0, 0, 0
        };

        m_OutputLessOrEqual =
        {
            1, 1, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1
        };

        m_OutputNotEqual =
        {
            0, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 0, 0, 0
        };
    }
};

class Broadcast1ElementTestData : public ComparisonTestData
{
public:
    Broadcast1ElementTestData() : ComparisonTestData()
    {
        m_InputShape0 = { 1, 2, 2, 2 };
        m_InputShape1 = { 1, 1, 1, 1 };

        m_OutputShape = m_InputShape0;

        m_InputData0 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
        m_InputData1 = { 3.f };

        m_OutputEqual          = { 0, 0, 1, 0, 0, 0, 0, 0 };
        m_OutputGreater        = { 0, 0, 0, 1, 1, 1, 1, 1 };
        m_OutputGreaterOrEqual = { 0, 0, 1, 1, 1, 1, 1, 1 };
        m_OutputLess           = { 1, 1, 0, 0, 0, 0, 0, 0 };
        m_OutputLessOrEqual    = { 1, 1, 1, 0, 0, 0, 0, 0 };
        m_OutputNotEqual       = { 1, 1, 0, 1, 1, 1, 1, 1 };
    }
};

class Broadcast1dVectorTestData : public ComparisonTestData
{
public:
    Broadcast1dVectorTestData() : ComparisonTestData()
    {
        m_InputShape0 = { 1, 2, 2, 3 };
        m_InputShape1 = { 1, 1, 1, 3 };

        m_OutputShape = m_InputShape0;

        m_InputData0 =
        {
            1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f
        };

        m_InputData1 = { 4.f, 5.f, 6.f };

        m_OutputEqual =
        {
            0, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0
        };

        m_OutputGreater =
        {
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1
        };

        m_OutputGreaterOrEqual =
        {
            0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1
        };

        m_OutputLess =
        {
            1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        };

        m_OutputLessOrEqual =
        {
            1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0
        };

        m_OutputNotEqual =
        {
            1, 1, 1, 0, 0, 0,
            1, 1, 1, 1, 1, 1
        };
    }
};

static SimpleTestData            s_SimpleTestData;
static Broadcast1ElementTestData s_Broadcast1ElementTestData;
static Broadcast1dVectorTestData s_Broadcast1dVectorTestData;

} // anonymous namespace

// Equal
LayerTestResult<uint8_t, 4> EqualSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                            const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Equal);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Equal);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1dVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Equal);
}

LayerTestResult<uint8_t, 4> EqualSimpleFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Equal);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1ElementFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Equal);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1dVectorFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Equal);
}

LayerTestResult<uint8_t, 4> EqualSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Equal);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Equal);
}

LayerTestResult<uint8_t, 4> EqualBroadcast1dVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Equal);
}

// Greater
LayerTestResult<uint8_t, 4> GreaterSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                              const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Greater);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Greater);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1dVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Greater);
}

LayerTestResult<uint8_t, 4> GreaterSimpleFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Greater);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Greater);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1dVectorFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Greater);
}

LayerTestResult<uint8_t, 4> GreaterSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Greater);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Greater);
}

LayerTestResult<uint8_t, 4> GreaterBroadcast1dVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Greater);
}

// GreaterOrEqual
LayerTestResult<uint8_t, 4> GreaterOrEqualSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

LayerTestResult<uint8_t, 4> GreaterOrEqualBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

LayerTestResult<uint8_t, 4> GreaterOrEqualBroadcast1dVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

LayerTestResult<uint8_t, 4> GreaterOrEqualSimpleFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

LayerTestResult<uint8_t, 4> GreaterOrEqualBroadcast1ElementFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

LayerTestResult<uint8_t, 4> GreaterOrEqualBroadcast1dVectorFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

LayerTestResult<uint8_t, 4> GreaterOrEqualSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

LayerTestResult<uint8_t, 4> GreaterOrEqualBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

LayerTestResult<uint8_t, 4> GreaterOrEqualBroadcast1dVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::GreaterOrEqual);
}

// Less
LayerTestResult<uint8_t, 4> LessSimpleTest(armnn::IWorkloadFactory& workloadFactory,
                                           const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                           const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Less);
}

LayerTestResult<uint8_t, 4> LessBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Less);
}

LayerTestResult<uint8_t, 4> LessBroadcast1dVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Less);
}

LayerTestResult<uint8_t, 4> LessSimpleFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Less);
}

LayerTestResult<uint8_t, 4> LessBroadcast1ElementFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Less);
}

LayerTestResult<uint8_t, 4> LessBroadcast1dVectorFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Less);
}

LayerTestResult<uint8_t, 4> LessSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::Less);
}

LayerTestResult<uint8_t, 4> LessBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::Less);
}

LayerTestResult<uint8_t, 4> LessBroadcast1dVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::Less);
}

// LessOrEqual
LayerTestResult<uint8_t, 4> LessOrEqualSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

LayerTestResult<uint8_t, 4> LessOrEqualBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

LayerTestResult<uint8_t, 4> LessOrEqualBroadcast1dVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

LayerTestResult<uint8_t, 4> LessOrEqualSimpleFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

LayerTestResult<uint8_t, 4> LessOrEqualBroadcast1ElementFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

LayerTestResult<uint8_t, 4> LessOrEqualBroadcast1dVectorFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

LayerTestResult<uint8_t, 4> LessOrEqualSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

LayerTestResult<uint8_t, 4> LessOrEqualBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

LayerTestResult<uint8_t, 4> LessOrEqualBroadcast1dVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::LessOrEqual);
}

// NotEqual
LayerTestResult<uint8_t, 4> NotEqualSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::NotEqual);
}

LayerTestResult<uint8_t, 4> NotEqualBroadcast1ElementTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::NotEqual);
}

LayerTestResult<uint8_t, 4> NotEqualBroadcast1dVectorTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::NotEqual);
}

LayerTestResult<uint8_t, 4> NotEqualSimpleFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::NotEqual);
}

LayerTestResult<uint8_t, 4> NotEqualBroadcast1ElementFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::NotEqual);
}

LayerTestResult<uint8_t, 4> NotEqualBroadcast1dVectorFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::NotEqual);
}

LayerTestResult<uint8_t, 4> NotEqualSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_SimpleTestData,
        armnn::ComparisonOperation::NotEqual);
}

LayerTestResult<uint8_t, 4> NotEqualBroadcast1ElementUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1ElementTestData,
        armnn::ComparisonOperation::NotEqual);
}

LayerTestResult<uint8_t, 4> NotEqualBroadcast1dVectorUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ComparisonTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        s_Broadcast1dVectorTestData,
        armnn::ComparisonOperation::NotEqual);
}
