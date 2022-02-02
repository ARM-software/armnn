//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LogicalTestImpl.hpp"

#include <armnn/utility/Assert.hpp>
#include <ResolveType.hpp>

#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace {

template <std::size_t NumDims>
LayerTestResult<uint8_t, NumDims> LogicalUnaryTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::UnaryOperation op,
    const armnn::TensorShape& inputShape,
    std::vector<uint8_t> input,
    const armnn::TensorShape& outputShape,
    std::vector<uint8_t> expectedOutput,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    ARMNN_ASSERT(inputShape.GetNumDimensions() == NumDims);
    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Boolean);

    ARMNN_ASSERT(outputShape.GetNumDimensions() == NumDims);
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Boolean);

    std::vector<uint8_t> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr <armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ElementwiseUnaryDescriptor desc(op);
    armnn::ElementwiseUnaryQueueDescriptor qDesc;
    qDesc.m_Parameters = desc;

    armnn::WorkloadInfo info;
    AddInputToWorkload(qDesc, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(qDesc, info, outputTensorInfo, outputHandle.get());

    auto workload = workloadFactory.CreateWorkload(armnn::LayerType::ElementwiseUnary, qDesc, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<uint8_t, NumDims>(actualOutput,
                                             expectedOutput,
                                             outputHandle->GetShape(),
                                             outputTensorInfo.GetShape(),
                                             true);
}

template <std::size_t NumDims>
LayerTestResult<uint8_t, NumDims> LogicalBinaryTestHelper(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::LogicalBinaryOperation op,
    const armnn::TensorShape& inputShape0,
    const armnn::TensorShape& inputShape1,
    std::vector<uint8_t> input0,
    std::vector<uint8_t> input1,
    const armnn::TensorShape& outputShape,
    std::vector<uint8_t> expectedOutput,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    ARMNN_ASSERT(inputShape0.GetNumDimensions() == NumDims);
    armnn::TensorInfo inputTensorInfo0(inputShape0, armnn::DataType::Boolean);

    ARMNN_ASSERT(inputShape1.GetNumDimensions() == NumDims);
    armnn::TensorInfo inputTensorInfo1(inputShape1, armnn::DataType::Boolean);

    ARMNN_ASSERT(outputShape.GetNumDimensions() == NumDims);
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Boolean);

    std::vector<uint8_t> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr <armnn::ITensorHandle> inputHandle0 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr <armnn::ITensorHandle> inputHandle1 = tensorHandleFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::LogicalBinaryDescriptor desc(op);
    armnn::LogicalBinaryQueueDescriptor qDesc;
    qDesc.m_Parameters = desc;

    armnn::WorkloadInfo info;
    AddInputToWorkload(qDesc, info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(qDesc, info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(qDesc, info, outputTensorInfo, outputHandle.get());

    auto workload = workloadFactory.CreateWorkload(armnn::LayerType::LogicalBinary, qDesc, info);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), input0.data());
    CopyDataToITensorHandle(inputHandle1.get(), input1.data());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<uint8_t, NumDims>(actualOutput,
                                             expectedOutput,
                                             outputHandle->GetShape(),
                                             outputTensorInfo.GetShape(),
                                             true);
}

class UnaryTestData
{
public:
    UnaryTestData()          = default;
    virtual ~UnaryTestData() = default;

    armnn::TensorShape m_InputShape;
    armnn::TensorShape m_OutputShape;

    std::vector<uint8_t> m_InputData;

    std::vector<uint8_t> m_OutputNot;
};

class BinaryTestData
{
public:
    BinaryTestData()          = default;
    virtual ~BinaryTestData() = default;

    armnn::TensorShape m_InputShape0;
    armnn::TensorShape m_InputShape1;
    armnn::TensorShape m_OutputShape;

    std::vector<uint8_t> m_InputData0;
    std::vector<uint8_t> m_InputData1;

    std::vector<uint8_t> m_OutputAnd;
    std::vector<uint8_t> m_OutputOr;
};

class SimpleUnaryTestData : public UnaryTestData
{
public:
    SimpleUnaryTestData() : UnaryTestData()
    {
        m_InputShape = { 1, 1, 1, 4 };
        m_OutputShape = m_InputShape;

        m_InputData =
        {
            true, false, false, true
        };

        m_OutputNot =
        {
            false, true, true, false
        };
    }
};

class SimpleUnaryIntTestData : public UnaryTestData
{
public:
    SimpleUnaryIntTestData() : UnaryTestData()
    {
        m_InputShape = { 1, 1, 1, 4 };
        m_OutputShape = m_InputShape;

        m_InputData =
        {
            1, 11, 111, 0
        };

        m_OutputNot =
        {
            0, 0, 0, 1
        };
    }
};

class SimpleBinaryTestData : public BinaryTestData
{
public:
    SimpleBinaryTestData() : BinaryTestData()
    {
        m_InputShape0 = { 1, 1, 1, 4 };
        m_InputShape1 = m_InputShape0;
        m_OutputShape = m_InputShape1;

        m_InputData0 =
        {
            true, false, false, true
        };

        m_InputData1 =
        {
            true, false, true, false
        };

        m_OutputAnd =
        {
            true, false, false, false
        };

        m_OutputOr =
        {
            true, false, true, true
        };
    }
};

class SimpleBinaryIntTestData : public BinaryTestData
{
public:
    SimpleBinaryIntTestData() : BinaryTestData()
    {
        m_InputShape0 = { 1, 1, 1, 4 };
        m_InputShape1 = m_InputShape0;
        m_OutputShape = m_InputShape1;

        m_InputData0 =
        {
            1, 11, 111, 0
        };

        m_InputData1 =
        {
            0, 111, 111, 0
        };

        m_OutputAnd =
        {
            0, 1, 1, 0
        };

        m_OutputOr =
        {
            1, 1, 1, 0
        };
    }
};

class BroadcastBinary1TestData : public BinaryTestData
{
public:
    BroadcastBinary1TestData() : BinaryTestData()
    {
        m_InputShape0 = { 1, 1, 1, 4 };
        m_InputShape1 = { 1, 1, 1, 1 };
        m_OutputShape = m_InputShape0;

        m_InputData0 =
        {
            true, false, false, true
        };

        m_InputData1 =
        {
            true
        };

        m_OutputAnd =
        {
            true, false, false, true
        };

        m_OutputOr =
        {
            true, true, true, true
        };
    }
};

class BroadcastBinary2TestData : public BinaryTestData
{
public:
    BroadcastBinary2TestData() : BinaryTestData()
    {
        m_InputShape0 = { 1, 1, 1, 1 };
        m_InputShape1 = { 1, 1, 1, 4 };
        m_OutputShape = m_InputShape1;

        m_InputData0 =
        {
            true
        };

        m_InputData1 =
        {
            true, false, false, true
        };

        m_OutputAnd =
        {
            true, false, false, true
        };

        m_OutputOr =
        {
            true, true, true, true
        };
    }
};

class BroadcastBinary3TestData : public BinaryTestData
{
public:
    BroadcastBinary3TestData() : BinaryTestData()
    {
        m_InputShape0 = { 1, 1, 1, 4 };
        m_InputShape1 = { 1, 1, 1, 1 };
        m_OutputShape = m_InputShape0;

        m_InputData0 =
        {
            true, false, false, true
        };

        m_InputData1 =
        {
            false
        };

        m_OutputAnd =
        {
            false, false, false, false
        };

        m_OutputOr =
        {
            true, false, false, true
        };
    }
};

static SimpleUnaryTestData s_SimpleUnaryTestData;
static SimpleBinaryTestData s_SimpleBinaryTestData;

static SimpleUnaryIntTestData s_SimpleUnaryIntTestData;
static SimpleBinaryIntTestData s_SimpleBinaryIntTestData;

static BroadcastBinary1TestData s_BroadcastBinary1TestData;
static BroadcastBinary2TestData s_BroadcastBinary2TestData;
static BroadcastBinary3TestData s_BroadcastBinary3TestData;


} // anonymous namespace

// Unary - Not
LayerTestResult<uint8_t, 4> LogicalNotTest(armnn::IWorkloadFactory& workloadFactory,
                                           const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                           const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalUnaryTestHelper<4>(workloadFactory,
                                     memoryManager,
                                     armnn::UnaryOperation::LogicalNot,
                                     s_SimpleUnaryTestData.m_InputShape,
                                     s_SimpleUnaryTestData.m_InputData,
                                     s_SimpleUnaryTestData.m_OutputShape,
                                     s_SimpleUnaryTestData.m_OutputNot,
                                     tensorHandleFactory);
}

// Unary - Not with integers
LayerTestResult<uint8_t, 4> LogicalNotIntTest(armnn::IWorkloadFactory& workloadFactory,
                                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                              const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalUnaryTestHelper<4>(workloadFactory,
                                     memoryManager,
                                     armnn::UnaryOperation::LogicalNot,
                                     s_SimpleUnaryIntTestData.m_InputShape,
                                     s_SimpleUnaryIntTestData.m_InputData,
                                     s_SimpleUnaryIntTestData.m_OutputShape,
                                     s_SimpleUnaryIntTestData.m_OutputNot,
                                     tensorHandleFactory);
}

// Binary - And
LayerTestResult<uint8_t, 4> LogicalAndTest(armnn::IWorkloadFactory& workloadFactory,
                                           const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                           const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalAnd,
                                      s_SimpleBinaryTestData.m_InputShape0,
                                      s_SimpleBinaryTestData.m_InputShape1,
                                      s_SimpleBinaryTestData.m_InputData0,
                                      s_SimpleBinaryTestData.m_InputData1,
                                      s_SimpleBinaryTestData.m_OutputShape,
                                      s_SimpleBinaryTestData.m_OutputAnd,
                                      tensorHandleFactory);
}

// Binary - Or
LayerTestResult<uint8_t, 4> LogicalOrTest(armnn::IWorkloadFactory& workloadFactory,
                                          const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                          const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalOr,
                                      s_SimpleBinaryTestData.m_InputShape0,
                                      s_SimpleBinaryTestData.m_InputShape1,
                                      s_SimpleBinaryTestData.m_InputData0,
                                      s_SimpleBinaryTestData.m_InputData1,
                                      s_SimpleBinaryTestData.m_OutputShape,
                                      s_SimpleBinaryTestData.m_OutputOr,
                                      tensorHandleFactory);
}

// Binary - And with integers
LayerTestResult<uint8_t, 4> LogicalAndIntTest(armnn::IWorkloadFactory& workloadFactory,
                                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                              const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalAnd,
                                      s_SimpleBinaryIntTestData.m_InputShape0,
                                      s_SimpleBinaryIntTestData.m_InputShape1,
                                      s_SimpleBinaryIntTestData.m_InputData0,
                                      s_SimpleBinaryIntTestData.m_InputData1,
                                      s_SimpleBinaryIntTestData.m_OutputShape,
                                      s_SimpleBinaryIntTestData.m_OutputAnd,
                                      tensorHandleFactory);
}

// Binary - Or with integers
LayerTestResult<uint8_t, 4> LogicalOrIntTest(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalOr,
                                      s_SimpleBinaryIntTestData.m_InputShape0,
                                      s_SimpleBinaryIntTestData.m_InputShape1,
                                      s_SimpleBinaryIntTestData.m_InputData0,
                                      s_SimpleBinaryIntTestData.m_InputData1,
                                      s_SimpleBinaryIntTestData.m_OutputShape,
                                      s_SimpleBinaryIntTestData.m_OutputOr,
                                      tensorHandleFactory);
}

// Binary - And Broadcast
LayerTestResult<uint8_t, 4> LogicalAndBroadcast1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalAnd,
                                      s_BroadcastBinary1TestData.m_InputShape0,
                                      s_BroadcastBinary1TestData.m_InputShape1,
                                      s_BroadcastBinary1TestData.m_InputData0,
                                      s_BroadcastBinary1TestData.m_InputData1,
                                      s_BroadcastBinary1TestData.m_OutputShape,
                                      s_BroadcastBinary1TestData.m_OutputAnd,
                                      tensorHandleFactory);
}

// Binary - Or Broadcast
LayerTestResult<uint8_t, 4> LogicalOrBroadcast1Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalOr,
                                      s_BroadcastBinary1TestData.m_InputShape0,
                                      s_BroadcastBinary1TestData.m_InputShape1,
                                      s_BroadcastBinary1TestData.m_InputData0,
                                      s_BroadcastBinary1TestData.m_InputData1,
                                      s_BroadcastBinary1TestData.m_OutputShape,
                                      s_BroadcastBinary1TestData.m_OutputOr,
                                      tensorHandleFactory);
}

// Binary - And Broadcast
LayerTestResult<uint8_t, 4> LogicalAndBroadcast2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalAnd,
                                      s_BroadcastBinary2TestData.m_InputShape0,
                                      s_BroadcastBinary2TestData.m_InputShape1,
                                      s_BroadcastBinary2TestData.m_InputData0,
                                      s_BroadcastBinary2TestData.m_InputData1,
                                      s_BroadcastBinary2TestData.m_OutputShape,
                                      s_BroadcastBinary2TestData.m_OutputAnd,
                                      tensorHandleFactory);
}

// Binary - Or Broadcast
LayerTestResult<uint8_t, 4> LogicalOrBroadcast2Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalOr,
                                      s_BroadcastBinary2TestData.m_InputShape0,
                                      s_BroadcastBinary2TestData.m_InputShape1,
                                      s_BroadcastBinary2TestData.m_InputData0,
                                      s_BroadcastBinary2TestData.m_InputData1,
                                      s_BroadcastBinary2TestData.m_OutputShape,
                                      s_BroadcastBinary2TestData.m_OutputOr,
                                      tensorHandleFactory);
}

// Binary - And Broadcast
LayerTestResult<uint8_t, 4> LogicalAndBroadcast3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalAnd,
                                      s_BroadcastBinary3TestData.m_InputShape0,
                                      s_BroadcastBinary3TestData.m_InputShape1,
                                      s_BroadcastBinary3TestData.m_InputData0,
                                      s_BroadcastBinary3TestData.m_InputData1,
                                      s_BroadcastBinary3TestData.m_OutputShape,
                                      s_BroadcastBinary3TestData.m_OutputAnd,
                                      tensorHandleFactory);
}

// Binary - Or Broadcast
LayerTestResult<uint8_t, 4> LogicalOrBroadcast3Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LogicalBinaryTestHelper<4>(workloadFactory,
                                      memoryManager,
                                      armnn::LogicalBinaryOperation::LogicalOr,
                                      s_BroadcastBinary3TestData.m_InputShape0,
                                      s_BroadcastBinary3TestData.m_InputShape1,
                                      s_BroadcastBinary3TestData.m_InputData0,
                                      s_BroadcastBinary3TestData.m_InputData1,
                                      s_BroadcastBinary3TestData.m_OutputShape,
                                      s_BroadcastBinary3TestData.m_OutputOr,
                                      tensorHandleFactory);
}