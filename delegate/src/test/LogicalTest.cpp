//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseUnaryTestHelper.hpp"
#include "LogicalTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void LogicalBinaryAndBoolTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2 };

    // Set input and output values
    std::vector<bool> input0Values { 0, 0, 1, 1 };
    std::vector<bool> input1Values { 0, 1, 0, 1 };
    std::vector<bool> expectedOutputValues { 0, 0, 0, 1 };

    LogicalBinaryTest<bool>(tflite::BuiltinOperator_LOGICAL_AND,
                            ::tflite::TensorType_BOOL,
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
}

void LogicalBinaryAndBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2 };

    std::vector<bool> input0Values { 0, 1, 0, 1 };
    std::vector<bool> input1Values { 1 };
    std::vector<bool> expectedOutputValues { 0, 1, 0, 1 };

    LogicalBinaryTest<bool>(tflite::BuiltinOperator_LOGICAL_AND,
                            ::tflite::TensorType_BOOL,
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
}

void LogicalBinaryOrBoolTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2 };

    std::vector<bool> input0Values { 0, 0, 1, 1 };
    std::vector<bool> input1Values { 0, 1, 0, 1 };
    std::vector<bool> expectedOutputValues { 0, 1, 1, 1 };

    LogicalBinaryTest<bool>(tflite::BuiltinOperator_LOGICAL_OR,
                            ::tflite::TensorType_BOOL,
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
}

void LogicalBinaryOrBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2 };

    std::vector<bool> input0Values { 0, 1, 0, 1 };
    std::vector<bool> input1Values { 1 };
    std::vector<bool> expectedOutputValues { 1, 1, 1, 1 };

    LogicalBinaryTest<bool>(tflite::BuiltinOperator_LOGICAL_OR,
                            ::tflite::TensorType_BOOL,
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
}

// LogicalNot operator uses ElementwiseUnary unary layer and descriptor but is still classed as logical operator.
void LogicalNotBoolTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 1, 2, 2 };

    std::vector<bool> inputValues { 0, 1, 0, 1 };
    std::vector<bool> expectedOutputValues { 1, 0, 1, 0 };

    ElementwiseUnaryBoolTest(tflite::BuiltinOperator_LOGICAL_NOT,
                             backends,
                             inputShape,
                             inputValues,
                             expectedOutputValues);
}

TEST_SUITE("LogicalBinaryTests_GpuAccTests")
{

TEST_CASE ("LogicalBinary_AND_Bool_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LogicalBinaryAndBoolTest(backends);
}

TEST_CASE ("LogicalBinary_AND_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LogicalBinaryAndBroadcastTest(backends);
}

TEST_CASE ("Logical_NOT_Bool_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LogicalNotBoolTest(backends);
}

TEST_CASE ("LogicalBinary_OR_Bool_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LogicalBinaryOrBoolTest(backends);
}

TEST_CASE ("LogicalBinary_OR_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LogicalBinaryOrBroadcastTest(backends);
}

}


TEST_SUITE("LogicalBinaryTests_CpuAccTests")
{

TEST_CASE ("LogicalBinary_AND_Bool_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LogicalBinaryAndBoolTest(backends);
}

TEST_CASE ("LogicalBinary_AND_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LogicalBinaryAndBroadcastTest(backends);
}

TEST_CASE ("Logical_NOT_Bool_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LogicalNotBoolTest(backends);
}

TEST_CASE ("LogicalBinary_OR_Bool_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LogicalBinaryOrBoolTest(backends);
}

TEST_CASE ("LogicalBinary_OR_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LogicalBinaryOrBroadcastTest(backends);
}

}


TEST_SUITE("LogicalBinaryTests_CpuRefTests")
{

TEST_CASE ("LogicalBinary_AND_Bool_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LogicalBinaryAndBoolTest(backends);
}

TEST_CASE ("LogicalBinary_AND_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LogicalBinaryAndBroadcastTest(backends);
}

TEST_CASE ("Logical_NOT_Bool_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LogicalNotBoolTest(backends);
}

TEST_CASE ("LogicalBinary_OR_Bool_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LogicalBinaryOrBoolTest(backends);
}

TEST_CASE ("LogicalBinary_OR_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LogicalBinaryOrBroadcastTest(backends);
}

}

} // namespace armnnDelegate