//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseUnaryTestHelper.hpp"
#include "LogicalTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void LogicalBinaryAndBoolTest()
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2 };

    // Set input and output values
    std::vector<bool> input0Values { 0, 0, 1, 1 };
    std::vector<bool> input1Values { 0, 1, 0, 1 };
    std::vector<bool> expectedOutputValues { 0, 0, 0, 1 };

    LogicalBinaryTest(tflite::BuiltinOperator_LOGICAL_AND,
                      ::tflite::TensorType_BOOL,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues);
}

void LogicalBinaryAndBroadcastTest()
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2 };

    std::vector<bool> input0Values { 0, 1, 0, 1 };
    std::vector<bool> input1Values { 1 };
    std::vector<bool> expectedOutputValues { 0, 1, 0, 1 };

    LogicalBinaryTest(tflite::BuiltinOperator_LOGICAL_AND,
                      ::tflite::TensorType_BOOL,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues);
}

void LogicalBinaryOrBoolTest()
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2 };

    std::vector<bool> input0Values { 0, 0, 1, 1 };
    std::vector<bool> input1Values { 0, 1, 0, 1 };
    std::vector<bool> expectedOutputValues { 0, 1, 1, 1 };

    LogicalBinaryTest(tflite::BuiltinOperator_LOGICAL_OR,
                      ::tflite::TensorType_BOOL,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues);
}

void LogicalBinaryOrBroadcastTest()
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2 };

    std::vector<bool> input0Values { 0, 1, 0, 1 };
    std::vector<bool> input1Values { 1 };
    std::vector<bool> expectedOutputValues { 1, 1, 1, 1 };

    LogicalBinaryTest(tflite::BuiltinOperator_LOGICAL_OR,
                      ::tflite::TensorType_BOOL,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues);
}

// LogicalNot operator uses ElementwiseUnary unary layer and descriptor but is still classed as logical operator.
void LogicalNotBoolTest()
{
    std::vector<int32_t> inputShape { 1, 2, 2 };

    std::vector<bool> inputValues { 0, 1, 0, 1 };
    std::vector<bool> expectedOutputValues { 1, 0, 1, 0 };

    ElementwiseUnaryBoolTest(tflite::BuiltinOperator_LOGICAL_NOT,
                             inputShape,
                             inputValues,
                             expectedOutputValues);
}

TEST_SUITE("LogicalBinaryTests_Tests")
{

TEST_CASE ("LogicalBinary_AND_Bool_Test")
{
    LogicalBinaryAndBoolTest();
}

TEST_CASE ("LogicalBinary_AND_Broadcast_Test")
{
    LogicalBinaryAndBroadcastTest();
}

TEST_CASE ("Logical_NOT_Bool_Test")
{
    LogicalNotBoolTest();
}

TEST_CASE ("LogicalBinary_OR_Bool_Test")
{
    LogicalBinaryOrBoolTest();
}

TEST_CASE ("LogicalBinary_OR_Broadcast_Test")
{
    LogicalBinaryOrBroadcastTest();
}

}

} // namespace armnnDelegate