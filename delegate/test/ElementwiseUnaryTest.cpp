//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseUnaryTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

TEST_SUITE("ElementwiseUnary_GpuAccTests")
{

TEST_CASE ("Abs_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        -0.1f, -0.2f, -0.3f,
        0.1f,  0.2f,  0.3f
    };
    // Calculate output data
    std::vector<float> expectedOutputValues(inputValues.size());
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        expectedOutputValues[i] = std::abs(inputValues[i]);
    }
    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_ABS, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Exp_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        5.0f, 4.0f,
        3.0f, 2.0f,
        1.0f, 1.1f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        148.413159102577f, 54.598150033144f,
        20.085536923188f,  7.389056098931f,
        2.718281828459f,  3.004166023946f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_EXP, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Log_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        1.0f, 1.0f,  2.0f,
        3.0f,  4.0f, 2.71828f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        0.f,  0.f,  0.69314718056f,
        1.09861228867f, 1.38629436112f, 0.99999932734f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_LOG, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Neg_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        1.f, 0.f, 3.f,
        25.f, 64.f, 100.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        -1.f, 0.f, -3.f,
        -25.f, -64.f, -100.f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_NEG, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Rsqrt_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        1.f, 4.f, 16.f,
        25.f, 64.f, 100.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f, 0.25f,
        0.2f, 0.125f, 0.1f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_RSQRT, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Sin_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
            0.0f, 1.0f, 16.0f,
            0.5f, 36.0f, -1.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
            0.0f, 0.8414709848f, -0.28790331666f,
            0.4794255386f, -0.99177885344f, -0.8414709848f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_SIN, backends, inputValues, expectedOutputValues);
}
} // TEST_SUITE("ElementwiseUnary_GpuAccTests")



TEST_SUITE("ElementwiseUnary_CpuAccTests")
{

TEST_CASE ("Abs_Float32_CpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        -0.1f, -0.2f, -0.3f,
        0.1f,  0.2f,  0.3f
    };
    // Calculate output data
    std::vector<float> expectedOutputValues(inputValues.size());
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        expectedOutputValues[i] = std::abs(inputValues[i]);
    }

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_ABS, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Exp_Float32_CpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        5.0f, 4.0f,
        3.0f, 2.0f,
        1.0f, 1.1f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        148.413159102577f, 54.598150033144f,
        20.085536923188f,  7.389056098931f,
        2.718281828459f,  3.004166023946f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_EXP, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Log_Float32_CpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        1.0f, 1.0f,  2.0f,
        3.0f,  4.0f, 2.71828f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        0.f,  0.f,  0.69314718056f,
        1.09861228867f, 1.38629436112f, 0.99999932734f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_LOG, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Neg_Float32_CpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        1.f, 0.f, 3.f,
        25.f, 64.f, 100.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        -1.f, 0.f, -3.f,
        -25.f, -64.f, -100.f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_NEG, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Rsqrt_Float32_CpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        1.f, 4.f, 16.f,
        25.f, 64.f, 100.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f, 0.25f,
        0.2f, 0.125f, 0.1f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_RSQRT, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Sin_Float32_CpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    // Set input data
    std::vector<float> inputValues
    {
        0.0f, 1.0f, 16.0f,
        0.5f, 36.0f, -1.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        0.0f, 0.8414709848f, -0.28790331666f,
        0.4794255386f, -0.99177885344f, -0.8414709848f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_SIN, backends, inputValues, expectedOutputValues);
}
} // TEST_SUITE("ElementwiseUnary_CpuAccTests")

TEST_SUITE("ElementwiseUnary_CpuRefTests")
{

TEST_CASE ("Abs_Float32_CpuRef_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    // Set input data
    std::vector<float> inputValues
    {
        -0.1f, -0.2f, -0.3f,
        0.1f,  0.2f,  0.3f
    };
    // Calculate output data
    std::vector<float> expectedOutputValues(inputValues.size());
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        expectedOutputValues[i] = std::abs(inputValues[i]);
    }

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_ABS, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Ceil_Float32_CpuRef_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    // Set input data
    std::vector<float> inputValues
    {
        0.0f, 1.1f, -16.1f,
        0.5f, -0.5f, -1.3f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        0.0f, 2.0f, -16.0f,
        1.0f, 0.0f, -1.0f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_CEIL, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Exp_Float32_CpuRef_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    // Set input data
    std::vector<float> inputValues
    {
        5.0f, 4.0f,
        3.0f, 2.0f,
        1.0f, 1.1f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        148.413159102577f, 54.598150033144f,
        20.085536923188f,  7.389056098931f,
        2.718281828459f,  3.004166023946f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_EXP, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Log_Float32_CpuRef_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    // Set input data
    std::vector<float> inputValues
    {
        1.0f, 1.0f,  2.0f,
        3.0f,  4.0f, 2.71828f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        0.f,  0.f,  0.69314718056f,
        1.09861228867f, 1.38629436112f, 0.99999932734f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_LOG, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Neg_Float32_CpuRef_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    // Set input data
    std::vector<float> inputValues
    {
        1.f, 0.f, 3.f,
        25.f, 64.f, 100.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        -1.f, 0.f, -3.f,
        -25.f, -64.f, -100.f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_NEG, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Rsqrt_Float32_CpuRef_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    // Set input data
    std::vector<float> inputValues
    {
        1.f, 4.f, 16.f,
        25.f, 64.f, 100.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f, 0.25f,
        0.2f, 0.125f, 0.1f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_RSQRT, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Sqrt_Float32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    // Set input data
    std::vector<float> inputValues
    {
        9.0f, 4.25f, 81.9f,
        0.1f,  0.9f,  169.0f
    };
    // Calculate output data
    std::vector<float> expectedOutputValues(inputValues.size());
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        expectedOutputValues[i] = std::sqrt(inputValues[i]);
    }

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_SQRT, backends, inputValues, expectedOutputValues);
}

TEST_CASE ("Sin_Float32_CpuRef_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    // Set input data
    std::vector<float> inputValues
    {
            0.0f, 1.0f, 16.0f,
            0.5f, 36.0f, -1.f
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
            0.0f, 0.8414709848f, -0.28790331666f,
            0.4794255386f, -0.99177885344f, -0.8414709848f
    };

    ElementwiseUnaryFP32Test(tflite::BuiltinOperator_SIN, backends, inputValues, expectedOutputValues);
}
} // TEST_SUITE("ElementwiseUnary_CpuRefTests")

} // namespace armnnDelegate