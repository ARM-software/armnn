//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ActivationTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

TEST_SUITE("Activation_Tests")
{

    TEST_CASE("Activation_ReLu_Test")
    {
        std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f, 0.2f, 0.3f, 0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f, 2.0f, 3.0f, 4.0f
        };

        // Calculate output values for input.
        auto f = [](float value) { return std::fmax(0.0f, value); };
        std::vector<float> outputExpectedData(inputData.size());
        std::transform(inputData.begin(), inputData.end(), outputExpectedData.begin(), f);
        ActivationTest(tflite::BuiltinOperator_RELU, inputData, outputExpectedData);
    }

    TEST_CASE("Activation_Bounded_Relu6_Test")
    {
        std::vector<float> inputData = { -0.1f, -0.2f, -0.3f, -0.4f,
                                         0.1f, 0.2f, 0.3f, 0.4f,
                                         -1.0f, -2.0f, -3.0f, -4.0f,
                                         1.0f, 2.0f, 3.0f, 4.0f
                                       };

        const float a = 6.0f;
        const float b = 0.0f;
        // Calculate output values for input.
        auto f = [a, b](float value)
        {
            return std::min(a, std::max(b, value));
        };
        std::vector<float> outputExpectedData(inputData.size());
        std::transform(inputData.begin(), inputData.end(), outputExpectedData.begin(), f);
        ActivationTest(tflite::BuiltinOperator_RELU6,
                       inputData, outputExpectedData);
    }

    TEST_CASE("Activation_Sigmoid_Test")
    {
        std::vector<float> inputData = { -0.1f, -0.2f, -0.3f, -0.4f,
                                         0.1f, 0.2f, 0.3f, 0.4f,
                                         -1.0f, -2.0f, -3.0f, -4.0f,
                                         1.0f, 2.0f, 3.0f, 4.0f
                                       };

        // Calculate output values for input.
        auto f = [](float value) { return 1.0f / (1.0f + std::exp(-value)); };
        std::vector<float> outputExpectedData(inputData.size());
        std::transform(inputData.begin(), inputData.end(), outputExpectedData.begin(), f);

        ActivationTest(tflite::BuiltinOperator_LOGISTIC, inputData, outputExpectedData);
    }

    TEST_CASE("Activation_TanH_Test")
    {
        std::vector<float> inputData = { -0.1f, -0.2f, -0.3f, -0.4f,
                                         0.1f, 0.2f, 0.3f, 0.4f,
                                         -1.0f, -2.0f, -3.0f, -4.0f,
                                         1.0f, 2.0f, 3.0f, 4.0f
                                       };

        // Calculate output values for input.
        auto f = [](float value) { return tanhf(value); };
        std::vector<float> outputExpectedData(inputData.size());
        std::transform(inputData.begin(), inputData.end(), outputExpectedData.begin(), f);

        ActivationTest(tflite::BuiltinOperator_TANH, inputData, outputExpectedData);
    }

    TEST_CASE("Activation_Elu_Test")
    {
        std::vector<float> inputData = { -0.1f, -0.2f, -0.3f, -0.4f,
                                         0.1f, 0.2f, 0.3f, 0.4f,
                                         -1.0f, -2.0f, -3.0f, -4.0f,
                                         1.0f, 2.0f, 3.0f, 4.0f
                                       };

        // Calculate output values for input.
        auto f = [](float value) {
            if (value < 0)
            {
                // alpha * (exp(x) - 1)
                return 1 * (std::exp(value) - 1);
            }
            return value;
        };
        std::vector<float> outputExpectedData(inputData.size());
        std::transform(inputData.begin(), inputData.end(), outputExpectedData.begin(), f);

        ActivationTest(tflite::BuiltinOperator_ELU, inputData, outputExpectedData);
    }

    TEST_CASE("Activation_HardSwish_Test")
    {
        std::vector<float> inputData = { -0.1f, -0.2f, -0.3f, -0.4f,
                                         0.1f, 0.2f, 0.3f, 0.4f,
                                         -1.0f, -2.0f, -3.0f, -4.0f,
                                         1.0f, 2.0f, 3.0f, 4.0f
                                       };

        // Calculate output values for input.
        auto f = [](float x) {
            // Break down the calculation to help with verification.
            // hard_swish(x) = x * relu6(x+3) / 6
            // relu6(x) = min(max(x,0),6)
            float reLu6_step1     = std::max((x + 3), 0.0f);
            float reLu6Complete   = std::min(reLu6_step1, 6.0f);
            float hardSwish_step1 = x * reLu6Complete;
            float result          = hardSwish_step1 / 6;
            return result;
        };
        std::vector<float> outputExpectedData(inputData.size());
        std::transform(inputData.begin(), inputData.end(), outputExpectedData.begin(), f);

        ActivationTest(tflite::BuiltinOperator_HARD_SWISH, inputData, outputExpectedData);
    }

    TEST_CASE("Activation_LeakyRelu_Test")
    {
        std::vector<float> inputData = { -0.1f, -0.2f, -0.3f, -0.4f,
                                         0.1f, 0.2f, 0.3f, 0.4f,
                                         -1.0f, -2.0f, -3.0f, -4.0f,
                                         1.0f, 2.0f, 3.0f, 4.0f
                                       };

        float alpha = 0.3f;

        // Calculate output values for input.
        auto f = [alpha](float value) { return value > 0 ? value : value * alpha; };
        std::vector<float> outputExpectedData(inputData.size());
        std::transform(inputData.begin(), inputData.end(), outputExpectedData.begin(), f);

        ActivationTest(tflite::BuiltinOperator_LEAKY_RELU, inputData, outputExpectedData, alpha);
    }

    TEST_CASE("Activation_Gelu_Test")
    {
        std::vector<float> inputData = { -0.1f, -0.2f, -0.3f, -0.4f,
                                         0.1f, 0.2f, 0.3f, 0.4f,
                                         -1.0f, -2.0f, -3.0f, -4.0f,
                                         1.0f, 2.0f, 3.0f, 4.0f
                                       };

        // Calculate output values for input.
        auto f = [](float x) {
            // gelu(x) = x * 1/2 * (1 + erf(x / sqrt(2))),
            // where erf is Gaussian error function
            auto result = x * (0.5f * (1.0f + erff(static_cast<float>(x / std::sqrt(2)))));
            return result;
        };
        std::vector<float> outputExpectedData(inputData.size());
        std::transform(inputData.begin(), inputData.end(), outputExpectedData.begin(), f);

        ActivationTest(tflite::BuiltinOperator_GELU, inputData, outputExpectedData);
    }
}

}    // namespace armnnDelegate