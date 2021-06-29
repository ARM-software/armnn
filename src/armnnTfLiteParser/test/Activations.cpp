//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

#include <doctest/doctest.h>

TEST_SUITE("TensorflowLiteParser_Activations")
{
struct ActivationFixture : ParserFlatbuffersFixture
{

    explicit ActivationFixture(std::string activationFunction, std::string dataType)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": )" + activationFunction + R"( } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 7 ],
                            "type": )" + dataType + R"(,
                            "buffer": 0,
                            "name": "inputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ 1, 7 ],
                            "type": )" + dataType + R"(,
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                          "opcode_index": 0,
                          "inputs": [ 0 ],
                          "outputs": [ 1 ],
                          "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [ {}, {} ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }

};

struct ReLuFixture : ActivationFixture
{
    ReLuFixture() : ActivationFixture("RELU", "FLOAT32") {}
};
TEST_CASE_FIXTURE(ReLuFixture, "ParseReLu")
{
    RunTest<2, armnn::DataType::Float32>(0, { -1.0f, -0.5f, 1.25f, -3.0f, 0.0f, 0.5f, -0.75f },
                                         { 0.0f, 0.0f, 1.25f, 0.0f, 0.0f, 0.5f, 0.0f });
}

struct ReLu6Fixture : ActivationFixture
{
    ReLu6Fixture() : ActivationFixture("RELU6", "FLOAT32") {}
};
TEST_CASE_FIXTURE(ReLu6Fixture, "ParseReLu6")
{
    RunTest<2, armnn::DataType::Float32>(0, { -1.0f, -0.5f, 7.25f, -3.0f, 0.0f, 0.5f, -0.75f },
                                         { 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.5f, 0.0f });
}

struct SigmoidFixture : ActivationFixture
{
    SigmoidFixture() : ActivationFixture("LOGISTIC", "FLOAT32") {}
};
TEST_CASE_FIXTURE(SigmoidFixture, "ParseLogistic")
{
    RunTest<2, armnn::DataType::Float32>(0, { -1.0f,     -0.5f,      4.0f,       -4.0f,  0.0f,      0.5f,     -0.75f },
                                         {0.268941f, 0.377541f, 0.982013f,  0.0179862f,  0.5f, 0.622459f,  0.320821f });
}

struct TanHFixture : ActivationFixture
{
    TanHFixture() : ActivationFixture("TANH", "FLOAT32") {}
};

TEST_CASE_FIXTURE(TanHFixture, "ParseTanH")
{
    RunTest<2, armnn::DataType::Float32>(0,
        { -0.1f,       -0.2f,         -0.3f,       -0.4f,    0.1f,         0.2f,              0.3f },
        { -0.09966799f, -0.19737528f, -0.29131261f, -0.379949f, 0.09966799f, 0.19737528f, 0.29131261f });
}

struct EluFixture : ActivationFixture
{
    EluFixture() : ActivationFixture("ELU", "FLOAT32") {}
};

TEST_CASE_FIXTURE(EluFixture, "ParseElu")
{
    RunTest<2, armnn::DataType::Float32>(0,
                                         { -2.0f,           -1.0f,           -0.0f, 0.0f, 1.0f, 2.0f, 3.0f },
                                         { -0.86466471676f, -0.63212055882f, -0.0f, 0.0f, 1.0f, 2.0f, 3.0f });
}

struct HardSwishFixture : ActivationFixture
{
    HardSwishFixture() : ActivationFixture("HARD_SWISH", "FLOAT32") {}
};

TEST_CASE_FIXTURE(HardSwishFixture, "ParseHardSwish")
{
    RunTest<2, armnn::DataType::Float32>(0,
                                         { -4.0f, -3.0f,        -2.9f,  1.2f,        2.2f, 3.0f, 4.0f },
                                         { -0.0f, -0.0f, -0.04833334f, 0.84f, 1.90666667f, 3.0f, 4.0f });
}

}
