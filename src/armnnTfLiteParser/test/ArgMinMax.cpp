//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_ArgMinMax")
{
struct ArgMinMaxFixture : public ParserFlatbuffersFixture
{
    explicit ArgMinMaxFixture(const std::string& operatorCode,
                              const std::string& inputShape,
                              const std::string& outputShape,
                              const std::string& axisData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": )" + operatorCode + R"( } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "FLOAT32",
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
                            "shape": )" + outputShape + R"( ,
                            "type": "INT32",
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ 1 ],
                            "type": "INT32",
                            "buffer": 2,
                            "name": "axis",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0 , 2 ],
                            "outputs": [ 1 ],
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + axisData + R"(, },
                ]
            }
        )";

        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleArgMaxFixture : public ArgMinMaxFixture
{
    SimpleArgMaxFixture() : ArgMinMaxFixture("ARG_MAX",
                                             "[ 1, 1, 1, 5 ]",
                                             "[ 1, 1, 1 ]",
                                             "[ 3, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(SimpleArgMaxFixture, "ParseSimpleArgMax")
{
    RunTest<3, armnn::DataType::Float32, armnn::DataType::Signed32>(
            0,
            {{ "inputTensor",  { 6.0f, 2.0f, 8.0f, 10.0f, 9.0f } } },
            {{ "outputTensor", { 3l } } });
}

struct ArgMaxFixture : public ArgMinMaxFixture
{
    ArgMaxFixture() : ArgMinMaxFixture("ARG_MAX",
                                       "[ 3, 2, 1, 4 ]",
                                       "[ 2, 1, 4 ]",
                                       "[ 0, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(ArgMaxFixture, "ParseArgMax")
{
    RunTest<3, armnn::DataType::Float32, armnn::DataType::Signed32>(
            0,
            {{ "inputTensor", { 1.0f,   2.0f,   3.0f,   4.0f,
                                8.0f,   7.0f,   6.0f,   6.0f,
                                100.0f, 20.0f,  300.0f, 40.0f,
                                500.0f, 476.0f, 450.0f, 426.0f,
                                50.0f,  60.0f,  70.0f,  80.0f,
                                10.0f,  200.0f, 30.0f,  400.0f } } },
            {{ "outputTensor", { 1, 2, 1, 2,
                                 1, 1, 1, 1 } } });
}

struct SimpleArgMinFixture : public ArgMinMaxFixture
{
    SimpleArgMinFixture() : ArgMinMaxFixture("ARG_MIN",
                                             "[ 1, 1, 1, 5 ]",
                                             "[ 1, 1, 1 ]",
                                             "[ 3, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(SimpleArgMinFixture, "ParseSimpleArgMin")
{
    RunTest<3, armnn::DataType::Float32, armnn::DataType::Signed32>(
            0,
            {{ "inputTensor",  { 6.0f, 2.0f, 8.0f, 10.0f, 9.0f } } },
            {{ "outputTensor", { 1l } } });
}

struct ArgMinFixture : public ArgMinMaxFixture
{
    ArgMinFixture() : ArgMinMaxFixture("ARG_MIN",
                                       "[ 3, 2, 1, 4 ]",
                                       "[ 2, 1, 4 ]",
                                       "[ 0, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(ArgMinFixture, "ParseArgMin")
{
    RunTest<3, armnn::DataType::Float32, armnn::DataType::Signed32>(
            0,
            {{ "inputTensor", { 1.0f,   2.0f,   3.0f,   4.0f,
                                8.0f,   7.0f,   6.0f,   6.0f,
                                100.0f, 20.0f,  300.0f, 40.0f,
                                500.0f, 476.0f, 450.0f, 426.0f,
                                50.0f,  60.0f,  70.0f,  80.0f,
                                10.0f,  200.0f, 30.0f,  400.0f } } },
            {{ "outputTensor", { 0, 0, 0, 0,
                                 0, 0, 0, 0 } } });
}

}
