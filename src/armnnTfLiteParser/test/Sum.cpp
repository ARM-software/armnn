//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Sum")
{
struct SumFixture : public ParserFlatbuffersFixture
{
    explicit SumFixture(const std::string& inputShape,
                        const std::string& outputShape,
                        const std::string& axisShape,
                        const std::string& axisData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SUM" } ],
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
                            "type": "FLOAT32",
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
                            "shape": )" + axisShape + R"( ,
                            "type": "INT32",
                            "buffer": 2,
                            "name": "axis",
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
                            "inputs": [ 0 , 2 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "ReducerOptions",
                            "builtin_options": {
                              "keep_dims": true,
                            },
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

struct SimpleSumFixture : public SumFixture
{
    SimpleSumFixture() : SumFixture("[ 1, 3, 2, 4 ]", "[ 1, 1, 1, 4 ]", "[ 2 ]", "[ 1, 0, 0, 0,  2, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(SimpleSumFixture, "ParseSum")
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>
        (0, {{ "inputTensor", { 1.0f,   2.0f,   3.0f,   4.0f,
                                5.0f,   6.0f,   7.0f,   8.0f,

                                10.0f,  20.0f,  30.0f,  40.0f,
                                50.0f,  60.0f,  70.0f,  80.0f,

                                100.0f, 200.0f, 300.0f, 400.0f,
                                500.0f, 600.0f, 700.0f, 800.0f } } },
            {{ "outputTensor", { 666.0f, 888.0f, 1110.0f, 1332.0f } } });
}

}
