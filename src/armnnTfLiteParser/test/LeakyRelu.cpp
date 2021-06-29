//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_LeakyRelu")
{
struct LeakyReluFixture : public ParserFlatbuffersFixture
{
    explicit LeakyReluFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "LEAKY_RELU" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 7 ],
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
                            "shape": [ 1, 7 ],
                            "type": "FLOAT32",
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
                          "builtin_options_type": "LeakyReluOptions",
                          "builtin_options": {
                            "alpha": 0.01
                          },
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

TEST_CASE_FIXTURE(LeakyReluFixture, "ParseLeakyRelu")
{
    RunTest<2, armnn::DataType::Float32>(0,
                                         {{ "inputTensor",  { -0.1f, -0.2f, -0.3f, -0.4f, 0.1f, 0.2f, 0.3f }}},
                                         {{ "outputTensor", { -0.001f, -0.002f, -0.003f, -0.004f, 0.1f, 0.2f, 0.3f }}});
}

}
