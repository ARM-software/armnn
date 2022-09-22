//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_BatchMatMul")
{
struct BatchMatMulFixture : public ParserFlatbuffersFixture
{
    explicit BatchMatMulFixture(const std::string &inputXShape,
                                const std::string &inputYShape,
                                const std::string &outputShape,
                                const std::string &tranX,
                                const std::string &tranY)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "BATCH_MATMUL" } ],
                "subgraphs": [
                    {
                        "tensors": [
                            {
                                "shape": )" + inputXShape + R"(,
                                "type": "FLOAT32",
                                "buffer": 0,
                                "name": "inputXTensor",
                                "quantization": {
                                    "min": [ 0.0 ],
                                    "max": [ 255.0 ],
                                    "scale": [ 1.0 ],
                                    "zero_point": [ 0 ],
                                }
                            },
                            {
                                "shape": )" + inputYShape + R"(,
                                "type": "FLOAT32",
                                "buffer": 1,
                                "name": "inputYTensor",
                                "quantization": {
                                    "min": [ 0.0 ],
                                    "max": [ 255.0 ],
                                    "scale": [ 1.0 ],
                                    "zero_point": [ 0 ],
                                }
                            },
                            {
                                "shape": )" + outputShape + R"(,
                                "type": "FLOAT32",
                                "buffer": 2,
                                "name": "outputTensor",
                                "quantization": {
                                    "min": [ 0.0 ],
                                    "max": [ 255.0 ],
                                    "scale": [ 1.0 ],
                                    "zero_point": [ 0 ],
                                }
                            }
                        ],
                        "inputs": [ 0, 1 ],
                        "outputs": [ 2 ],
                        "operators": [
                            {
                                "opcode_index": 0,
                                "inputs": [ 0 , 1 ],
                                "outputs": [ 2 ],
                                "builtin_options_type": "BatchMatMulOptions",
                                "builtin_options": {
                                    adj_x: )" + tranX + R"(,
                                    adj_y: )" + tranY + R"(,
                                    "asymmetric_quantize_inputs": false
                                },
                                "custom_options_format": "FLEXBUFFERS"
                            }
                        ]
                    }
                ],
                "buffers": [{},{}]
            }
        )";
        Setup();
    }
};

struct BatchMatMulParamsFixture : BatchMatMulFixture
{
    BatchMatMulParamsFixture()
        : BatchMatMulFixture("[ 1, 3, 3 ]",
                             "[ 1, 3, 3 ]",
                             "[ 1, 3, 3 ]",
                             "false",
                             "true")
    {}
};

TEST_CASE_FIXTURE(BatchMatMulParamsFixture, "ParseBatchMatMulParams")
{
    RunTest<3, armnn::DataType::Float32>(
        0,
        {{"inputXTensor", {2.0f, 3.0f, 5.0f,
                           8.0f, 13.0f, 21.0f,
                           34.0f, 55.0f, 89.0f}},
         {"inputYTensor", {0.0f, 1.0f, 1.0f,
                           1.0f, 0.0f, 1.0f,
                           1.0f, 1.0f, 0.0f}}},
        {{"outputTensor", {8.0f, 7.0f, 5.0f,
                           34.0f, 29.0f, 21.0f,
                           144.0f, 123.0f, 89.0f}}}
        );
}

}