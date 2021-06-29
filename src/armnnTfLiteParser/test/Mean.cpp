//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Mean")
{
struct MeanNoReduceFixture : public ParserFlatbuffersFixture
{
    explicit MeanNoReduceFixture(const std::string & inputShape,
                                 const std::string & outputShape,
                                 const std::string & dimShape,
                                 const std::string & dimData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "MEAN" } ],
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
                            "shape": )" + dimShape + R"( ,
                            "type": "INT32",
                            "buffer": 2,
                            "name": "dimShape",
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
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + dimData + R"(, },
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleMeanNoReduceFixture : public MeanNoReduceFixture
{
    SimpleMeanNoReduceFixture() : MeanNoReduceFixture("[ 2, 2 ]", "[ 1, 1 ]", "[ 0 ]", "[ ]") {}
};

TEST_CASE_FIXTURE(SimpleMeanNoReduceFixture, "ParseMeanNoReduce")
{
    RunTest<2, armnn::DataType::Float32>(0, {{ "inputTensor", { 1.0f, 1.0f, 2.0f, 2.0f } } },
                                            {{ "outputTensor", { 1.5f } } });
}

}
