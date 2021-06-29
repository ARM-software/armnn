//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Cast")
{
struct CastFixture : public ParserFlatbuffersFixture
{
    explicit CastFixture(const std::string& inputShape,
                         const std::string& outputShape,
                         const std::string& inputDataType,
                         const std::string& outputDataType)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "CAST" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": )" + inputDataType + R"(,
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
                            "shape": )" + outputShape + R"(,
                            "type": )" + outputDataType + R"(,
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

struct SimpleCastFixture : CastFixture
{
    SimpleCastFixture() : CastFixture("[ 1, 6 ]",
                                      "[ 1, 6 ]",
                                      "INT32",
                                      "FLOAT32") {}
};

TEST_CASE_FIXTURE(SimpleCastFixture, "SimpleCast")
{
RunTest<2, armnn::DataType::Signed32 , armnn::DataType::Float32>(
0,
{{"inputTensor",  { 0,   -1,   5,   -100,   200,   -255 }}},
{{"outputTensor", { 0.0f, -1.0f, 5.0f, -100.0f, 200.0f, -255.0f }}});
}

}