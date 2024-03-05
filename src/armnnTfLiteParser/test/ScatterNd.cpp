//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_ScatterNd")
{
struct ScatterNdFixture : public ParserFlatbuffersFixture
{
    explicit ScatterNdFixture(const std::string& shapeShape,
                              const std::string& outputShape,
                              const std::string& dataType = "FLOAT32",
                              const std::string& scale = "1.0",
                              const std::string& offset = "0")
    {
        const std::string& indicesShape = "[ 3, 1 ]";
        const std::string& updatesShape = "[ 3 ]";

        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SCATTER_ND" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + indicesShape + R"(,
                            "type": "INT32",
                            "buffer": 0,
                            "name": "indices",
                            "quantization": {
                                "details_type": "NONE",
                                "quantized_dimension": 0
                              },
                        },
                        {
                             "shape": )" + updatesShape + R"( ,
                             "type": )" + dataType + R"(,
                             "buffer": 1,
                             "name": "updates",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                            }
                        },
                        {
                             "shape": )" + shapeShape + R"( ,
                              "type": "INT32",
                              "buffer": 2,
                              "name": "shape",
                              "quantization": {
                                "details_type": "NONE",
                                "quantized_dimension": 0
                              },
                              "is_variable": false,
                              "has_rank": true
                            },
                        {
                             "shape": )" + outputShape + R"(,
                             "type": )" + dataType + R"(,
                             "buffer": 3,
                             "name": "outputTensor",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                            }
                        }
                    ],
                    "inputs": [ 0, 1, 2 ],
                    "outputs": [ 3 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1, 2 ],
                            "outputs": [ 3 ],
                            "builtin_options_type": "ScatterNdOptions",
                            "builtin_options": {},
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { "data": [ 0, 1, 2 ],
                      "offset": 0,
                      "size": 0 },
                    { "data": [ 1, 2, 3 ],
                      "offset": 0,
                      "size": 0 },
                   { },
                   { },
                ]
            }
        )";
        Setup();
    }
};

struct SimpleScatterNdFixture : public ScatterNdFixture
{
    SimpleScatterNdFixture() : ScatterNdFixture("[ 1 ]", "[ 5 ]") {}
};

TEST_CASE_FIXTURE(SimpleScatterNdFixture, "ParseScatterNd")
{
    RunTest<2, armnn::DataType::Signed32, armnn::DataType::Float32>
        (0,
         {{ "shape", { 5 }}},
         {{ "outputTensor", {1, 2, 3, 0, 0 }}});
}

struct ScatterNdUint8Fixture : public ScatterNdFixture
{
    ScatterNdUint8Fixture() : ScatterNdFixture("[ 2 ]", "[ 3, 3 ]", "UINT8") {}
};

TEST_CASE_FIXTURE(ScatterNdUint8Fixture, "ParseScatterNdUint8")
{
    RunTest<2, armnn::DataType::Signed32, armnn::DataType::QAsymmU8>
        (0,
         {{ "shape", { 3, 3 }}},
         {{ "outputTensor", { 1, 0, 0, 0, 2, 0, 0, 0, 3 }}});
}

}
