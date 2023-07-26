//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_Tile")
{
struct TileFixture : public ParserFlatbuffersFixture
{
    explicit TileFixture(const std::string& inputShape,
                         const std::string& outputShape,
                         const std::string& multiplesShape,
                         const std::string& multiplesData,
                         const std::string& dataType = "FLOAT32",
                         const std::string& scale = "1.0",
                         const std::string& offset = "0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [
                  {
                    "deprecated_builtin_code": 69,
                    "version": 1,
                    "builtin_code": "TILE"
                  }
                ],
                "subgraphs": [
                {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": )" + dataType + R"(,
                            "buffer": 1,
                            "name": "inputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                            },
                            "is_variable": false,
                            "has_rank": true
                        },
                        {
                            "shape": )" + multiplesShape + R"(,
                            "type": "INT32",
                            "buffer": 2,
                            "name": "multiples",
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
                            },
                            "is_variable": false,
                            "has_rank": true
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 2 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1 ],
                            "outputs": [ 2 ],
                            "builtin_options_type": "NONE",
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + multiplesData + R"(, },
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleTileFixture : public TileFixture
{
    SimpleTileFixture() : TileFixture("[ 2, 2 ]", "[ 4, 6 ]", "[ 2 ]", "[ 2, 0, 0, 0, 3, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(SimpleTileFixture, "ParseTile")
{
    RunTest<2, armnn::DataType::Float32, armnn::DataType::Float32>
        (0,
        {{ "inputTensor",  { 1, 2,
                             3, 4 }}},
        {{ "outputTensor", { 1, 2, 1, 2, 1, 2,
                             3, 4, 3, 4, 3, 4,
                             1, 2, 1, 2, 1, 2,
                             3, 4, 3, 4, 3, 4, }}});
}

}