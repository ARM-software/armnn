//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_GatherNd")
{
struct GatherNdFixture : public ParserFlatbuffersFixture
{
    explicit GatherNdFixture(const std::string& paramsShape,
                             const std::string& indicesShape,
                             const std::string& outputShape,
                             const std::string& dataType = "FLOAT32",
                             const std::string& scale = "1.0",
                             const std::string& offset = "0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "GATHER_ND" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + paramsShape + R"(,
                            "type": )" + dataType + R"(,
                            "buffer": 0,
                            "name": "params",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                                "details_type": "NONE",
                                "quantized_dimension": 0
                            },
                            "is_variable": false,
                            "shape_signature": )" + paramsShape + R"(
                        },
                        {
                             "shape": )" + indicesShape + R"( ,
                             "type": "INT32",
                             "buffer": 1,
                             "name": "indices",
                              "quantization": {
                                "details_type": "NONE",
                                "quantized_dimension": 0
                              },
                             "is_variable": false
                        },
                        {
                             "shape": )" + outputShape + R"(,
                             "type": )" + dataType + R"(,
                             "buffer": 2,
                             "name": "output",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                                "details_type": "NONE",
                                "quantized_dimension": 0
                              },
                            "is_variable": false,
                            "shape_signature": )" + outputShape + R"(
                        }
                    ],
                    "inputs": [ 0, 1 ],
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
                    { },
                ]
            }
        )";
        Setup();
    }
};

struct SimpleGatherNdFixture : public GatherNdFixture
{
    SimpleGatherNdFixture() : GatherNdFixture("[ 5, 2 ]", "[ 3, 1 ]", "[ 3, 2 ]" ) {}
};

TEST_CASE_FIXTURE(SimpleGatherNdFixture, "ParseGatherNd")
{
    RunTest<2, armnn::DataType::Float32, armnn::DataType::Signed32, armnn::DataType::Float32>
        (0,
         {{ "params",  { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }}},
         {{ "indices", { 1, 3, 4 }}},
         {{ "output", { 3, 4, 7, 8, 9, 10 }}});
}

struct GatherNdUint8Fixture : public GatherNdFixture
{
    GatherNdUint8Fixture() : GatherNdFixture("[ 8 ]", "[ 3, 1 ]", "[ 3 ]", "UINT8") {}
};

TEST_CASE_FIXTURE(GatherNdUint8Fixture, "ParseGatherNdUint8")
{
    RunTest<1, armnn::DataType::QAsymmU8, armnn::DataType::Signed32, armnn::DataType::QAsymmU8>
        (0,
         {{ "params",  { 1, 2, 3, 4, 5, 6, 7, 8 }}},
         {{ "indices", { 7, 6, 5 }}},
         {{ "output", { 8, 7, 6 }}});
}

}
