//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Gather")
{
struct GatherFixture : public ParserFlatbuffersFixture
{
    explicit GatherFixture(const std::string& paramsShape,
                           const std::string& outputShape,
                           const std::string& indicesShape,
                           const std::string& dataType = "FLOAT32",
                           const std::string& scale = "1.0",
                           const std::string& offset = "0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "GATHER" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + paramsShape + R"(,
                            "type": )" + dataType + R"(,
                            "buffer": 0,
                            "name": "inputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                            }
                        },
                        {
                             "shape": )" + indicesShape + R"( ,
                             "type": "INT32",
                             "buffer": 1,
                             "name": "indices",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                             }
                        },
                        {
                             "shape": )" + outputShape + R"(,
                             "type": )" + dataType + R"(,
                             "buffer": 2,
                             "name": "outputTensor",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                            }
                        }
                    ],
                    "inputs": [ 0, 1 ],
                    "outputs": [ 2 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1 ],
                            "outputs": [ 2 ],
                            "builtin_options_type": "GatherOptions",
                            "builtin_options": {
                              "axis": 0
                            },
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

struct SimpleGatherFixture : public GatherFixture
{
    SimpleGatherFixture() : GatherFixture("[ 5, 2 ]", "[ 3, 2 ]", "[ 3 ]") {}
};

TEST_CASE_FIXTURE(SimpleGatherFixture, "ParseGather")
{
    RunTest<2, armnn::DataType::Float32, armnn::DataType::Signed32, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }}},
         {{ "indices", { 1, 3, 4 }}},
         {{ "outputTensor", { 3, 4, 7, 8, 9, 10 }}});
}

struct GatherUint8Fixture : public GatherFixture
{
    GatherUint8Fixture() : GatherFixture("[ 8 ]", "[ 3 ]", "[ 3 ]", "UINT8") {}
};

TEST_CASE_FIXTURE(GatherUint8Fixture, "ParseGatherUint8")
{
    RunTest<1, armnn::DataType::QAsymmU8, armnn::DataType::Signed32, armnn::DataType::QAsymmU8>
        (0,
         {{ "inputTensor",  { 1, 2, 3, 4, 5, 6, 7, 8 }}},
         {{ "indices", { 7, 6, 5 }}},
         {{ "outputTensor", { 8, 7, 6 }}});
}

}
