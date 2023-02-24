//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_SpaceToDepth")
{
struct SpaceToDepthFixture : public ParserFlatbuffersFixture
{
    explicit SpaceToDepthFixture(const std::string& inputShape,
                                 const std::string& outputShape,
                                 const std::string& dataType = "FLOAT32",
                                 const std::string& scale = "1.0",
                                 const std::string& offset = "0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "DEPTH_TO_SPACE" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
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
                             "shape": )" + outputShape + R"(,
                             "type": )" + dataType + R"(,
                             "buffer": 1,
                             "name": "outputTensor",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
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
                            "builtin_options_type": "SpaceToDepthOptions",
                            "builtin_options": {
                              "block_size": 2
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                ]
            }
        )";
      SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleSpaceToDepthFixture : public SpaceToDepthFixture
{
    SimpleSpaceToDepthFixture() : SpaceToDepthFixture("[ 1, 2, 2, 1 ]", "[ 1, 1, 1, 4 ]") {}
};

TEST_CASE_FIXTURE(SimpleSpaceToDepthFixture, "ParseSpaceToDepth")
{
    RunTest<4, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.f,  2.f,  3.f,  4.f }}},
         {{ "outputTensor", { 1.f,  2.f,  3.f,  4.f }}} );
}

}
