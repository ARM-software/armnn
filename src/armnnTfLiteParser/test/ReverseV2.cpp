//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_ReverseV2")
{
struct ReverseV2Fixture : public ParserFlatbuffersFixture
{
    explicit ReverseV2Fixture(const std::string& inputShape,
                              const std::string& outputShape,
                              const std::string& axisShape,
                              const std::string& axisData,
                              const std::string& dataType = "FLOAT32",
                              const std::string& scale = "1.0",
                              const std::string& offset = "0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "REVERSE_V2" } ],
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
                             "shape": )" + axisShape + R"( ,
                             "type": "INT32",
                             "buffer": 1,
                             "name": "axis",
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
                            "builtin_options_type": "ReverseV2Options",
                            "builtin_options": {},
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { "data": )" + axisData + R"(, },
                    { },
                ]
            }
        )";
        Setup();
    }
};

struct SimpleReverseV2Fixture : public ReverseV2Fixture
{
    SimpleReverseV2Fixture() : ReverseV2Fixture("[ 2, 2, 2 ]", "[ 2, 2, 2 ]", "[ 2 ]", "[ 0,0,0,0, 1,0,0,0 ]") {}
};

TEST_CASE_FIXTURE(SimpleReverseV2Fixture, "ParseReverseV2")
{
    RunTest<3, armnn::DataType::Float32>
        (0,
        {{ "inputTensor",  { 1, 2, 3, 4, 5, 6, 7, 8 }}},
        {{ "outputTensor", { 7, 8, 5, 6, 3, 4, 1, 2 }}});
}

}