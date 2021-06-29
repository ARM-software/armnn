//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Pad")
{
struct PadFixture : public ParserFlatbuffersFixture
{
    explicit PadFixture(const std::string& inputShape,
                        const std::string& outputShape,
                        const std::string& padListShape,
                        const std::string& padListData,
                        const std::string& dataType = "FLOAT32",
                        const std::string& scale = "1.0",
                        const std::string& offset = "0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "PAD" } ],
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
                        },
                        {
                             "shape": )" + padListShape + R"( ,
                             "type": "INT32",
                             "buffer": 2,
                             "name": "padList",
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
                            "inputs": [ 0, 2 ],
                            "outputs": [ 1 ],
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + padListData + R"(, },
                ]
            }
        )";
      SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimplePadFixture : public PadFixture
{
    SimplePadFixture() : PadFixture("[ 2, 3 ]", "[ 4, 7 ]", "[ 2, 2 ]",
                                    "[  1,0,0,0, 1,0,0,0, 2,0,0,0, 2,0,0,0 ]") {}
};

TEST_CASE_FIXTURE(SimplePadFixture, "ParsePad")
{
    RunTest<2, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}},
         {{ "outputTensor", { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }}});
}

struct Uint8PadFixture : public PadFixture
{
    Uint8PadFixture() : PadFixture("[ 2, 3 ]", "[ 4, 7 ]", "[ 2, 2 ]",
                                  "[  1,0,0,0, 1,0,0,0, 2,0,0,0, 2,0,0,0 ]",
                                  "UINT8", "-2.0", "3") {}
};

TEST_CASE_FIXTURE(Uint8PadFixture, "ParsePadUint8")
{
    RunTest<2, armnn::DataType::QAsymmU8>
        (0,
         {{ "inputTensor",  { 1, 2, 3, 4, 5, 6 }}},
         {{ "outputTensor", { 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 1, 2, 3, 3, 3,
                              3, 3, 4, 5, 6, 3, 3,
                              3, 3, 3, 3, 3, 3, 3 }}});
}

struct Int8PadFixture : public PadFixture
{
    Int8PadFixture() : PadFixture("[ 2, 3 ]", "[ 4, 7 ]", "[ 2, 2 ]",
                                    "[  1,0,0,0, 1,0,0,0, 2,0,0,0, 2,0,0,0 ]",
                                    "INT8", "-2.0", "3") {}
};

TEST_CASE_FIXTURE(Int8PadFixture, "ParsePadInt8")
{
    RunTest<2, armnn::DataType::QAsymmS8>
        (0,
         {{ "inputTensor",  { 1, -2, 3, 4, 5, -6 }}},
         {{ "outputTensor", { 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 1, -2, 3, 3, 3,
                              3, 3, 4, 5, -6, 3, 3,
                              3, 3, 3, 3, 3, 3, 3 }}});
}

}
