//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Pad")
{
struct PadV2Fixture : public ParserFlatbuffersFixture
{
    explicit PadV2Fixture(const std::string& inputShape,
                          const std::string& outputShape,
                          const std::string& padListShape,
                          const std::string& padListData,
                          const std::string& constantValuesShape,
                          const std::string& constantValuesData,
                          const std::string& dataType = "FLOAT32",
                          const std::string& scale = "1.0",
                          const std::string& offset = "0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "PADV2" } ],
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
                             "type": "INT64",
                             "buffer": 2,
                             "name": "padList",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                             }
                        },
                        {
                             "shape": )" + constantValuesShape + R"( ,
                             "type": )" + dataType + R"(,
                             "buffer": 3,
                             "name": "constantValues",
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
                            "inputs": [ 0, 2, 3 ],
                            "outputs": [ 1 ],
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + padListData + R"(, },
                    { "data": )" + constantValuesData + R"(, },
                ]
            }
        )";
      SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimplePadV2Fixture : public PadV2Fixture
{
    SimplePadV2Fixture() : PadV2Fixture("[ 2,3 ]", "[ 4,7 ]", "[ 2,2 ]",
                                        "[ 1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0 ]",
                                        "[1]", "[0,0,160,64]") {}
};

TEST_CASE_FIXTURE(SimplePadV2Fixture, "ParsePadV2")
{
    RunTest<2, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}},
         {{ "outputTensor", { 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
                              5.0f, 5.0f, 1.0f, 2.0f, 3.0f, 5.0f, 5.0f,
                              5.0f, 5.0f, 4.0f, 5.0f, 6.0f, 5.0f, 5.0f,
                              5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f }}});
}

struct NoConstValuePadV2Fixture : public PadV2Fixture
{
    NoConstValuePadV2Fixture() : PadV2Fixture("[ 2,3 ]", "[ 4,7 ]", "[ 2,2 ]",
                                              "[ 1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0 ]",
                                              "[]", "[]") {}
};

TEST_CASE_FIXTURE(NoConstValuePadV2Fixture, "ParsePadV2NoConstValue")
{
    RunTest<2, armnn::DataType::Float32>
            (0,
             {{ "inputTensor",  { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}},
             {{ "outputTensor", { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, .0f,
                                  0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }}});
}

struct Uint8PadV2Fixture : public PadV2Fixture
{
    Uint8PadV2Fixture() : PadV2Fixture("[ 2,3 ]", "[ 4,7 ]", "[ 2,2 ]",
                                       "[ 1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0 ]",
                                       "[1]", "[1]","UINT8", "-2.0", "3") {}
};

TEST_CASE_FIXTURE(Uint8PadV2Fixture, "ParsePadV2Uint8")
{
    RunTest<2, armnn::DataType::QAsymmU8>
        (0,
         {{ "inputTensor",  { 1, 2, 3, 4, 5, 6 }}},
         {{ "outputTensor", { 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 2, 3, 1, 1,
                              1, 1, 4, 5, 6, 1, 1,
                              1, 1, 1, 1, 1, 1, 1 }}});
}

struct Int8PadV2Fixture : public PadV2Fixture
{
    Int8PadV2Fixture() : PadV2Fixture("[ 2,3 ]", "[ 4,7 ]", "[ 2,2 ]",
                                      "[ 1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0 ]",
                                      "[1]", "[2]","INT8", "-2.0", "3") {}
};

TEST_CASE_FIXTURE(Int8PadV2Fixture, "ParsePadV2Int8")
{
    RunTest<2, armnn::DataType::QAsymmS8>
        (0,
         {{ "inputTensor",  { 1, -2, 3, 4, 5, -6 }}},
         {{ "outputTensor", { 2, 2, 2, 2, 2, 2, 2,
                              2, 2, 1, -2, 3, 2, 2,
                              2, 2, 4, 5, -6, 2, 2,
                              2, 2, 2, 2, 2, 2, 2 }}});
}

}
