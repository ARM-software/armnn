//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Concatenation")
{
struct ConcatenationFixture : public ParserFlatbuffersFixture
{
    explicit ConcatenationFixture(const std::string & inputShape1,
                                  const std::string & inputShape2,
                                  const std::string & outputShape,
                                  const std::string & axis,
                                  const std::string & activation="NONE")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "CONCATENATION" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape1 + R"(,
                            "type": "UINT8",
                            "buffer": 0,
                            "name": "inputTensor1",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + inputShape2 + R"(,
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "inputTensor2",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + outputShape + R"( ,
                            "type": "UINT8",
                            "buffer": 2,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
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
                            "builtin_options_type": "ConcatenationOptions",
                            "builtin_options": {
                                "axis": )" + axis + R"(,
                                "fused_activation_function": )" + activation + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { }
                ]
            }
        )";
        Setup();
    }
};


struct ConcatenationFixtureNegativeDim : ConcatenationFixture
{
    ConcatenationFixtureNegativeDim() : ConcatenationFixture("[ 1, 1, 2, 2 ]",
                                                             "[ 1, 1, 2, 2 ]",
                                                             "[ 1, 2, 2, 2 ]",
                                                             "-3" ) {}
};

TEST_CASE_FIXTURE(ConcatenationFixtureNegativeDim, "ParseConcatenationNegativeDim")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        {{"inputTensor1", { 0, 1, 2, 3 }},
        {"inputTensor2", { 4, 5, 6, 7 }}},
        {{"outputTensor", { 0, 1, 2, 3, 4, 5, 6, 7 }}});
}

struct ConcatenationFixtureNCHW : ConcatenationFixture
{
    ConcatenationFixtureNCHW() : ConcatenationFixture("[ 1, 1, 2, 2 ]", "[ 1, 1, 2, 2 ]", "[ 1, 2, 2, 2 ]", "1" ) {}
};

TEST_CASE_FIXTURE(ConcatenationFixtureNCHW, "ParseConcatenationNCHW")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        {{"inputTensor1", { 0, 1, 2, 3 }},
        {"inputTensor2", { 4, 5, 6, 7 }}},
        {{"outputTensor", { 0, 1, 2, 3, 4, 5, 6, 7 }}});
}

struct ConcatenationFixtureNHWC : ConcatenationFixture
{
    ConcatenationFixtureNHWC() : ConcatenationFixture("[ 1, 1, 2, 2 ]", "[ 1, 1, 2, 2 ]", "[ 1, 1, 2, 4 ]", "3" ) {}
};

TEST_CASE_FIXTURE(ConcatenationFixtureNHWC, "ParseConcatenationNHWC")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        {{"inputTensor1", { 0, 1, 2, 3 }},
        {"inputTensor2", { 4, 5, 6, 7 }}},
        {{"outputTensor", { 0, 1, 4, 5, 2, 3, 6, 7 }}});
}

struct ConcatenationFixtureDim1 : ConcatenationFixture
{
    ConcatenationFixtureDim1() : ConcatenationFixture("[ 1, 2, 3, 4 ]", "[ 1, 2, 3, 4 ]", "[ 1, 4, 3, 4 ]", "1" ) {}
};

TEST_CASE_FIXTURE(ConcatenationFixtureDim1, "ParseConcatenationDim1")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        { { "inputTensor1", {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 } },
        { "inputTensor2", {  50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                             62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73 } } },
        { { "outputTensor", {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                               62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73 } } });
}

struct ConcatenationFixtureDim3 : ConcatenationFixture
{
    ConcatenationFixtureDim3() : ConcatenationFixture("[ 1, 2, 3, 4 ]", "[ 1, 2, 3, 4 ]", "[ 1, 2, 3, 8 ]", "3" ) {}
};

TEST_CASE_FIXTURE(ConcatenationFixtureDim3, "ParseConcatenationDim3")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        { { "inputTensor1", {  0,  1,  2,  3,
                               4,  5,  6,  7,
                               8,  9, 10, 11,
                               12, 13, 14, 15,
                               16, 17, 18, 19,
                               20, 21, 22, 23 } },
        { "inputTensor2", {  50, 51, 52, 53,
                             54, 55, 56, 57,
                             58, 59, 60, 61,
                             62, 63, 64, 65,
                             66, 67, 68, 69,
                             70, 71, 72, 73 } } },
        { { "outputTensor", {  0,  1,  2,  3,
                               50, 51, 52, 53,
                               4,  5,  6,  7,
                               54, 55, 56, 57,
                               8,  9,  10, 11,
                               58, 59, 60, 61,
                               12, 13, 14, 15,
                               62, 63, 64, 65,
                               16, 17, 18, 19,
                               66, 67, 68, 69,
                               20, 21, 22, 23,
                               70, 71, 72, 73 } } });
}

struct ConcatenationFixture3DDim0 : ConcatenationFixture
{
    ConcatenationFixture3DDim0() : ConcatenationFixture("[ 1, 2, 3]", "[ 2, 2, 3]", "[ 3, 2, 3]", "0" ) {}
};

TEST_CASE_FIXTURE(ConcatenationFixture3DDim0, "ParseConcatenation3DDim0")
{
    RunTest<3, armnn::DataType::QAsymmU8>(
        0,
        { { "inputTensor1", { 0,  1,  2,  3,  4,  5 } },
          { "inputTensor2", { 6,  7,  8,  9, 10, 11,
                             12, 13, 14, 15, 16, 17 } } },
        { { "outputTensor", { 0,  1,  2,  3,  4,  5,
                              6,  7,  8,  9, 10, 11,
                             12, 13, 14, 15, 16, 17 } } });
}

struct ConcatenationFixture3DDim1 : ConcatenationFixture
{
    ConcatenationFixture3DDim1() : ConcatenationFixture("[ 1, 2, 3]", "[ 1, 4, 3]", "[ 1, 6, 3]", "1" ) {}
};

TEST_CASE_FIXTURE(ConcatenationFixture3DDim1, "ParseConcatenation3DDim1")
{
    RunTest<3, armnn::DataType::QAsymmU8>(
        0,
        { { "inputTensor1", { 0,  1,  2,  3,  4,  5 } },
          { "inputTensor2", { 6,  7,  8,  9, 10, 11,
                             12, 13, 14, 15, 16, 17 } } },
        { { "outputTensor", { 0,  1,  2,  3,  4,  5,
                              6,  7,  8,  9, 10, 11,
                             12, 13, 14, 15, 16, 17 } } });
}

struct ConcatenationFixture3DDim2 : ConcatenationFixture
{
    ConcatenationFixture3DDim2() : ConcatenationFixture("[ 1, 2, 3]", "[ 1, 2, 6]", "[ 1, 2, 9]", "2" ) {}
};

TEST_CASE_FIXTURE(ConcatenationFixture3DDim2, "ParseConcatenation3DDim2")
{
    RunTest<3, armnn::DataType::QAsymmU8>(
        0,
        { { "inputTensor1", { 0,  1,  2,
                              3,  4,  5 } },
          { "inputTensor2", { 6,  7,  8,  9, 10, 11,
                             12, 13, 14, 15, 16, 17 } } },
        { { "outputTensor", { 0,  1,  2,  6,  7,  8,  9, 10, 11,
                              3,  4,  5, 12, 13, 14, 15, 16, 17 } } });
}

}
