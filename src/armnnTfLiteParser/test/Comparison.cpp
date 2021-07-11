//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>

TEST_SUITE("TensorflowLiteParser_Comparison")
{
struct ComparisonFixture : public ParserFlatbuffersFixture
{
    explicit ComparisonFixture(const std::string& operatorCode,
                                     const std::string& dataType,
                                     const std::string& inputShape,
                                     const std::string& inputShape2,
                                     const std::string& outputShape)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": )" + operatorCode + R"( } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": )" + dataType + R"( ,
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
                            "shape": )" + inputShape2 + R"(,
                            "type": )" + dataType + R"( ,
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
                            "type": "BOOL",
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

struct SimpleEqualFixture : public ComparisonFixture
{
    SimpleEqualFixture() : ComparisonFixture("EQUAL", "UINT8", "[ 2, 2 ]", "[ 2, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(SimpleEqualFixture, "SimpleEqual")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 0, 1, 2, 3 }},
                    {"inputTensor2", { 0, 1, 5, 6 }}},
                   {{"outputTensor", { 1, 1, 0, 0 }}});
}

struct BroadcastEqualFixture : public ComparisonFixture
{
    BroadcastEqualFixture() : ComparisonFixture("EQUAL", "UINT8", "[ 2, 2 ]", "[ 1, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(BroadcastEqualFixture, "BroadcastEqual")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 0, 1, 2, 3 }},
                    {"inputTensor2", { 0, 1 }}},
                   {{"outputTensor", { 1, 1, 0, 0 }}});
}

struct SimpleNotEqualFixture : public ComparisonFixture
{
    SimpleNotEqualFixture() : ComparisonFixture("NOT_EQUAL", "UINT8", "[ 2, 2 ]", "[ 2, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(SimpleNotEqualFixture, "SimpleNotEqual")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 0, 1, 2, 3 }},
                    {"inputTensor2", { 0, 1, 5, 6 }}},
                   {{"outputTensor", { 0, 0, 1, 1 }}});
}

struct BroadcastNotEqualFixture : public ComparisonFixture
{
    BroadcastNotEqualFixture() : ComparisonFixture("NOT_EQUAL", "UINT8", "[ 2, 2 ]", "[ 1, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(BroadcastNotEqualFixture, "BroadcastNotEqual")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 0, 1, 2, 3 }},
                    {"inputTensor2", { 0, 1 }}},
                   {{"outputTensor", { 0, 0, 1, 1 }}});
}

struct SimpleGreaterFixture : public ComparisonFixture
{
    SimpleGreaterFixture() : ComparisonFixture("GREATER", "UINT8", "[ 2, 2 ]", "[ 2, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(SimpleGreaterFixture, "SimpleGreater")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 0, 2, 3, 6 }},
                    {"inputTensor2", { 0, 1, 5, 3 }}},
                   {{"outputTensor", { 0, 1, 0, 1 }}});
}

struct BroadcastGreaterFixture : public ComparisonFixture
{
    BroadcastGreaterFixture() : ComparisonFixture("GREATER", "UINT8", "[ 2, 2 ]", "[ 1, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(BroadcastGreaterFixture, "BroadcastGreater")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 5, 4, 1, 0 }},
                    {"inputTensor2", { 2, 3 }}},
                   {{"outputTensor", { 1, 1, 0, 0 }}});
}

struct SimpleGreaterOrEqualFixture : public ComparisonFixture
{
    SimpleGreaterOrEqualFixture() : ComparisonFixture("GREATER_EQUAL", "UINT8", "[ 2, 2 ]", "[ 2, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(SimpleGreaterOrEqualFixture, "SimpleGreaterOrEqual")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 0, 2, 3, 6 }},
                    {"inputTensor2", { 0, 1, 5, 3 }}},
                   {{"outputTensor", { 1, 1, 0, 1 }}});
}

struct BroadcastGreaterOrEqualFixture : public ComparisonFixture
{
    BroadcastGreaterOrEqualFixture() : ComparisonFixture("GREATER_EQUAL", "UINT8",
                                                         "[ 2, 2 ]", "[ 1, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(BroadcastGreaterOrEqualFixture, "BroadcastGreaterOrEqual")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 5, 4, 1, 0 }},
                    {"inputTensor2", { 2, 4 }}},
                   {{"outputTensor", { 1, 1, 0, 0 }}});
}

struct SimpleLessFixture : public ComparisonFixture
{
    SimpleLessFixture() : ComparisonFixture("LESS", "UINT8", "[ 2, 2 ]", "[ 2, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(SimpleLessFixture, "SimpleLess")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 0, 2, 3, 6 }},
                    {"inputTensor2", { 0, 1, 5, 3 }}},
                   {{"outputTensor", { 0, 0, 1, 0 }}});
}

struct BroadcastLessFixture : public ComparisonFixture
{
    BroadcastLessFixture() : ComparisonFixture("LESS", "UINT8", "[ 2, 2 ]", "[ 1, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(BroadcastLessFixture, "BroadcastLess")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 5, 4, 1, 0 }},
                    {"inputTensor2", { 2, 3 }}},
                   {{"outputTensor", { 0, 0, 1, 1 }}});
}

struct SimpleLessOrEqualFixture : public ComparisonFixture
{
    SimpleLessOrEqualFixture() : ComparisonFixture("LESS_EQUAL", "UINT8", "[ 2, 2 ]", "[ 2, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(SimpleLessOrEqualFixture, "SimpleLessOrEqual")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 0, 2, 3, 6 }},
                    {"inputTensor2", { 0, 1, 5, 3 }}},
                   {{"outputTensor", { 1, 0, 1, 0 }}});
}

struct BroadcastLessOrEqualFixture : public ComparisonFixture
{
    BroadcastLessOrEqualFixture() : ComparisonFixture("LESS_EQUAL", "UINT8", "[ 2, 2 ]", "[ 1, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(BroadcastLessOrEqualFixture, "BroadcastLessOrEqual")
{
    RunTest<2, armnn::DataType::QAsymmU8,
               armnn::DataType::Boolean>(
                   0,
                   {{"inputTensor",  { 5, 4, 1, 0 }},
                    {"inputTensor2", { 1, 3 }}},
                   {{"outputTensor", { 0, 0, 1, 1 }}});
}

}
