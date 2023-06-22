//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"
#include <doctest/doctest.h>

TEST_SUITE("TensorflowLiteParser")
{
TEST_CASE("ParseEmptyBinaryData")
{
    ITfLiteParser::TfLiteParserOptions options;
    ITfLiteParserPtr m_Parser(ITfLiteParser::Create(armnn::Optional<ITfLiteParser::TfLiteParserOptions>(options)));
    // Should throw armnn::ParseException: Buffer doesn't conform to the expected Tensorflow Lite flatbuffers format.
    CHECK_THROWS_AS(m_Parser->CreateNetworkFromBinary({0}), armnn::ParseException);
}

struct NoInputBindingsFixture : public ParserFlatbuffersFixture
{
    explicit NoInputBindingsFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "CONV_2D" } ],
                "subgraphs": [ { } ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

TEST_CASE_FIXTURE(NoInputBindingsFixture, "ParseBadInputBindings")
{
    // Should throw armnn::ParseException: No input binding found for subgraph:0 and name:inputTensor.
    CHECK_THROWS_AS((RunTest<4, armnn::DataType::QAsymmU8>(0, { }, { 0 })), armnn::ParseException);
}

TEST_CASE("ParseInvalidFileName")
{
    // Nullptr should throw InvalidArgumentException
    CHECK_THROWS_AS(armnnTfLiteParser::TfLiteParserImpl::LoadModelFromFile(nullptr), armnn::InvalidArgumentException);
    // Empty string should throw FileNotFoundException.
    CHECK_THROWS_AS(armnnTfLiteParser::TfLiteParserImpl::LoadModelFromFile(""), armnn::FileNotFoundException);
    // Garbage string should throw FileNotFoundException.
    CHECK_THROWS_AS(armnnTfLiteParser::TfLiteParserImpl::LoadModelFromFile("askjfhseuirwqeuiy"),
                    armnn::FileNotFoundException);
    // Valid directory should throw InvalidArgumentException
    CHECK_THROWS_AS(armnnTfLiteParser::TfLiteParserImpl::LoadModelFromFile("."), armnn::InvalidArgumentException);
    // Valid file but not a regular file should throw InvalidArgumentException
    CHECK_THROWS_AS(armnnTfLiteParser::TfLiteParserImpl::LoadModelFromFile("/dev/null"),
                    armnn::InvalidArgumentException);
}

}
