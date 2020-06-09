//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

BOOST_AUTO_TEST_CASE(ParseEmptyBinaryData)
{
    ITfLiteParser::TfLiteParserOptions options;
    ITfLiteParserPtr m_Parser(ITfLiteParser::Create(armnn::Optional<ITfLiteParser::TfLiteParserOptions>(options)));
    // Should throw armnn::ParseException: Buffer doesn't conform to the expected Tensorflow Lite flatbuffers format.
    BOOST_CHECK_THROW(m_Parser->CreateNetworkFromBinary({0}), armnn::ParseException);
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

BOOST_FIXTURE_TEST_CASE( ParseBadInputBindings, NoInputBindingsFixture )
{
    // Should throw armnn::ParseException: No input binding found for subgraph:0 and name:inputTensor.
    BOOST_CHECK_THROW( (RunTest<4, armnn::DataType::QAsymmU8>(0, { }, { 0 })), armnn::ParseException);
}

BOOST_AUTO_TEST_SUITE_END()
