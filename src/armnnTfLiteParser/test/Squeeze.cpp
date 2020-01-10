//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct SqueezeFixture : public ParserFlatbuffersFixture
{
    explicit SqueezeFixture(const std::string& inputShape,
                            const std::string& outputShape,
                            const std::string& squeezeDims)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SQUEEZE" } ],
                "subgraphs": [ {
                    "tensors": [
                        {)";
        m_JsonString += R"(
                            "shape" : )" + inputShape + ",";
        m_JsonString += R"(
                            "type": "UINT8",
                            "buffer": 0,
                            "name": "inputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {)";
        m_JsonString += R"(
                            "shape" : )" + outputShape;
        m_JsonString += R"(,
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "outputTensor",
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
                            "inputs": [ 0 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "SqueezeOptions",
                            "builtin_options": {)";
        if (!squeezeDims.empty())
        {
            m_JsonString += R"("squeeze_dims" : )" + squeezeDims;
        }
        m_JsonString += R"(},
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [ {}, {} ]
            }
        )";
    }
};

struct SqueezeFixtureWithSqueezeDims : SqueezeFixture
{
    SqueezeFixtureWithSqueezeDims() : SqueezeFixture("[ 1, 2, 2, 1 ]", "[ 2, 2, 1 ]", "[ 0, 1, 2 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseSqueezeWithSqueezeDims, SqueezeFixtureWithSqueezeDims)
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<3, armnn::DataType::QAsymmU8>(0, { 1, 2, 3, 4 }, { 1, 2, 3, 4 });
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
        == armnn::TensorShape({2,2,1})));

}

struct SqueezeFixtureWithoutSqueezeDims : SqueezeFixture
{
    SqueezeFixtureWithoutSqueezeDims() : SqueezeFixture("[ 1, 2, 2, 1 ]", "[ 2, 2 ]", "") {}
};

BOOST_FIXTURE_TEST_CASE(ParseSqueezeWithoutSqueezeDims, SqueezeFixtureWithoutSqueezeDims)
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<2, armnn::DataType::QAsymmU8>(0, { 1, 2, 3, 4 }, { 1, 2, 3, 4 });
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
        == armnn::TensorShape({2,2})));
}

struct SqueezeFixtureWithInvalidInput : SqueezeFixture
{
    SqueezeFixtureWithInvalidInput() : SqueezeFixture("[ 1, 2, 2, 1, 2, 2 ]", "[ 1, 2, 2, 1, 2 ]", "[ ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseSqueezeInvalidInput, SqueezeFixtureWithInvalidInput)
{
    static_assert(armnn::MaxNumOfTensorDimensions == 5, "Please update SqueezeFixtureWithInvalidInput");
    BOOST_CHECK_THROW((SetupSingleInputSingleOutput("inputTensor", "outputTensor")),
                      armnn::InvalidArgumentException);
}

struct SqueezeFixtureWithSqueezeDimsSizeInvalid : SqueezeFixture
{
    SqueezeFixtureWithSqueezeDimsSizeInvalid() : SqueezeFixture("[ 1, 2, 2, 1 ]",
                                                                "[ 1, 2, 2, 1 ]",
                                                                "[ 1, 2, 2, 2, 2 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseSqueezeInvalidSqueezeDims, SqueezeFixtureWithSqueezeDimsSizeInvalid)
{
    BOOST_CHECK_THROW((SetupSingleInputSingleOutput("inputTensor", "outputTensor")), armnn::ParseException);
}


struct SqueezeFixtureWithNegativeSqueezeDims : SqueezeFixture
{
    SqueezeFixtureWithNegativeSqueezeDims() : SqueezeFixture("[ 1, 2, 2, 1 ]",
                                                             "[ 1, 2, 2, 1 ]",
                                                             "[ -2 , 2 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseSqueezeNegativeSqueezeDims, SqueezeFixtureWithNegativeSqueezeDims)
{
    BOOST_CHECK_THROW((SetupSingleInputSingleOutput("inputTensor", "outputTensor")), armnn::ParseException);
}


BOOST_AUTO_TEST_SUITE_END()
