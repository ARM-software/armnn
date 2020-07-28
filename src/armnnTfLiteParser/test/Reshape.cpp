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

struct ReshapeFixture : public ParserFlatbuffersFixture
{
    explicit ReshapeFixture(const std::string& inputShape,
                            const std::string& outputShape,
                            const std::string& newShape)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "RESHAPE" } ],
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
                            "builtin_options_type": "ReshapeOptions",
                            "builtin_options": {)";
        if (!newShape.empty())
        {
            m_JsonString += R"("new_shape" : )" + newShape;
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

struct ReshapeFixtureWithReshapeDims : ReshapeFixture
{
    ReshapeFixtureWithReshapeDims() : ReshapeFixture("[ 1, 9 ]", "[ 3, 3 ]", "[ 3, 3 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseReshapeWithReshapeDims, ReshapeFixtureWithReshapeDims)
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<2, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                                                 { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({3,3})));
}

struct ReshapeFixtureWithReshapeDimsFlatten : ReshapeFixture
{
    ReshapeFixtureWithReshapeDimsFlatten() : ReshapeFixture("[ 3, 3 ]", "[ 9 ]", "[ -1 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseReshapeWithReshapeDimsFlatten, ReshapeFixtureWithReshapeDimsFlatten)
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<1, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                                                 { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({9})));
}

struct ReshapeFixtureWithReshapeDimsFlattenTwoDims : ReshapeFixture
{
    ReshapeFixtureWithReshapeDimsFlattenTwoDims() : ReshapeFixture("[ 3, 2, 3 ]", "[ 2, 9 ]", "[ 2, -1 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseReshapeWithReshapeDimsFlattenTwoDims, ReshapeFixtureWithReshapeDimsFlattenTwoDims)
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<2, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 },
                                                 { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 });
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({2,9})));
}

struct ReshapeFixtureWithReshapeDimsFlattenOneDim : ReshapeFixture
{
    ReshapeFixtureWithReshapeDimsFlattenOneDim() : ReshapeFixture("[ 2, 9 ]", "[ 2, 3, 3 ]", "[ 2, -1, 3 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseReshapeWithReshapeDimsFlattenOneDim, ReshapeFixtureWithReshapeDimsFlattenOneDim)
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<3, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 },
                                                 { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 });
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({2,3,3})));
}

struct DynamicReshapeFixtureWithReshapeDimsFlattenOneDim : ReshapeFixture
{
    DynamicReshapeFixtureWithReshapeDimsFlattenOneDim() : ReshapeFixture("[ 2, 9 ]",
                                                                         "[ ]",
                                                                         "[ 2, -1, 3 ]") {}
};

BOOST_FIXTURE_TEST_CASE(DynParseReshapeWithReshapeDimsFlattenOneDim, DynamicReshapeFixtureWithReshapeDimsFlattenOneDim)
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
     RunTest<3,
        armnn::DataType::QAsymmU8,
        armnn::DataType::QAsymmU8>(0,
                                   { { "inputTensor", {  1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 } } },
                                   { { "outputTensor", { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 } } },
                                   true);
}

BOOST_AUTO_TEST_SUITE_END()
