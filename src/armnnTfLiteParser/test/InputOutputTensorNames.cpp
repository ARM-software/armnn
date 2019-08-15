//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct EmptyNetworkFixture : public ParserFlatbuffersFixture
{
    explicit EmptyNetworkFixture() {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [],
                "subgraphs": [ {} ]
            })";
    }
};

BOOST_FIXTURE_TEST_CASE(EmptyNetworkHasNoInputsAndOutputs, EmptyNetworkFixture)
{
    Setup();
    BOOST_TEST(m_Parser->GetSubgraphCount() == 1);
    BOOST_TEST(m_Parser->GetSubgraphInputTensorNames(0).size() == 0);
    BOOST_TEST(m_Parser->GetSubgraphOutputTensorNames(0).size() == 0);
}

struct MissingTensorsFixture : public ParserFlatbuffersFixture
{
    explicit MissingTensorsFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [],
                "subgraphs": [{
                    "inputs" : [ 0, 1 ],
                    "outputs" : [ 2, 3 ],
                }]
            })";
    }
};

BOOST_FIXTURE_TEST_CASE(MissingTensorsThrowException, MissingTensorsFixture)
{
    // this throws because it cannot do the input output tensor connections
    BOOST_CHECK_THROW(Setup(), armnn::ParseException);
}

struct InvalidTensorsFixture : public ParserFlatbuffersFixture
{
    explicit InvalidTensorsFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ ],
                "subgraphs": [{
                    "tensors": [ {
                        "shape": [ 1, 1, 1, 1, 1, 1 ],
                        "type": "FLOAT32",
                        "name": "In",
                        "buffer": 0
                    }, {
                        "shape": [ 1, 1, 1, 1, 1, 1 ],
                        "type": "FLOAT32",
                        "name": "Out",
                        "buffer": 1
                    }],
                    "inputs" : [ 0 ],
                    "outputs" : [ 1 ],
                }]
            })";
    }
};

BOOST_FIXTURE_TEST_CASE(InvalidTensorsThrowException, InvalidTensorsFixture)
{
    // Tensor numDimensions must be less than or equal to MaxNumOfTensorDimensions
    static_assert(armnn::MaxNumOfTensorDimensions == 5, "Please update InvalidTensorsFixture");
    BOOST_CHECK_THROW(Setup(), armnn::InvalidArgumentException);
}

struct ValidTensorsFixture : public ParserFlatbuffersFixture
{
    explicit ValidTensorsFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "AVERAGE_POOL_2D" } ],
                "subgraphs": [{
                    "tensors": [ {
                        "shape": [ 1, 1, 1, 1 ],
                        "type": "FLOAT32",
                        "name": "In",
                        "buffer": 0,
                    }, {
                        "shape": [ 1, 1, 1, 1 ],
                        "type": "FLOAT32",
                        "name": "Out",
                        "buffer": 1,
                    }],
                    "inputs" : [ 0 ],
                    "outputs" : [ 1 ],
                    "operators": [{
                        "opcode_index": 0,
                        "inputs": [ 0 ],
                        "outputs": [ 1 ],
                        "builtin_options_type": "Pool2DOptions",
                        "builtin_options":
                        {
                            "padding": "VALID",
                            "stride_w": 1,
                            "stride_h": 1,
                            "filter_width": 1,
                            "filter_height": 1,
                            "fused_activation_function": "NONE"
                        },
                        "custom_options_format": "FLEXBUFFERS"
                    }]
                }]
            })";
    }
};

BOOST_FIXTURE_TEST_CASE(GetValidInputOutputTensorNames, ValidTensorsFixture)
{
    Setup();
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphInputTensorNames(0).size(), 1u);
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphOutputTensorNames(0).size(), 1u);
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphInputTensorNames(0)[0], "In");
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphOutputTensorNames(0)[0], "Out");
}

BOOST_FIXTURE_TEST_CASE(ThrowIfSubgraphIdInvalidForInOutNames, ValidTensorsFixture)
{
    Setup();

    // these throw because of the invalid subgraph id
    BOOST_CHECK_THROW(m_Parser->GetSubgraphInputTensorNames(1), armnn::ParseException);
    BOOST_CHECK_THROW(m_Parser->GetSubgraphOutputTensorNames(1), armnn::ParseException);
}

struct Rank0TensorFixture : public ParserFlatbuffersFixture
{
    explicit Rank0TensorFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "MINIMUM" } ],
                "subgraphs": [{
                    "tensors": [ {
                        "shape": [  ],
                        "type": "FLOAT32",
                        "name": "In0",
                        "buffer": 0,
                    }, {
                        "shape": [  ],
                        "type": "FLOAT32",
                        "name": "In1",
                        "buffer": 1,
                    }, {
                        "shape": [ ],
                        "type": "FLOAT32",
                        "name": "Out",
                        "buffer": 2,
                    }],
                    "inputs" : [ 0, 1 ],
                    "outputs" : [ 2 ],
                    "operators": [{
                        "opcode_index": 0,
                        "inputs": [ 0, 1 ],
                        "outputs": [ 2 ],
                        "custom_options_format": "FLEXBUFFERS"
                    }]
                }]
            }
        )";
    }
};

BOOST_FIXTURE_TEST_CASE(Rank0Tensor, Rank0TensorFixture)
{
    Setup();
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphInputTensorNames(0).size(), 2u);
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphOutputTensorNames(0).size(), 1u);
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphInputTensorNames(0)[0], "In0");
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphInputTensorNames(0)[1], "In1");
    BOOST_CHECK_EQUAL(m_Parser->GetSubgraphOutputTensorNames(0)[0], "Out");
}

BOOST_AUTO_TEST_SUITE_END()
