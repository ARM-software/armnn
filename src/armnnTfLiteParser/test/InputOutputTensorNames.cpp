//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

TEST_SUITE("TensorflowLiteParser_InputOutputTensorNames")
{
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

TEST_CASE_FIXTURE(EmptyNetworkFixture, "EmptyNetworkHasNoInputsAndOutputs")
{
    Setup(false);
    CHECK(m_Parser->GetSubgraphCount() == 1);
    CHECK(m_Parser->GetSubgraphInputTensorNames(0).size() == 0);
    CHECK(m_Parser->GetSubgraphOutputTensorNames(0).size() == 0);
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

TEST_CASE_FIXTURE(MissingTensorsFixture, "MissingTensorsThrowException")
{
    // this throws because it cannot do the input output tensor connections
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
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

TEST_CASE_FIXTURE(InvalidTensorsFixture, "InvalidTensorsThrowException")
{
    // Tensor numDimensions must be less than or equal to MaxNumOfTensorDimensions
    static_assert(armnn::MaxNumOfTensorDimensions == 5, "Please update InvalidTensorsFixture");
    CHECK_THROWS_AS(Setup(), armnn::InvalidArgumentException);
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

TEST_CASE_FIXTURE(ValidTensorsFixture, "GetValidInputOutputTensorNames")
{
    Setup();
    CHECK_EQ(m_Parser->GetSubgraphInputTensorNames(0).size(), 1u);
    CHECK_EQ(m_Parser->GetSubgraphOutputTensorNames(0).size(), 1u);
    CHECK_EQ(m_Parser->GetSubgraphInputTensorNames(0)[0], "In");
    CHECK_EQ(m_Parser->GetSubgraphOutputTensorNames(0)[0], "Out");
}

TEST_CASE_FIXTURE(ValidTensorsFixture, "ThrowIfSubgraphIdInvalidForInOutNames")
{
    Setup();

    // these throw because of the invalid subgraph id
    CHECK_THROWS_AS(m_Parser->GetSubgraphInputTensorNames(1), armnn::ParseException);
    CHECK_THROWS_AS(m_Parser->GetSubgraphOutputTensorNames(1), armnn::ParseException);
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

TEST_CASE_FIXTURE(Rank0TensorFixture, "Rank0Tensor")
{
    Setup();
    CHECK_EQ(m_Parser->GetSubgraphInputTensorNames(0).size(), 2u);
    CHECK_EQ(m_Parser->GetSubgraphOutputTensorNames(0).size(), 1u);
    CHECK_EQ(m_Parser->GetSubgraphInputTensorNames(0)[0], "In0");
    CHECK_EQ(m_Parser->GetSubgraphInputTensorNames(0)[1], "In1");
    CHECK_EQ(m_Parser->GetSubgraphOutputTensorNames(0)[0], "Out");
}

}
