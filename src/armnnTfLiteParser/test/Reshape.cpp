//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Reshape")
{
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

TEST_CASE_FIXTURE(ReshapeFixtureWithReshapeDims, "ParseReshapeWithReshapeDims")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<2, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                                                 { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({3,3})));
}

struct ReshapeFixtureWithReshapeDimsFlatten : ReshapeFixture
{
    ReshapeFixtureWithReshapeDimsFlatten() : ReshapeFixture("[ 3, 3 ]", "[ 9 ]", "[ -1 ]") {}
};

TEST_CASE_FIXTURE(ReshapeFixtureWithReshapeDimsFlatten, "ParseReshapeWithReshapeDimsFlatten")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<1, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                                                 { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({9})));
}

struct ReshapeFixtureWithReshapeDimsFlattenTwoDims : ReshapeFixture
{
    ReshapeFixtureWithReshapeDimsFlattenTwoDims() : ReshapeFixture("[ 3, 2, 3 ]", "[ 2, 9 ]", "[ 2, -1 ]") {}
};

TEST_CASE_FIXTURE(ReshapeFixtureWithReshapeDimsFlattenTwoDims, "ParseReshapeWithReshapeDimsFlattenTwoDims")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<2, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 },
                                                 { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 });
    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({2,9})));
}

struct ReshapeFixtureWithReshapeDimsFlattenOneDim : ReshapeFixture
{
    ReshapeFixtureWithReshapeDimsFlattenOneDim() : ReshapeFixture("[ 2, 9 ]", "[ 2, 3, 3 ]", "[ 2, -1, 3 ]") {}
};

TEST_CASE_FIXTURE(ReshapeFixtureWithReshapeDimsFlattenOneDim, "ParseReshapeWithReshapeDimsFlattenOneDim")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<3, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 },
                                                 { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 });
    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({2,3,3})));
}

struct DynamicReshapeFixtureWithReshapeDimsFlattenOneDim : ReshapeFixture
{
    DynamicReshapeFixtureWithReshapeDimsFlattenOneDim() : ReshapeFixture("[ 2, 9 ]",
                                                                         "[ ]",
                                                                         "[ 2, -1, 3 ]") {}
};

TEST_CASE_FIXTURE(DynamicReshapeFixtureWithReshapeDimsFlattenOneDim, "DynParseReshapeWithReshapeDimsFlattenOneDim")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
     RunTest<3,
        armnn::DataType::QAsymmU8,
        armnn::DataType::QAsymmU8>(0,
                                   { { "inputTensor", {  1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 } } },
                                   { { "outputTensor", { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 } } },
                                   true);
}

}
