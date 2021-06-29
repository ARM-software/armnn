//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Squeeze")
{
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

TEST_CASE_FIXTURE(SqueezeFixtureWithSqueezeDims, "ParseSqueezeWithSqueezeDims")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<3, armnn::DataType::QAsymmU8>(0, { 1, 2, 3, 4 }, { 1, 2, 3, 4 });
    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
        == armnn::TensorShape({2,2,1})));

}

struct SqueezeFixtureWithoutSqueezeDims : SqueezeFixture
{
    SqueezeFixtureWithoutSqueezeDims() : SqueezeFixture("[ 1, 2, 2, 1 ]", "[ 2, 2 ]", "") {}
};

TEST_CASE_FIXTURE(SqueezeFixtureWithoutSqueezeDims, "ParseSqueezeWithoutSqueezeDims")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<2, armnn::DataType::QAsymmU8>(0, { 1, 2, 3, 4 }, { 1, 2, 3, 4 });
    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
        == armnn::TensorShape({2,2})));
}

struct SqueezeFixtureWithInvalidInput : SqueezeFixture
{
    SqueezeFixtureWithInvalidInput() : SqueezeFixture("[ 1, 2, 2, 1, 2, 2 ]", "[ 1, 2, 2, 1, 2 ]", "[ ]") {}
};

TEST_CASE_FIXTURE(SqueezeFixtureWithInvalidInput, "ParseSqueezeInvalidInput")
{
    static_assert(armnn::MaxNumOfTensorDimensions == 5, "Please update SqueezeFixtureWithInvalidInput");
    CHECK_THROWS_AS((SetupSingleInputSingleOutput("inputTensor", "outputTensor")),
                      armnn::InvalidArgumentException);
}

struct SqueezeFixtureWithSqueezeDimsSizeInvalid : SqueezeFixture
{
    SqueezeFixtureWithSqueezeDimsSizeInvalid() : SqueezeFixture("[ 1, 2, 2, 1 ]",
                                                                "[ 1, 2, 2, 1 ]",
                                                                "[ 1, 2, 2, 2, 2 ]") {}
};

TEST_CASE_FIXTURE(SqueezeFixtureWithSqueezeDimsSizeInvalid, "ParseSqueezeInvalidSqueezeDims")
{
    CHECK_THROWS_AS((SetupSingleInputSingleOutput("inputTensor", "outputTensor")), armnn::ParseException);
}


struct SqueezeFixtureWithNegativeSqueezeDims1 : SqueezeFixture
{
    SqueezeFixtureWithNegativeSqueezeDims1() : SqueezeFixture("[ 1, 2, 2, 1 ]",
                                                             "[ 2, 2, 1 ]",
                                                             "[ -1 ]") {}
};

TEST_CASE_FIXTURE(SqueezeFixtureWithNegativeSqueezeDims1, "ParseSqueezeNegativeSqueezeDims1")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<3, armnn::DataType::QAsymmU8>(0, { 1, 2, 3, 4 }, { 1, 2, 3, 4 });
            CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                   == armnn::TensorShape({ 2, 2, 1 })));
}

struct SqueezeFixtureWithNegativeSqueezeDims2 : SqueezeFixture
{
    SqueezeFixtureWithNegativeSqueezeDims2() : SqueezeFixture("[ 1, 2, 2, 1 ]",
                                                              "[ 1, 2, 2 ]",
                                                              "[ -1 ]") {}
};

TEST_CASE_FIXTURE(SqueezeFixtureWithNegativeSqueezeDims2, "ParseSqueezeNegativeSqueezeDims2")
{
    SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    RunTest<3, armnn::DataType::QAsymmU8>(0, { 1, 2, 3, 4 }, { 1, 2, 3, 4 });
            CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                   == armnn::TensorShape({ 1, 2, 2 })));
}

struct SqueezeFixtureWithNegativeSqueezeDimsInvalid : SqueezeFixture
{
    SqueezeFixtureWithNegativeSqueezeDimsInvalid() : SqueezeFixture("[ 1, 2, 2, 1 ]",
                                                                    "[ 1, 2, 2, 1 ]",
                                                                    "[ -2 , 2 ]") {}
};

TEST_CASE_FIXTURE(SqueezeFixtureWithNegativeSqueezeDimsInvalid, "ParseSqueezeNegativeSqueezeDimsInvalid")
{
    CHECK_THROWS_AS((SetupSingleInputSingleOutput("inputTensor", "outputTensor")), armnn::ParseException);
}


}
