//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_ElementwiseUnary")
{
struct ElementWiseUnaryFixture : public ParserFlatbuffersFixture
{
    explicit ElementWiseUnaryFixture(const std::string& operatorCode,
                                     const std::string& dataType,
                                     const std::string& inputShape,
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
                            "shape": )" + outputShape + R"( ,
                            "type": )" + dataType + R"( ,
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

struct SimpleAbsFixture : public ElementWiseUnaryFixture
{
    SimpleAbsFixture() : ElementWiseUnaryFixture("ABS", "FLOAT32", "[ 2, 2 ]", "[ 2, 2 ]") {}
};

TEST_CASE_FIXTURE(SimpleAbsFixture, "ParseAbs")
{
    std::vector<float> inputValues
    {
        -0.1f, 0.2f,
        0.3f, -0.4f
    };

    // Calculate output data
    std::vector<float> expectedOutputValues(inputValues.size());
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        expectedOutputValues[i] = std::abs(inputValues[i]);
    }

    RunTest<2, armnn::DataType::Float32>(0, {{ "inputTensor", { inputValues } }},
                                            {{ "outputTensor",{ expectedOutputValues } } });
}

struct SimpleExpFixture : public ElementWiseUnaryFixture
{
    SimpleExpFixture() : ElementWiseUnaryFixture("EXP", "FLOAT32", "[ 1, 2, 3, 1 ]", "[ 1, 2, 3, 1 ]") {}
};

TEST_CASE_FIXTURE(SimpleExpFixture, "ParseExp")
{
    RunTest<4, armnn::DataType::Float32>(0, {{ "inputTensor", { 0.0f,  1.0f,  2.0f,
                                                                3.0f,  4.0f,  5.0f} }},
                                            {{ "outputTensor",{ 1.0f,  2.718281f,  7.3890515f,
                                                                20.0855185f, 54.5980834f, 148.4129329f} } });
}

struct SimpleLogicalNotFixture : public ElementWiseUnaryFixture
{
    SimpleLogicalNotFixture() : ElementWiseUnaryFixture("LOGICAL_NOT", "BOOL", "[ 1, 1, 1, 4 ]", "[ 1, 1, 1, 4 ]") {}
};

TEST_CASE_FIXTURE(SimpleLogicalNotFixture, "ParseLogicalNot")
{
    RunTest<4, armnn::DataType::Boolean>(0, {{ "inputTensor", { 0, 1, 0, 1 } }},
                                            {{ "outputTensor",{ 1, 0, 1, 0 } } });
}

struct SimpleNegFixture : public ElementWiseUnaryFixture
{
    SimpleNegFixture() : ElementWiseUnaryFixture("NEG", "FLOAT32", "[ 1, 2, 3, 1 ]", "[ 1, 2, 3, 1 ]") {}
};

TEST_CASE_FIXTURE(SimpleNegFixture, "ParseNeg")
{
    RunTest<4, armnn::DataType::Float32>(0, {{ "inputTensor", { 0.0f, 1.0f, -2.0f,
                                                                20.0855185f, -54.5980834f, 5.0f} }},
                                            {{ "outputTensor",{ 0.0f, -1.0f, 2.0f,
                                                                -20.0855185f, 54.5980834f, -5.0f} }});
}

struct SimpleRsqrtFixture : public ElementWiseUnaryFixture
{
    SimpleRsqrtFixture() : ElementWiseUnaryFixture("RSQRT", "FLOAT32", "[ 1, 2, 3, 1 ]", "[ 1, 2, 3, 1 ]") {}
};

TEST_CASE_FIXTURE(SimpleRsqrtFixture, "ParseRsqrt")
{
    RunTest<4, armnn::DataType::Float32>(0, {{ "inputTensor", { 1.0f, 4.0f, 16.0f,
                                                                25.0f, 64.0f, 100.0f } }},
                                            {{ "outputTensor",{ 1.0f, 0.5f, 0.25f,
                                                                0.2f, 0.125f, 0.1f} }});
}

}
