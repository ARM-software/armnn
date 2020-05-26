//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct NegFixture : public ParserFlatbuffersFixture
{
    explicit NegFixture(const std::string & inputShape,
                        const std::string & outputShape)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "NEG" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "FLOAT32",
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
                            "type": "FLOAT32",
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

struct SimpleNegFixture : public NegFixture
{
    SimpleNegFixture() : NegFixture("[ 1, 2, 3, 1 ]", "[ 1, 2, 3, 1 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseNeg, SimpleNegFixture)
{
    using armnn::DataType;
    RunTest<4, DataType::Float32>(0, {{ "inputTensor", { 0.0f, 1.0f, -2.0f,
                                                         20.0855185f, -54.5980834f, 5.0f} }},
                                     {{ "outputTensor",{ 0.0f, -1.0f, 2.0f,
                                                         -20.0855185f, 54.5980834f, -5.0f} }});
}

BOOST_AUTO_TEST_SUITE_END()
