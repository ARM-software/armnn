//
// Copyright © 2023, 2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

#include <doctest/doctest.h>


TEST_SUITE("TensorflowLiteParser_SquaredDifference")
{
    struct SquaredDifferenceFixture : public ParserFlatbuffersFixture
    {
        explicit SquaredDifferenceFixture(const std::string & inputShape1,
                                          const std::string & inputShape2,
                                          const std::string & outputShape)
        {
            m_JsonString = R"(
                {
                    "version": 3,
                    "operator_codes": [ { "builtin_code": "SQUARED_DIFFERENCE" } ],
                    "subgraphs": [ {
                        "tensors": [
                            {
                                "shape": )" + inputShape1 + R"(,
                                "type": "UINT8",
                                "buffer": 0,
                                "name": "inputTensor1",
                                "quantization": {
                                    "min": [ 0.0 ],
                                    "max": [ 255.0 ],
                                    "scale": [ 1.0 ],
                                    "zero_point": [ 0 ],
                                }
                            },
                            {
                                "shape": )" + inputShape2 + R"(,
                                "type": "UINT8",
                                "buffer": 1,
                                "name": "inputTensor2",
                                "quantization": {
                                    "min": [ 0.0 ],
                                    "max": [ 255.0 ],
                                    "scale": [ 1.0 ],
                                    "zero_point": [ 0 ],
                                }
                            },
                            {
                                "shape": )" + outputShape + R"( ,
                                "type": "UINT8",
                                "buffer": 2,
                                "name": "outputTensor",
                                "quantization": {
                                    "min": [ 0.0 ],
                                    "max": [ 255.0 ],
                                    "scale": [ 1.0 ],
                                    "zero_point": [ 0 ],
                                }
                            }
                        ],
                        "inputs": [ 0, 1 ],
                        "outputs": [ 2 ],
                        "operators": [
                            {
                                "opcode_index": 0,
                                "inputs": [ 0, 1 ],
                                "outputs": [ 2 ],
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


    struct SimpleSquaredDifferenceFixture : SquaredDifferenceFixture
    {
        SimpleSquaredDifferenceFixture() : SquaredDifferenceFixture("[ 2, 2 ]",
                                                                    "[ 2, 2 ]",
                                                                    "[ 2, 2 ]") {}
    };

    TEST_CASE_FIXTURE(SimpleSquaredDifferenceFixture, "SimpleSquaredDifference")
    {
        RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            {{"inputTensor1", { 4, 1, 8, 9 }},
            {"inputTensor2", { 0, 5, 6, 3 }}},
            {{"outputTensor", { 16, 16, 4, 36 }}});
    }

}