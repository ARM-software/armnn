//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

#include <doctest/doctest.h>


TEST_SUITE("TensorflowLiteParser_Power")
{
    struct PowerFixture : public ParserFlatbuffersFixture
    {
        explicit PowerFixture(const std::string & inputShape1,
                              const std::string & inputShape2,
                              const std::string & outputShape,
                              const std::string & activation="NONE")
        {
            m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "POW" } ],
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


    struct SimplePowerFixture : PowerFixture
    {
        SimplePowerFixture() : PowerFixture("[ 2, 2 ]",
                                            "[ 2, 2 ]",
                                            "[ 2, 2 ]") {}
    };

    TEST_CASE_FIXTURE(SimplePowerFixture, "SimplePower")
    {
        RunTest<2, armnn::DataType::QAsymmU8>(
                0,
                {{"inputTensor1", { 0, 1, 2, 3 }},
                 {"inputTensor2", { 4, 5, 6, 3 }}},
                {{"outputTensor", { 0, 1, 64, 27 }}});
    }

}