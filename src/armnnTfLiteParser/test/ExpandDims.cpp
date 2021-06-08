//
// Copyright © 2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>
#include <iostream>

TEST_SUITE("TensorflowLiteParser_ExpandDims")
{
struct ExpandDimsFixture : public ParserFlatbuffersFixture
{
    explicit ExpandDimsFixture(const std::string& inputShape,
                               const std::string& outputShape,
                               const std::string& axis)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "EXPAND_DIMS" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
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
                        {
                            "shape": )" + outputShape + R"( ,
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ 1 ],
                            "type": "UINT8",
                            "buffer": 2,
                            "name": "expand_dims",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0 , 2 ],
                            "outputs": [ 1 ],
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + axis + R"(, },
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct ExpandDimsFixture3dto4Daxis0 : ExpandDimsFixture
{
    ExpandDimsFixture3dto4Daxis0() : ExpandDimsFixture("[ 2, 2, 1 ]", "[ 1, 2, 2, 1 ]", "[ 0, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(ExpandDimsFixture3dto4Daxis0, "ParseExpandDims3Dto4Daxis0")
{
    RunTest<4, armnn::DataType::QAsymmU8>(0, {{ "inputTensor",  { 1, 2, 3, 4 } } },
                                             {{ "outputTensor", { 1, 2, 3, 4 } } });
}

struct ExpandDimsFixture3dto4Daxis3 : ExpandDimsFixture
{
    ExpandDimsFixture3dto4Daxis3() : ExpandDimsFixture("[ 1, 2, 2 ]", "[ 1, 2, 2, 1 ]", "[ 3, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(ExpandDimsFixture3dto4Daxis3, "ParseExpandDims3Dto4Daxis3")
{
    RunTest<4, armnn::DataType::QAsymmU8>(0, {{ "inputTensor",  { 1, 2, 3, 4 } } },
                                             {{ "outputTensor", { 1, 2, 3, 4 } } });
}

}