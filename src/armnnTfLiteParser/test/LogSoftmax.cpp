//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_LogSoftmax")
{
struct LogSoftmaxFixture : public ParserFlatbuffersFixture
{
    explicit LogSoftmaxFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "LOG_SOFTMAX" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 7 ],
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
                            "shape": [ 1, 7 ],
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 0.00390625 ],
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
                          "builtin_options_type": "LogSoftmaxOptions",
                          "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [ {}, {} ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

TEST_CASE_FIXTURE(LogSoftmaxFixture, "ParseLogSoftmaxLite")
{
    RunTest<2, armnn::DataType::QAsymmU8>(0, { 0, 0, 100, 0, 0, 0, 0 }, { 0, 0, 255, 0, 0, 0, 0 });
}

}
