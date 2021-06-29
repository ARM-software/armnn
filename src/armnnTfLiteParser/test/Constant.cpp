//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


using armnnTfLiteParser::TfLiteParserImpl;

TEST_SUITE("TensorflowLiteParser_Constant")
{
struct ConstantAddFixture : public ParserFlatbuffersFixture
{
    explicit ConstantAddFixture(const std::string & inputShape,
                                const std::string & outputShape,
                                const std::string & constShape,
                                const std::string & constData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "ADD" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + constShape + R"( ,
                            "type": "UINT8",
                            "buffer": 3,
                            "name": "ConstTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "InputTensor",
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
                            "name": "OutputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                "inputs": [ 1 ],
                "outputs": [ 2 ],
                "operators": [
                    {
                        "opcode_index": 0,
                        "inputs": [ 1, 0 ],
                        "outputs": [ 2 ],
                        "builtin_options_type": "AddOptions",
                        "builtin_options": {
                        },
                        "custom_options_format": "FLEXBUFFERS"
                    }
                ],
              } ],
              "buffers" : [
                  { },
                  { },
                  { },
                  { "data": )" + constData + R"(, },
              ]
            }
      )";
      Setup();
    }
};


struct SimpleConstantAddFixture : ConstantAddFixture
{
    SimpleConstantAddFixture()
        : ConstantAddFixture("[ 2, 2 ]",        // inputShape
                             "[ 2, 2 ]",        // outputShape
                             "[ 2, 2 ]",        // constShape
                             "[  4,5, 6,7 ]")   // constData
    {}
};

TEST_CASE_FIXTURE(SimpleConstantAddFixture, "SimpleConstantAdd")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
                0,
                {{"InputTensor", { 0, 1, 2, 3 }}},
                {{"OutputTensor", { 4, 6, 8, 10 }}}
                );
}

}
