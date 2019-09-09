//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct TransposeFixture : public ParserFlatbuffersFixture
{
    explicit TransposeFixture(const std::string & inputShape,
                              const std::string & outputShape)
    {
        m_JsonString = R"(
            {
                  "version": 3,
                  "operator_codes": [
                    {
                      "builtin_code": "TRANSPOSE",
                      "version": 1
                    }
                  ],
                  "subgraphs": [
                    {
                      "tensors": [
                        {
                          "shape": )" + inputShape + R"(,
                          "type": "FLOAT32",
                          "buffer": 3,
                          "name": "Placeholder",
                          "quantization": {
                            "min": [
                              0.0
                            ],
                            "max": [
                              255.0
                            ],
                            "details_type": 0,
                            "quantized_dimension": 0
                          },
                          "is_variable": false
                        },
                        {
                          "shape": )" + outputShape + R"(,
                          "type": "FLOAT32",
                          "buffer": 2,
                          "name": "transpose",
                          "quantization": {
                            "details_type": 0,
                            "quantized_dimension": 0
                          },
                          "is_variable": false
                        },
                        {
                          "shape": [
                            3
                          ],
                          "type": "INT32",
                          "buffer": 1,
                          "name": "transpose/perm",
                          "quantization": {
                            "details_type": 0,
                            "quantized_dimension": 0
                          },
                          "is_variable": false
                        }
                      ],
                      "inputs": [
                        0
                      ],
                      "outputs": [
                        1
                      ],
                      "operators": [
                        {
                          "opcode_index": 0,
                          "inputs": [
                            0,
                            2
                          ],
                          "outputs": [
                            1
                          ],
                          "builtin_options_type": "TransposeOptions",
                          "builtin_options": {
                          },
                          "custom_options_format": "FLEXBUFFERS"
                        }
                      ]
                    }
                  ],
                  "description": "TOCO Converted.",
                  "buffers": [
                    { },
                    { },
                    { },
                    { }
                  ]
                }
        )";
        Setup();
    }
};

struct SimpleTransposeFixture : TransposeFixture
{
    SimpleTransposeFixture() : TransposeFixture("[ 2, 2, 3 ]",
                                                "[ 2, 3, 2 ]") {}
};

BOOST_FIXTURE_TEST_CASE(SimpleTranspose, SimpleTransposeFixture)
{
    RunTest<3, armnn::DataType::Float32>(
      0,
      {{"Placeholder", {  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }}},

      {{"transpose", {  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }}});
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo(0, "transpose").second.GetShape()
                == armnn::TensorShape({2,3,2})));
}

BOOST_AUTO_TEST_SUITE_END()