//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_ResizeBilinear")
{
struct ResizeBilinearFixture : public ParserFlatbuffersFixture
{
    explicit ResizeBilinearFixture(const std::string & inputShape,
                                   const std::string & outputShape,
                                   const std::string & sizeShape,
                                   const std::string & sizeData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "RESIZE_BILINEAR" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + sizeShape + R"( ,
                            "type": "INT32",
                            "buffer": 0,
                            "name": "sizeTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "FLOAT32",
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
                            "type": "FLOAT32",
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
                        "builtin_options_type": "ResizeBilinearOptions",
                        "builtin_options": {
                        },
                        "custom_options_format": "FLEXBUFFERS"
                    }
                ],
              } ],
              "buffers" : [
                  { "data": )" + sizeData + R"(, },
                  { },
                  { },
              ]
            }
      )";
      Setup();
    }
};


struct SimpleResizeBilinearFixture : ResizeBilinearFixture
{
    SimpleResizeBilinearFixture()
        : ResizeBilinearFixture("[ 1, 3, 3, 1 ]",         // inputShape
                                "[ 1, 5, 5, 1 ]",         // outputShape
                                "[ 2 ]",                  // sizeShape
                                "[  5,0,0,0, 5,0,0,0 ]")  // sizeData
    {}
};

TEST_CASE_FIXTURE(SimpleResizeBilinearFixture, "ParseResizeBilinear")
{
    RunTest<4, armnn::DataType::Float32>(
                0,
                {{"InputTensor", { 0.0f, 1.0f, 2.0f,
                                   3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f }}},
                {{"OutputTensor", { 0.0f, 0.6f, 1.2f, 1.8f, 2.0f,
                                    1.8f, 2.4f, 3.0f, 3.6f, 3.8f,
                                    3.6f, 4.2f, 4.8f, 5.4f, 5.6f,
                                    5.4f, 6.0f, 6.6f, 7.2f, 7.4f,
                                    6.0f, 6.6f, 7.2f, 7.8f, 8.0f }}}
                );
}

}
