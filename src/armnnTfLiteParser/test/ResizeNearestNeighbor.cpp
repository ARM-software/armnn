//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_ResizeNearestNeighbor")
{
struct ResizeNearestNeighborFixture : public ParserFlatbuffersFixture
{
    explicit ResizeNearestNeighborFixture(const std::string & inputShape,
                                          const std::string & outputShape,
                                          const std::string & sizeShape,
                                          const std::string & sizeData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "RESIZE_NEAREST_NEIGHBOR" } ],
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
                        "builtin_options_type": "ResizeNearestNeighborOptions",
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


struct SimpleResizeNearestNeighborFixture : ResizeNearestNeighborFixture
{
    SimpleResizeNearestNeighborFixture()
        : ResizeNearestNeighborFixture("[ 1, 2, 2, 1 ]",         // inputShape
                                       "[ 1, 1, 1, 1 ]",         // outputShape
                                       "[ 2 ]",                  // sizeShape
                                       "[ 1,0,0,0, 1,0,0,0 ]")   // sizeData
    {}
};

TEST_CASE_FIXTURE(SimpleResizeNearestNeighborFixture, "ParseResizeNearestNeighbor")
{
    RunTest<4, armnn::DataType::Float32>(
                0,
                {{"InputTensor", {  1.0f, 2.0f, 3.0f, 4.0f }}},
                {{"OutputTensor", {  1.0f }}});
}

}
