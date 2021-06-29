//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_StridedSlice")
{
struct StridedSliceFixture : public ParserFlatbuffersFixture
{
    explicit StridedSliceFixture(const std::string & inputShape,
                                 const std::string & outputShape,
                                 const std::string & beginData,
                                 const std::string & endData,
                                 const std::string & stridesData,
                                 int beginMask = 0,
                                 int endMask = 0)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "STRIDED_SLICE" } ],
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
                            "shape": [ 4 ],
                            "type": "INT32",
                            "buffer": 1,
                            "name": "beginTensor",
                            "quantization": {
                            }
                        },
                        {
                           "shape": [ 4 ],
                            "type": "INT32",
                            "buffer": 2,
                            "name": "endTensor",
                            "quantization": {
                            }
                        },
                        {
                           "shape": [ 4 ],
                            "type": "INT32",
                            "buffer": 3,
                            "name": "stridesTensor",
                            "quantization": {
                            }
                        },
                        {
                            "shape": )" + outputShape + R"( ,
                            "type": "FLOAT32",
                            "buffer": 4,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0, 1, 2, 3 ],
                    "outputs": [ 4 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1, 2, 3 ],
                            "outputs": [ 4 ],
                            "builtin_options_type": "StridedSliceOptions",
                            "builtin_options": {
                               "begin_mask": )"       + std::to_string(beginMask)      + R"(,
                               "end_mask": )"         + std::to_string(endMask)        + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { "data": )" + beginData + R"(, },
                    { "data": )" + endData + R"(, },
                    { "data": )" + stridesData + R"(, },
                    { }
                ]
            }
        )";
        Setup();
    }
};

struct StridedSlice4DFixture : StridedSliceFixture
{
    StridedSlice4DFixture() : StridedSliceFixture("[ 3, 2, 3, 1 ]",  // inputShape
                                                  "[ 1, 2, 3, 1 ]",  // outputShape
                                                  "[ 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]",  // beginData
                                                  "[ 2,0,0,0, 2,0,0,0, 3,0,0,0, 1,0,0,0 ]",  // endData
                                                  "[ 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0 ]"   // stridesData
                                                 ) {}
};

TEST_CASE_FIXTURE(StridedSlice4DFixture, "StridedSlice4D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

                         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

                         5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}},

      {{"outputTensor", { 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f }}});
}

struct StridedSlice4DReverseFixture : StridedSliceFixture
{
    StridedSlice4DReverseFixture() : StridedSliceFixture("[ 3, 2, 3, 1 ]",    // inputShape
                                                         "[ 1, 2, 3, 1 ]",    // outputShape
                                                         "[ 1,0,0,0, "
                                                         "255,255,255,255, "
                                                         "0,0,0,0, "
                                                         "0,0,0,0 ]",  // beginData    [ 1 -1 0 0 ]
                                                         "[ 2,0,0,0, "
                                                         "253,255,255,255, "
                                                         "3,0,0,0, "
                                                         "1,0,0,0 ]",  // endData      [ 2 -3 3 1 ]
                                                         "[ 1,0,0,0, "
                                                         "255,255,255,255, "
                                                         "1,0,0,0, "
                                                         "1,0,0,0 ]"   // stridesData  [ 1 -1 1 1 ]
                                                        ) {}
};

TEST_CASE_FIXTURE(StridedSlice4DReverseFixture, "StridedSlice4DReverse")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

                         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

                         5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}},

      {{"outputTensor", { 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f }}});
}

struct StridedSliceSimpleStrideFixture : StridedSliceFixture
{
    StridedSliceSimpleStrideFixture() : StridedSliceFixture("[ 3, 2, 3, 1 ]",  // inputShape
                                                            "[ 2, 1, 2, 1 ]",  // outputShape
                                                            "[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]",  // beginData
                                                            "[ 3,0,0,0, 2,0,0,0, 3,0,0,0, 1,0,0,0 ]",  // endData
                                                            "[ 2,0,0,0, 2,0,0,0, 2,0,0,0, 1,0,0,0 ]"   // stridesData
                                                 ) {}
};

TEST_CASE_FIXTURE(StridedSliceSimpleStrideFixture, "StridedSliceSimpleStride")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

                         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

                         5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}},

      {{"outputTensor", { 1.0f, 1.0f,

                          5.0f, 5.0f }}});
}

struct StridedSliceSimpleRangeMaskFixture : StridedSliceFixture
{
    StridedSliceSimpleRangeMaskFixture() : StridedSliceFixture("[ 3, 2, 3, 1 ]",  // inputShape
                                                               "[ 3, 2, 3, 1 ]",  // outputShape
                                                               "[ 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0 ]",  // beginData
                                                               "[ 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0 ]",  // endData
                                                               "[ 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0 ]",  // stridesData
                                                               (1 << 4) - 1,  // beginMask
                                                               (1 << 4) - 1   // endMask
                                                 ) {}
};

TEST_CASE_FIXTURE(StridedSliceSimpleRangeMaskFixture, "StridedSliceSimpleRangeMask")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

                         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

                         5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}},

      {{"outputTensor", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

                          5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f }}});
}

}
