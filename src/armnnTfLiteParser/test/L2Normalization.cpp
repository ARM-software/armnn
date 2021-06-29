//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

#include <numeric>

TEST_SUITE("TensorflowLiteParser_L2Normalization")
{
struct L2NormalizationFixture : public ParserFlatbuffersFixture
{
    explicit L2NormalizationFixture(const std::string & inputOutputShape)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "L2_NORMALIZATION" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputOutputShape + R"(,
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
                            "shape": )" + inputOutputShape + R"(,
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
                    { }
                ]
            }
        )";
        Setup();
    }
};

float CalcL2Norm(std::initializer_list<float> elements)
{
    const float reduction = std::accumulate(elements.begin(), elements.end(), 0.0f,
        [](float acc, float element) { return acc + element * element; });
    const float eps = 1e-12f;
    const float max = reduction < eps ? eps : reduction;
    return sqrtf(max);
}

struct L2NormalizationFixture4D : L2NormalizationFixture
{
    // TfLite uses NHWC shape
    L2NormalizationFixture4D() : L2NormalizationFixture("[ 1, 1, 4, 3 ]") {}
};

TEST_CASE_FIXTURE(L2NormalizationFixture4D, "ParseL2Normalization4D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor", { 1.0f,  2.0f,  3.0f,
                         4.0f,  5.0f,  6.0f,
                         7.0f,  8.0f,  9.0f,
                         10.0f, 11.0f, 12.0f }}},

      {{"outputTensor", { 1.0f  / CalcL2Norm({ 1.0f,  2.0f,  3.0f }),
                          2.0f  / CalcL2Norm({ 1.0f,  2.0f,  3.0f }),
                          3.0f  / CalcL2Norm({ 1.0f,  2.0f,  3.0f }),

                          4.0f  / CalcL2Norm({ 4.0f,  5.0f,  6.0f }),
                          5.0f  / CalcL2Norm({ 4.0f,  5.0f,  6.0f }),
                          6.0f  / CalcL2Norm({ 4.0f,  5.0f,  6.0f }),

                          7.0f  / CalcL2Norm({ 7.0f,  8.0f,  9.0f }),
                          8.0f  / CalcL2Norm({ 7.0f,  8.0f,  9.0f }),
                          9.0f  / CalcL2Norm({ 7.0f,  8.0f,  9.0f }),

                          10.0f / CalcL2Norm({ 10.0f, 11.0f, 12.0f }),
                          11.0f / CalcL2Norm({ 10.0f, 11.0f, 12.0f }),
                          12.0f / CalcL2Norm({ 10.0f, 11.0f, 12.0f }) }}});
}

struct L2NormalizationSimpleFixture4D : L2NormalizationFixture
{
    L2NormalizationSimpleFixture4D() : L2NormalizationFixture("[ 1, 1, 1, 4 ]") {}
};

TEST_CASE_FIXTURE(L2NormalizationSimpleFixture4D, "ParseL2NormalizationEps4D")
{
      RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor", { 0.00000001f, 0.00000002f, 0.00000003f, 0.00000004f }}},

      {{"outputTensor", { 0.00000001f / CalcL2Norm({ 0.00000001f, 0.00000002f, 0.00000003f, 0.00000004f }),
                          0.00000002f / CalcL2Norm({ 0.00000001f, 0.00000002f, 0.00000003f, 0.00000004f }),
                          0.00000003f / CalcL2Norm({ 0.00000001f, 0.00000002f, 0.00000003f, 0.00000004f }),
                          0.00000004f / CalcL2Norm({ 0.00000001f, 0.00000002f, 0.00000003f, 0.00000004f }) }}});
}

}
