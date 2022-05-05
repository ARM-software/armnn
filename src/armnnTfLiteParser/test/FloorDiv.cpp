//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_FloorDiv")
{
struct FloorDivFixture : public ParserFlatbuffersFixture
{
    explicit FloorDivFixture(const std::string& inputShape1,
                             const std::string& inputShape2,
                             const std::string& outputShape,
                             const std::string& inputShapeSignature1,
                             const std::string& inputShapeSignature2,
                             const std::string& outputShapeSignature,
                             const std::string& dataType = "FLOAT32")
    {
        m_JsonString = R"(
            {
              "version": 3,
              "operator_codes": [
                {
                  "deprecated_builtin_code": 90,
                  "version": 2,
                  "builtin_code": "FLOOR_DIV"
                }
              ],
              "subgraphs": [
                {
                  "tensors": [
                    {
                      "shape": )" + inputShape1 + R"(,
                      "type": )" + dataType + R"(,
                      "buffer": 1,
                      "name": "inputTensor1",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      },
                      "is_variable": false,
                      "shape_signature": )" + inputShapeSignature1 + R"(,
                    },
                    {
                      "shape": )" + inputShape2 + R"(,
                      "type": )" + dataType + R"(,
                      "buffer": 2,
                      "name": "inputTensor2",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      },
                      "is_variable": false,
                      "shape_signature": )" + inputShapeSignature2 + R"(,
                    },
                    {
                      "shape": )" + outputShape + R"(,
                      "type": )" + dataType + R"(,
                      "buffer": 3,
                      "name": "outputTensor",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      },
                      "is_variable": false,
                      "shape_signature": )" + outputShapeSignature + R"(,
                    }
                  ],
                  "inputs": [
                    0,
                    1
                  ],
                  "outputs": [
                    2
                  ],
                  "operators": [
                    {
                      "opcode_index": 0,
                      "inputs": [
                        0,
                        1
                      ],
                      "outputs": [
                        2
                      ],
                      "builtin_options_type": "NONE",
                      "custom_options_format": "FLEXBUFFERS"
                    }
                  ],
                  "name": "main"
                }
              ],
              "description": "MLIR Converted.",
              "buffers": [ {}, {}, {}, {},
                {
                  "data": [
                    49,
                    46,
                    49,
                    52,
                    46,
                    48,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                  ]
                }
              ],
              "metadata": [
                {
                  "name": "min_runtime_version",
                  "buffer": 4
                }
              ],
              "signature_defs": [

              ]
            }
        )";
        Setup();
    }
};

struct SimpleFloorDivFixture : public FloorDivFixture
{
    SimpleFloorDivFixture() : FloorDivFixture("[ 1, 3, 4 ]", "[ 1, 3, 4 ]", "[ 1, 3, 4 ]",
                                              "[ -1, 3, 4 ]", "[ -1, 3, 4 ]", "[ -1, 3, 4 ]") {}
};

TEST_CASE_FIXTURE(SimpleFloorDivFixture, "ParseFloorDiv")
{
    using armnn::DataType;
    float Inf = std::numeric_limits<float>::infinity();
    float NaN = std::numeric_limits<float>::quiet_NaN();

    RunTest<3, DataType::Float32>(0, {{ "inputTensor1", { 0.0f,  1.0f,  2.0f,
                                                          3.0f,  4.0f,  5.0f,
                                                          6.0f,  -7.0f,  8.0f,
                                                          9.0f, 10.0f, -11.0f } },
                                      { "inputTensor2", { 0.0f,  0.0f,  4.0f,
                                                          3.0f,  40.0f,  5.0f,
                                                          6.0f,  2.0f,  8.0f,
                                                          9.0f,  10.0f,  11.0f} } },
                                     {{ "outputTensor", { NaN,   Inf,  0.0f,
                                                          1.0f,  0.0f, 1.0f,
                                                          1.0f,  -4.0f, 1.0f,
                                                          1.0f,  1.0f, -1.0f } } });
}

struct SimpleFloorDivInt32Fixture : public FloorDivFixture
{
    SimpleFloorDivInt32Fixture() : FloorDivFixture("[ 1, 3, 4 ]", "[ 1, 3, 4 ]", "[ 1, 3, 4 ]",
                                                   "[ -1, 3, 4 ]", "[ -1, 3, 4 ]", "[ -1, 3, 4 ]", "INT32") {}
};
TEST_CASE_FIXTURE(SimpleFloorDivInt32Fixture, "ParseFloorDivInt32")
{
    using armnn::DataType;

    RunTest<3, DataType::Signed32>(0, {{ "inputTensor1", { 1,  1,  2,
                                                                3,  4,  5,
                                                                6,  -7,  8,
                                                                9, 10, -11 } },
                                      { "inputTensor2", { 1,  1,  4,
                                                                3,  40,  5,
                                                                6,  2,  8,
                                                                9,  10,  11} } },
                                   {{ "outputTensor", { 1,  1,  0,
                                                       1,  0, 1,
                                                       1,  -4, 1,
                                                       1,  1, -1 } } });
}


struct DynamicFloorDivFixture : public FloorDivFixture
{
    DynamicFloorDivFixture() : FloorDivFixture("[ 1, 3, 4 ]", "[ 1, 3, 4 ]", "[ 1, 3, 4 ]",
                                               "[ -1, 3, 4 ]", "[ -1, 3, 4 ]", "[ -1, 3, 4 ]") {}
};

TEST_CASE_FIXTURE(DynamicFloorDivFixture, "ParseDynamicFloorDiv")
{
    using armnn::DataType;
    float Inf = std::numeric_limits<float>::infinity();
    float NaN = std::numeric_limits<float>::quiet_NaN();

    RunTest<3, DataType::Float32, DataType::Float32>(0, {{ "inputTensor1", { 0.0f,  1.0f,  2.0f,
                                                                             3.0f,  4.0f,  5.0f,
                                                                             6.0f,  -7.0f,  8.0f,
                                                                             9.0f, 10.0f, -11.0f } },
                                                         { "inputTensor2", { 0.0f,  0.0f,  4.0f,
                                                                             3.0f,  40.0f,  5.0f,
                                                                             6.0f,  2.0f,  8.0f,
                                                                             9.0f,  10.0f,  11.0f} } },
                                                     {{ "outputTensor", { NaN,   Inf,  0.0f,
                                                                          1.0f,  0.0f, 1.0f,
                                                                          1.0f,  -4.0f, 1.0f,
                                                                          1.0f,  1.0f, -1.0f } } }, true);
}

}
