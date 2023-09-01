//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_BroadcastTo")
{
struct BroadcastToFixture : public ParserFlatbuffersFixture
{
    explicit BroadcastToFixture(const std::string& inputShape1,
                                const std::string& inputShape2,
                                const std::string& shapeShape,
                                const std::string& outputBroadcastToShape,
                                const std::string& outputShape,
                                const std::string& shapeData,
                                const bool checkThrows,
                                const std::string& scale = "1.0",
                                const std::string& offset = "0")
    {
        m_JsonString = R"(
        {
          "version": 3,
          "operator_codes": [
            {
              "deprecated_builtin_code": 127,
              "version": 2,
              "builtin_code": "BROADCAST_TO"
            },
            {
              "version": 1,
              "builtin_code": "MUL"
            }
          ],
          "subgraphs": [
            {
              "tensors": [
                {
                  "shape": )" + inputShape1 + R"(,
                  "type": "FLOAT32",
                  "buffer": 1,
                  "name": "inputTensor1",
                  "quantization":{
                        "min": [ 0.0 ],
                        "max": [ 255.0 ],
                        "scale": [ )" + scale + R"( ],
                        "zero_point": [ )" + offset + R"( ],
                  },
                  "is_variable": false,
                  "shape_signature": [
                    -1,
                    3
                  ],
                  "has_rank": true
                },
                {
                  "shape": )" + inputShape2 + R"(,
                  "type": "FLOAT32",
                  "buffer": 2,
                  "name": "inputTensor2",
                  "quantization": {
                        "min": [ 0.0 ],
                        "max": [ 255.0 ],
                        "scale": [ )" + scale + R"( ],
                        "zero_point": [ )" + offset + R"( ],
                  },
                  "is_variable": false,
                  "shape_signature": [
                    -1,
                    3
                  ],
                  "has_rank": true
                },
                {
                  "shape": )" + shapeShape + R"(,
                  "type": "INT32",
                  "buffer": 3,
                  "name": "shape",
                  "quantization": {
                    "details_type": "NONE",
                    "quantized_dimension": 0
                  },
                  "is_variable": false,
                  "shape_signature": [
                    -1
                  ],
                  "has_rank": true
                },
                {
                  "shape":  )" + outputBroadcastToShape + R"(,
                  "type": "FLOAT32",
                  "buffer": 4,
                  "name": "model/tf.broadcast_to/BroadcastTo",
                  "quantization": {
                        "min": [ 0.0 ],
                        "max": [ 255.0 ],
                        "scale": [ )" + scale + R"( ],
                        "zero_point": [ )" + offset + R"( ],
                  },
                  "is_variable": false,
                  "has_rank": false
                },
                {
                  "shape": )" + outputShape + R"(,
                  "type": "FLOAT32",
                  "buffer": 5,
                  "name": "outputTensor",
                  "quantization": {
                        "min": [ 0.0 ],
                        "max": [ 255.0 ],
                        "scale": [ )" + scale + R"( ],
                        "zero_point": [ )" + offset + R"( ],
                  },
                  "is_variable": false,
                  "has_rank": false
                }
              ],
              "inputs": [
                0,
                1,
                2
              ],
              "outputs": [
                4
              ],
              "operators": [
                {
                  "opcode_index": 0,
                  "inputs": [
                    0,
                    2
                  ],
                  "outputs": [
                    3
                  ],
                  "builtin_options_type": "NONE",
                  "custom_options_format": "FLEXBUFFERS"
                },
                {
                  "opcode_index": 1,
                  "inputs": [
                    1,
                    3
                  ],
                  "outputs": [
                    4
                  ],
                  "builtin_options_type": "MulOptions",
                  "builtin_options": {
                    "fused_activation_function": "NONE"
                  },
                  "custom_options_format": "FLEXBUFFERS"
                }
              ],
              "name": "main"
            }
          ],
          "description": "MLIR Converted.",
          "buffers": [
            {
            },
            {
            },
            {
            },
            {
             )" + shapeData + R"(
            },
            {
            },
            {
            },
            {
            },
            {
            }
          ],
          "metadata": [
            {
              "name": "min_runtime_version",
              "buffer": 6
            },
            {
              "name": "CONVERSION_METADATA",
              "buffer": 7
            }
          ],
          "signature_defs": [

          ]
        }
        )";
        if(checkThrows)
        {
            CHECK_THROWS_AS(Setup(), armnn::ParseException);
        }
        else
        {
            Setup();
        }
    }
};

struct BroadcastToSimpleFixture : public ParserFlatbuffersFixture
{
    explicit BroadcastToSimpleFixture(const std::string& inputShape,
                                      const std::string& shapeShape,
                                      const std::string& outputShape,
                                      const std::string& shapeData,
                                      const std::string& dataType,
                                      const std::string& scale = "1.0",
                                      const std::string& offset = "0")
    {
        m_JsonString = R"(
        {
          "version": 3,
          "operator_codes": [
            {
              "deprecated_builtin_code": 127,
              "version": 2,
              "builtin_code": "BROADCAST_TO"
            }
          ],
          "subgraphs": [
            {
              "tensors": [
                {
                  "shape": )" + inputShape + R"(,
                  "type": )" + dataType + R"(,
                  "buffer": 1,
                  "name": "input1",
                  "quantization": {
                        "min": [ 0.0 ],
                        "max": [ 255.0 ],
                        "scale": [ )" + scale + R"( ],
                        "zero_point": [ )" + offset + R"( ],
                  },
                  "is_variable": false,
                  "shape_signature": [
                    -1,
                    3
                  ],
                  "has_rank": true
                },
                {
                  "shape": )" + shapeShape + R"(,
                  "type": "INT32",
                  "buffer": 2,
                  "name": "shape",
                  "quantization": {
                    "details_type": "NONE",
                    "quantized_dimension": 0
                  },
                  "is_variable": false,
                  "shape_signature": [
                    -1
                  ],
                  "has_rank": true
                },
                {
                  "shape": )" + outputShape + R"(,
                  "type": )" + dataType + R"(,
                  "buffer": 3,
                  "name": "Identity",
                  "quantization": {
                        "min": [ 0.0 ],
                        "max": [ 255.0 ],
                        "scale": [ )" + scale + R"( ],
                        "zero_point": [ )" + offset + R"( ],
                  },
                  "is_variable": false,
                  "has_rank": false
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
          "buffers": [
            {
            },
            {
            },
            {
                "data": )" + shapeData + R"(,
            },
            {
            },
            {
            },
            {
            }
          ],
          "metadata": [
            {
              "name": "min_runtime_version",
              "buffer": 4
            },
            {
              "name": "CONVERSION_METADATA",
              "buffer": 5
            }
          ],
          "signature_defs": [

          ]
        }
    )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleBroadcastToSimpleFixtureFloat32 : public BroadcastToSimpleFixture
{
    SimpleBroadcastToSimpleFixtureFloat32() : BroadcastToSimpleFixture("[1, 4]",
                                                                       "[2]",
                                                                       "[3, 4]",
                                                                       "[3, 0, 0, 0, 4, 0, 0, 0]",
                                                                       "FLOAT32") {}
};

TEST_CASE_FIXTURE(SimpleBroadcastToSimpleFixtureFloat32, "SimpleParseBroadcastToFloat32")
{
    RunTest<2, armnn::DataType::Float32, armnn::DataType::Float32>
            (0, {{ "input1",  { 1.f, 2.f, 3.f, 4.f }}},
             {{ "Identity", { 1.f, 2.f, 3.f, 4.f,
                              1.f, 2.f, 3.f, 4.f,
                              1.f, 2.f, 3.f, 4.f}}});

}
struct SimpleBroadcastToSimpleFixtureSigned32 : public BroadcastToSimpleFixture
{
    SimpleBroadcastToSimpleFixtureSigned32() : BroadcastToSimpleFixture("[1, 4]",
                                                                        "[2]",
                                                                        "[3, 4]",
                                                                        "[3, 0, 0, 0, 4, 0, 0, 0]",
                                                                        "INT32") {}
};

TEST_CASE_FIXTURE(SimpleBroadcastToSimpleFixtureSigned32, "SimpleParseBroadcastToSigned32")
{
    RunTest<2, armnn::DataType::Signed32, armnn::DataType::Signed32>
            (0, {{ "input1",  { 1, 2, 3, -4 }}},
             {{ "Identity", { 1, 2, 3, -4,
                              1, 2, 3, -4,
                              1, 2, 3, -4}}});

}

struct SimpleBroadcastToSimpleFixtureQAsymmU8 : public BroadcastToSimpleFixture
{
    SimpleBroadcastToSimpleFixtureQAsymmU8() : BroadcastToSimpleFixture("[1, 4]",
                                                                        "[2]",
                                                                        "[3, 4]",
                                                                        "[3, 0, 0, 0, 4, 0, 0, 0]",
                                                                        "UINT8") {}
};

TEST_CASE_FIXTURE(SimpleBroadcastToSimpleFixtureQAsymmU8, "SimpleParseBroadcastToQAsymmU8")
{
    RunTest<2, armnn::DataType::QAsymmU8, armnn::DataType::QAsymmU8>
            (0, {{ "input1",  { 1, 2, 3, 4 }}},
             {{ "Identity", { 1, 2, 3, 4,
                              1, 2, 3, 4,
                              1, 2, 3, 4}}});

}

struct SimpleBroadcastToFixture : public BroadcastToFixture
{
    SimpleBroadcastToFixture() : BroadcastToFixture("[1, 4]",
                                                    "[3, 4]",
                                                    "[2]",
                                                    "[3, 4]",
                                                    "[3, 4]",
                                                    "\"data\":[3, 0, 0, 0, 4, 0, 0, 0]",
                                                    false) {}
};

TEST_CASE_FIXTURE(SimpleBroadcastToFixture, "ParseBroadcastTo")
{
    RunTest<2, armnn::DataType::Float32, armnn::DataType::Float32, armnn::DataType::Float32>
            (
                    0, {{ "inputTensor1",  { 1, 2, 3, 4 }}},
                    {{"inputTensor2", {1, 1, 1, 1,
                                       1, 1, 1, 1,
                                       1, 1, 1, 1}}},
                    {{ "outputTensor", { 1, 2, 3, 4,
                                         1, 2, 3, 4,
                                         1, 2, 3, 4}}}
            );

}

struct DynamicBroadcastToFixture : public BroadcastToFixture
{
    DynamicBroadcastToFixture() : BroadcastToFixture("[1, 4]",
                                                     "[3, 4]",
                                                     "[2]",
                                                     "[3, 4]",
                                                     "[3, 4]",
                                                     "\"data\":[3, 0, 0, 0, 4, 0, 0, 0]",
                                                     false) {}
};

TEST_CASE_FIXTURE(DynamicBroadcastToFixture, "DynamicParseBroadcastTo")
{
    RunTest<2, armnn::DataType::Float32, armnn::DataType::Float32, armnn::DataType::Float32>
            (
                    0, {{ "inputTensor1",  { 1, 2, 3, 4 }}},
                    {{"inputTensor2", {1, 1, 1, 1,
                                       1, 1, 1, 1,
                                       1, 1, 1, 1}}},
                    {{ "outputTensor", { 1, 2, 3, 4,
                                         1, 2, 3, 4,
                                         1, 2, 3, 4}}}
            );

}

struct DynamicBroadcastToFixtureNoOutputShape : public BroadcastToFixture
{
    DynamicBroadcastToFixtureNoOutputShape() : BroadcastToFixture("[1, 4]",
                                                     "[3, 4]",
                                                     "[2]",
                                                     "[]",
                                                     "[3, 4]",
                                                     "\"data\":[3, 0, 0, 0, 4, 0, 0, 0]",
                                                     false) {}
};

TEST_CASE_FIXTURE(DynamicBroadcastToFixtureNoOutputShape, "DynamicParseBroadcastToNoOutputShape")
{
    RunTest<2, armnn::DataType::Float32, armnn::DataType::Float32, armnn::DataType::Float32>
            (
                    0, {{ "inputTensor1",  { 1, 2, 3, 4 }}},
                    {{"inputTensor2", {1, 1, 1, 1,
                                       1, 1, 1, 1,
                                       1, 1, 1, 1}}},
                    {{ "outputTensor", { 1, 2, 3, 4,
                                         1, 2, 3, 4,
                                         1, 2, 3, 4}}}
            );

}

struct DynamicBroadcastToFixtureNoData : public BroadcastToFixture
{
    DynamicBroadcastToFixtureNoData() : BroadcastToFixture("[1, 4]",
                                                                  "[3, 4]",
                                                                  "[2]",
                                                                  "[3, 4]",
                                                                  "[3, 4]",
                                                                  "",
                                                                  false) {}
};

TEST_CASE_FIXTURE(DynamicBroadcastToFixtureNoData, "DynamicParseBroadcastToNoData")
{
    RunTest<2, armnn::DataType::Float32, armnn::DataType::Float32, armnn::DataType::Float32>
            (
                    0, {{ "inputTensor1",  { 1, 2, 3, 4 }}},
                    {{"inputTensor2", {1, 1, 1, 1,
                                       1, 1, 1, 1,
                                       1, 1, 1, 1}}},
                    {{ "outputTensor", { 1, 2, 3, 4,
                                         1, 2, 3, 4,
                                         1, 2, 3, 4}}}
            );

}

struct DynamicBroadcastToFixtureNoDataNoOutputShape : public BroadcastToFixture
{
    DynamicBroadcastToFixtureNoDataNoOutputShape() : BroadcastToFixture("[1, 4]",
                                                           "[3, 4]",
                                                           "[2]",
                                                           "[]",
                                                           "[3, 4]",
                                                           "", true) {  }
};

TEST_CASE_FIXTURE(DynamicBroadcastToFixtureNoDataNoOutputShape, "DynamicParseBroadcastToNoDataNoOutputShape")
{
}
}