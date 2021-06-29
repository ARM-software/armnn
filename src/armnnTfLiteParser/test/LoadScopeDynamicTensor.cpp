//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfLiteParser/ITfLiteParser.hpp"
#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_LoadScopeDynamicTensor")
{
struct LoadScopeDynamicTensorFixture : public ParserFlatbuffersFixture
{
    explicit LoadScopeDynamicTensorFixture(const std::string& shape0,
                                           const std::string& shape1,
                                           const std::string& shape2)
    {
        m_JsonString = R"(
        {
            "version": 3,
            "operator_codes": [
                {
                    "builtin_code": "AVERAGE_POOL_2D",
                    "version": 1
                },
                {
                    "builtin_code": "SOFTMAX",
                    "version": 1
                }
            ],
            "subgraphs": [
                {
                    "tensors": [
                        {
                            "shape": )" + shape0 + R"(,
                            "type": "FLOAT32",
                            "buffer": 1,
                            "name": "input0",
                            "quantization": {
                                "details_type": 0,
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                        {
                            "shape": )" + shape1 + R"(,
                            "type": "FLOAT32",
                            "buffer": 3,
                            "name": "output",
                            "quantization": {
                                "details_type": 0,
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                        {
                            "shape": )" + shape2 + R"(,
                            "type": "FLOAT32",
                            "buffer": 2,
                            "name": "model/average_pooling2d/AvgPool",
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
                            "opcode_index": 1,
                            "inputs": [
                                2
                            ],
                            "outputs": [
                                1
                            ],
                            "builtin_options_type": "SoftmaxOptions",
                            "builtin_options": {
                                "beta": 1.0
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        },
                        {
                            "opcode_index": 0,
                            "inputs": [
                                0
                            ],
                            "outputs": [
                                2
                            ],
                            "builtin_options_type": "Pool2DOptions",
                            "builtin_options": {
                                "padding": "VALID",
                                "stride_w": 2,
                                "stride_h": 2,
                                "filter_width": 2,
                                "filter_height": 2,
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
                }
            ]
        }
        )";
        Setup();
    }
};

struct LoadScopeDynamicTensor0Fixture : LoadScopeDynamicTensorFixture
{
    LoadScopeDynamicTensor0Fixture() : LoadScopeDynamicTensorFixture("[ 1, 2, 3, 2 ]", "[]", "[]") {}
};

struct LoadScopeDynamicTensor1Fixture : LoadScopeDynamicTensorFixture
{
    LoadScopeDynamicTensor1Fixture() : LoadScopeDynamicTensorFixture("[ 1, 2, 4, 1 ]", "[ 1, 1, 2, 1 ]", "[]") {}
};

struct LoadScopeDynamicTensor2Fixture : LoadScopeDynamicTensorFixture
{
    LoadScopeDynamicTensor2Fixture() : LoadScopeDynamicTensorFixture("[ 1, 3, 3, 2 ]", "[ ]", "[ 1, 1, 1, 2 ]") {}
};

TEST_CASE_FIXTURE(LoadScopeDynamicTensor0Fixture, "LoadScopeDynamicTensor0")
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>(
        0,
        { {"input0", { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f }} },
        { {"output", { 0.26894143f, 0.7310586f }} },
        true);
}

TEST_CASE_FIXTURE(LoadScopeDynamicTensor1Fixture, "LoadScopeDynamicTensor1")
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>(
        0,
        { {"input0", { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f }} },
        { {"output", { 1.f, 1.f }} },
        true);
}

TEST_CASE_FIXTURE(LoadScopeDynamicTensor2Fixture, "LoadScopeDynamicTensor2")
{
  RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>(
        0,
        { {"input0", { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f }} },
        { {"output", { 0.7772999f, 0.22270015f }} },
        true);
}

struct LoadScopeDynamicTensorBroadcastingFixture : public ParserFlatbuffersFixture
{
    explicit LoadScopeDynamicTensorBroadcastingFixture(const std::string& inputShape0,
                                                       const std::string& inputShape1,
                                                       const std::string& inputShape2,
                                                       const std::string& addShape,
                                                       const std::string& outputShape)
    {
        m_JsonString = R"(
        {
            "version": 3,
            "operator_codes": [
                {
                    "builtin_code": "ADD",
                    "version": 1
                },
                {
                    "builtin_code": "SUB",
                    "version": 1
                }
            ],
            "subgraphs": [
                {
                    "tensors": [
                        {
                            "shape": )" + inputShape0 + R"(,
                            "type": "FLOAT32",
                            "buffer": 1,
                            "name": "input0",
                            "quantization": {
                                "details_type": 0,
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                        {
                            "shape": )" + inputShape1 + R"(,
                            "type": "FLOAT32",
                            "buffer": 2,
                            "name": "input1",
                            "quantization": {
                                "details_type": 0,
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                        {
                            "shape": )" + outputShape + R"(,
                            "type": "FLOAT32",
                            "buffer": 5,
                            "name": "output",
                            "quantization": {
                                "details_type": 0,
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },

                        {
                            "shape": )" + addShape + R"(,
                            "type": "FLOAT32",
                            "buffer": 4,
                            "name": "model/add/add",
                            "quantization": {
                                "details_type": 0,
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                        {
                            "shape": )" + inputShape2 + R"(,
                            "type": "FLOAT32",
                            "buffer": 3,
                            "name": "input2",
                            "quantization": {
                                "details_type": 0,
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                    ],
                    "inputs": [
                        0,
                        1,
                        4
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
                                3
                            ],
                            "builtin_options_type": "AddOptions",
                            "builtin_options": {
                                "fused_activation_function": "NONE"
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        },
                        {
                            "opcode_index": 1,
                            "inputs": [
                                3,
                                4
                            ],
                            "outputs": [
                                2
                            ],
                            "builtin_options_type": "SubOptions",
                            "builtin_options": {
                                "fused_activation_function": "NONE"
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                    "name": "main"
                }
            ],
            "buffers": [
                {
                },
                {
                },
                {
                },
                {
                },
                {
                },
                {
                }
            ]
        }
        )";
        Setup();
    }
};

struct LoadScopeDynamicTensorBroadcasting3DFixture : LoadScopeDynamicTensorBroadcastingFixture
{
    LoadScopeDynamicTensorBroadcasting3DFixture() : LoadScopeDynamicTensorBroadcastingFixture("[ 1, 2, 3, 2 ]",
                                                                                              "[ 2, 3, 2 ]",
                                                                                              "[ 2, 3, 2 ]",
                                                                                              "[ 1, 2, 3, 2 ]", "[]") {}
};

struct LoadScopeDynamicTensorBroadcasting2DFixture : LoadScopeDynamicTensorBroadcastingFixture
{
    LoadScopeDynamicTensorBroadcasting2DFixture() : LoadScopeDynamicTensorBroadcastingFixture("[ 1, 2, 3, 2 ]",
                                                                                              "[ 3, 2 ]",
                                                                                              "[ 3, 2 ]",
                                                                                              "[]", "[]") {}
};

struct LoadScopeDynamicTensorBroadcasting1DFixture : LoadScopeDynamicTensorBroadcastingFixture
{
    LoadScopeDynamicTensorBroadcasting1DFixture() : LoadScopeDynamicTensorBroadcastingFixture("[ 1, 2, 3, 2 ]",
                                                                                              "[ 1 ]",
                                                                                              "[ 1 ]",
                                                                                              "[]",
                                                                                              "[ 1, 2, 3, 2 ]") {}
};

TEST_CASE_FIXTURE(LoadScopeDynamicTensorBroadcasting3DFixture, "LoadScopeDynamicTensorBroadcasting3D")
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>(
        0,
        { {"input0", { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f }},
          {"input1", { 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f }},
          {"input2", { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f }}
        },
        { {"output", { 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f }} },
        true);
}

TEST_CASE_FIXTURE(LoadScopeDynamicTensorBroadcasting2DFixture, "LoadScopeDynamicTensorBroadcasting2D")
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>(
        0,
        { {"input0", { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f }},
          {"input1", { 3.f, 4.f, 5.f, 6.f, 7.f, 8.f }},
          {"input2", { -1.f, -2.f, 3.f, 4.f, 5.f, 6.f }}
        },
        { {"output", { 4.f, 7.f, 4.f, 5.f, 6.f, 7.f, 10.f, 13.f, 10.f, 11.f, 12.f, 13.f }} },
        true);
}

TEST_CASE_FIXTURE(LoadScopeDynamicTensorBroadcasting1DFixture, "LoadScopeDynamicTensorBroadcasting1D")
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>(
        0,
        { {"input0", { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f }},
          {"input1", { 5.f }},
          {"input2", { 1.f }}
        },
        { {"output", { 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f }} },
        true);
}

}
