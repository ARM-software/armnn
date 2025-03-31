//
// Copyright © 2023-2025 Arm Ltd and contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "CommonTestUtils.hpp"

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <doctest/doctest.h>

#include <vector>

namespace
{

template<armnn::DataType ArmnnTypeInput>
INetworkPtr CreateElementwiseBinaryNetwork(const TensorShape& input1Shape,
                                          const TensorShape& input2Shape,
                                          const TensorShape& outputShape,
                                          BinaryOperation operation,
                                          const float qScale = 1.0f,
                                          const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr net(INetwork::Create());

    TensorInfo input1TensorInfo(input1Shape, ArmnnTypeInput, qScale, qOffset, true);
    TensorInfo input2TensorInfo(input2Shape, ArmnnTypeInput, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, ArmnnTypeInput, qScale, qOffset);

    IConnectableLayer* input1 = net->AddInputLayer(armnn::numeric_cast<LayerBindingId>(0));
    IConnectableLayer* input2 = net->AddInputLayer(armnn::numeric_cast<LayerBindingId>(1));
    IConnectableLayer* elementwiseBinaryLayer = net->AddElementwiseBinaryLayer(operation, "elementwiseUnary");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    Connect(input1, elementwiseBinaryLayer, input1TensorInfo, 0, 0);
    Connect(input2, elementwiseBinaryLayer, input2TensorInfo, 0, 1);
    Connect(elementwiseBinaryLayer, output, outputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnTypeInput>
INetworkPtr CreateElementwiseBinaryNetworkDifferentScales(const TensorShape& input1Shape,
                                                          const TensorShape& input2Shape,
                                                          const TensorShape& outputShape,
                                                          BinaryOperation operation,
                                                          const float input1Scale = 1.0f,
                                                          const float input2Scale = 1.0f,
                                                          const float outputScale = 1.0f,
                                                          const int32_t input1Offset = 0,
                                                          const int32_t input2Offset = 0,
                                                          const int32_t outputOffset = 0)
{
    using namespace armnn;

    INetworkPtr net(INetwork::Create());

    TensorInfo input1TensorInfo(input1Shape, ArmnnTypeInput, input1Scale, input1Offset, true);
    TensorInfo input2TensorInfo(input2Shape, ArmnnTypeInput, input2Scale, input2Offset, true);
    TensorInfo outputTensorInfo(outputShape, ArmnnTypeInput, outputScale, outputOffset);

    IConnectableLayer* input1 = net->AddInputLayer(armnn::numeric_cast<LayerBindingId>(0));
    IConnectableLayer* input2 = net->AddInputLayer(armnn::numeric_cast<LayerBindingId>(1));
    IConnectableLayer* elementwiseBinaryLayer = net->AddElementwiseBinaryLayer(operation, "elementwiseUnary");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    Connect(input1, elementwiseBinaryLayer, input1TensorInfo, 0, 0);
    Connect(input2, elementwiseBinaryLayer, input2TensorInfo, 0, 1);
    Connect(elementwiseBinaryLayer, output, outputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnInType,
         typename TInput = armnn::ResolveType<ArmnnInType>>
void ElementwiseBinarySimpleEndToEnd(const std::vector<BackendId>& backends,
                                     BinaryOperation operation)
{
    using namespace armnn;

    const float   qScale  = IsQuantizedType<TInput>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<TInput>() ? 50    : 0;

    const TensorShape& input1Shape  = { 2, 2, 2, 2 };
    const TensorShape& input2Shape  = { 1 };
    const TensorShape& outputShape = { 2, 2, 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateElementwiseBinaryNetwork<ArmnnInType>(input1Shape, input2Shape, outputShape,
                                                                  operation, qScale, qOffset);

    CHECK(net);

    const std::vector<float> input1({ 1, -1, 1, 1,  5, -5, 5, 5,  -3, 3, 3, 3,  4, 4, -4, 4 });

    const std::vector<float> input2({ 2 });
    std::vector<float> expectedOutput;
    switch (operation) {
        case armnn::BinaryOperation::Add:
            expectedOutput = { 3, 1, 3, 3,  7, -3, 7, 7,  -1, 5, 5, 5,  6, 6, -2, 6 };
            break;
        case armnn::BinaryOperation::Div:
            expectedOutput = {0.5f, -0.5f, 0.5f, 0.5f, 2.5f, -2.5f, 2.5f, 2.5f, -1.5f, 1.5f, 1.5f, 1.5f, 2, 2, -2, 2};
            break;
        case armnn::BinaryOperation::Maximum:
            expectedOutput = { 2, 2, 2, 2,  5, 2, 5, 5,  2, 3, 3, 3,  4, 4, 2, 4 };
            break;
        case armnn::BinaryOperation::Minimum:
            expectedOutput = { 1, -1, 1, 1,  2, -5, 2, 2,  -3, 2, 2, 2,  2, 2, -4, 2 };
            break;
        case armnn::BinaryOperation::Mul:
            expectedOutput = { 2, -2, 2, 2,  10, -10, 10, 10,  -6, 6, 6, 6,  8, 8, -8, 8 };
            break;
        case armnn::BinaryOperation::Sub:
            expectedOutput = { -1, -3, -1, -1,  3, -7, 3, 3,  -5, 1, 1, 1,  2, 2, -6, 2 };
            break;
        case armnn::BinaryOperation::SqDiff:
            expectedOutput = { 1, 9, 1, 1,  9, 49, 9, 9, 25, 1, 1, 1,  4, 4, 36, 4  };
            break;
        case armnn::BinaryOperation::Power:
            expectedOutput = { 1, 1, 1, 1, 25, 25, 25, 25,  9, 9, 9, 9,  16, 16, 16, 16 };
            break;
        default:
            throw("Invalid Elementwise Binary operation");
    }
    const std::vector<float> expectedOutput_const = expectedOutput;
    // quantize data
    std::vector<TInput> qInput1Data     = armnnUtils::QuantizedVector<TInput>(input1, qScale, qOffset);
    std::vector<TInput> qInput2Data     = armnnUtils::QuantizedVector<TInput>(input2, qScale, qOffset);
    std::vector<TInput> qExpectedOutput = armnnUtils::QuantizedVector<TInput>(expectedOutput_const, qScale, qOffset);

    std::map<int, std::vector<TInput>> inputTensorData    = {{ 0, qInput1Data }, { 1, qInput2Data }};
    std::map<int, std::vector<TInput>> expectedOutputData = {{ 0, qExpectedOutput }};

    EndToEndLayerTestImpl<ArmnnInType, ArmnnInType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnInType, typename TInput = armnn::ResolveType<ArmnnInType>>
void ElementwiseBinarySimpleWithScalesEndToEnd(const std::vector<BackendId>& backends,
                                     BinaryOperation operation)
{
    using namespace armnn;

    const auto input1Scale  = static_cast<float>(0.0683157816529274);
    const auto input2Scale  = static_cast<float>(0.29619300365448);
    const auto outputScale  = static_cast<float>(2.6329452991485596);

    const int32_t input1Offset  = 22;
    const int32_t input2Offset  = -125;
    const int32_t outputOffset  = 31;

    const TensorShape& input1Shape  = { 1, 1, 2, 2 };
    const TensorShape& input2Shape  = { 1, 1, 2, 2 };
    const TensorShape& outputShape  = { 1, 1, 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateElementwiseBinaryNetworkDifferentScales<ArmnnInType>(input1Shape, input2Shape, outputShape,
                                                                                 operation, input1Scale,
                                                                                 input2Scale, outputScale, input1Offset,
                                                                                 input2Offset, outputOffset);

    CHECK(net);

    const std::vector<float> input1({ 2, -2, 4, -4 });
    const std::vector<float> input2({ 2, 2, 2, 2 });

    std::vector<float> expectedOutput;
    switch (operation) {
        case armnn::BinaryOperation::Add:
            expectedOutput = { 4, 0, 6, -2 };
            break;
        case armnn::BinaryOperation::Mul:
            expectedOutput = { 4, -4, 8, -8 };
            break;
        case armnn::BinaryOperation::Sub:
            expectedOutput = { 0, -4, 2, -6 };
            break;
        default:
            throw("Invalid Elementwise Binary operation");
    }
    const std::vector<float> expectedOutput_const = expectedOutput;
    // quantize data
    std::vector<TInput> qInput1Data     = armnnUtils::QuantizedVector<TInput>(input1, input1Scale, input1Offset);
    std::vector<TInput> qInput2Data     = armnnUtils::QuantizedVector<TInput>(input2, input2Scale, input2Offset);
    std::vector<TInput> qExpectedOutput = armnnUtils::QuantizedVector<TInput>(expectedOutput_const, outputScale,
                                                                              outputOffset);

    std::map<int, std::vector<TInput>> inputTensorData    = {{ 0, qInput1Data }, { 1, qInput2Data }};
    std::map<int, std::vector<TInput>> expectedOutputData = {{ 0, qExpectedOutput }};

    // Setting tolerance for int8 to 1.
    auto tolerance = 1;
    EndToEndLayerTestImpl<ArmnnInType, ArmnnInType>(std::move(net), inputTensorData, expectedOutputData, backends,
                                                    static_cast<float>(tolerance));
}

template<armnn::DataType ArmnnInType,
        typename TInput = armnn::ResolveType<ArmnnInType>>
void ElementwiseBinarySimple3DEndToEnd(const std::vector<BackendId>& backends,
                                       BinaryOperation operation)
{
    using namespace armnn;

    const float   qScale  = IsQuantizedType<TInput>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<TInput>() ? 50    : 0;

    const TensorShape& input1Shape  = { 2, 2 };
    const TensorShape& input2Shape  = { 2, 2 };
    const TensorShape& outputShape  = { 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateElementwiseBinaryNetwork<ArmnnInType>(input1Shape, input2Shape, outputShape,
                                                                  operation, qScale, qOffset);

    CHECK(net);

    const std::vector<float> input1({ 1, -1, 1, 1 });

    const std::vector<float> input2({ 2, 2, 2, 2 });
    std::vector<float> expectedOutput;
    switch (operation)
    {
        case armnn::BinaryOperation::Add:
            expectedOutput = { 3, 1, 3, 3 };
            break;
        case armnn::BinaryOperation::Div:
            expectedOutput = {0.5f, -0.5f, 0.5f, 0.5f };
            break;
        case armnn::BinaryOperation::Maximum:
            expectedOutput = { 2, 2, 2, 2 };
            break;
        case armnn::BinaryOperation::Minimum:
            expectedOutput = { 1, -1, 1, 1 };
            break;
        case armnn::BinaryOperation::Mul:
            expectedOutput = { 2, -2, 2, 2 };
            break;
        case armnn::BinaryOperation::Sub:
            expectedOutput = { -1, -3, -1, -1 };
            break;
        case armnn::BinaryOperation::SqDiff:
            expectedOutput = { 1, 9, 1, 1 };
            break;
        case armnn::BinaryOperation::Power:
            expectedOutput = { 1, 1, 1, 1 };
            break;
        default:
            throw("Invalid Elementwise Binary operation");
    }
    const std::vector<float> expectedOutput_const = expectedOutput;
    // quantize data
    std::vector<TInput> qInput1Data     = armnnUtils::QuantizedVector<TInput>(input1, qScale, qOffset);
    std::vector<TInput> qInput2Data     = armnnUtils::QuantizedVector<TInput>(input2, qScale, qOffset);
    std::vector<TInput> qExpectedOutput = armnnUtils::QuantizedVector<TInput>(expectedOutput_const, qScale, qOffset);

    std::map<int, std::vector<TInput>> inputTensorData    = {{ 0, qInput1Data }, { 1, qInput2Data }};
    std::map<int, std::vector<TInput>> expectedOutputData = {{ 0, qExpectedOutput }};

    EndToEndLayerTestImpl<ArmnnInType, ArmnnInType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnInType,
        typename TInput = armnn::ResolveType<ArmnnInType>>
void ElementwiseBinarySimpleNoReshapeEndToEnd(const std::vector<BackendId>& backends,
                                              BinaryOperation operation)
{
    using namespace armnn;

    const float   qScale  = 1.0f;
    const int32_t qOffset = 0;

    const TensorShape& input1Shape  = { 2, 2, 2, 2 };
    const TensorShape& input2Shape  = { 2, 2, 2, 2 };
    const TensorShape& outputShape = { 2, 2, 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateElementwiseBinaryNetwork<ArmnnInType>(input1Shape, input2Shape, outputShape,
                                                                  operation, qScale, qOffset);

    CHECK(net);

    const std::vector<float> input1({ 1, -1, 1, 1,  5, -5, 5, 5,  -3, 3, 3, 3,  4, 4, -4, 4 });

    const std::vector<float> input2({ 2, 2, 2, 2,   2, 2, 2, 2,    2, 2, 2, 2,  2, 2, 2, 2 });

    std::vector<float> expectedOutput;
    switch (operation) {
        case armnn::BinaryOperation::Add:
            expectedOutput = { 3, 1, 3, 3,  7, -3, 7, 7,  -1, 5, 5, 5,  6, 6, -2, 6 };
            break;
        case armnn::BinaryOperation::Div:
            expectedOutput = {0.5f, -0.5f, 0.5f, 0.5f, 2.5f, -2.5f, 2.5f, 2.5f, -1.5f, 1.5f, 1.5f, 1.5f, 2, 2, -2, 2};
            break;
        case armnn::BinaryOperation::Maximum:
            expectedOutput = { 2, 2, 2, 2,  5, 2, 5, 5,  2, 3, 3, 3,  4, 4, 2, 4 };
            break;
        case armnn::BinaryOperation::Minimum:
            expectedOutput = { 1, -1, 1, 1,  2, -5, 2, 2,  -3, 2, 2, 2,  2, 2, -4, 2 };
            break;
        case armnn::BinaryOperation::Mul:
            expectedOutput = { 2, -2, 2, 2,  10, -10, 10, 10,  -6, 6, 6, 6,  8, 8, -8, 8 };
            break;
        case armnn::BinaryOperation::Sub:
            expectedOutput = { -1, -3, -1, -1,  3, -7, 3, 3,  -5, 1, 1, 1,  2, 2, -6, 2 };
            break;
        case armnn::BinaryOperation::SqDiff:
            expectedOutput = { 1, 9, 1, 1,  9, 49, 9, 9, 25, 1, 1, 1,  4, 4, 36, 4  };
            break;
        case armnn::BinaryOperation::Power:
            expectedOutput = { 1, 1, 1, 1, 25, 25, 25, 25,  9, 9, 9, 9,  16, 16, 16, 16 };
            break;
        default:
            throw("Invalid Elementwise Binary operation");
    }

    const std::vector<float> expectedOutput_const = expectedOutput;
    // quantize data
    std::vector<TInput> qInput1Data     = armnnUtils::QuantizedVector<TInput>(input1, qScale, qOffset);
    std::vector<TInput> qInput2Data     = armnnUtils::QuantizedVector<TInput>(input2, qScale, qOffset);
    std::vector<TInput> qExpectedOutput = armnnUtils::QuantizedVector<TInput>(expectedOutput_const, qScale, qOffset);

    std::map<int, std::vector<TInput>> inputTensorData    = {{ 0, qInput1Data }, { 1, qInput2Data }};
    std::map<int, std::vector<TInput>> expectedOutputData = {{ 0, qExpectedOutput }};

    EndToEndLayerTestImpl<ArmnnInType, ArmnnInType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace
