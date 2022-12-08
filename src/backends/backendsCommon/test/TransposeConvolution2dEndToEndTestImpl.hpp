//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnnUtils/Permute.hpp>

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <CommonTestUtils.hpp>

#include <map>
#include <vector>

namespace
{

INetworkPtr CreateTransposeConvolution2dNetwork(const armnn::TransposeConvolution2dDescriptor& descriptor,
                                                const armnn::TensorInfo& inputInfo,
                                                const armnn::TensorInfo& outputInfo,
                                                const armnn::ConstTensor& weights,
                                                const armnn::Optional<armnn::ConstTensor>& biases)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());
    IConnectableLayer* input = network->AddInputLayer(0, "input");
    IConnectableLayer* transposeConvolution2d =
        network->AddTransposeConvolution2dLayer(descriptor, weights, biases, "transposeConvolution2d");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, transposeConvolution2d, inputInfo, 0, 0);
    Connect(transposeConvolution2d, output, outputInfo, 0, 0);

    return network;
}

} // anonymous namespace

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType>
void TransposeConvolution2dEndToEnd(const std::vector<armnn::BackendId>& backends,
                                    armnn::DataLayout dataLayout)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    constexpr unsigned int batches  = 1u;
    constexpr unsigned int channels = 1u;

    constexpr unsigned int wInput = 3u;
    constexpr unsigned int hInput = wInput;

    constexpr unsigned int wOutput = 5u;
    constexpr unsigned int hOutput = wOutput;

    constexpr unsigned int wWeights = 3u;
    constexpr unsigned int hWeights = wWeights;

    TensorShape inputShape   = MakeTensorShape(batches, channels, hInput, wInput, dataLayout);
    TensorShape outputShape  = MakeTensorShape(batches, channels, hOutput, wOutput, dataLayout);
    TensorShape weightsShape = MakeTensorShape(batches, channels, hWeights, wWeights, dataLayout);

    const float   qScale  = IsQuantizedType<T>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<T>() ? 50    : 0;

    TensorInfo inputInfo(inputShape, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo(outputShape, ArmnnType, qScale, qOffset);
    TensorInfo weightsInfo(weightsShape, ArmnnType, qScale, qOffset, true);
    TensorInfo biasesInfo({ channels }, ArmnnBType, qScale * qScale, 0, true);

    std::vector<float> inputData =
    {
       1.f, 1.f, 1.f,
       1.f, 1.f, 1.f,
       1.f, 1.f, 1.f
    };

    std::vector<float> weightsData =
    {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f
    };

    std::vector<float> biasesData = { 1.f };

    std::vector<float> expectedOutputData =
    {
         6.f, 11.f,  6.f, 11.f,  6.f,
        11.f, 21.f, 11.f, 21.f, 11.f,
         6.f, 11.f,  6.f, 11.f,  6.f,
        11.f, 21.f, 11.f, 21.f, 11.f,
         6.f, 11.f,  6.f, 11.f,  6.f
    };

    TransposeConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 1;
    descriptor.m_PadRight    = 1;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 1;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = dataLayout;

    // swizzle data if needed
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        constexpr size_t dataTypeSize = sizeof(float);
        const armnn::PermutationVector nchwToNhwc = { 0, 3, 1, 2 };

        std::vector<float> tmp(inputData.size());
        armnnUtils::Permute(inputInfo.GetShape(), nchwToNhwc, inputData.data(), tmp.data(), dataTypeSize);
        inputData = tmp;

        tmp.resize(weightsData.size());
        armnnUtils::Permute(weightsInfo.GetShape(), nchwToNhwc, weightsData.data(), tmp.data(), dataTypeSize);
        weightsData = tmp;

        tmp.resize(expectedOutputData.size());
        armnnUtils::Permute(outputInfo.GetShape(), nchwToNhwc, expectedOutputData.data(), tmp.data(), dataTypeSize);
        expectedOutputData = tmp;
    }

    // quantize data
    std::vector<T> qInputData          = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> qWeightsData        = armnnUtils::QuantizedVector<T>(weightsData, qScale, qOffset);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);

    using BT = ResolveType<ArmnnBType>;
    std::vector<BT> qBiasesData = armnnUtils::QuantizedVector<BT>(biasesData, qScale * qScale, 0);

    ConstTensor weights(weightsInfo, qWeightsData);
    ConstTensor biases(biasesInfo, qBiasesData);

    INetworkPtr network = CreateTransposeConvolution2dNetwork(descriptor,
                                                              inputInfo,
                                                              outputInfo,
                                                              weights,
                                                              Optional<ConstTensor>(biases));


    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                { { 0, qInputData } },
                                                { { 0, qExpectedOutputData } },
                                                backends);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType>
void SimpleTransposeConvolution2dEndToEnd(const std::vector<armnn::BackendId>& backends,
                                          armnn::DataLayout dataLayout)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    const float   qScale  = IsQuantizedType<T>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<T>() ? 50    : 0;

    TensorInfo inputInfo({1, 2, 2, 1}, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({1, 3, 3, 1}, ArmnnType, qScale, qOffset);
    TensorInfo weightsInfo({1, 2, 2, 1}, ArmnnType, qScale, qOffset, true);
    TensorInfo biasesInfo({ 1 }, ArmnnBType, qScale * qScale, 0, true);

    std::vector<float> inputData =
    {
        1, 2, 3, 4
    };

    std::vector<float> weightsData =
    {
        0, 1, 2, 4
    };
    std::vector<float> biasesData = { 0.f };

    std::vector<float> expectedOutputData =
    {
        0, 1,  2,
        2, 11, 12,
        6, 20, 16
    };

    TransposeConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = dataLayout;
    descriptor.m_OutputShapeEnabled = true;
    descriptor.m_OutputShape = { 1, 3, 3, 1 };

    // quantize data
    std::vector<T> qInputData          = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> qWeightsData        = armnnUtils::QuantizedVector<T>(weightsData, qScale, qOffset);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);

    using BT = ResolveType<ArmnnBType>;
    std::vector<BT> qBiasesData = armnnUtils::QuantizedVector<BT>(biasesData, qScale * qScale, 0);

    ConstTensor weights(weightsInfo, qWeightsData);
    ConstTensor biases(biasesInfo, qBiasesData);

    INetworkPtr network = CreateTransposeConvolution2dNetwork(descriptor,
                                                              inputInfo,
                                                              outputInfo,
                                                              weights,
                                                              Optional<ConstTensor>(biases));

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                { { 0, qInputData } },
                                                { { 0, qExpectedOutputData } },
                                                backends);
}
