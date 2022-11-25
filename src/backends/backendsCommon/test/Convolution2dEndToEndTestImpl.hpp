//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "EndToEndTestImpl.hpp"
#include <armnnUtils/QuantizeHelper.hpp>

#include <ResolveType.hpp>

#include <CommonTestUtils.hpp>
#include <armnnTestUtils/DataLayoutUtils.hpp>

#include <map>
#include <vector>

namespace
{

armnn::INetworkPtr CreateConstConvolution2dNetwork(const armnn::Convolution2dDescriptor& descriptor,
                                                   const armnn::TensorInfo& inputInfo,
                                                   const armnn::TensorInfo& weightsInfo,
                                                   const armnn::TensorInfo& biasInfo,
                                                   const armnn::TensorInfo& outputInfo,
                                                   const armnn::ConstTensor& weights,
                                                   const armnn::ConstTensor& biases,
                                                   bool biasEnabled)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());
    IConnectableLayer* input = network->AddInputLayer(0, "input");
    IConnectableLayer* weightsLayer = network->AddConstantLayer(weights, "Weights");
    IConnectableLayer* convolution2d = network->AddConvolution2dLayer(descriptor, "convolution2d");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, convolution2d, inputInfo, 0, 0);
    Connect(weightsLayer, convolution2d, weightsInfo, 0, 1);

    if(biasEnabled)
    {
        armnn::IConnectableLayer* biasLayer = network->AddConstantLayer(biases, "Bias");
        Connect(biasLayer, convolution2d, biasInfo, 0, 2);
    }

    Connect(convolution2d, output, outputInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void Convolution2dEndToEnd(const std::vector<armnn::BackendId>& backends,
                           armnn::DataLayout dataLayout,
                           bool biasEnabled = true)
{
    using namespace armnn;

    const float   qScale  = IsQuantizedType<T>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<T>() ? 50    : 0;

    TensorInfo inputInfo({ 1, 5, 5, 1 }, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({ 1, 3, 3, 1 }, ArmnnType, qScale, qOffset);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, ArmnnType, qScale, qOffset, true);
    TensorInfo biasesInfo({ 1 }, ArmnnType, qScale * qScale, 0, true);

    std::vector<float> inputData =
    {
        1.0f, 5.0f, 2.0f, 3.0f, 5.0f,
        8.0f, 7.0f, 3.0f, 6.0f, 3.0f,
        3.0f, 3.0f, 9.0f, 1.0f, 9.0f,
        4.0f, 1.0f, 8.0f, 1.0f, 3.0f,
        6.0f, 8.0f, 1.0f, 9.0f, 2.0f
    };

    std::vector<float> weightsData =
    {
        4.0f, 5.0f, 6.0f,
        0.0f, 0.0f, 0.0f,
        3.0f, 2.0f, 1.0f
    };

    std::vector<float> biasesData = { 1.0f };

    float bias = biasEnabled ? biasesData[0] : 0.0f;
    std::vector<float> expectedOutputData =
    {
        65.0f + bias,  76.0f + bias,  91.0f + bias,
        107.0f + bias, 99.0f + bias,  89.0f + bias,
        116.0f + bias, 98.0f + bias,  118.0f + bias,
    };

    Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_DataLayout  = dataLayout;

    if (dataLayout == DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw(inputInfo, inputData);
        PermuteTensorNhwcToNchw(weightsInfo, weightsData);
        PermuteTensorNhwcToNchw(outputInfo, expectedOutputData);
    }

    // Quantize data
    std::vector<T> qInputData          = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> qWeightsData        = armnnUtils::QuantizedVector<T>(weightsData, qScale, qOffset);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);
    std::vector<T> qBiasesData         = armnnUtils::QuantizedVector<T>(biasesData, qScale * qScale, 0);

    ConstTensor weights(weightsInfo, qWeightsData);
    ConstTensor biases(biasesInfo, qBiasesData);

    INetworkPtr network = CreateConstConvolution2dNetwork(descriptor,
                                                          inputInfo,
                                                          weightsInfo,
                                                          biasesInfo,
                                                          outputInfo,
                                                          weights,
                                                          biases,
                                                          biasEnabled);

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                {{ 0, qInputData }},
                                                {{ 0, qExpectedOutputData }},
                                                backends);
}

} // anonymous namespace
