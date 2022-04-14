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

armnn::INetworkPtr CreateDepthwiseConvolution2dNetwork(const armnn::DepthwiseConvolution2dDescriptor& descriptor,
                                                       const armnn::TensorInfo& inputInfo,
                                                       const armnn::TensorInfo& weightsInfo,
                                                       const armnn::TensorInfo& biasInfo,
                                                       const armnn::TensorInfo& outputInfo,
                                                       const armnn::ConstTensor& weights,
                                                       const armnn::ConstTensor& biases)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());
    IConnectableLayer* input = network->AddInputLayer(0, "input");
    armnn::IConnectableLayer* weightsLayer = network->AddConstantLayer(weights, "Weights");
    armnn::IConnectableLayer* biasLayer = network->AddConstantLayer(biases, "Bias");
    IConnectableLayer* convolution2d = network->AddDepthwiseConvolution2dLayer(descriptor, "depthwiseConvolution2d");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, convolution2d, inputInfo, 0, 0);
    Connect(weightsLayer, convolution2d, weightsInfo, 0, 1);
    Connect(biasLayer, convolution2d, biasInfo, 0, 2);
    Connect(convolution2d, output, outputInfo, 0, 0);

    return network;
}

} // anonymous namespace

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType>
void DepthwiseConvolution2dEndToEnd(const std::vector<armnn::BackendId>& backends,
                                    armnn::DataLayout dataLayout)
{
    using namespace armnn;
    using T  = ResolveType<ArmnnType>;
    using BT = ResolveType<ArmnnBType>;

    const float   qScale  = IsQuantizedType<T>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<T>() ? 50    : 0;

    unsigned int depthMultiplier = 2;

    unsigned int inputHeight    = 8;
    unsigned int inputWidth     = 16;
    unsigned int inputChannels  = 2;
    unsigned int inputBatchSize = 1;

    unsigned int kernelHeight = 5;
    unsigned int kernelWidth  = 3;

    unsigned int outputHeight    = inputHeight - kernelHeight + 1 + 2;
    unsigned int outputWidth     = (inputWidth - kernelWidth + 1)/2;
    unsigned int outputChannels  = inputChannels * depthMultiplier;
    unsigned int outputBatchSize = inputBatchSize;

    TensorInfo inputInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth }, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth }, ArmnnType, qScale, qOffset);
    TensorInfo weightsInfo({1, kernelHeight, kernelWidth, outputChannels}, ArmnnType, qScale, qOffset, true);
    TensorInfo biasesInfo({outputChannels}, ArmnnBType, qScale * qScale, 0, true);

    std::vector<float> inputData =
    {
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
   };

    std::vector<float> weightsData =
    {
        1.0f,  1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,

        2.0f,  2.0f, 2.0f,
        2.0f,  2.0f, 2.0f,
        2.0f,  2.0f, 2.0f,
        2.0f,  2.0f, 2.0f,
        2.0f,  2.0f, 2.0f,

        0.0f,  0.0f, 0.0f,
        0.0f, -1.0f, 0.0f,
        0.0f,  0.0f, 0.0f,
        0.0f,  0.0f, 0.0f,
        0.0f,  0.0f, 0.0f,

        0.0f,  0.0f, 0.0f,
        0.0f,  0.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
        0.0f,  0.0f, 0.0f,
        0.0f,  0.0f, 0.0f
    };

    std::vector<float> biasesData = { 0.0f, 2.0f, 1.0f, -1.0f };

    std::vector<float> expectedOutputData =
    {
        3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f,
        5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 3.5f, 3.5f, 3.5f, 3.5f, 3.5f, 3.5f, 3.5f,
        4.5f, 4.5f, 4.5f, 4.5f, 4.5f, 4.5f, 4.5f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f,
        6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f,
        1.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        3.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        3.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
   };

    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 0;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 1;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = dataLayout;

    // Permute input and output if NCDHW.
    if (dataLayout == DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw(inputInfo, inputData);
        PermuteTensorNhwcToNchw(outputInfo, expectedOutputData);
    }

    // Quantize data
    std::vector<T> qInputData          = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> qWeightsData        = armnnUtils::QuantizedVector<T>(weightsData, qScale, qOffset);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);

    std::vector<BT> qBiasesData = armnnUtils::QuantizedVector<BT>(biasesData, qScale * qScale, 0);

    ConstTensor weights(weightsInfo, qWeightsData);
    ConstTensor biases(biasesInfo, qBiasesData);

    INetworkPtr network = CreateDepthwiseConvolution2dNetwork(descriptor,
                                                              inputInfo,
                                                              weightsInfo,
                                                              biasesInfo,
                                                              outputInfo,
                                                              weights,
                                                              biases);

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                { { 0, qInputData } },
                                                { { 0, qExpectedOutputData } },
                                                backends);
}
