//
// Copyright © 2022, 2024 Arm Ltd and Contributors. All rights reserved.
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
    IConnectableLayer* convolution2d = network->AddDepthwiseConvolution2dLayer(descriptor, "depthwiseConvolution2d");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, convolution2d, inputInfo, 0, 0);
    Connect(weightsLayer, convolution2d, weightsInfo, 0, 1);
    Connect(convolution2d, output, outputInfo, 0, 0);

    if(descriptor.m_BiasEnabled)
    {
        armnn::IConnectableLayer* biasLayer = network->AddConstantLayer(biases, "Bias");
        Connect(biasLayer, convolution2d, biasInfo, 0, 2);
    }

    return network;
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType = ArmnnType>
void DepthwiseConvolution2dEndToEnd(const std::vector<armnn::BackendId>& backends,
                                    DepthwiseConvolution2dDescriptor & descriptor,
                                    std::vector<float>& inputData,
                                    std::vector<float>& expectedOutputData,
                                    float quantizedScale=0.25f,
                                    int32_t quantizedBias=50)
{
    using namespace armnn;
    using T  = ResolveType<ArmnnType>;
    using BT = ResolveType<ArmnnBType>;

    const float   qScale  = IsQuantizedType<T>() ? quantizedScale : 1.0f;
    const int32_t qOffset = IsQuantizedType<T>() ? quantizedBias    : 0;

    unsigned int depthMultiplier = 2;

    unsigned int inputHeight    = 8;
    unsigned int inputWidth     = 16;
    unsigned int inputChannels  = 2;
    unsigned int inputBatchSize = 1;

    unsigned int kernelHeight = 5;
    unsigned int kernelWidth  = 3;

    unsigned int outputHeight    = inputHeight - kernelHeight + 1 + 2;
    unsigned int outputWidth     = inputWidth - kernelWidth + 1;
    unsigned int outputChannels  = inputChannels * depthMultiplier;
    unsigned int outputBatchSize = inputBatchSize;

    TensorInfo inputInfo({ inputBatchSize, inputHeight, inputWidth, inputChannels }, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({ outputBatchSize, outputHeight, outputWidth, outputChannels }, ArmnnType, qScale, qOffset);
    TensorInfo weightsInfo({ 1, kernelHeight, kernelWidth, outputChannels }, ArmnnType, qScale, qOffset, true);
    TensorInfo biasesInfo({ outputChannels }, ArmnnBType, qScale * qScale, 0, true);

    // Layer weights and biases data
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

    // Permute input if NCHW, the original input and output are in NHWC format.
    if (descriptor.m_DataLayout == DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw(inputInfo, inputData);
        PermuteTensorNhwcToNchw(outputInfo, expectedOutputData);
        PermuteTensorNhwcToNchw(weightsInfo, weightsData);
    }

    // Quantize data
    std::vector<T> qInputData          = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> qWeightsData        = armnnUtils::QuantizedVector<T>(weightsData, qScale, qOffset);
    std::vector<BT> qBiasesData        = armnnUtils::QuantizedVector<BT>(biasesData, qScale * qScale, 0);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);

    ConstTensor weights(weightsInfo, qWeightsData);
    ConstTensor biases(biasesInfo, qBiasesData);

    // Create the network
    INetworkPtr network = CreateDepthwiseConvolution2dNetwork(descriptor,
                                                              inputInfo,
                                                              weightsInfo,
                                                              biasesInfo,
                                                              outputInfo,
                                                              weights,
                                                              biases);

    // Run the end-to-end test
    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                { { 0, qInputData } },
                                                { { 0, qExpectedOutputData } },
                                                backends);
}

} // anonymous namespace
