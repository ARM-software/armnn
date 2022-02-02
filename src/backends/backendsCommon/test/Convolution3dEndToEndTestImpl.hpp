//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
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

armnn::INetworkPtr CreateConvolution3dNetwork(const armnn::Convolution3dDescriptor& descriptor,
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
    IConnectableLayer* convolution3d = network->AddConvolution3dLayer(descriptor, "convolution3d");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, convolution3d, inputInfo, 0, 0);
    Connect(weightsLayer, convolution3d, weightsInfo, 0, 1);
    Connect(biasLayer, convolution3d, biasInfo, 0, 2);
    Connect(convolution3d, output, outputInfo, 0, 0);

    return network;
}

} // anonymous namespace

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType>
void Convolution3dEndToEnd(const std::vector<armnn::BackendId>& backends,
                           armnn::DataLayout dataLayout)
{
    using namespace armnn;
    using T  = ResolveType<ArmnnType>;
    using BT = ResolveType<ArmnnBType>;

    const float   qScale  = IsQuantizedType<T>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<T>() ? 50    : 0;

    TensorInfo inputInfo({ 1, 5, 5, 5, 1 }, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({ 1, 2, 2, 2, 1 }, ArmnnType, qScale, qOffset);
    TensorInfo weightsInfo({ 3, 3, 3, 1, 1 }, ArmnnType, qScale, qOffset, true);
    TensorInfo biasesInfo({ 1 }, ArmnnBType, qScale * qScale, 0, true);

    std::vector<float> inputData =
    {
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f, 18.0f, 19.0f,

        20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
        30.0f, 31.0f, 32.0f, 33.0f, 34.0f,
        35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
        40.0f, 41.0f, 42.0f, 43.0f, 44.0f,

        45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
        50.0f, 51.0f, 52.0f, 53.0f, 54.0f,
        55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
        60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f,

        70.0f, 71.0f, 72.0f, 73.0f, 74.0f,
        75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
        80.0f, 81.0f, 82.0f, 83.0f, 84.0f,
        85.0f, 86.0f, 87.0f, 88.0f, 89.0f,
        90.0f, 91.0f, 92.0f, 93.0f, 94.0f,
        95.0f, 96.0f, 97.0f, 98.0f, 99.0f,

        100.0f, 101.0f, 102.0f, 103.0f, 104.0f,
        105.0f, 106.0f, 107.0f, 108.0f, 109.0f,
        110.0f, 111.0f, 112.0f, 113.0f, 114.0f,
        115.0f, 116.0f, 117.0f, 118.0f, 119.0f,
        120.0f, 121.0f, 122.0f, 123.0f, 124.0f
    };

    std::vector<float> weightsData =
    {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
    };

    std::vector<float> biasesData = { 1.f };

    std::vector<float> expectedOutputData =
    {
        559.0f, 595.0f,

        739.0f, 775.0f,

        1459.0f, 1495.0f,

        1639.0f, 1675.0f,
    };

    Convolution3dDescriptor descriptor;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_PadFront    = 0;
    descriptor.m_PadBack     = 0;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_StrideZ     = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = dataLayout;

    // Permute input and output if NCDHW.
    if (dataLayout == DataLayout::NCDHW)
    {
        PermuteTensorNdhwcToNcdhw(inputInfo, inputData);
        PermuteTensorNdhwcToNcdhw(outputInfo, expectedOutputData);
    }

    // Quantize data
    std::vector<T> qInputData          = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> qWeightsData        = armnnUtils::QuantizedVector<T>(weightsData, qScale, qOffset);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);

    std::vector<BT> qBiasesData = armnnUtils::QuantizedVector<BT>(biasesData, qScale * qScale, 0);

    ConstTensor weights(weightsInfo, qWeightsData);
    ConstTensor biases(biasesInfo, qBiasesData);

    INetworkPtr network = CreateConvolution3dNetwork(descriptor,
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
