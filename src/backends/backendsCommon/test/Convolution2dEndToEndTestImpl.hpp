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

armnn::INetworkPtr CreateConstConvolution2dNetwork(const armnn::Convolution2dDescriptor& descriptor,
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
    IConnectableLayer* weightsLayer = network->AddConstantLayer(weights, "Weights");
    IConnectableLayer* convolution2d = network->AddConvolution2dLayer(descriptor, "convolution2d");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, convolution2d, inputInfo, 0, 0);
    Connect(weightsLayer, convolution2d, weightsInfo, 0, 1);

    if(descriptor.m_BiasEnabled)
    {
        armnn::IConnectableLayer* biasLayer = network->AddConstantLayer(biases, "Bias");
        Connect(biasLayer, convolution2d, biasInfo, 0, 2);
    }

    Connect(convolution2d, output, outputInfo, 0, 0);

    return network;
}

template<DataType ArmnnIType, DataType ArmnnWType = ArmnnIType, DataType ArmnnBType = ArmnnIType,
        DataType ArmnnOType = ArmnnIType>
void Convolution2dEndToEnd(const std::vector<armnn::BackendId>& backends,
                           armnn::DataLayout dataLayout,
                           bool biasEnabled = true)
{
    using namespace armnn;
    using IT = ResolveType<ArmnnIType>;
    using WT = ResolveType<ArmnnWType>;
    using BT = ResolveType<ArmnnBType>;
    using OT = ResolveType<ArmnnOType>;

    const float   qScale  = 1.0f;
    const int32_t qOffset = IsQuantizedType<IT>() ? 10 : 0; // offset must be zero for non-quantized types

    TensorInfo inputInfo(  { 1, 5, 5, 1 }, ArmnnIType, qScale,          qOffset, true);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, ArmnnWType, qScale,          qOffset, true);
    TensorInfo biasesInfo( { 1 },          ArmnnBType, qScale * qScale, 0,       true);
    TensorInfo outputInfo( { 1, 3, 3, 1 }, ArmnnOType, qScale,          qOffset);

    std::vector<float> inputData =
            {
                    1, 5, 2, 3, 5,
                    8, 7, 3, 6, 3,
                    3, 3, 9, 1, 9,
                    4, 1, 8, 1, 3,
                    6, 8, 1, 9, 2
            };

    std::vector<float> weightsData =
            {
                    4, 5, 6,
                    0, 0, 0,
                    3, 2, 1
            };

    std::vector<float> biasesData = biasEnabled ? std::vector<float>({ 1.0f })
                                                : std::vector<float>({ 0.f });

    std::vector<float> expectedOutputData =
    {
         65 + biasesData[0], 76 + biasesData[0],  91 + biasesData[0],
        107 + biasesData[0], 99 + biasesData[0],  89 + biasesData[0],
        116 + biasesData[0], 98 + biasesData[0], 118 + biasesData[0]
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
        PermuteTensorNhwcToNchw(inputInfo,   inputData);
        PermuteTensorNhwcToNchw(weightsInfo, weightsData);
        PermuteTensorNhwcToNchw(outputInfo,  expectedOutputData);
    }

    // Convert data
    std::vector<IT> qInputData = armnnUtils::QuantizedVector<IT>(inputData, qScale, qOffset);
    std::vector<WT> qWeightsData = armnnUtils::QuantizedVector<WT>(weightsData, qScale, qOffset);
    std::vector<BT> qBiasesData = armnnUtils::QuantizedVector<BT>(biasesData, qScale * qScale, 0);
    std::vector<OT> qExpectedOutputData = armnnUtils::QuantizedVector<OT>(expectedOutputData, qScale, qOffset);

    ConstTensor weights(weightsInfo, qWeightsData);
    ConstTensor biases(biasesInfo, qBiasesData);

    INetworkPtr network = CreateConstConvolution2dNetwork(descriptor,
                                                          inputInfo,
                                                          weightsInfo,
                                                          biasesInfo,
                                                          outputInfo,
                                                          weights,
                                                          biases);

    EndToEndLayerTestImpl<ArmnnIType, ArmnnOType>(std::move(network),
                                                  {{ 0, qInputData }},
                                                  {{ 0, qExpectedOutputData }},
                                                  backends);
}

} // anonymous namespace
