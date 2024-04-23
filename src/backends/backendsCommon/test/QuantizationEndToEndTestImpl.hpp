//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>

#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

armnn::INetworkPtr CreateQuantizationNetwork(const armnn::TensorInfo& inputInfo,
                                             const armnn::TensorInfo& outputInfo)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    IConnectableLayer *input= network->AddInputLayer(0, "input");
    IConnectableLayer *quantization = network->AddQuantizeLayer("quantization");
    IConnectableLayer *output = network->AddOutputLayer(0, "output");

    Connect(input, quantization, inputInfo, 0, 0);
    Connect(quantization, output, outputInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnIType, armnn::DataType ArmnnOType,
        typename Tin = armnn::ResolveType<ArmnnIType>, typename Tout = armnn::ResolveType<ArmnnOType>>
void QuantizeEndToEndLayerTestImpl(const std::vector<armnn::BackendId>& backends,
                                   const armnn::TensorShape& tensorShape,
                                   const std::vector<Tin>& input,
                                   const std::vector<Tout>& expectedOutput,
                                   float scale,
                                   int32_t offset)
{
    using namespace armnn;

    TensorInfo inputInfo(tensorShape, ArmnnIType);
    TensorInfo outputInfo(tensorShape, ArmnnOType, scale, offset);

    inputInfo.SetConstant(true);

    // Builds up the structure of the network
    INetworkPtr net = CreateQuantizationNetwork(inputInfo, outputInfo);

    CHECK(net);

    const std::map<int, std::vector<Tin>> inputTensorData = { { 0, input } };
    const std::map<int, std::vector<Tout>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnIType, ArmnnOType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnOType, typename Tout = armnn::ResolveType<ArmnnOType>>
void QuantizationEndToEndFloat32(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape tensorShape({ 1, 1, 1, 5 });

    std::vector<float> inputData = { 63.5f, 49.5f, 14.0f, 0.0f, 50.0f };

    float qScale = 0.5f;
    int32_t qOffset = 127;
    std::vector<Tout> expectedOutputData = armnnUtils::QuantizedVector<Tout>(inputData, qScale, qOffset);

    QuantizeEndToEndLayerTestImpl<DataType::Float32, ArmnnOType>(backends,
                                                                 tensorShape,
                                                                 inputData,
                                                                 expectedOutputData,
                                                                 qScale,
                                                                 qOffset);
};

template<armnn::DataType ArmnnOType, typename Tout = armnn::ResolveType<ArmnnOType>>
void QuantizationEndToEndFloat16(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;
    using namespace half_float::literal;
    using Half = half_float::half;

    const TensorShape tensorShape({ 1, 1, 1, 5 });

    std::vector<float> floatInputData = { 63.f, 49.f, 14.f, 0.f, 50.f };
    std::vector<Half> inputData = { 63._h, 49._h, 14._h, 0._h, 50._h };

    float qScale = 0.25f;
    int32_t qOffset = 1;
    std::vector<Tout> expectedOutputData = armnnUtils::QuantizedVector<Tout>(floatInputData, qScale, qOffset);

    QuantizeEndToEndLayerTestImpl<DataType::Float16, ArmnnOType>(backends,
                                                                 tensorShape,
                                                                 inputData,
                                                                 expectedOutputData,
                                                                 qScale,
                                                                 qOffset);
};

inline void QuantizationEndToEndInt8(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape tensorShape({ 1, 1, 1, 5 });

    std::vector<int8_t> inputData = { 113, 16, 13, 101, 13 };
    std::vector<int8_t> expectedOutputData = { 127, 45, 41, 127, 41 };

    float qScale = 0.75f;
    int32_t qOffset = 24;

    QuantizeEndToEndLayerTestImpl<DataType::QSymmS8, DataType::QSymmS8>(backends,
                                                                        tensorShape,
                                                                        inputData,
                                                                        expectedOutputData,
                                                                        qScale,
                                                                        qOffset);
};

}