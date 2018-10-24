//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ArmNN.hpp>

#include <backends/test/QuantizeHelper.hpp>

#include <vector>

namespace
{

using namespace armnn;

template<typename T>
bool ConstantUsageTest(const std::vector<BackendId>& computeDevice,
                       const TensorInfo& commonTensorInfo,
                       const std::vector<T>& inputData,
                       const std::vector<T>& constantData,
                       const std::vector<T>& expectedOutputData)
{
    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);
    IConnectableLayer* constant = net->AddConstantLayer(ConstTensor(commonTensorInfo, constantData));
    IConnectableLayer* add = net->AddAdditionLayer();
    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the tensors in the network.
    input->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    constant->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, computeDevice, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    // Creates structures for input & output.
    std::vector<T> outputData(inputData.size());

    InputTensors inputTensors
    {
        {0, ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())}
    };
    OutputTensors outputTensors
    {
        {0, Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    return outputData == expectedOutputData;
}

inline bool ConstantUsageFloat32Test(const std::vector<BackendId>& backends)
{
    const TensorInfo commonTensorInfo({ 2, 3 }, DataType::Float32);

    return ConstantUsageTest(backends,
        commonTensorInfo,
        std::vector<float>{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }, // Input.
        std::vector<float>{ 6.f, 5.f, 4.f, 3.f, 2.f, 1.f }, // Const input.
        std::vector<float>{ 7.f, 7.f, 7.f, 7.f, 7.f, 7.f }  // Expected output.
    );
}

inline bool ConstantUsageUint8Test(const std::vector<BackendId>& backends)
{
    TensorInfo commonTensorInfo({ 2, 3 }, DataType::QuantisedAsymm8);

    const float scale = 0.023529f;
    const int8_t offset = -43;

    commonTensorInfo.SetQuantizationScale(scale);
    commonTensorInfo.SetQuantizationOffset(offset);

    return ConstantUsageTest(backends,
        commonTensorInfo,
        QuantizedVector<uint8_t>(scale, offset, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }), // Input.
        QuantizedVector<uint8_t>(scale, offset, { 6.f, 5.f, 4.f, 3.f, 2.f, 1.f }), // Const input.
        QuantizedVector<uint8_t>(scale, offset, { 7.f, 7.f, 7.f, 7.f, 7.f, 7.f })  // Expected output.
    );
}

} // anonymous namespace