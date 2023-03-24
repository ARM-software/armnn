//
// Copyright © 2017, 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/Types.hpp>
#include <Runtime.hpp>

#include <doctest/doctest.h>

TEST_SUITE("DebugCallback")
{
namespace
{

using namespace armnn;

INetworkPtr CreateSimpleNetwork()
{
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0, "Input");

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::ReLu;
    IConnectableLayer* activationLayer = net->AddActivationLayer(descriptor, "Activation:ReLu");

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 5 }, DataType::Float32));
    activationLayer->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 5 }, DataType::Float32));

    return net;
}

TEST_CASE("RuntimeRegisterDebugCallback")
{
    INetworkPtr net = CreateSimpleNetwork();

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Optimize the network with debug option
    OptimizerOptionsOpaque optimizerOptions(false, true);
    std::vector<BackendId> backends = { "CpuRef" };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);

    NetworkId netId;
    CHECK(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);

    // Set up callback function
    int callCount = 0;
    std::vector<TensorShape> tensorShapes;
    std::vector<unsigned int> slotIndexes;
    auto mockCallback = [&](LayerGuid guid, unsigned int slotIndex, ITensorHandle* tensor)
    {
        armnn::IgnoreUnused(guid);
        slotIndexes.push_back(slotIndex);
        tensorShapes.push_back(tensor->GetShape());
        callCount++;
    };

    runtime->RegisterDebugCallback(netId, mockCallback);

    std::vector<float> inputData({-2, -1, 0, 1, 2});
    std::vector<float> outputData(5);

    TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
    {
        {0, ConstTensor(inputTensorInfo, inputData.data())}
    };
    OutputTensors outputTensors
    {
        {0, Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Check that the callback was called twice
    CHECK(callCount == 2);

    // Check that tensor handles passed to callback have correct shapes
    const std::vector<TensorShape> expectedShapes({TensorShape({1, 1, 1, 5}), TensorShape({1, 1, 1, 5})});
    CHECK(tensorShapes == expectedShapes);

    // Check that slot indexes passed to callback are correct
    const std::vector<unsigned int> expectedSlotIndexes({0, 0});
    CHECK(slotIndexes == expectedSlotIndexes);
}

} // anonymous namespace

}
