//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once


#include <Runtime.hpp>

namespace
{

inline void CreateAndDropDummyNetwork(const std::vector<armnn::BackendId>& backends, armnn::RuntimeImpl& runtime)
{
    armnn::NetworkId networkIdentifier;
    {
        armnn::TensorInfo inputTensorInfo(armnn::TensorShape({ 7, 7 }), armnn::DataType::Float32);
        armnn::TensorInfo outputTensorInfo(armnn::TensorShape({ 7, 7 }), armnn::DataType::Float32);

        armnn::INetworkPtr network(armnn::INetwork::Create());

        armnn::IConnectableLayer* input = network->AddInputLayer(0, "input");
        armnn::IConnectableLayer* layer = network->AddActivationLayer(armnn::ActivationDescriptor(), "test");
        armnn::IConnectableLayer* output = network->AddOutputLayer(0, "output");

        input->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
        layer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        // Sets the tensors in the network.
        input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
        layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        // optimize the network
        armnn::IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime.GetDeviceSpec());

        runtime.LoadNetwork(networkIdentifier, std::move(optNet));
    }

    runtime.UnloadNetwork(networkIdentifier);
}

} // anonymous namespace
