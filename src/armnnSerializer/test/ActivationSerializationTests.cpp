//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Serializer.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>

#include <sstream>

TEST_SUITE("SerializerTests")
{
class VerifyActivationName : public armnn::IStrategy
{
public:
    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        IgnoreUnused(layer, descriptor, constants, id);
        if (layer->GetType() == armnn::LayerType::Activation)
        {
            CHECK(std::string(name) == "activation");
        }
    }
};

TEST_CASE("ActivationSerialization")
{
    armnnDeserializer::IDeserializerPtr parser = armnnDeserializer::IDeserializer::Create();

    armnn::TensorInfo inputInfo(armnn::TensorShape({1, 2, 2, 1}), armnn::DataType::Float32, 1.0f, 0);
    armnn::TensorInfo outputInfo(armnn::TensorShape({1, 2, 2, 1}), armnn::DataType::Float32, 4.0f, 0);

    // Construct network
    armnn::INetworkPtr network = armnn::INetwork::Create();

    armnn::ActivationDescriptor descriptor;
    descriptor.m_Function = armnn::ActivationFunction::ReLu;
    descriptor.m_A = 0;
    descriptor.m_B = 0;

    armnn::IConnectableLayer* const inputLayer      = network->AddInputLayer(0, "input");
    armnn::IConnectableLayer* const activationLayer = network->AddActivationLayer(descriptor, "activation");
    armnn::IConnectableLayer* const outputLayer     = network->AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    activationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnnSerializer::ISerializerPtr serializer = armnnSerializer::ISerializer::Create();

    serializer->Serialize(*network);

    std::stringstream stream;
    serializer->SaveSerializedToStream(stream);

    std::string const serializerString{stream.str()};
    std::vector<std::uint8_t> const serializerVector{serializerString.begin(), serializerString.end()};

    armnn::INetworkPtr deserializedNetwork = parser->CreateNetworkFromBinary(serializerVector);

    VerifyActivationName visitor;
    deserializedNetwork->ExecuteStrategy(visitor);

    armnn::IRuntime::CreationOptions options; // default options
    armnn::IRuntimePtr run = armnn::IRuntime::Create(options);
    auto deserializedOptimized = Optimize(*deserializedNetwork, { armnn::Compute::CpuRef }, run->GetDeviceSpec());

    armnn::NetworkId networkIdentifier;

    // Load graph into runtime
    run->LoadNetwork(networkIdentifier, std::move(deserializedOptimized));

    std::vector<float> inputData {0.0f, -5.3f, 42.0f, -42.0f};
    armnn::TensorInfo inputTensorInfo = run->GetInputTensorInfo(networkIdentifier, 0);
    inputTensorInfo.SetConstant(true);
    armnn::InputTensors inputTensors
    {
        {0, armnn::ConstTensor(inputTensorInfo, inputData.data())}
    };

    std::vector<float> expectedOutputData {0.0f, 0.0f, 42.0f, 0.0f};

    std::vector<float> outputData(4);
    armnn::OutputTensors outputTensors
    {
        {0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputData.data())}
    };
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
    CHECK(std::equal(outputData.begin(), outputData.end(), expectedOutputData.begin(), expectedOutputData.end()));
}

}
