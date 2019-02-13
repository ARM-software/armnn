//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ArmNN.hpp>
#include <armnn/INetwork.hpp>

#include "../Serializer.hpp"

#include <armnnDeserializeParser/IDeserializeParser.hpp>

#include <numeric>
#include <sstream>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <flatbuffers/idl.h>

BOOST_AUTO_TEST_SUITE(SerializerTests)

armnnDeserializeParser::IDeserializeParserPtr g_Parser = armnnDeserializeParser::IDeserializeParser::Create();

BOOST_AUTO_TEST_CASE(SimpleNetworkSerialization)
{
    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    armnn::IConnectableLayer* const additionLayer0 = network->AddAdditionLayer();
    inputLayer0->GetOutputSlot(0).Connect(additionLayer0->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(additionLayer0->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer0 = network->AddOutputLayer(0);
    additionLayer0->GetOutputSlot(0).Connect(outputLayer0->GetInputSlot(0));

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
}

BOOST_AUTO_TEST_CASE(SimpleNetworkWithMultiplicationSerialization)
{
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    const char* multLayerName = "mult_0";

    armnn::IConnectableLayer* const multiplicationLayer0 = network->AddMultiplicationLayer(multLayerName);
    inputLayer0->GetOutputSlot(0).Connect(multiplicationLayer0->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(multiplicationLayer0->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer0 = network->AddOutputLayer(0);
    multiplicationLayer0->GetOutputSlot(0).Connect(outputLayer0->GetInputSlot(0));

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find(multLayerName) != stream.str().npos);
}

BOOST_AUTO_TEST_CASE(SimpleSoftmaxIntegration)
{
    armnn::TensorInfo tensorInfo({1, 10}, armnn::DataType::Float32);

    armnn::SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;

    // Create test network
    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer *const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer *const softmaxLayer = network->AddSoftmaxLayer(descriptor, "softmax");
    armnn::IConnectableLayer *const outputLayer  = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    softmaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    softmaxLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // Serialize
    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);
    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    const std::string serializerString{stream.str()};

    // Deserialize
    armnn::INetworkPtr deserializedNetwork =
        g_Parser->CreateNetworkFromBinary({serializerString.begin(), serializerString.end()});
    BOOST_CHECK(deserializedNetwork);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr run = armnn::IRuntime::Create(options);

    armnn::IOptimizedNetworkPtr optimizedNetwork =
        armnn::Optimize(*network, {armnn::Compute::CpuRef}, run->GetDeviceSpec());
    BOOST_CHECK(optimizedNetwork);

    armnn::IOptimizedNetworkPtr deserializedOptimizedNetwork =
        armnn::Optimize(*deserializedNetwork, {armnn::Compute::CpuRef}, run->GetDeviceSpec());
    BOOST_CHECK(deserializedOptimizedNetwork);

    armnn::NetworkId networkId1;
    armnn::NetworkId networkId2;

    run->LoadNetwork(networkId1, std::move(optimizedNetwork));
    run->LoadNetwork(networkId2, std::move(deserializedOptimizedNetwork));

    std::vector<float> inputData(tensorInfo.GetNumElements());
    std::iota(inputData.begin(), inputData.end(), 0);

    armnn::InputTensors inputTensors1
    {
         {0, armnn::ConstTensor(run->GetInputTensorInfo(networkId1, 0), inputData.data())}
    };

    armnn::InputTensors inputTensors2
    {
         {0, armnn::ConstTensor(run->GetInputTensorInfo(networkId2, 0), inputData.data())}
    };

    std::vector<float> outputData1(inputData.size());
    std::vector<float> outputData2(inputData.size());

    armnn::OutputTensors outputTensors1
    {
         {0, armnn::Tensor(run->GetOutputTensorInfo(networkId1, 0), outputData1.data())}
    };

    armnn::OutputTensors outputTensors2
    {
         {0, armnn::Tensor(run->GetOutputTensorInfo(networkId2, 0), outputData2.data())}
    };

    run->EnqueueWorkload(networkId1, inputTensors1, outputTensors1);
    run->EnqueueWorkload(networkId2, inputTensors2, outputTensors2);

    BOOST_CHECK_EQUAL_COLLECTIONS(outputData1.begin(), outputData1.end(),
                                  outputData2.begin(), outputData2.end());
}

BOOST_AUTO_TEST_SUITE_END()
