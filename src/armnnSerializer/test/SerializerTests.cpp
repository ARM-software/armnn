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

using armnnDeserializeParser::IDeserializeParser;

namespace
{

armnn::INetworkPtr DeserializeNetwork(const std::string& serializerString)
{
    std::vector<std::uint8_t> const serializerVector{serializerString.begin(), serializerString.end()};
    return armnnDeserializeParser::IDeserializeParser::Create()->CreateNetworkFromBinary(serializerVector);
}

std::string SerializeNetwork(const armnn::INetwork& network)
{
    armnnSerializer::Serializer serializer;
    serializer.Serialize(network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);

    std::string serializerString{stream.str()};
    return serializerString;
}

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(SerializerTests)

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

BOOST_AUTO_TEST_CASE(SimpleReshapeIntegration)
{
    armnn::NetworkId networkIdentifier;
    armnn::IRuntime::CreationOptions options; // default options
    armnn::IRuntimePtr run = armnn::IRuntime::Create(options);

    unsigned int inputShape[] = {1, 9};
    unsigned int outputShape[] = {3, 3};

    auto inputTensorInfo = armnn::TensorInfo(2, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(2, outputShape, armnn::DataType::Float32);
    auto reshapeOutputTensorInfo = armnn::TensorInfo(2, outputShape, armnn::DataType::Float32);

    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = reshapeOutputTensorInfo.GetShape();

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer *const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer *const reshapeLayer = network->AddReshapeLayer(reshapeDescriptor, "ReshapeLayer");
    armnn::IConnectableLayer *const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(reshapeLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    reshapeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);
    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    std::string const serializerString{stream.str()};

    //Deserialize network.
    auto deserializedNetwork = DeserializeNetwork(serializerString);

    //Optimize the deserialized network
    auto deserializedOptimized = Optimize(*deserializedNetwork, {armnn::Compute::CpuRef},
                                          run->GetDeviceSpec());

    // Load graph into runtime
    run->LoadNetwork(networkIdentifier, std::move(deserializedOptimized));

    std::vector<float> input1Data(inputTensorInfo.GetNumElements());
    std::iota(input1Data.begin(), input1Data.end(), 8);

    armnn::InputTensors inputTensors
    {
         {0, armnn::ConstTensor(run->GetInputTensorInfo(networkIdentifier, 0), input1Data.data())}
    };

    std::vector<float> outputData(input1Data.size());
    armnn::OutputTensors outputTensors
    {
         {0,armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputData.data())}
    };

    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    BOOST_CHECK_EQUAL_COLLECTIONS(outputData.begin(),outputData.end(), input1Data.begin(),input1Data.end());
}

BOOST_AUTO_TEST_CASE(SimpleSoftmaxIntegration)
{
    armnn::TensorInfo tensorInfo({1, 10}, armnn::DataType::Float32);

    armnn::SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;

    // Create test network
    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const softmaxLayer = network->AddSoftmaxLayer(descriptor, "softmax");
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    softmaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    softmaxLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // Serialize & deserialize network
    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);

    armnn::IOptimizedNetworkPtr optimizedNetwork =
        armnn::Optimize(*network, {armnn::Compute::CpuRef}, runtime->GetDeviceSpec());
    BOOST_CHECK(optimizedNetwork);

    armnn::IOptimizedNetworkPtr deserializedOptimizedNetwork =
        armnn::Optimize(*deserializedNetwork, {armnn::Compute::CpuRef}, runtime->GetDeviceSpec());
    BOOST_CHECK(deserializedOptimizedNetwork);

    armnn::NetworkId networkId1;
    armnn::NetworkId networkId2;

    runtime->LoadNetwork(networkId1, std::move(optimizedNetwork));
    runtime->LoadNetwork(networkId2, std::move(deserializedOptimizedNetwork));

    std::vector<float> inputData(tensorInfo.GetNumElements());
    std::iota(inputData.begin(), inputData.end(), 0);

    armnn::InputTensors inputTensors1
    {
         {0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId1, 0), inputData.data())}
    };

    armnn::InputTensors inputTensors2
    {
         {0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId2, 0), inputData.data())}
    };

    std::vector<float> outputData1(inputData.size());
    std::vector<float> outputData2(inputData.size());

    armnn::OutputTensors outputTensors1
    {
         {0, armnn::Tensor(runtime->GetOutputTensorInfo(networkId1, 0), outputData1.data())}
    };

    armnn::OutputTensors outputTensors2
    {
         {0, armnn::Tensor(runtime->GetOutputTensorInfo(networkId2, 0), outputData2.data())}
    };

    runtime->EnqueueWorkload(networkId1, inputTensors1, outputTensors1);
    runtime->EnqueueWorkload(networkId2, inputTensors2, outputTensors2);

    BOOST_CHECK_EQUAL_COLLECTIONS(outputData1.begin(), outputData1.end(),
                                  outputData2.begin(), outputData2.end());
}

BOOST_AUTO_TEST_CASE(SimplePooling2dIntegration)
{
    armnn::NetworkId networkIdentifier;
    armnn::IRuntime::CreationOptions options; // default options
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);

    unsigned int inputShape[]  = {1, 2, 2, 1};
    unsigned int outputShape[] = {1, 1, 1, 1};

    auto inputTensorInfo  = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    armnn::Pooling2dDescriptor desc;
    desc.m_DataLayout          = armnn::DataLayout::NHWC;
    desc.m_PadTop              = 0;
    desc.m_PadBottom           = 0;
    desc.m_PadLeft             = 0;
    desc.m_PadRight            = 0;
    desc.m_PoolType            = armnn::PoolingAlgorithm::Average;
    desc.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    desc.m_PaddingMethod       = armnn::PaddingMethod::Exclude;
    desc.m_PoolHeight          = 2;
    desc.m_PoolWidth           = 2;
    desc.m_StrideX             = 2;
    desc.m_StrideY             = 2;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer *const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer *const pooling2dLayer = network->AddPooling2dLayer(desc, "ReshapeLayer");
    armnn::IConnectableLayer *const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(pooling2dLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    pooling2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    pooling2dLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto deserializeNetwork = DeserializeNetwork(SerializeNetwork(*network));

    //Optimize the deserialized network
    auto deserializedOptimized = Optimize(*deserializeNetwork, {armnn::Compute::CpuRef},
                                          runtime->GetDeviceSpec());

    // Load graph into runtime
    runtime->LoadNetwork(networkIdentifier, std::move(deserializedOptimized));

    std::vector<float> input1Data(inputTensorInfo.GetNumElements());
    std::iota(input1Data.begin(), input1Data.end(), 4);

    armnn::InputTensors inputTensors
    {
          {0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkIdentifier, 0), input1Data.data())}
    };

    std::vector<float> outputData(input1Data.size());
    armnn::OutputTensors outputTensors
    {
           {0, armnn::Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0), outputData.data())}
    };

    runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    BOOST_CHECK_EQUAL(outputData[0], 5.5);
}

BOOST_AUTO_TEST_SUITE_END()
