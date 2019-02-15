//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ArmNN.hpp>
#include <armnn/INetwork.hpp>

#include "../Serializer.hpp"

#include <armnnDeserializer/IDeserializer.hpp>

#include <random>
#include <sstream>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <flatbuffers/idl.h>

using armnnDeserializer::IDeserializer;

namespace
{

armnn::INetworkPtr DeserializeNetwork(const std::string& serializerString)
{
    std::vector<std::uint8_t> const serializerVector{serializerString.begin(), serializerString.end()};
    return IDeserializer::Create()->CreateNetworkFromBinary(serializerVector);
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

template<typename DataType>
static std::vector<DataType> GenerateRandomData(size_t size)
{
    constexpr bool isIntegerType = std::is_integral<DataType>::value;
    using Distribution =
        typename std::conditional<isIntegerType,
                                  std::uniform_int_distribution<DataType>,
                                  std::uniform_real_distribution<DataType>>::type;

    static constexpr DataType lowerLimit = std::numeric_limits<DataType>::min();
    static constexpr DataType upperLimit = std::numeric_limits<DataType>::max();

    static Distribution distribution(lowerLimit, upperLimit);
    static std::default_random_engine generator;

    std::vector<DataType> randomData(size);
    std::generate(randomData.begin(), randomData.end(), []() { return distribution(generator); });

    return randomData;
}

void CheckDeserializedNetworkAgainstOriginal(const armnn::INetwork& deserializedNetwork,
                                             const armnn::INetwork& originalNetwork,
                                             const armnn::TensorShape& inputShape,
                                             const armnn::TensorShape& outputShape,
                                             armnn::LayerBindingId inputBindingId = 0,
                                             armnn::LayerBindingId outputBindingId = 0)
{
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);

    std::vector<armnn::BackendId> preferredBackends = { armnn::BackendId("CpuRef") };

    // Optimize original network
    armnn::IOptimizedNetworkPtr optimizedOriginalNetwork =
        armnn::Optimize(originalNetwork, preferredBackends, runtime->GetDeviceSpec());
    BOOST_CHECK(optimizedOriginalNetwork);

    // Optimize deserialized network
    armnn::IOptimizedNetworkPtr optimizedDeserializedNetwork =
        armnn::Optimize(deserializedNetwork, preferredBackends, runtime->GetDeviceSpec());
    BOOST_CHECK(optimizedDeserializedNetwork);

    armnn::NetworkId networkId1;
    armnn::NetworkId networkId2;

    // Load original and deserialized network
    armnn::Status status1 = runtime->LoadNetwork(networkId1, std::move(optimizedOriginalNetwork));
    BOOST_CHECK(status1 == armnn::Status::Success);

    armnn::Status status2 = runtime->LoadNetwork(networkId2, std::move(optimizedDeserializedNetwork));
    BOOST_CHECK(status2 == armnn::Status::Success);

    // Generate some input data
    std::vector<float> inputData = GenerateRandomData<float>(inputShape.GetNumElements());

    armnn::InputTensors inputTensors1
    {
         { 0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId1, inputBindingId), inputData.data()) }
    };

    armnn::InputTensors inputTensors2
    {
         { 0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId2, inputBindingId), inputData.data()) }
    };

    std::vector<float> outputData1(outputShape.GetNumElements());
    std::vector<float> outputData2(outputShape.GetNumElements());

    armnn::OutputTensors outputTensors1
    {
         { 0, armnn::Tensor(runtime->GetOutputTensorInfo(networkId1, outputBindingId), outputData1.data()) }
    };

    armnn::OutputTensors outputTensors2
    {
         { 0, armnn::Tensor(runtime->GetOutputTensorInfo(networkId2, outputBindingId), outputData2.data()) }
    };

    // Run original and deserialized network
    runtime->EnqueueWorkload(networkId1, inputTensors1, outputTensors1);
    runtime->EnqueueWorkload(networkId2, inputTensors2, outputTensors2);

    // Compare output data
    BOOST_CHECK_EQUAL_COLLECTIONS(outputData1.begin(), outputData1.end(),
                                  outputData2.begin(), outputData2.end());
}

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(SerializerTests)

BOOST_AUTO_TEST_CASE(SerializeAddition)
{
    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    armnn::IConnectableLayer* const additionLayer0 = network->AddAdditionLayer();
    inputLayer0->GetOutputSlot(0).Connect(additionLayer0->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(additionLayer0->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer0 = network->AddOutputLayer(0);
    additionLayer0->GetOutputSlot(0).Connect(outputLayer0->GetInputSlot(0));

    armnn::TensorShape shape{1U};
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    additionLayer0->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
}

BOOST_AUTO_TEST_CASE(SerializeMultiplication)
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

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    multiplicationLayer0->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find(multLayerName) != stream.str().npos);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeConvolution2d)
{
    armnn::TensorInfo inputInfo ({ 1, 5, 5, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo outputInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);

    armnn::TensorInfo weightsInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32);

    // Construct network
    armnn::INetworkPtr network = armnn::INetwork::Create();

    armnn::Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 1;
    descriptor.m_PadRight    = 1;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 1;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NHWC;

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);

    std::vector<float> biasesData = GenerateRandomData<float>(biasesInfo.GetNumElements());
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::IConnectableLayer* const inputLayer  = network->AddInputLayer(0, "input");
    armnn::IConnectableLayer* const convLayer   =
        network->AddConvolution2dLayer(descriptor, weights, biases, "convolution");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputInfo.GetShape(),
                                            outputInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeReshape)
{
    unsigned int inputShape[]  = { 1, 9 };
    unsigned int outputShape[] = { 3, 3 };

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

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputTensorInfo.GetShape(),
                                            outputTensorInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeDepthwiseConvolution2d)
{
    armnn::TensorInfo inputInfo ({ 1, 5, 5, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo outputInfo({ 1, 3, 3, 3 }, armnn::DataType::Float32);

    armnn::TensorInfo weightsInfo({ 1, 3, 3, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo biasesInfo ({ 3 }, armnn::DataType::Float32);

    armnn::DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NHWC;

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);

    std::vector<int32_t> biasesData = GenerateRandomData<int32_t>(biasesInfo.GetNumElements());
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const depthwiseConvLayer =
        network->AddDepthwiseConvolution2dLayer(descriptor, weights, biases, "depthwiseConv");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(depthwiseConvLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    depthwiseConvLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    depthwiseConvLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputInfo.GetShape(),
                                            outputInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeSoftmax)
{
    armnn::TensorInfo tensorInfo({1, 10}, armnn::DataType::Float32);

    armnn::SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const softmaxLayer = network->AddSoftmaxLayer(descriptor, "softmax");
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    softmaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    softmaxLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            tensorInfo.GetShape(),
                                            tensorInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializePooling2d)
{
    unsigned int inputShape[]  = {1, 2, 2, 1};
    unsigned int outputShape[] = {1, 1, 1, 1};

    auto inputInfo  = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

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
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    pooling2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    pooling2dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputInfo.GetShape(),
                                            outputInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializePermute)
{
    unsigned int inputShape[]  = { 4, 3, 2, 1 };
    unsigned int outputShape[] = { 1, 2, 3, 4 };
    unsigned int dimsMapping[] = { 3, 2, 1, 0 };

    auto inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    armnn::PermuteDescriptor permuteDescriptor(armnn::PermutationVector(dimsMapping, 4));

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer *const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer *const permuteLayer = network->AddPermuteLayer(permuteDescriptor, "PermuteLayer");
    armnn::IConnectableLayer *const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(permuteLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    permuteLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    permuteLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputTensorInfo.GetShape(),
                                            outputTensorInfo.GetShape());
}

BOOST_AUTO_TEST_SUITE_END()
