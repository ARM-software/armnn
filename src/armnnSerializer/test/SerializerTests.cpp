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
    class VerifyAdditionName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitAdditionLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == "addition");
        }
    };

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    armnn::IConnectableLayer* const additionLayer = network->AddAdditionLayer("addition");
    inputLayer0->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);
    additionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    armnn::TensorShape shape{1U};
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    additionLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(stream.str());
    BOOST_CHECK(deserializedNetwork);

    VerifyAdditionName nameChecker;
    deserializedNetwork->Accept(nameChecker);
}

BOOST_AUTO_TEST_CASE(SerializeConstant)
{
    armnn::INetworkPtr network = armnn::INetwork::Create();

    armnn::ConstTensor inputTensor;

    armnn::IConnectableLayer* const inputLayer0 = network->AddConstantLayer(inputTensor, "constant");
    armnn::IConnectableLayer* const outputLayer0 = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(outputLayer0->GetInputSlot(0));

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find("constant") != stream.str().npos);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeConstant)
{
    class VerifyConstantName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitConstantLayer(const armnn::IConnectableLayer*, const armnn::ConstTensor&, const char* name) override
        {
            BOOST_TEST(name == "constant");
        }
    };

    armnn::TensorInfo commonTensorInfo({ 2, 3 }, armnn::DataType::Float32);

    std::vector<float> constantData = GenerateRandomData<float>(commonTensorInfo.GetNumElements());
    armnn::ConstTensor constTensor(commonTensorInfo, constantData);

    // Builds up the structure of the network.
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);
    armnn::IConnectableLayer* constant = net->AddConstantLayer(constTensor, "constant");
    armnn::IConnectableLayer* add = net->AddAdditionLayer();
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the tensors in the network.
    input->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    constant->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*net));
    BOOST_CHECK(deserializedNetwork);

    VerifyConstantName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*net,
                                            *deserializedNetwork,
                                            commonTensorInfo.GetShape(),
                                            commonTensorInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeMultiplication)
{
    class VerifyMultiplicationName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitMultiplicationLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == "multiplication");
        }
    };

    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    const char* multLayerName = "multiplication";

    armnn::IConnectableLayer* const multiplicationLayer = network->AddMultiplicationLayer(multLayerName);
    inputLayer0->GetOutputSlot(0).Connect(multiplicationLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(multiplicationLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);
    multiplicationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    multiplicationLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find(multLayerName) != stream.str().npos);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(stream.str());
    BOOST_CHECK(deserializedNetwork);

    VerifyMultiplicationName nameChecker;
    deserializedNetwork->Accept(nameChecker);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeConvolution2d)
{

    class VerifyConvolution2dName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitConvolution2dLayer(const armnn::IConnectableLayer*,
                                     const armnn::Convolution2dDescriptor&,
                                     const armnn::ConstTensor&,
                                     const armnn::Optional<armnn::ConstTensor>&,
                                     const char* name) override
        {
            BOOST_TEST(name == "convolution");
        }
    };

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

    VerifyConvolution2dName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputInfo.GetShape(),
                                            outputInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeReshape)
{
    class VerifyReshapeName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitReshapeLayer(const armnn::IConnectableLayer*, const armnn::ReshapeDescriptor&, const char* name)
        {
            BOOST_TEST(name == "reshape");
        }
    };

    unsigned int inputShape[]  = { 1, 9 };
    unsigned int outputShape[] = { 3, 3 };

    auto inputTensorInfo = armnn::TensorInfo(2, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(2, outputShape, armnn::DataType::Float32);
    auto reshapeOutputTensorInfo = armnn::TensorInfo(2, outputShape, armnn::DataType::Float32);

    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = reshapeOutputTensorInfo.GetShape();

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const reshapeLayer = network->AddReshapeLayer(reshapeDescriptor, "reshape");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(reshapeLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    reshapeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyReshapeName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputTensorInfo.GetShape(),
                                            outputTensorInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeDepthwiseConvolution2d)
{
    class VerifyDepthwiseConvolution2dName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitDepthwiseConvolution2dLayer(const armnn::IConnectableLayer*,
                                              const armnn::DepthwiseConvolution2dDescriptor&,
                                              const armnn::ConstTensor&,
                                              const armnn::Optional<armnn::ConstTensor>&,
                                              const char* name) override
        {
            BOOST_TEST(name == "depthwise_convolution");
        }
    };

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
        network->AddDepthwiseConvolution2dLayer(descriptor, weights, biases, "depthwise_convolution");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(depthwiseConvLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    depthwiseConvLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    depthwiseConvLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyDepthwiseConvolution2dName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputInfo.GetShape(),
                                            outputInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeSoftmax)
{
    class VerifySoftmaxName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitSoftmaxLayer(const armnn::IConnectableLayer*, const armnn::SoftmaxDescriptor&, const char* name)
        {
            BOOST_TEST(name == "softmax");
        }
    };

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

    VerifySoftmaxName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            tensorInfo.GetShape(),
                                            tensorInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializePooling2d)
{
    class VerifyPooling2dName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
        void VisitPooling2dLayer(const armnn::IConnectableLayer*, const armnn::Pooling2dDescriptor&, const char* name)
        {
            BOOST_TEST(name == "pooling2d");
        }
    };

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
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const pooling2dLayer = network->AddPooling2dLayer(desc, "pooling2d");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(pooling2dLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    pooling2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    pooling2dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyPooling2dName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputInfo.GetShape(),
                                            outputInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializePermute)
{
    class VerifyPermuteName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitPermuteLayer(const armnn::IConnectableLayer*, const armnn::PermuteDescriptor&, const char* name)
        {
            BOOST_TEST(name == "permute");
        }
    };

    unsigned int inputShape[]  = { 4, 3, 2, 1 };
    unsigned int outputShape[] = { 1, 2, 3, 4 };
    unsigned int dimsMapping[] = { 3, 2, 1, 0 };

    auto inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    armnn::PermuteDescriptor permuteDescriptor(armnn::PermutationVector(dimsMapping, 4));

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const permuteLayer = network->AddPermuteLayer(permuteDescriptor, "permute");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(permuteLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    permuteLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    permuteLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyPermuteName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputTensorInfo.GetShape(),
                                            outputTensorInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeFullyConnected)
{
    class VerifyFullyConnectedName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitFullyConnectedLayer(const armnn::IConnectableLayer*,
                                      const armnn::FullyConnectedDescriptor&,
                                      const armnn::ConstTensor&,
                                      const armnn::Optional<armnn::ConstTensor>&,
                                      const char* name) override
        {
            BOOST_TEST(name == "fully_connected");
        }
    };

    armnn::TensorInfo inputInfo ({ 2, 5, 1, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo outputInfo({ 2, 3 }, armnn::DataType::Float32);

    armnn::TensorInfo weightsInfo({ 5, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo biasesInfo ({ 3 }, armnn::DataType::Float32);

    armnn::FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = true;
    descriptor.m_TransposeWeightMatrix = false;

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    std::vector<float> biasesData  = GenerateRandomData<float>(biasesInfo.GetNumElements());

    armnn::ConstTensor weights(weightsInfo, weightsData);
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0, "input");
    armnn::IConnectableLayer* const fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor,
                                                                                          weights,
                                                                                          biases,
                                                                                          "fully_connected");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyFullyConnectedName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputInfo.GetShape(),
                                            outputInfo.GetShape());
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeSpaceToBatchNd)
{
    class VerifySpaceToBatchNdName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitSpaceToBatchNdLayer(const armnn::IConnectableLayer*,
                                      const armnn::SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                      const char* name) override
        {
            BOOST_TEST(name == "SpaceToBatchNdLayer");
        }
    };

    unsigned int inputShape[] = {2, 1, 2, 4};
    unsigned int outputShape[] = {8, 1, 1, 3};

    armnn::SpaceToBatchNdDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;
    desc.m_BlockShape = {2, 2};
    desc.m_PadList = {{0, 0}, {2, 0}};

    auto inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const spaceToBatchNdLayer = network->AddSpaceToBatchNdLayer(desc, "SpaceToBatchNdLayer");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(spaceToBatchNdLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    spaceToBatchNdLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    spaceToBatchNdLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifySpaceToBatchNdName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal(*network,
                                            *deserializedNetwork,
                                            inputTensorInfo.GetShape(),
                                            outputTensorInfo.GetShape());
}

BOOST_AUTO_TEST_SUITE_END()
