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

template<typename T>
void CheckDeserializedNetworkAgainstOriginal(const armnn::INetwork& deserializedNetwork,
                                             const armnn::INetwork& originalNetwork,
                                             const std::vector<armnn::TensorShape>& inputShapes,
                                             const std::vector<armnn::TensorShape>& outputShapes,
                                             const std::vector<armnn::LayerBindingId>& inputBindingIds = {0},
                                             const std::vector<armnn::LayerBindingId>& outputBindingIds = {0})
{
    BOOST_CHECK(inputShapes.size() == inputBindingIds.size());
    BOOST_CHECK(outputShapes.size() == outputBindingIds.size());

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
    armnn::InputTensors inputTensors1;
    armnn::InputTensors inputTensors2;
    std::vector<std::vector<T>> inputData;
    inputData.reserve(inputShapes.size());

    for (unsigned int i = 0; i < inputShapes.size(); i++) {
        inputData.push_back(GenerateRandomData<T>(inputShapes[i].GetNumElements()));

        inputTensors1.emplace_back(
            i, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId1, inputBindingIds[i]), inputData[i].data()));

        inputTensors2.emplace_back(
            i, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId2, inputBindingIds[i]), inputData[i].data()));
    }

    armnn::OutputTensors outputTensors1;
    armnn::OutputTensors outputTensors2;
    std::vector<std::vector<T>> outputData1;
    std::vector<std::vector<T>> outputData2;
    outputData1.reserve(outputShapes.size());
    outputData2.reserve(outputShapes.size());

    for (unsigned int i = 0; i < outputShapes.size(); i++)
    {
        outputData1.emplace_back(outputShapes[i].GetNumElements());
        outputData2.emplace_back(outputShapes[i].GetNumElements());

        outputTensors1.emplace_back(
            i, armnn::Tensor(runtime->GetOutputTensorInfo(networkId1, outputBindingIds[i]), outputData1[i].data()));

        outputTensors2.emplace_back(
            i, armnn::Tensor(runtime->GetOutputTensorInfo(networkId2, outputBindingIds[i]), outputData2[i].data()));
    }

    // Run original and deserialized network
    runtime->EnqueueWorkload(networkId1, inputTensors1, outputTensors1);
    runtime->EnqueueWorkload(networkId2, inputTensors2, outputTensors2);

    // Compare output data
    for (unsigned int i = 0; i < outputShapes.size(); i++)
    {
        BOOST_CHECK_EQUAL_COLLECTIONS(
            outputData1[i].begin(), outputData1[i].end(), outputData2[i].begin(), outputData2[i].end());
    }
}

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(SerializerTests)

BOOST_AUTO_TEST_CASE(SerializeDeserializeAddition)
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {info.GetShape(), info.GetShape()},
                                                   {info.GetShape()},
                                                   {0, 1});
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
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = network->AddInputLayer(0);
    armnn::IConnectableLayer* constant = network->AddConstantLayer(constTensor, "constant");
    armnn::IConnectableLayer* add = network->AddAdditionLayer();
    armnn::IConnectableLayer* output = network->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the tensors in the network.
    input->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    constant->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyConstantName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {commonTensorInfo.GetShape()},
                                                   {commonTensorInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeFloor)
{
    class VerifyFloorName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitMultiplicationLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == "floor");
        }
    };

    const armnn::TensorInfo info({4,4}, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);

    const char* floorLayerName = "floor";

    armnn::IConnectableLayer* const floorLayer = network->AddFloorLayer(floorLayerName);
    inputLayer->GetOutputSlot(0).Connect(floorLayer->GetInputSlot(0));

    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);
    floorLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    floorLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find(floorLayerName) != stream.str().npos);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(stream.str());
    BOOST_CHECK(deserializedNetwork);

    VerifyFloorName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {info.GetShape()},
                                                   {info.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeMinimum)
{
    class VerifyMinimumName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        explicit VerifyMinimumName(const std::string& expectedMinimumLayerName)
            : m_ExpectedMinimumLayerName(expectedMinimumLayerName) {}

        void VisitMinimumLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == m_ExpectedMinimumLayerName.c_str());
        }

    private:
        std::string m_ExpectedMinimumLayerName;
    };

    const armnn::TensorInfo info({ 1, 2, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    const std::string minimumLayerName("minimum");

    armnn::IConnectableLayer* const minimumLayer = network->AddMinimumLayer(minimumLayerName.c_str());
    inputLayer0->GetOutputSlot(0).Connect(minimumLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(minimumLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);
    minimumLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    minimumLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find(minimumLayerName) != stream.str().npos);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(stream.str());
    BOOST_CHECK(deserializedNetwork);

    VerifyMinimumName nameChecker(minimumLayerName);
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {info.GetShape(), info.GetShape()},
                                                   {info.GetShape()},
                                                   {0, 1});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeMaximum)
{
    class VerifyMaximumName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        explicit VerifyMaximumName(const std::string& expectedMaximumLayerName)
            : m_ExpectedMaximumLayerName(expectedMaximumLayerName) {}

        void VisitMaximumLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == m_ExpectedMaximumLayerName.c_str());
        }

    private:
        std::string m_ExpectedMaximumLayerName;
    };

    const armnn::TensorInfo info({ 1, 2, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    const std::string maximumLayerName("maximum");

    armnn::IConnectableLayer* const maximumLayer = network->AddMaximumLayer(maximumLayerName.c_str());
    inputLayer0->GetOutputSlot(0).Connect(maximumLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(maximumLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);
    maximumLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    maximumLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find(maximumLayerName) != stream.str().npos);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(stream.str());
    BOOST_CHECK(deserializedNetwork);

    VerifyMaximumName nameChecker(maximumLayerName);
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {info.GetShape(), info.GetShape()},
                                                   {info.GetShape()},
                                                   {0, 1});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeL2Normalization)
{
    class VerifyL2NormalizationName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        explicit VerifyL2NormalizationName(const std::string& expectedL2NormalizationLayerName)
        : m_ExpectedL2NormalizationLayerName(expectedL2NormalizationLayerName) {}

        void VisitL2NormalizationLayer(const armnn::IConnectableLayer*,
                                       const armnn::L2NormalizationDescriptor&,
                                       const char* name) override
        {
            BOOST_TEST(name == m_ExpectedL2NormalizationLayerName.c_str());
        }
    private:
        std::string m_ExpectedL2NormalizationLayerName;
    };

    const armnn::TensorInfo info({ 1, 2, 1, 5 }, armnn::DataType::Float32);

    armnn::L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);

    const char* l2NormLayerName = "l2Normalization";

    armnn::IConnectableLayer* const l2NormLayer = network->AddL2NormalizationLayer(desc, l2NormLayerName);
    inputLayer0->GetOutputSlot(0).Connect(l2NormLayer->GetInputSlot(0));

    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);
        l2NormLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    l2NormLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find(l2NormLayerName) != stream.str().npos);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(stream.str());
    BOOST_CHECK(deserializedNetwork);

    VerifyL2NormalizationName nameChecker(l2NormLayerName);
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*network,
                                                   *deserializedNetwork,
                                                   { info.GetShape() },
                                                   { info.GetShape() });
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeMultiplication)
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {info.GetShape(), info.GetShape()},
                                                   {info.GetShape()},
                                                   {0, 1});
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputInfo.GetShape()},
                                                   {outputInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeGreater)
{
    class VerifyGreaterName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitGreaterLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == "greater");
        }
    };

    const armnn::TensorInfo inputTensorInfo1({ 1, 2, 2, 2 }, armnn::DataType::Float32);
    const armnn::TensorInfo inputTensorInfo2({ 1, 2, 2, 2 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo({ 1, 2, 2, 2 }, armnn::DataType::Boolean);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer2 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const greaterLayer = network->AddGreaterLayer("greater");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer1->GetOutputSlot(0).Connect(greaterLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputTensorInfo1);
    inputLayer2->GetOutputSlot(0).Connect(greaterLayer->GetInputSlot(1));
    inputLayer2->GetOutputSlot(0).SetTensorInfo(inputTensorInfo2);
    greaterLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    greaterLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyGreaterName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo1.GetShape(), inputTensorInfo2.GetShape()},
                                                   {outputTensorInfo.GetShape()},
                                                   {0, 1});
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()});
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputInfo.GetShape()},
                                                   {outputInfo.GetShape()});
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {tensorInfo.GetShape()},
                                                   {tensorInfo.GetShape()});
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputInfo.GetShape()},
                                                   {outputInfo.GetShape()});
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()});
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputInfo.GetShape()},
                                                   {outputInfo.GetShape()});
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

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeGather)
{
    class VerifyGatherName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VerifyGatherLayer(const armnn::IConnectableLayer *, const char *name)
        {
            BOOST_TEST(name == "gatherLayer");
        }
    };

    armnn::TensorInfo paramsInfo({ 8 }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo indicesInfo({ 3 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 3 }, armnn::DataType::QuantisedAsymm8);

    paramsInfo.SetQuantizationScale(1.0f);
    paramsInfo.SetQuantizationOffset(0);
    outputInfo.SetQuantizationScale(1.0f);
    outputInfo.SetQuantizationOffset(0);

    const std::vector<int32_t>& indicesData = {7, 6, 5};
    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer *const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer *const constantLayer =
            network->AddConstantLayer(armnn::ConstTensor(indicesInfo, indicesData));
    armnn::IConnectableLayer *const gatherLayer = network->AddGatherLayer("gatherLayer");
    armnn::IConnectableLayer *const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(gatherLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(paramsInfo);
    constantLayer->GetOutputSlot(0).Connect(gatherLayer->GetInputSlot(1));
    constantLayer->GetOutputSlot(0).SetTensorInfo(indicesInfo);
    gatherLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    gatherLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyGatherName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<uint8_t>(*deserializedNetwork,
                                                     *network,
                                                     {paramsInfo.GetShape()},
                                                     {outputInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeBatchNormalization)
{
    class VerifyBatchNormalizationName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitBatchNormalizationLayer(const armnn::IConnectableLayer*,
                                      const armnn::BatchNormalizationDescriptor&,
                                      const armnn::ConstTensor&,
                                      const armnn::ConstTensor&,
                                      const armnn::ConstTensor&,
                                      const armnn::ConstTensor&,
                                      const char* name) override
        {
            BOOST_TEST(name == "BatchNormalizationLayer");
        }
    };

    armnn::TensorInfo inputInfo ({ 1, 3, 3, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo outputInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);

    armnn::TensorInfo meanInfo({1}, armnn::DataType::Float32);
    armnn::TensorInfo varianceInfo({1}, armnn::DataType::Float32);
    armnn::TensorInfo scaleInfo({1}, armnn::DataType::Float32);
    armnn::TensorInfo offsetInfo({1}, armnn::DataType::Float32);

    armnn::BatchNormalizationDescriptor descriptor;
    descriptor.m_Eps = 0.0010000000475f;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    std::vector<float> meanData({5.0});
    std::vector<float> varianceData({2.0});
    std::vector<float> scaleData({1.0});
    std::vector<float> offsetData({0.0});

    armnn::ConstTensor mean(meanInfo, meanData);
    armnn::ConstTensor variance(varianceInfo, varianceData);
    armnn::ConstTensor scale(scaleInfo, scaleData);
    armnn::ConstTensor offset(offsetInfo, offsetData);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const batchNormalizationLayer = network->AddBatchNormalizationLayer(
                                                                       descriptor,
                                                                       mean,
                                                                       variance,
                                                                       scale,
                                                                       offset,
                                                                       "BatchNormalizationLayer");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(batchNormalizationLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    batchNormalizationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    batchNormalizationLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyBatchNormalizationName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputInfo.GetShape()},
                                                   {outputInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeDivision)
{
    class VerifyDivisionName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitDivisionLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == "division");
        }
    };

    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);

    const char* divLayerName = "division";

    armnn::IConnectableLayer* const divisionLayer = network->AddDivisionLayer(divLayerName);
    inputLayer0->GetOutputSlot(0).Connect(divisionLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(divisionLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);
    divisionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    divisionLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnnSerializer::Serializer serializer;
    serializer.Serialize(*network);

    std::stringstream stream;
    serializer.SaveSerializedToStream(stream);
    BOOST_TEST(stream.str().length() > 0);
    BOOST_TEST(stream.str().find(divLayerName) != stream.str().npos);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(stream.str());
    BOOST_CHECK(deserializedNetwork);

    VerifyDivisionName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {info.GetShape(), info.GetShape()},
                                                   {info.GetShape()},
                                                   {0, 1});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeNormalization)
{
    class VerifyNormalizationName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
        public:
        void VisitNormalizationLayer(const armnn::IConnectableLayer*,
                                     const armnn::NormalizationDescriptor& normalizationDescriptor,
                                     const char* name) override
        {
            BOOST_TEST(name == "NormalizationLayer");
        }
    };

    unsigned int inputShape[] = {2, 1, 2, 2};
    unsigned int outputShape[] = {2, 1, 2, 2};

    armnn::NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;
    desc.m_NormSize = 3;
    desc.m_Alpha = 1;
    desc.m_Beta = 1;
    desc.m_K = 1;

    auto inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    auto outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const normalizationLayer = network->AddNormalizationLayer(desc, "NormalizationLayer");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(normalizationLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    normalizationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    normalizationLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyNormalizationName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeEqual)
{
    class VerifyEqualName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitEqualLayer(const armnn::IConnectableLayer*,
                             const char* name) override
        {
            BOOST_TEST(name == "EqualLayer");
        }
    };

    const armnn::TensorInfo inputTensorInfo1 = armnn::TensorInfo({2, 1, 2, 4}, armnn::DataType::Float32);
    const armnn::TensorInfo inputTensorInfo2 = armnn::TensorInfo({2, 1, 2, 4}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({2, 1, 2, 4}, armnn::DataType::Boolean);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer2 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const equalLayer = network->AddEqualLayer("EqualLayer");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer1->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputTensorInfo1);
    inputLayer2->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(1));
    inputLayer2->GetOutputSlot(0).SetTensorInfo(inputTensorInfo2);
    equalLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    equalLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyEqualName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo1.GetShape(), inputTensorInfo2.GetShape()},
                                                   {outputTensorInfo.GetShape()},
                                                   {0, 1});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializePad)
{
    class VerifyPadName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitPadLayer(const armnn::IConnectableLayer*,
                           const armnn::PadDescriptor& descriptor,
                           const char* name) override
        {
            BOOST_TEST(name == "PadLayer");
        }
    };

    armnn::PadDescriptor desc({{0, 0}, {1, 0}, {1, 1}, {1, 2}});

    const armnn::TensorInfo inputTensorInfo = armnn::TensorInfo({1, 2, 3, 4}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({1, 3, 5, 7}, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const padLayer = network->AddPadLayer(desc, "PadLayer");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(padLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    padLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    padLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyPadName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeRsqrt)
{
    class VerifyRsqrtName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitRsqrtLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == "rsqrt");
        }
    };

    const armnn::TensorInfo tensorInfo({ 3, 1, 2 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer  = network->AddInputLayer(0);
    armnn::IConnectableLayer* const rsqrtLayer  = network->AddRsqrtLayer("rsqrt");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(rsqrtLayer->GetInputSlot(0));
    rsqrtLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    rsqrtLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyRsqrtName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {tensorInfo.GetShape()},
                                                   {tensorInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeResizeBilinear)
{
    class VerifyResizeBilinearName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitResizeBilinearLayer(const armnn::IConnectableLayer*,
                                      const armnn::ResizeBilinearDescriptor& descriptor,
                                      const char* name) override
        {
            BOOST_TEST(name == "ResizeBilinearLayer");
        }
    };

    armnn::ResizeBilinearDescriptor desc;
    desc.m_TargetWidth = 4;
    desc.m_TargetHeight = 2;

    const armnn::TensorInfo inputTensorInfo = armnn::TensorInfo({1, 3, 5, 5}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({1, 3, 2, 4}, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const resizeLayer = network->AddResizeBilinearLayer(desc, "ResizeBilinearLayer");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(resizeLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    resizeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    resizeLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyResizeBilinearName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeSubtraction)
{
    class VerifySubtractionName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitSubtractionLayer(const armnn::IConnectableLayer*, const char* name) override
        {
            BOOST_TEST(name == "subtraction");
        }
    };

    const armnn::TensorInfo info = armnn::TensorInfo({ 1, 4 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const subtractionLayer = network->AddSubtractionLayer("subtraction");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(subtractionLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(subtractionLayer->GetInputSlot(1));
    subtractionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    subtractionLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifySubtractionName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {info.GetShape(), info.GetShape()},
                                                   {info.GetShape()},
                                                   {0, 1});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeStridedSlice)
{
    class VerifyStridedSliceName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitStridedSliceLayer(const armnn::IConnectableLayer*,
                                    const armnn::StridedSliceDescriptor& descriptor,
                                    const char* name) override
        {
            BOOST_TEST(name == "StridedSliceLayer");
        }
    };

    armnn::StridedSliceDescriptor desc({0, 0, 1, 0}, {1, 1, 1, 1}, {1, 1, 1, 1});
    desc.m_EndMask = (1 << 4) - 1;
    desc.m_ShrinkAxisMask = (1 << 1) | (1 << 2);
    desc.m_DataLayout = armnn::DataLayout::NCHW;

    const armnn::TensorInfo inputTensorInfo = armnn::TensorInfo({3, 2, 3, 1}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({3, 1}, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const stridedSliceLayer = network->AddStridedSliceLayer(desc, "StridedSliceLayer");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(stridedSliceLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    stridedSliceLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    stridedSliceLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyStridedSliceName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeMean)
{
    class VerifyMeanName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitMeanLayer(const armnn::IConnectableLayer*, const armnn::MeanDescriptor&, const char* name)
        {
            BOOST_TEST(name == "mean");
        }
    };

    armnn::TensorInfo inputTensorInfo({1, 1, 3, 2}, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo({1, 1, 1, 2}, armnn::DataType::Float32);

    armnn::MeanDescriptor descriptor;
    descriptor.m_Axis = { 2 };
    descriptor.m_KeepDims = true;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const meanLayer = network->AddMeanLayer(descriptor, "mean");
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(meanLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    meanLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    meanLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyMeanName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()});
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeMerger)
{
    class VerifyMergerName : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
    {
    public:
        void VisitMergerLayer(const armnn::IConnectableLayer* layer,
                              const armnn::OriginsDescriptor& mergerDescriptor,
                              const char* name = nullptr) override
        {
            BOOST_TEST(name == "MergerLayer");
        }
    };

    unsigned int inputShapeOne[] = {2, 3, 2, 2};
    unsigned int inputShapeTwo[] = {2, 3, 2, 2};
    unsigned int outputShape[] = {4, 3, 2, 2};

    const armnn::TensorInfo inputOneTensorInfo = armnn::TensorInfo(4, inputShapeOne, armnn::DataType::Float32);
    const armnn::TensorInfo inputTwoTensorInfo = armnn::TensorInfo(4, inputShapeTwo, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<armnn::TensorShape> shapes;
    shapes.push_back(inputOneTensorInfo.GetShape());
    shapes.push_back(inputTwoTensorInfo.GetShape());

    armnn::MergerDescriptor descriptor =
        armnn::CreateMergerDescriptorForConcatenation(shapes.begin(), shapes.end(), 0);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayerOne = network->AddInputLayer(0);
    inputLayerOne->GetOutputSlot(0).SetTensorInfo(inputOneTensorInfo);
    armnn::IConnectableLayer* const inputLayerTwo = network->AddInputLayer(1);
    inputLayerTwo->GetOutputSlot(0).SetTensorInfo(inputTwoTensorInfo);
    armnn::IConnectableLayer* const mergerLayer = network->AddMergerLayer(descriptor, "MergerLayer");
    mergerLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayerOne->GetOutputSlot(0).Connect(mergerLayer->GetInputSlot(0));
    inputLayerTwo->GetOutputSlot(0).Connect(mergerLayer->GetInputSlot(1));
    mergerLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyMergerName nameChecker;
    deserializedNetwork->Accept(nameChecker);

    CheckDeserializedNetworkAgainstOriginal<float>(*deserializedNetwork,
                                                   *network,
                                                   {inputOneTensorInfo.GetShape(), inputTwoTensorInfo.GetShape()},
                                                   {outputTensorInfo.GetShape()},
                                                   {0, 1});
}

BOOST_AUTO_TEST_SUITE_END()
