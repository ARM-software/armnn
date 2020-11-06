//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Serializer.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/QuantizedLstmParams.hpp>
#include <armnnDeserializer/IDeserializer.hpp>

#include <random>
#include <vector>

#include <boost/test/unit_test.hpp>

using armnnDeserializer::IDeserializer;

namespace
{

#define DECLARE_LAYER_VERIFIER_CLASS(name) \
class name##LayerVerifier : public LayerVerifierBase \
{ \
public: \
    name##LayerVerifier(const std::string& layerName, \
                        const std::vector<armnn::TensorInfo>& inputInfos, \
                        const std::vector<armnn::TensorInfo>& outputInfos) \
        : LayerVerifierBase(layerName, inputInfos, outputInfos) {} \
\
    void Visit##name##Layer(const armnn::IConnectableLayer* layer, const char* name) override \
    { \
        VerifyNameAndConnections(layer, name); \
    } \
};

#define DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(name) \
class name##LayerVerifier : public LayerVerifierBaseWithDescriptor<armnn::name##Descriptor> \
{ \
public: \
    name##LayerVerifier(const std::string& layerName, \
                        const std::vector<armnn::TensorInfo>& inputInfos, \
                        const std::vector<armnn::TensorInfo>& outputInfos, \
                        const armnn::name##Descriptor& descriptor) \
        : LayerVerifierBaseWithDescriptor<armnn::name##Descriptor>( \
            layerName, inputInfos, outputInfos, descriptor) {} \
\
    void Visit##name##Layer(const armnn::IConnectableLayer* layer, \
                            const armnn::name##Descriptor& descriptor, \
                            const char* name) override \
    { \
        VerifyNameAndConnections(layer, name); \
        VerifyDescriptor(descriptor); \
    } \
};

struct DefaultLayerVerifierPolicy
{
    static void Apply(const std::string)
    {
        BOOST_TEST_MESSAGE("Unexpected layer found in network");
        BOOST_TEST(false);
    }
};

class LayerVerifierBase : public armnn::LayerVisitorBase<DefaultLayerVerifierPolicy>
{
public:
    LayerVerifierBase(const std::string& layerName,
                      const std::vector<armnn::TensorInfo>& inputInfos,
                      const std::vector<armnn::TensorInfo>& outputInfos)
    : m_LayerName(layerName)
    , m_InputTensorInfos(inputInfos)
    , m_OutputTensorInfos(outputInfos) {}

    void VisitInputLayer(const armnn::IConnectableLayer*, armnn::LayerBindingId, const char*) override {}

    void VisitOutputLayer(const armnn::IConnectableLayer*, armnn::LayerBindingId, const char*) override {}

protected:
    void VerifyNameAndConnections(const armnn::IConnectableLayer* layer, const char* name)
    {
        BOOST_TEST(name == m_LayerName.c_str());

        BOOST_TEST(layer->GetNumInputSlots() == m_InputTensorInfos.size());
        BOOST_TEST(layer->GetNumOutputSlots() == m_OutputTensorInfos.size());

        for (unsigned int i = 0; i < m_InputTensorInfos.size(); i++)
        {
            const armnn::IOutputSlot* connectedOutput = layer->GetInputSlot(i).GetConnection();
            BOOST_CHECK(connectedOutput);

            const armnn::TensorInfo& connectedInfo = connectedOutput->GetTensorInfo();
            BOOST_TEST(connectedInfo.GetShape() == m_InputTensorInfos[i].GetShape());
            BOOST_TEST(
                GetDataTypeName(connectedInfo.GetDataType()) == GetDataTypeName(m_InputTensorInfos[i].GetDataType()));

            BOOST_TEST(connectedInfo.GetQuantizationScale() == m_InputTensorInfos[i].GetQuantizationScale());
            BOOST_TEST(connectedInfo.GetQuantizationOffset() == m_InputTensorInfos[i].GetQuantizationOffset());
        }

        for (unsigned int i = 0; i < m_OutputTensorInfos.size(); i++)
        {
            const armnn::TensorInfo& outputInfo = layer->GetOutputSlot(i).GetTensorInfo();
            BOOST_TEST(outputInfo.GetShape() == m_OutputTensorInfos[i].GetShape());
            BOOST_TEST(
                GetDataTypeName(outputInfo.GetDataType()) == GetDataTypeName(m_OutputTensorInfos[i].GetDataType()));

            BOOST_TEST(outputInfo.GetQuantizationScale() == m_OutputTensorInfos[i].GetQuantizationScale());
            BOOST_TEST(outputInfo.GetQuantizationOffset() == m_OutputTensorInfos[i].GetQuantizationOffset());
        }
    }

    void VerifyConstTensors(const std::string& tensorName,
                            const armnn::ConstTensor* expectedPtr,
                            const armnn::ConstTensor* actualPtr)
    {
        if (expectedPtr == nullptr)
        {
            BOOST_CHECK_MESSAGE(actualPtr == nullptr, tensorName + " should not exist");
        }
        else
        {
            BOOST_CHECK_MESSAGE(actualPtr != nullptr, tensorName + " should have been set");
            if (actualPtr != nullptr)
            {
                const armnn::TensorInfo& expectedInfo = expectedPtr->GetInfo();
                const armnn::TensorInfo& actualInfo = actualPtr->GetInfo();

                BOOST_CHECK_MESSAGE(expectedInfo.GetShape() == actualInfo.GetShape(),
                                    tensorName + " shapes don't match");
                BOOST_CHECK_MESSAGE(
                        GetDataTypeName(expectedInfo.GetDataType()) == GetDataTypeName(actualInfo.GetDataType()),
                        tensorName + " data types don't match");

                BOOST_CHECK_MESSAGE(expectedPtr->GetNumBytes() == actualPtr->GetNumBytes(),
                                    tensorName + " (GetNumBytes) data sizes do not match");
                if (expectedPtr->GetNumBytes() == actualPtr->GetNumBytes())
                {
                    //check the data is identical
                    const char* expectedData = static_cast<const char*>(expectedPtr->GetMemoryArea());
                    const char* actualData = static_cast<const char*>(actualPtr->GetMemoryArea());
                    bool same = true;
                    for (unsigned int i = 0; i < expectedPtr->GetNumBytes(); ++i)
                    {
                        same = expectedData[i] == actualData[i];
                        if (!same)
                        {
                            break;
                        }
                    }
                    BOOST_CHECK_MESSAGE(same, tensorName + " data does not match");
                }
            }
        }
    }

private:
    std::string m_LayerName;
    std::vector<armnn::TensorInfo> m_InputTensorInfos;
    std::vector<armnn::TensorInfo> m_OutputTensorInfos;
};

template<typename Descriptor>
class LayerVerifierBaseWithDescriptor : public LayerVerifierBase
{
public:
    LayerVerifierBaseWithDescriptor(const std::string& layerName,
                                    const std::vector<armnn::TensorInfo>& inputInfos,
                                    const std::vector<armnn::TensorInfo>& outputInfos,
                                    const Descriptor& descriptor)
        : LayerVerifierBase(layerName, inputInfos, outputInfos)
        , m_Descriptor(descriptor) {}

protected:
    void VerifyDescriptor(const Descriptor& descriptor)
    {
        BOOST_CHECK(descriptor == m_Descriptor);
    }

    Descriptor m_Descriptor;
};

template<typename T>
void CompareConstTensorData(const void* data1, const void* data2, unsigned int numElements)
{
    T typedData1 = static_cast<T>(data1);
    T typedData2 = static_cast<T>(data2);
    BOOST_CHECK(typedData1);
    BOOST_CHECK(typedData2);

    for (unsigned int i = 0; i < numElements; i++)
    {
        BOOST_TEST(typedData1[i] == typedData2[i]);
    }
}

void CompareConstTensor(const armnn::ConstTensor& tensor1, const armnn::ConstTensor& tensor2)
{
    BOOST_TEST(tensor1.GetShape() == tensor2.GetShape());
    BOOST_TEST(GetDataTypeName(tensor1.GetDataType()) == GetDataTypeName(tensor2.GetDataType()));

    switch (tensor1.GetDataType())
    {
        case armnn::DataType::Float32:
            CompareConstTensorData<const float*>(
                tensor1.GetMemoryArea(), tensor2.GetMemoryArea(), tensor1.GetNumElements());
            break;
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::Boolean:
            CompareConstTensorData<const uint8_t*>(
                tensor1.GetMemoryArea(), tensor2.GetMemoryArea(), tensor1.GetNumElements());
            break;
        case armnn::DataType::QSymmS8:
            CompareConstTensorData<const int8_t*>(
                tensor1.GetMemoryArea(), tensor2.GetMemoryArea(), tensor1.GetNumElements());
            break;
        case armnn::DataType::Signed32:
            CompareConstTensorData<const int32_t*>(
                tensor1.GetMemoryArea(), tensor2.GetMemoryArea(), tensor1.GetNumElements());
            break;
        default:
            // Note that Float16 is not yet implemented
            BOOST_TEST_MESSAGE("Unexpected datatype");
            BOOST_TEST(false);
    }
}

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

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(SerializerTests)

BOOST_AUTO_TEST_CASE(SerializeAddition)
{
    DECLARE_LAYER_VERIFIER_CLASS(Addition)

    const std::string layerName("addition");
    const armnn::TensorInfo tensorInfo({1, 2, 3}, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const additionLayer = network->AddAdditionLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(1));
    additionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    additionLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    AdditionLayerVerifier verifier(layerName, {tensorInfo, tensorInfo}, {tensorInfo});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeArgMinMax)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(ArgMinMax)

    const std::string layerName("argminmax");
    const armnn::TensorInfo inputInfo({1, 2, 3}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1, 3}, armnn::DataType::Signed32);

    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Function = armnn::ArgMinMaxFunction::Max;
    descriptor.m_Axis = 1;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const argMinMaxLayer = network->AddArgMinMaxLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(argMinMaxLayer->GetInputSlot(0));
    argMinMaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    argMinMaxLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    ArgMinMaxLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeBatchNormalization)
{
    using Descriptor = armnn::BatchNormalizationDescriptor;
    class BatchNormalizationLayerVerifier : public LayerVerifierBaseWithDescriptor<Descriptor>
    {
    public:
        BatchNormalizationLayerVerifier(const std::string& layerName,
                                        const std::vector<armnn::TensorInfo>& inputInfos,
                                        const std::vector<armnn::TensorInfo>& outputInfos,
                                        const Descriptor& descriptor,
                                        const armnn::ConstTensor& mean,
                                        const armnn::ConstTensor& variance,
                                        const armnn::ConstTensor& beta,
                                        const armnn::ConstTensor& gamma)
            : LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor)
            , m_Mean(mean)
            , m_Variance(variance)
            , m_Beta(beta)
            , m_Gamma(gamma) {}

        void VisitBatchNormalizationLayer(const armnn::IConnectableLayer* layer,
                                          const Descriptor& descriptor,
                                          const armnn::ConstTensor& mean,
                                          const armnn::ConstTensor& variance,
                                          const armnn::ConstTensor& beta,
                                          const armnn::ConstTensor& gamma,
                                          const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            VerifyDescriptor(descriptor);

            CompareConstTensor(mean, m_Mean);
            CompareConstTensor(variance, m_Variance);
            CompareConstTensor(beta, m_Beta);
            CompareConstTensor(gamma, m_Gamma);
        }

    private:
        armnn::ConstTensor m_Mean;
        armnn::ConstTensor m_Variance;
        armnn::ConstTensor m_Beta;
        armnn::ConstTensor m_Gamma;
    };

    const std::string layerName("batchNormalization");
    const armnn::TensorInfo inputInfo ({ 1, 3, 3, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);

    const armnn::TensorInfo meanInfo({1}, armnn::DataType::Float32);
    const armnn::TensorInfo varianceInfo({1}, armnn::DataType::Float32);
    const armnn::TensorInfo betaInfo({1}, armnn::DataType::Float32);
    const armnn::TensorInfo gammaInfo({1}, armnn::DataType::Float32);

    armnn::BatchNormalizationDescriptor descriptor;
    descriptor.m_Eps = 0.0010000000475f;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    std::vector<float> meanData({5.0});
    std::vector<float> varianceData({2.0});
    std::vector<float> betaData({1.0});
    std::vector<float> gammaData({0.0});

    armnn::ConstTensor mean(meanInfo, meanData);
    armnn::ConstTensor variance(varianceInfo, varianceData);
    armnn::ConstTensor beta(betaInfo, betaData);
    armnn::ConstTensor gamma(gammaInfo, gammaData);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const batchNormalizationLayer =
        network->AddBatchNormalizationLayer(descriptor, mean, variance, beta, gamma, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(batchNormalizationLayer->GetInputSlot(0));
    batchNormalizationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    batchNormalizationLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    BatchNormalizationLayerVerifier verifier(
        layerName, {inputInfo}, {outputInfo}, descriptor, mean, variance, beta, gamma);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeBatchToSpaceNd)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(BatchToSpaceNd)

    const std::string layerName("spaceToBatchNd");
    const armnn::TensorInfo inputInfo({4, 1, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1, 1, 4, 4}, armnn::DataType::Float32);

    armnn::BatchToSpaceNdDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;
    desc.m_BlockShape = {2, 2};
    desc.m_Crops = {{0, 0}, {0, 0}};

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const batchToSpaceNdLayer = network->AddBatchToSpaceNdLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(batchToSpaceNdLayer->GetInputSlot(0));
    batchToSpaceNdLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    batchToSpaceNdLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    BatchToSpaceNdLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeComparison)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Comparison)

    const std::string layerName("comparison");

    const armnn::TensorShape shape{2, 1, 2, 4};

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    armnn::ComparisonDescriptor descriptor(armnn::ComparisonOperation::NotEqual);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0     = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1     = network->AddInputLayer(1);
    armnn::IConnectableLayer* const comparisonLayer = network->AddComparisonLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer     = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(comparisonLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(comparisonLayer->GetInputSlot(1));
    comparisonLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(inputInfo);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputInfo);
    comparisonLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    ComparisonLayerVerifier verifier(layerName, { inputInfo, inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeConstant)
{
    class ConstantLayerVerifier : public LayerVerifierBase
    {
    public:
        ConstantLayerVerifier(const std::string& layerName,
                              const std::vector<armnn::TensorInfo>& inputInfos,
                              const std::vector<armnn::TensorInfo>& outputInfos,
                              const armnn::ConstTensor& layerInput)
            : LayerVerifierBase(layerName, inputInfos, outputInfos)
            , m_LayerInput(layerInput) {}

        void VisitConstantLayer(const armnn::IConnectableLayer* layer,
                                const armnn::ConstTensor& input,
                                const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            CompareConstTensor(input, m_LayerInput);
        }

        void VisitAdditionLayer(const armnn::IConnectableLayer*, const char*) override {}

    private:
        armnn::ConstTensor m_LayerInput;
    };

    const std::string layerName("constant");
    const armnn::TensorInfo info({ 2, 3 }, armnn::DataType::Float32);

    std::vector<float> constantData = GenerateRandomData<float>(info.GetNumElements());
    armnn::ConstTensor constTensor(info, constantData);

    armnn::INetworkPtr network(armnn::INetwork::Create());
    armnn::IConnectableLayer* input = network->AddInputLayer(0);
    armnn::IConnectableLayer* constant = network->AddConstantLayer(constTensor, layerName.c_str());
    armnn::IConnectableLayer* add = network->AddAdditionLayer();
    armnn::IConnectableLayer* output = network->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(info);
    constant->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    ConstantLayerVerifier verifier(layerName, {}, {info}, constTensor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeConvolution2d)
{
    using Descriptor = armnn::Convolution2dDescriptor;
    class Convolution2dLayerVerifier : public LayerVerifierBaseWithDescriptor<Descriptor>
    {
    public:
        Convolution2dLayerVerifier(const std::string& layerName,
                                   const std::vector<armnn::TensorInfo>& inputInfos,
                                   const std::vector<armnn::TensorInfo>& outputInfos,
                                   const Descriptor& descriptor,
                                   const armnn::ConstTensor& weights,
                                   const armnn::Optional<armnn::ConstTensor>& biases)
            : LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor)
            , m_Weights(weights)
            , m_Biases(biases) {}

        void VisitConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                     const Descriptor& descriptor,
                                     const armnn::ConstTensor& weights,
                                     const armnn::Optional<armnn::ConstTensor>& biases,
                                     const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            VerifyDescriptor(descriptor);

            // check weights
            CompareConstTensor(weights, m_Weights);

            // check biases
            BOOST_CHECK(biases.has_value() == descriptor.m_BiasEnabled);
            BOOST_CHECK(biases.has_value() == m_Biases.has_value());

            if (biases.has_value() && m_Biases.has_value())
            {
                CompareConstTensor(biases.value(), m_Biases.value());
            }
        }

    private:
        armnn::ConstTensor                  m_Weights;
        armnn::Optional<armnn::ConstTensor> m_Biases;
    };

    const std::string layerName("convolution2d");
    const armnn::TensorInfo inputInfo ({ 1, 5, 5, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32);

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);

    std::vector<float> biasesData = GenerateRandomData<float>(biasesInfo.GetNumElements());
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 1;
    descriptor.m_PadRight    = 1;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 1;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_DilationX   = 2;
    descriptor.m_DilationY   = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer  = network->AddInputLayer(0);
    armnn::IConnectableLayer* const convLayer   =
            network->AddConvolution2dLayer(descriptor,
                                           weights,
                                           armnn::Optional<armnn::ConstTensor>(biases),
                                           layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    Convolution2dLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor, weights, biases);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeConvolution2dWithPerAxisParams)
{
    using Descriptor = armnn::Convolution2dDescriptor;
    class Convolution2dLayerVerifier : public LayerVerifierBaseWithDescriptor<Descriptor>
    {
    public:
        Convolution2dLayerVerifier(const std::string& layerName,
                                   const std::vector<armnn::TensorInfo>& inputInfos,
                                   const std::vector<armnn::TensorInfo>& outputInfos,
                                   const Descriptor& descriptor,
                                   const armnn::ConstTensor& weights,
                                   const armnn::Optional<armnn::ConstTensor>& biases)
            : LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor)
            , m_Weights(weights)
            , m_Biases(biases) {}

        void VisitConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                     const Descriptor& descriptor,
                                     const armnn::ConstTensor& weights,
                                     const armnn::Optional<armnn::ConstTensor>& biases,
                                     const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            VerifyDescriptor(descriptor);

            // check weights
            CompareConstTensor(weights, m_Weights);

            // check biases
            BOOST_CHECK(biases.has_value() == descriptor.m_BiasEnabled);
            BOOST_CHECK(biases.has_value() == m_Biases.has_value());

            if (biases.has_value() && m_Biases.has_value())
            {
                CompareConstTensor(biases.value(), m_Biases.value());
            }
        }

    private:
        armnn::ConstTensor                  m_Weights;
        armnn::Optional<armnn::ConstTensor> m_Biases;
    };

    using namespace armnn;

    const std::string layerName("convolution2dWithPerAxis");
    const TensorInfo inputInfo ({ 1, 3, 1, 2 }, DataType::QAsymmU8, 0.55f, 128);
    const TensorInfo outputInfo({ 1, 3, 1, 3 }, DataType::QAsymmU8, 0.75f, 128);

    const std::vector<float> quantScales{ 0.75f, 0.65f, 0.85f };
    constexpr unsigned int quantDimension = 0;

    const TensorInfo kernelInfo({ 3, 1, 1, 2 }, DataType::QSymmS8, quantScales, quantDimension);

    const std::vector<float> biasQuantScales{ 0.25f, 0.50f, 0.75f };
    const TensorInfo biasInfo({ 3 }, DataType::Signed32, biasQuantScales, quantDimension);

    std::vector<int8_t> kernelData = GenerateRandomData<int8_t>(kernelInfo.GetNumElements());
    armnn::ConstTensor weights(kernelInfo, kernelData);
    std::vector<int32_t> biasData = GenerateRandomData<int32_t>(biasInfo.GetNumElements());
    armnn::ConstTensor biases(biasInfo, biasData);

    Convolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer  = network->AddInputLayer(0);
    armnn::IConnectableLayer* const convLayer   =
        network->AddConvolution2dLayer(descriptor,
                                       weights,
                                       armnn::Optional<armnn::ConstTensor>(biases),
                                       layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    Convolution2dLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor, weights, biases);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeDepthToSpace)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(DepthToSpace)

    const std::string layerName("depthToSpace");

    const armnn::TensorInfo inputInfo ({ 1,  8, 4, 12 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 16, 8,  3 }, armnn::DataType::Float32);

    armnn::DepthToSpaceDescriptor desc;
    desc.m_BlockSize  = 2;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer        = network->AddInputLayer(0);
    armnn::IConnectableLayer* const depthToSpaceLayer = network->AddDepthToSpaceLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer       = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(depthToSpaceLayer->GetInputSlot(0));
    depthToSpaceLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    depthToSpaceLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    DepthToSpaceLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeDepthwiseConvolution2d)
{
    using Descriptor = armnn::DepthwiseConvolution2dDescriptor;
    class DepthwiseConvolution2dLayerVerifier : public LayerVerifierBaseWithDescriptor<Descriptor>
    {
    public:
        DepthwiseConvolution2dLayerVerifier(const std::string& layerName,
                                            const std::vector<armnn::TensorInfo>& inputInfos,
                                            const std::vector<armnn::TensorInfo>& outputInfos,
                                            const Descriptor& descriptor,
                                            const armnn::ConstTensor& weights,
                                            const armnn::Optional<armnn::ConstTensor>& biases) :
            LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor),
            m_Weights(weights),
            m_Biases(biases) {}

        void VisitDepthwiseConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                              const Descriptor& descriptor,
                                              const armnn::ConstTensor& weights,
                                              const armnn::Optional<armnn::ConstTensor>& biases,
                                              const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            VerifyDescriptor(descriptor);

            // check weights
            CompareConstTensor(weights, m_Weights);

            // check biases
            BOOST_CHECK(biases.has_value() == descriptor.m_BiasEnabled);
            BOOST_CHECK(biases.has_value() == m_Biases.has_value());

            if (biases.has_value() && m_Biases.has_value())
            {
                CompareConstTensor(biases.value(), m_Biases.value());
            }
        }

    private:
        armnn::ConstTensor                      m_Weights;
        armnn::Optional<armnn::ConstTensor>     m_Biases;
    };

    const std::string layerName("depwiseConvolution2d");
    const armnn::TensorInfo inputInfo ({ 1, 5, 5, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 3, 3, 3 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 1, 3, 3, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo biasesInfo ({ 3 }, armnn::DataType::Float32);

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);

    std::vector<int32_t> biasesData = GenerateRandomData<int32_t>(biasesInfo.GetNumElements());
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 1;
    descriptor.m_PadRight    = 1;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 1;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_DilationX   = 2;
    descriptor.m_DilationY   = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const depthwiseConvLayer =
        network->AddDepthwiseConvolution2dLayer(descriptor,
                                                weights,
                                                armnn::Optional<armnn::ConstTensor>(biases),
                                                layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(depthwiseConvLayer->GetInputSlot(0));
    depthwiseConvLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    depthwiseConvLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    DepthwiseConvolution2dLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor, weights, biases);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeDepthwiseConvolution2dWithPerAxisParams)
{
    using Descriptor = armnn::DepthwiseConvolution2dDescriptor;
    class DepthwiseConvolution2dLayerVerifier : public LayerVerifierBaseWithDescriptor<Descriptor>
    {
    public:
        DepthwiseConvolution2dLayerVerifier(const std::string& layerName,
                                            const std::vector<armnn::TensorInfo>& inputInfos,
                                            const std::vector<armnn::TensorInfo>& outputInfos,
                                            const Descriptor& descriptor,
                                            const armnn::ConstTensor& weights,
                                            const armnn::Optional<armnn::ConstTensor>& biases) :
            LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor),
            m_Weights(weights),
            m_Biases(biases) {}

        void VisitDepthwiseConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                              const Descriptor& descriptor,
                                              const armnn::ConstTensor& weights,
                                              const armnn::Optional<armnn::ConstTensor>& biases,
                                              const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            VerifyDescriptor(descriptor);

            // check weights
            CompareConstTensor(weights, m_Weights);

            // check biases
            BOOST_CHECK(biases.has_value() == descriptor.m_BiasEnabled);
            BOOST_CHECK(biases.has_value() == m_Biases.has_value());

            if (biases.has_value() && m_Biases.has_value())
            {
                CompareConstTensor(biases.value(), m_Biases.value());
            }
        }

    private:
        armnn::ConstTensor                      m_Weights;
        armnn::Optional<armnn::ConstTensor>     m_Biases;
    };

    using namespace armnn;

    const std::string layerName("depwiseConvolution2dWithPerAxis");
    const TensorInfo inputInfo ({ 1, 3, 3, 2 }, DataType::QAsymmU8, 0.55f, 128);
    const TensorInfo outputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 0.75f, 128);

    const std::vector<float> quantScales{ 0.75f, 0.80f, 0.90f, 0.95f };
    const unsigned int quantDimension = 0;
    TensorInfo kernelInfo({ 2, 2, 2, 2 }, DataType::QSymmS8, quantScales, quantDimension);

    const std::vector<float> biasQuantScales{ 0.25f, 0.35f, 0.45f, 0.55f };
    constexpr unsigned int biasQuantDimension = 0;
    TensorInfo biasInfo({ 4 }, DataType::Signed32, biasQuantScales, biasQuantDimension);

    std::vector<int8_t> kernelData = GenerateRandomData<int8_t>(kernelInfo.GetNumElements());
    armnn::ConstTensor weights(kernelInfo, kernelData);
    std::vector<int32_t> biasData = GenerateRandomData<int32_t>(biasInfo.GetNumElements());
    armnn::ConstTensor biases(biasInfo, biasData);

    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_DilationX   = 1;
    descriptor.m_DilationY   = 1;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const depthwiseConvLayer =
        network->AddDepthwiseConvolution2dLayer(descriptor,
                                                weights,
                                                armnn::Optional<armnn::ConstTensor>(biases),
                                                layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(depthwiseConvLayer->GetInputSlot(0));
    depthwiseConvLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    depthwiseConvLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    DepthwiseConvolution2dLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor, weights, biases);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeDequantize)
{
    DECLARE_LAYER_VERIFIER_CLASS(Dequantize)

    const std::string layerName("dequantize");
    const armnn::TensorInfo inputInfo({ 1, 5, 2, 3 }, armnn::DataType::QAsymmU8, 0.5f, 1);
    const armnn::TensorInfo outputInfo({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const dequantizeLayer = network->AddDequantizeLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(dequantizeLayer->GetInputSlot(0));
    dequantizeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    dequantizeLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    DequantizeLayerVerifier verifier(layerName, {inputInfo}, {outputInfo});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeDetectionPostProcess)
{
    using Descriptor = armnn::DetectionPostProcessDescriptor;
    class DetectionPostProcessLayerVerifier : public LayerVerifierBaseWithDescriptor<Descriptor>
    {
    public:
        DetectionPostProcessLayerVerifier(const std::string& layerName,
                                          const std::vector<armnn::TensorInfo>& inputInfos,
                                          const std::vector<armnn::TensorInfo>& outputInfos,
                                          const Descriptor& descriptor,
                                          const armnn::ConstTensor& anchors)
            : LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor)
            , m_Anchors(anchors) {}

        void VisitDetectionPostProcessLayer(const armnn::IConnectableLayer* layer,
                                            const Descriptor& descriptor,
                                            const armnn::ConstTensor& anchors,
                                            const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            VerifyDescriptor(descriptor);

            CompareConstTensor(anchors, m_Anchors);
        }

    private:
        armnn::ConstTensor m_Anchors;
    };

    const std::string layerName("detectionPostProcess");

    const std::vector<armnn::TensorInfo> inputInfos({
        armnn::TensorInfo({ 1, 6, 4 }, armnn::DataType::Float32),
        armnn::TensorInfo({ 1, 6, 3}, armnn::DataType::Float32)
    });

    const std::vector<armnn::TensorInfo> outputInfos({
        armnn::TensorInfo({ 1, 3, 4 }, armnn::DataType::Float32),
        armnn::TensorInfo({ 1, 3 }, armnn::DataType::Float32),
        armnn::TensorInfo({ 1, 3 }, armnn::DataType::Float32),
        armnn::TensorInfo({ 1 }, armnn::DataType::Float32)
    });

    armnn::DetectionPostProcessDescriptor descriptor;
    descriptor.m_UseRegularNms = true;
    descriptor.m_MaxDetections = 3;
    descriptor.m_MaxClassesPerDetection = 1;
    descriptor.m_DetectionsPerClass =1;
    descriptor.m_NmsScoreThreshold = 0.0;
    descriptor.m_NmsIouThreshold = 0.5;
    descriptor.m_NumClasses = 2;
    descriptor.m_ScaleY = 10.0;
    descriptor.m_ScaleX = 10.0;
    descriptor.m_ScaleH = 5.0;
    descriptor.m_ScaleW = 5.0;

    const armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32);
    const std::vector<float> anchorsData({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });
    armnn::ConstTensor anchors(anchorsInfo, anchorsData);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const detectionLayer =
        network->AddDetectionPostProcessLayer(descriptor, anchors, layerName.c_str());

    for (unsigned int i = 0; i < 2; i++)
    {
        armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(static_cast<int>(i));
        inputLayer->GetOutputSlot(0).Connect(detectionLayer->GetInputSlot(i));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfos[i]);
    }

    for (unsigned int i = 0; i < 4; i++)
    {
        armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(static_cast<int>(i));
        detectionLayer->GetOutputSlot(i).Connect(outputLayer->GetInputSlot(0));
        detectionLayer->GetOutputSlot(i).SetTensorInfo(outputInfos[i]);
    }

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    DetectionPostProcessLayerVerifier verifier(layerName, inputInfos, outputInfos, descriptor, anchors);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeDivision)
{
    DECLARE_LAYER_VERIFIER_CLASS(Division)

    const std::string layerName("division");
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const divisionLayer = network->AddDivisionLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(divisionLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(divisionLayer->GetInputSlot(1));
    divisionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    divisionLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    DivisionLayerVerifier verifier(layerName, {info, info}, {info});
    deserializedNetwork->Accept(verifier);
}

class EqualLayerVerifier : public LayerVerifierBase
{
public:
    EqualLayerVerifier(const std::string& layerName,
                       const std::vector<armnn::TensorInfo>& inputInfos,
                       const std::vector<armnn::TensorInfo>& outputInfos)
        : LayerVerifierBase(layerName, inputInfos, outputInfos) {}

    void VisitComparisonLayer(const armnn::IConnectableLayer* layer,
                              const armnn::ComparisonDescriptor& descriptor,
                              const char* name) override
    {
        VerifyNameAndConnections(layer, name);
        BOOST_CHECK(descriptor.m_Operation == armnn::ComparisonOperation::Equal);
    }

    void VisitEqualLayer(const armnn::IConnectableLayer*, const char*) override
    {
        throw armnn::Exception("EqualLayer should have translated to ComparisonLayer");
    }
};

// NOTE: Until the deprecated AddEqualLayer disappears this test checks that calling
//       AddEqualLayer places a ComparisonLayer into the serialized format and that
//       when this deserialises we have a ComparisonLayer
BOOST_AUTO_TEST_CASE(SerializeEqual)
{
    const std::string layerName("equal");

    const armnn::TensorShape shape{2, 1, 2, 4};

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const equalLayer = network->AddEqualLayer(layerName.c_str());
    ARMNN_NO_DEPRECATE_WARN_END
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(1));
    equalLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(inputInfo);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputInfo);
    equalLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    EqualLayerVerifier verifier(layerName, { inputInfo, inputInfo }, { outputInfo });
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(EnsureEqualBackwardCompatibility)
{
    // The hex data below is a flat buffer containing a simple network with two inputs,
    // an EqualLayer (now deprecated) and an output
    //
    // This test verifies that we can still deserialize this old-style model by replacing
    // the EqualLayer with an equivalent ComparisonLayer
    const std::vector<uint8_t> equalModel =
    {
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x0A, 0x00,
        0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0xCC, 0x01, 0x00, 0x00, 0x20, 0x01, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x60, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0B, 0x04, 0x00, 0x00, 0x00, 0xFE, 0xFE, 0xFF, 0xFF, 0x04, 0x00,
        0x00, 0x00, 0x06, 0xFF, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0xEA, 0xFE, 0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x64, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xB4, 0xFE, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x13, 0x04, 0x00, 0x00, 0x00, 0x52, 0xFF, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x36, 0xFF, 0xFF, 0xFF,
        0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1C, 0x00,
        0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x65, 0x71, 0x75, 0x61, 0x6C, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x5C, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x34, 0xFF,
        0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x92, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x04, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0E, 0x00,
        0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0E, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0E, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0E, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x66, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
        0x04, 0x00, 0x00, 0x00, 0xF6, 0xFF, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x0A, 0x00,
        0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x14, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0A, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x08, 0x00,
        0x07, 0x00, 0x0C, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00
    };

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(std::string(equalModel.begin(), equalModel.end()));
    BOOST_CHECK(deserializedNetwork);

    const armnn::TensorShape shape{ 2, 1, 2, 4 };

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    EqualLayerVerifier verifier("equal", { inputInfo, inputInfo }, { outputInfo });
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeFill)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Fill)

    const std::string layerName("fill");
    const armnn::TensorInfo inputInfo({4}, armnn::DataType::Signed32);
    const armnn::TensorInfo outputInfo({1, 3, 3, 1}, armnn::DataType::Float32);

    armnn::FillDescriptor descriptor(1.0f);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const fillLayer = network->AddFillLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fillLayer->GetInputSlot(0));
    fillLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    fillLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    FillLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor);

    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeFloor)
{
    DECLARE_LAYER_VERIFIER_CLASS(Floor)

    const std::string layerName("floor");
    const armnn::TensorInfo info({4,4}, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const floorLayer = network->AddFloorLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(floorLayer->GetInputSlot(0));
    floorLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    floorLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    FloorLayerVerifier verifier(layerName, {info}, {info});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeFullyConnected)
{
    using Descriptor = armnn::FullyConnectedDescriptor;
    class FullyConnectedLayerVerifier : public LayerVerifierBaseWithDescriptor<Descriptor>
    {
    public:
        FullyConnectedLayerVerifier(const std::string& layerName,
                                    const std::vector<armnn::TensorInfo>& inputInfos,
                                    const std::vector<armnn::TensorInfo>& outputInfos,
                                    const Descriptor& descriptor,
                                    const armnn::ConstTensor& weight,
                                    const armnn::Optional<armnn::ConstTensor>& bias)
            : LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor)
            , m_Weight(weight)
            , m_Bias(bias) {}

        void VisitFullyConnectedLayer(const armnn::IConnectableLayer* layer,
                                      const Descriptor& descriptor,
                                      const armnn::ConstTensor& weight,
                                      const armnn::Optional<armnn::ConstTensor>& bias,
                                      const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            VerifyDescriptor(descriptor);

            CompareConstTensor(weight, m_Weight);

            BOOST_TEST(bias.has_value() == descriptor.m_BiasEnabled);
            BOOST_TEST(bias.has_value() == m_Bias.has_value());

            if (bias.has_value() && m_Bias.has_value())
            {
                CompareConstTensor(bias.value(), m_Bias.value());
            }
        }

    private:
        armnn::ConstTensor m_Weight;
        armnn::Optional<armnn::ConstTensor> m_Bias;
    };

    const std::string layerName("fullyConnected");
    const armnn::TensorInfo inputInfo ({ 2, 5, 1, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 2, 3 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 5, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo biasesInfo ({ 3 }, armnn::DataType::Float32);
    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    std::vector<float> biasesData  = GenerateRandomData<float>(biasesInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = true;
    descriptor.m_TransposeWeightMatrix = false;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const fullyConnectedLayer =
        network->AddFullyConnectedLayer(descriptor,
                                        weights,
                                        armnn::Optional<armnn::ConstTensor>(biases),
                                        layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    FullyConnectedLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor, weights, biases);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeGather)
{
    using GatherDescriptor = armnn::GatherDescriptor;
    class GatherLayerVerifier : public LayerVerifierBaseWithDescriptor<GatherDescriptor>
    {
    public:
        GatherLayerVerifier(const std::string& layerName,
                            const std::vector<armnn::TensorInfo>& inputInfos,
                            const std::vector<armnn::TensorInfo>& outputInfos,
                            const GatherDescriptor& descriptor)
            : LayerVerifierBaseWithDescriptor<GatherDescriptor>(layerName, inputInfos, outputInfos, descriptor) {}

        void VisitGatherLayer(const armnn::IConnectableLayer* layer,
                              const GatherDescriptor& descriptor,
                              const char *name) override
        {
            VerifyNameAndConnections(layer, name);
            BOOST_CHECK(descriptor.m_Axis == m_Descriptor.m_Axis);
        }

        void VisitConstantLayer(const armnn::IConnectableLayer*,
                                const armnn::ConstTensor&,
                                const char*) override {}
    };

    const std::string layerName("gather");
    armnn::TensorInfo paramsInfo({ 8 }, armnn::DataType::QAsymmU8);
    armnn::TensorInfo outputInfo({ 3 }, armnn::DataType::QAsymmU8);
    const armnn::TensorInfo indicesInfo({ 3 }, armnn::DataType::Signed32);
    GatherDescriptor descriptor;
    descriptor.m_Axis = 1;

    paramsInfo.SetQuantizationScale(1.0f);
    paramsInfo.SetQuantizationOffset(0);
    outputInfo.SetQuantizationScale(1.0f);
    outputInfo.SetQuantizationOffset(0);

    const std::vector<int32_t>& indicesData = {7, 6, 5};

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer *const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer *const constantLayer =
            network->AddConstantLayer(armnn::ConstTensor(indicesInfo, indicesData));
    armnn::IConnectableLayer *const gatherLayer = network->AddGatherLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer *const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(gatherLayer->GetInputSlot(0));
    constantLayer->GetOutputSlot(0).Connect(gatherLayer->GetInputSlot(1));
    gatherLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(paramsInfo);
    constantLayer->GetOutputSlot(0).SetTensorInfo(indicesInfo);
    gatherLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    GatherLayerVerifier verifier(layerName, {paramsInfo, indicesInfo}, {outputInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

class GreaterLayerVerifier : public LayerVerifierBase
{
public:
    GreaterLayerVerifier(const std::string& layerName,
                         const std::vector<armnn::TensorInfo>& inputInfos,
                         const std::vector<armnn::TensorInfo>& outputInfos)
        : LayerVerifierBase(layerName, inputInfos, outputInfos) {}

    void VisitComparisonLayer(const armnn::IConnectableLayer* layer,
                              const armnn::ComparisonDescriptor& descriptor,
                              const char* name) override
    {
        VerifyNameAndConnections(layer, name);
        BOOST_CHECK(descriptor.m_Operation == armnn::ComparisonOperation::Greater);
    }

    void VisitGreaterLayer(const armnn::IConnectableLayer*, const char*) override
    {
        throw armnn::Exception("GreaterLayer should have translated to ComparisonLayer");
    }
};

// NOTE: Until the deprecated AddGreaterLayer disappears this test checks that calling
//       AddGreaterLayer places a ComparisonLayer into the serialized format and that
//       when this deserialises we have a ComparisonLayer
BOOST_AUTO_TEST_CASE(SerializeGreater)
{
    const std::string layerName("greater");

    const armnn::TensorShape shape{2, 1, 2, 4};

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const equalLayer = network->AddGreaterLayer(layerName.c_str());
    ARMNN_NO_DEPRECATE_WARN_END
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(1));
    equalLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(inputInfo);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputInfo);
    equalLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    GreaterLayerVerifier verifier(layerName, { inputInfo, inputInfo }, { outputInfo });
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(EnsureGreaterBackwardCompatibility)
{
    // The hex data below is a flat buffer containing a simple network with two inputs,
    // an GreaterLayer (now deprecated) and an output
    //
    // This test verifies that we can still deserialize this old-style model by replacing
    // the GreaterLayer with an equivalent ComparisonLayer
    const std::vector<uint8_t> greaterModel =
    {
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x0A, 0x00,
        0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0xCC, 0x01, 0x00, 0x00, 0x20, 0x01, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x60, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0B, 0x04, 0x00, 0x00, 0x00, 0xFE, 0xFE, 0xFF, 0xFF, 0x04, 0x00,
        0x00, 0x00, 0x06, 0xFF, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0xEA, 0xFE, 0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x64, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xB4, 0xFE, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x19, 0x04, 0x00, 0x00, 0x00, 0x52, 0xFF, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x36, 0xFF, 0xFF, 0xFF,
        0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1C, 0x00,
        0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x67, 0x72, 0x65, 0x61, 0x74, 0x65, 0x72, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x5C, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x34, 0xFF,
        0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x92, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x04, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x08, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0E, 0x00,
        0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0E, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0E, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0E, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x66, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
        0x04, 0x00, 0x00, 0x00, 0xF6, 0xFF, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x0A, 0x00,
        0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x14, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0A, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x08, 0x00,
        0x07, 0x00, 0x0C, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00
    };

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(std::string(greaterModel.begin(), greaterModel.end()));
    BOOST_CHECK(deserializedNetwork);

    const armnn::TensorShape shape{ 1, 2, 2, 2 };

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    GreaterLayerVerifier verifier("greater", { inputInfo, inputInfo }, { outputInfo });
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeInstanceNormalization)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(InstanceNormalization)

    const std::string layerName("instanceNormalization");
    const armnn::TensorInfo info({ 1, 2, 1, 5 }, armnn::DataType::Float32);

    armnn::InstanceNormalizationDescriptor descriptor;
    descriptor.m_Gamma      = 1.1f;
    descriptor.m_Beta       = 0.1f;
    descriptor.m_Eps        = 0.0001f;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer        = network->AddInputLayer(0);
    armnn::IConnectableLayer* const instanceNormLayer =
        network->AddInstanceNormalizationLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer       = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(instanceNormLayer->GetInputSlot(0));
    instanceNormLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    instanceNormLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    InstanceNormalizationLayerVerifier verifier(layerName, {info}, {info}, descriptor);
    deserializedNetwork->Accept(verifier);
}

DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(L2Normalization)

BOOST_AUTO_TEST_CASE(SerializeL2Normalization)
{
    const std::string l2NormLayerName("l2Normalization");
    const armnn::TensorInfo info({1, 2, 1, 5}, armnn::DataType::Float32);

    armnn::L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;
    desc.m_Eps = 0.0001f;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const l2NormLayer = network->AddL2NormalizationLayer(desc, l2NormLayerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(l2NormLayer->GetInputSlot(0));
    l2NormLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    l2NormLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    L2NormalizationLayerVerifier verifier(l2NormLayerName, {info}, {info}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(EnsureL2NormalizationBackwardCompatibility)
{
    // The hex data below is a flat buffer containing a simple network with one input
    // a L2Normalization layer and an output layer with dimensions as per the tensor infos below.
    //
    // This test verifies that we can still read back these old style
    // models without the normalization epsilon value.
    const std::vector<uint8_t> l2NormalizationModel =
    {
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x0A, 0x00,
        0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x3C, 0x01, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xE8, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0B,
        0x04, 0x00, 0x00, 0x00, 0xD6, 0xFE, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00,
        0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x9E, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x4C, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0xFF, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x20, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x20, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x06, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0E, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0E, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x20, 0x00,
        0x00, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x6C, 0x32, 0x4E, 0x6F, 0x72, 0x6D, 0x61, 0x6C, 0x69, 0x7A, 0x61, 0x74,
        0x69, 0x6F, 0x6E, 0x00, 0x01, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0C, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x52, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
        0x04, 0x00, 0x00, 0x00, 0xF6, 0xFF, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x0A, 0x00,
        0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x14, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0A, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x08, 0x00,
        0x07, 0x00, 0x0C, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x05, 0x00, 0x00, 0x00, 0x00
    };

    armnn::INetworkPtr deserializedNetwork =
        DeserializeNetwork(std::string(l2NormalizationModel.begin(), l2NormalizationModel.end()));
    BOOST_CHECK(deserializedNetwork);

    const std::string layerName("l2Normalization");
    const armnn::TensorInfo inputInfo = armnn::TensorInfo({1, 2, 1, 5}, armnn::DataType::Float32);

    armnn::L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;
    // Since this variable does not exist in the l2NormalizationModel dump, the default value will be loaded
    desc.m_Eps = 1e-12f;

    L2NormalizationLayerVerifier verifier(layerName, {inputInfo}, {inputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeLogicalBinary)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(LogicalBinary)

    const std::string layerName("logicalBinaryAnd");

    const armnn::TensorShape shape{2, 1, 2, 2};

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Boolean);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    armnn::LogicalBinaryDescriptor descriptor(armnn::LogicalBinaryOperation::LogicalAnd);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0        = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1        = network->AddInputLayer(1);
    armnn::IConnectableLayer* const logicalBinaryLayer = network->AddLogicalBinaryLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer        = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(logicalBinaryLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(logicalBinaryLayer->GetInputSlot(1));
    logicalBinaryLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(inputInfo);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputInfo);
    logicalBinaryLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    LogicalBinaryLayerVerifier verifier(layerName, { inputInfo, inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeLogicalUnary)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(ElementwiseUnary)

    const std::string layerName("elementwiseUnaryLogicalNot");

    const armnn::TensorShape shape{2, 1, 2, 2};

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Boolean);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    armnn::ElementwiseUnaryDescriptor descriptor(armnn::UnaryOperation::LogicalNot);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const elementwiseUnaryLayer =
        network->AddElementwiseUnaryLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(elementwiseUnaryLayer->GetInputSlot(0));
    elementwiseUnaryLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    elementwiseUnaryLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));

    BOOST_CHECK(deserializedNetwork);

    ElementwiseUnaryLayerVerifier verifier(layerName, { inputInfo }, { outputInfo }, descriptor);

    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeLogSoftmax)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(LogSoftmax)

    const std::string layerName("log_softmax");
    const armnn::TensorInfo info({1, 10}, armnn::DataType::Float32);

    armnn::LogSoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;
    descriptor.m_Axis = -1;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer      = network->AddInputLayer(0);
    armnn::IConnectableLayer* const logSoftmaxLayer = network->AddLogSoftmaxLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer     = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(logSoftmaxLayer->GetInputSlot(0));
    logSoftmaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    logSoftmaxLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    LogSoftmaxLayerVerifier verifier(layerName, {info}, {info}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeMaximum)
{
    DECLARE_LAYER_VERIFIER_CLASS(Maximum)

    const std::string layerName("maximum");
    const armnn::TensorInfo info({ 1, 2, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const maximumLayer = network->AddMaximumLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(maximumLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(maximumLayer->GetInputSlot(1));
    maximumLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    maximumLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    MaximumLayerVerifier verifier(layerName, {info, info}, {info});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeMean)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Mean)

    const std::string layerName("mean");
    const armnn::TensorInfo inputInfo({1, 1, 3, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1, 1, 1, 2}, armnn::DataType::Float32);

    armnn::MeanDescriptor descriptor;
    descriptor.m_Axis = { 2 };
    descriptor.m_KeepDims = true;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const meanLayer = network->AddMeanLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(meanLayer->GetInputSlot(0));
    meanLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    meanLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    MeanLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeMerge)
{
    DECLARE_LAYER_VERIFIER_CLASS(Merge)

    const std::string layerName("merge");
    const armnn::TensorInfo info({ 1, 2, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const mergeLayer = network->AddMergeLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(mergeLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(mergeLayer->GetInputSlot(1));
    mergeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    mergeLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    MergeLayerVerifier verifier(layerName, {info, info}, {info});
    deserializedNetwork->Accept(verifier);
}

class MergerLayerVerifier : public LayerVerifierBaseWithDescriptor<armnn::OriginsDescriptor>
{
public:
    MergerLayerVerifier(const std::string& layerName,
                        const std::vector<armnn::TensorInfo>& inputInfos,
                        const std::vector<armnn::TensorInfo>& outputInfos,
                        const armnn::OriginsDescriptor& descriptor)
        : LayerVerifierBaseWithDescriptor<armnn::OriginsDescriptor>(layerName, inputInfos, outputInfos, descriptor) {}

    void VisitMergerLayer(const armnn::IConnectableLayer*,
                          const armnn::OriginsDescriptor&,
                          const char*) override
    {
        throw armnn::Exception("MergerLayer should have translated to ConcatLayer");
    }

    void VisitConcatLayer(const armnn::IConnectableLayer* layer,
                          const armnn::OriginsDescriptor& descriptor,
                          const char* name) override
    {
        VerifyNameAndConnections(layer, name);
        VerifyDescriptor(descriptor);
    }
};

// NOTE: Until the deprecated AddMergerLayer disappears this test checks that calling
//       AddMergerLayer places a ConcatLayer into the serialized format and that
//       when this deserialises we have a ConcatLayer
BOOST_AUTO_TEST_CASE(SerializeMerger)
{
    const std::string layerName("merger");
    const armnn::TensorInfo inputInfo = armnn::TensorInfo({2, 3, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({4, 3, 2, 2}, armnn::DataType::Float32);

    const std::vector<armnn::TensorShape> shapes({inputInfo.GetShape(), inputInfo.GetShape()});

    armnn::OriginsDescriptor descriptor =
        armnn::CreateDescriptorForConcatenation(shapes.begin(), shapes.end(), 0);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayerOne = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayerTwo = network->AddInputLayer(1);
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const mergerLayer = network->AddMergerLayer(descriptor, layerName.c_str());
    ARMNN_NO_DEPRECATE_WARN_END
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayerOne->GetOutputSlot(0).Connect(mergerLayer->GetInputSlot(0));
    inputLayerTwo->GetOutputSlot(0).Connect(mergerLayer->GetInputSlot(1));
    mergerLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayerOne->GetOutputSlot(0).SetTensorInfo(inputInfo);
    inputLayerTwo->GetOutputSlot(0).SetTensorInfo(inputInfo);
    mergerLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    std::string mergerLayerNetwork = SerializeNetwork(*network);
    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(mergerLayerNetwork);
    BOOST_CHECK(deserializedNetwork);

    MergerLayerVerifier verifier(layerName, {inputInfo, inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(EnsureMergerLayerBackwardCompatibility)
{
    // The hex data below is a flat buffer containing a simple network with two inputs
    // a merger layer (now deprecated) and an output layer with dimensions as per the tensor infos below.
    //
    // This test verifies that we can still read back these old style
    // models replacing the MergerLayers with ConcatLayers with the same parameters.
    const std::vector<uint8_t> mergerModel =
    {
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x0A, 0x00,
        0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x38, 0x02, 0x00, 0x00, 0x8C, 0x01, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0xF4, 0xFD, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0B, 0x04, 0x00, 0x00, 0x00, 0x92, 0xFE, 0xFF, 0xFF, 0x04, 0x00,
        0x00, 0x00, 0x9A, 0xFE, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x7E, 0xFE, 0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xF8, 0xFE, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xFE, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x1F, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x68, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00,
        0x0C, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x22, 0xFF, 0xFF, 0xFF, 0x04, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x3E, 0xFF, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0xFF, 0xFF, 0xFF,
        0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1C, 0x00,
        0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x6D, 0x65, 0x72, 0x67, 0x65, 0x72, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x5C, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x34, 0xFF,
        0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x92, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x08, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0E, 0x00,
        0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0E, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0E, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0E, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x66, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
        0x04, 0x00, 0x00, 0x00, 0xF6, 0xFF, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x0A, 0x00,
        0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x14, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0A, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x08, 0x00,
        0x07, 0x00, 0x0C, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00
    };

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(std::string(mergerModel.begin(), mergerModel.end()));
    BOOST_CHECK(deserializedNetwork);

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({ 2, 3, 2, 2 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({ 4, 3, 2, 2 }, armnn::DataType::Float32);

    const std::vector<armnn::TensorShape> shapes({inputInfo.GetShape(), inputInfo.GetShape()});

    armnn::OriginsDescriptor descriptor =
            armnn::CreateDescriptorForConcatenation(shapes.begin(), shapes.end(), 0);

    MergerLayerVerifier verifier("merger", { inputInfo, inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeConcat)
{
    const std::string layerName("concat");
    const armnn::TensorInfo inputInfo = armnn::TensorInfo({2, 3, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({4, 3, 2, 2}, armnn::DataType::Float32);

    const std::vector<armnn::TensorShape> shapes({inputInfo.GetShape(), inputInfo.GetShape()});

    armnn::OriginsDescriptor descriptor =
        armnn::CreateDescriptorForConcatenation(shapes.begin(), shapes.end(), 0);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayerOne = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayerTwo = network->AddInputLayer(1);
    armnn::IConnectableLayer* const concatLayer = network->AddConcatLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayerOne->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    inputLayerTwo->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayerOne->GetOutputSlot(0).SetTensorInfo(inputInfo);
    inputLayerTwo->GetOutputSlot(0).SetTensorInfo(inputInfo);
    concatLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    std::string concatLayerNetwork = SerializeNetwork(*network);
    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(concatLayerNetwork);
    BOOST_CHECK(deserializedNetwork);

    // NOTE: using the MergerLayerVerifier to ensure that it is a concat layer and not a
    //       merger layer that gets placed into the graph.
    MergerLayerVerifier verifier(layerName, {inputInfo, inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeMinimum)
{
    DECLARE_LAYER_VERIFIER_CLASS(Minimum)

    const std::string layerName("minimum");
    const armnn::TensorInfo info({ 1, 2, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const minimumLayer = network->AddMinimumLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(minimumLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(minimumLayer->GetInputSlot(1));
    minimumLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    minimumLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    MinimumLayerVerifier verifier(layerName, {info, info}, {info});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeMultiplication)
{
    DECLARE_LAYER_VERIFIER_CLASS(Multiplication)

    const std::string layerName("multiplication");
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const multiplicationLayer = network->AddMultiplicationLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(multiplicationLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(multiplicationLayer->GetInputSlot(1));
    multiplicationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    multiplicationLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    MultiplicationLayerVerifier verifier(layerName, {info, info}, {info});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializePrelu)
{
    DECLARE_LAYER_VERIFIER_CLASS(Prelu)

    const std::string layerName("prelu");

    armnn::TensorInfo inputTensorInfo ({ 4, 1, 2 }, armnn::DataType::Float32);
    armnn::TensorInfo alphaTensorInfo ({ 5, 4, 3, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 5, 4, 3, 2 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const alphaLayer = network->AddInputLayer(1);
    armnn::IConnectableLayer* const preluLayer = network->AddPreluLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(preluLayer->GetInputSlot(0));
    alphaLayer->GetOutputSlot(0).Connect(preluLayer->GetInputSlot(1));
    preluLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    alphaLayer->GetOutputSlot(0).SetTensorInfo(alphaTensorInfo);
    preluLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    PreluLayerVerifier verifier(layerName, {inputTensorInfo, alphaTensorInfo}, {outputTensorInfo});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeNormalization)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Normalization)

    const std::string layerName("normalization");
    const armnn::TensorInfo info({2, 1, 2, 2}, armnn::DataType::Float32);

    armnn::NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;
    desc.m_NormSize = 3;
    desc.m_Alpha = 1;
    desc.m_Beta = 1;
    desc.m_K = 1;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const normalizationLayer = network->AddNormalizationLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(normalizationLayer->GetInputSlot(0));
    normalizationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    normalizationLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    NormalizationLayerVerifier verifier(layerName, {info}, {info}, desc);
    deserializedNetwork->Accept(verifier);
}

DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Pad)

BOOST_AUTO_TEST_CASE(SerializePad)
{
    const std::string layerName("pad");
    const armnn::TensorInfo inputTensorInfo = armnn::TensorInfo({1, 2, 3, 4}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({1, 3, 5, 7}, armnn::DataType::Float32);

    armnn::PadDescriptor desc({{0, 0}, {1, 0}, {1, 1}, {1, 2}});

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const padLayer = network->AddPadLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    padLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    PadLayerVerifier verifier(layerName, {inputTensorInfo}, {outputTensorInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(EnsurePadBackwardCompatibility)
{
    // The PadDescriptor is being extended with a float PadValue (so a value other than 0
    // can be used to pad the tensor.
    //
    // This test contains a binary representation of a simple input->pad->output network
    // prior to this change to test that the descriptor has been updated in a backward
    // compatible way with respect to Deserialization of older binary dumps
    const std::vector<uint8_t> padModel =
    {
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x0A, 0x00,
        0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x54, 0x01, 0x00, 0x00, 0x6C, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xD0, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0B,
        0x04, 0x00, 0x00, 0x00, 0x96, 0xFF, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x9E, 0xFF, 0xFF, 0xFF, 0x04, 0x00,
        0x00, 0x00, 0x72, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2C, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x24, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x16, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00,
        0x0E, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x4C, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0E, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0E, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x70, 0x61, 0x64, 0x00, 0x01, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x52, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x05, 0x00,
        0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x07, 0x00, 0x08, 0x00, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x04, 0x00, 0x00, 0x00, 0xF6, 0xFF, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x06, 0x00, 0x0A, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0E, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x0E, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0A, 0x00, 0x10, 0x00, 0x08, 0x00, 0x07, 0x00, 0x0C, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00
    };

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(std::string(padModel.begin(), padModel.end()));
    BOOST_CHECK(deserializedNetwork);

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({ 1, 2, 3, 4 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({ 1, 3, 5, 7 }, armnn::DataType::Float32);

    armnn::PadDescriptor descriptor({{ 0, 0 }, { 1, 0 }, { 1, 1 }, { 1, 2 }});

    PadLayerVerifier verifier("pad", { inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializePermute)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Permute)

    const std::string layerName("permute");
    const armnn::TensorInfo inputTensorInfo({4, 3, 2, 1}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo({1, 2, 3, 4}, armnn::DataType::Float32);

    armnn::PermuteDescriptor descriptor(armnn::PermutationVector({3, 2, 1, 0}));

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const permuteLayer = network->AddPermuteLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(permuteLayer->GetInputSlot(0));
    permuteLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    permuteLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    PermuteLayerVerifier verifier(layerName, {inputTensorInfo}, {outputTensorInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializePooling2d)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Pooling2d)

    const std::string layerName("pooling2d");
    const armnn::TensorInfo inputInfo({1, 2, 2, 1}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1, 1, 1, 1}, armnn::DataType::Float32);

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
    armnn::IConnectableLayer* const pooling2dLayer = network->AddPooling2dLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(pooling2dLayer->GetInputSlot(0));
    pooling2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    pooling2dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    Pooling2dLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeQuantize)
{
    DECLARE_LAYER_VERIFIER_CLASS(Quantize)

    const std::string layerName("quantize");
    const armnn::TensorInfo info({ 1, 2, 2, 3 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const quantizeLayer = network->AddQuantizeLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(quantizeLayer->GetInputSlot(0));
    quantizeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    quantizeLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    QuantizeLayerVerifier verifier(layerName, {info}, {info});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeRank)
{
    DECLARE_LAYER_VERIFIER_CLASS(Rank)

    const std::string layerName("rank");
    const armnn::TensorInfo inputInfo({1, 9}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1}, armnn::DataType::Signed32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const rankLayer = network->AddRankLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(rankLayer->GetInputSlot(0));
    rankLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    rankLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    RankLayerVerifier verifier(layerName, {inputInfo}, {outputInfo});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeReshape)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Reshape)

    const std::string layerName("reshape");
    const armnn::TensorInfo inputInfo({1, 9}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({3, 3}, armnn::DataType::Float32);

    armnn::ReshapeDescriptor descriptor({3, 3});

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const reshapeLayer = network->AddReshapeLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(reshapeLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    ReshapeLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeResize)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Resize)

    const std::string layerName("resize");
    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({1, 3, 5, 5}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({1, 3, 2, 4}, armnn::DataType::Float32);

    armnn::ResizeDescriptor desc;
    desc.m_TargetWidth  = 4;
    desc.m_TargetHeight = 2;
    desc.m_Method       = armnn::ResizeMethod::NearestNeighbor;
    desc.m_AlignCorners = true;
    desc.m_HalfPixelCenters = true;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const resizeLayer = network->AddResizeLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(resizeLayer->GetInputSlot(0));
    resizeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    resizeLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    ResizeLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

class ResizeBilinearLayerVerifier : public LayerVerifierBaseWithDescriptor<armnn::ResizeBilinearDescriptor>
{
public:
    ResizeBilinearLayerVerifier(const std::string& layerName,
                                const std::vector<armnn::TensorInfo>& inputInfos,
                                const std::vector<armnn::TensorInfo>& outputInfos,
                                const armnn::ResizeBilinearDescriptor& descriptor)
        : LayerVerifierBaseWithDescriptor<armnn::ResizeBilinearDescriptor>(
            layerName, inputInfos, outputInfos, descriptor) {}

    void VisitResizeLayer(const armnn::IConnectableLayer* layer,
                          const armnn::ResizeDescriptor& descriptor,
                          const char* name) override
    {
        VerifyNameAndConnections(layer, name);

        BOOST_CHECK(descriptor.m_Method             == armnn::ResizeMethod::Bilinear);
        BOOST_CHECK(descriptor.m_TargetWidth        == m_Descriptor.m_TargetWidth);
        BOOST_CHECK(descriptor.m_TargetHeight       == m_Descriptor.m_TargetHeight);
        BOOST_CHECK(descriptor.m_DataLayout         == m_Descriptor.m_DataLayout);
        BOOST_CHECK(descriptor.m_AlignCorners       == m_Descriptor.m_AlignCorners);
        BOOST_CHECK(descriptor.m_HalfPixelCenters   == m_Descriptor.m_HalfPixelCenters);
    }

    void VisitResizeBilinearLayer(const armnn::IConnectableLayer*,
                                  const armnn::ResizeBilinearDescriptor&,
                                  const char*) override
    {
        throw armnn::Exception("ResizeBilinearLayer should have translated to ResizeLayer");
    }
};

// NOTE: Until the deprecated AddResizeBilinearLayer disappears this test checks that
//       calling AddResizeBilinearLayer places a ResizeLayer into the serialized format
//       and that when this deserialises we have a ResizeLayer
BOOST_AUTO_TEST_CASE(SerializeResizeBilinear)
{
    const std::string layerName("resizeBilinear");
    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({1, 3, 5, 5}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({1, 3, 2, 4}, armnn::DataType::Float32);

    armnn::ResizeBilinearDescriptor desc;
    desc.m_TargetWidth  = 4u;
    desc.m_TargetHeight = 2u;
    desc.m_AlignCorners = true;
    desc.m_HalfPixelCenters = true;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const resizeLayer = network->AddResizeBilinearLayer(desc, layerName.c_str());
    ARMNN_NO_DEPRECATE_WARN_END
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(resizeLayer->GetInputSlot(0));
    resizeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    resizeLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    ResizeBilinearLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(EnsureResizeBilinearBackwardCompatibility)
{
    // The hex data below is a flat buffer containing a simple network with an input,
    // a ResizeBilinearLayer (now deprecated) and an output
    //
    // This test verifies that we can still deserialize this old-style model by replacing
    // the ResizeBilinearLayer with an equivalent ResizeLayer
    const std::vector<uint8_t> resizeBilinearModel =
    {
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x0A, 0x00,
        0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x50, 0x01, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xD4, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0B,
        0x04, 0x00, 0x00, 0x00, 0xC2, 0xFE, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00,
        0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x8A, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x38, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0xFF, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x1A, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0E, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x34, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x12, 0x00, 0x08, 0x00, 0x0C, 0x00,
        0x07, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x0E, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0E, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00,
        0x20, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x72, 0x65, 0x73, 0x69, 0x7A, 0x65, 0x42, 0x69, 0x6C, 0x69,
        0x6E, 0x65, 0x61, 0x72, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x52, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x09, 0x04, 0x00, 0x00, 0x00, 0xF6, 0xFF, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
        0x0A, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x14, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0A, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00,
        0x08, 0x00, 0x07, 0x00, 0x0C, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x05, 0x00,
        0x00, 0x00, 0x05, 0x00, 0x00, 0x00
    };

    armnn::INetworkPtr deserializedNetwork =
        DeserializeNetwork(std::string(resizeBilinearModel.begin(), resizeBilinearModel.end()));
    BOOST_CHECK(deserializedNetwork);

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({1, 3, 5, 5}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({1, 3, 2, 4}, armnn::DataType::Float32);

    armnn::ResizeBilinearDescriptor descriptor;
    descriptor.m_TargetWidth  = 4u;
    descriptor.m_TargetHeight = 2u;

    ResizeBilinearLayerVerifier verifier("resizeBilinear", { inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeSlice)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Slice)

    const std::string layerName{"slice"};

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({3, 2, 3, 1}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({2, 2, 2, 1}, armnn::DataType::Float32);

    armnn::SliceDescriptor descriptor({ 0, 0, 1, 0}, {2, 2, 2, 1});

    armnn::INetworkPtr network = armnn::INetwork::Create();

    armnn::IConnectableLayer* const inputLayer  = network->AddInputLayer(0);
    armnn::IConnectableLayer* const sliceLayer  = network->AddSliceLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(sliceLayer->GetInputSlot(0));
    sliceLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    sliceLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    SliceLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeSoftmax)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Softmax)

    const std::string layerName("softmax");
    const armnn::TensorInfo info({1, 10}, armnn::DataType::Float32);

    armnn::SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const softmaxLayer = network->AddSoftmaxLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));
    softmaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    softmaxLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    SoftmaxLayerVerifier verifier(layerName, {info}, {info}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeSpaceToBatchNd)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(SpaceToBatchNd)

    const std::string layerName("spaceToBatchNd");
    const armnn::TensorInfo inputInfo({2, 1, 2, 4}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({8, 1, 1, 3}, armnn::DataType::Float32);

    armnn::SpaceToBatchNdDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;
    desc.m_BlockShape = {2, 2};
    desc.m_PadList = {{0, 0}, {2, 0}};

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const spaceToBatchNdLayer = network->AddSpaceToBatchNdLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(spaceToBatchNdLayer->GetInputSlot(0));
    spaceToBatchNdLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    spaceToBatchNdLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    SpaceToBatchNdLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeSpaceToDepth)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(SpaceToDepth)

    const std::string layerName("spaceToDepth");

    const armnn::TensorInfo inputInfo ({ 1, 16, 8,  3 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1,  8, 4, 12 }, armnn::DataType::Float32);

    armnn::SpaceToDepthDescriptor desc;
    desc.m_BlockSize  = 2;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer        = network->AddInputLayer(0);
    armnn::IConnectableLayer* const spaceToDepthLayer = network->AddSpaceToDepthLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer       = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(spaceToDepthLayer->GetInputSlot(0));
    spaceToDepthLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    spaceToDepthLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    SpaceToDepthLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeSplitter)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Splitter)

    const unsigned int numViews = 3;
    const unsigned int numDimensions = 4;
    const unsigned int inputShape[] = {1, 18, 4, 4};
    const unsigned int outputShape[] = {1, 6, 4, 4};

    // This is modelled on how the caffe parser sets up a splitter layer to partition an input along dimension one.
    unsigned int splitterDimSizes[4] = {static_cast<unsigned int>(inputShape[0]),
                                        static_cast<unsigned int>(inputShape[1]),
                                        static_cast<unsigned int>(inputShape[2]),
                                        static_cast<unsigned int>(inputShape[3])};
    splitterDimSizes[1] /= numViews;
    armnn::ViewsDescriptor desc(numViews, numDimensions);

    for (unsigned int g = 0; g < numViews; ++g)
    {
        desc.SetViewOriginCoord(g, 1, splitterDimSizes[1] * g);

        for (unsigned int dimIdx=0; dimIdx < 4; dimIdx++)
        {
            desc.SetViewSize(g, dimIdx, splitterDimSizes[dimIdx]);
        }
    }

    const std::string layerName("splitter");
    const armnn::TensorInfo inputInfo(numDimensions, inputShape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo(numDimensions, outputShape, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const splitterLayer = network->AddSplitterLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer0 = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const outputLayer1 = network->AddOutputLayer(1);
    armnn::IConnectableLayer* const outputLayer2 = network->AddOutputLayer(2);

    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(0).Connect(outputLayer0->GetInputSlot(0));
    splitterLayer->GetOutputSlot(1).Connect(outputLayer1->GetInputSlot(0));
    splitterLayer->GetOutputSlot(2).Connect(outputLayer2->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    splitterLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    splitterLayer->GetOutputSlot(1).SetTensorInfo(outputInfo);
    splitterLayer->GetOutputSlot(2).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    SplitterLayerVerifier verifier(layerName, {inputInfo}, {outputInfo, outputInfo, outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeStack)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Stack)

    const std::string layerName("stack");

    armnn::TensorInfo inputTensorInfo ({4, 3, 5}, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo({4, 3, 2, 5}, armnn::DataType::Float32);

    armnn::StackDescriptor descriptor(2, 2, {4, 3, 5});

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer2 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const stackLayer = network->AddStackLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer1->GetOutputSlot(0).Connect(stackLayer->GetInputSlot(0));
    inputLayer2->GetOutputSlot(0).Connect(stackLayer->GetInputSlot(1));
    stackLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    inputLayer2->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    stackLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    StackLayerVerifier verifier(layerName, {inputTensorInfo, inputTensorInfo}, {outputTensorInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeStandIn)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(StandIn)

    const std::string layerName("standIn");

    armnn::TensorInfo tensorInfo({ 1u }, armnn::DataType::Float32);
    armnn::StandInDescriptor descriptor(2u, 2u);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0  = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1  = network->AddInputLayer(1);
    armnn::IConnectableLayer* const standInLayer = network->AddStandInLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer0 = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const outputLayer1 = network->AddOutputLayer(1);

    inputLayer0->GetOutputSlot(0).Connect(standInLayer->GetInputSlot(0));
    inputLayer0->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    inputLayer1->GetOutputSlot(0).Connect(standInLayer->GetInputSlot(1));
    inputLayer1->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    standInLayer->GetOutputSlot(0).Connect(outputLayer0->GetInputSlot(0));
    standInLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    standInLayer->GetOutputSlot(1).Connect(outputLayer1->GetInputSlot(0));
    standInLayer->GetOutputSlot(1).SetTensorInfo(tensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    StandInLayerVerifier verifier(layerName, { tensorInfo, tensorInfo }, { tensorInfo, tensorInfo }, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeStridedSlice)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(StridedSlice)

    const std::string layerName("stridedSlice");
    const armnn::TensorInfo inputInfo = armnn::TensorInfo({3, 2, 3, 1}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({3, 1}, armnn::DataType::Float32);

    armnn::StridedSliceDescriptor desc({0, 0, 1, 0}, {1, 1, 1, 1}, {1, 1, 1, 1});
    desc.m_EndMask = (1 << 4) - 1;
    desc.m_ShrinkAxisMask = (1 << 1) | (1 << 2);
    desc.m_DataLayout = armnn::DataLayout::NCHW;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const stridedSliceLayer = network->AddStridedSliceLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(stridedSliceLayer->GetInputSlot(0));
    stridedSliceLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    stridedSliceLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    StridedSliceLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeSubtraction)
{
    DECLARE_LAYER_VERIFIER_CLASS(Subtraction)

    const std::string layerName("subtraction");
    const armnn::TensorInfo info({ 1, 4 }, armnn::DataType::Float32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* const subtractionLayer = network->AddSubtractionLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(subtractionLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(subtractionLayer->GetInputSlot(1));
    subtractionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(info);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(info);
    subtractionLayer->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    SubtractionLayerVerifier verifier(layerName, {info, info}, {info});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeSwitch)
{
    class SwitchLayerVerifier : public LayerVerifierBase
    {
    public:
        SwitchLayerVerifier(const std::string& layerName,
                            const std::vector<armnn::TensorInfo>& inputInfos,
                            const std::vector<armnn::TensorInfo>& outputInfos)
            : LayerVerifierBase(layerName, inputInfos, outputInfos) {}

        void VisitSwitchLayer(const armnn::IConnectableLayer* layer, const char* name) override
        {
            VerifyNameAndConnections(layer, name);
        }

        void VisitConstantLayer(const armnn::IConnectableLayer*,
                                const armnn::ConstTensor&,
                                const char*) override {}
    };

    const std::string layerName("switch");
    const armnn::TensorInfo info({ 1, 4 }, armnn::DataType::Float32);

    std::vector<float> constantData = GenerateRandomData<float>(info.GetNumElements());
    armnn::ConstTensor constTensor(info, constantData);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const constantLayer = network->AddConstantLayer(constTensor, "constant");
    armnn::IConnectableLayer* const switchLayer = network->AddSwitchLayer(layerName.c_str());
    armnn::IConnectableLayer* const trueOutputLayer = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const falseOutputLayer = network->AddOutputLayer(1);

    inputLayer->GetOutputSlot(0).Connect(switchLayer->GetInputSlot(0));
    constantLayer->GetOutputSlot(0).Connect(switchLayer->GetInputSlot(1));
    switchLayer->GetOutputSlot(0).Connect(trueOutputLayer->GetInputSlot(0));
    switchLayer->GetOutputSlot(1).Connect(falseOutputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    constantLayer->GetOutputSlot(0).SetTensorInfo(info);
    switchLayer->GetOutputSlot(0).SetTensorInfo(info);
    switchLayer->GetOutputSlot(1).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    SwitchLayerVerifier verifier(layerName, {info, info}, {info, info});
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeTranspose)
{
    DECLARE_LAYER_VERIFIER_CLASS_WITH_DESCRIPTOR(Transpose)

    const std::string layerName("transpose");
    const armnn::TensorInfo inputTensorInfo({4, 3, 2, 1}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo({1, 2, 3, 4}, armnn::DataType::Float32);

    armnn::TransposeDescriptor descriptor(armnn::PermutationVector({3, 2, 1, 0}));

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const transposeLayer = network->AddTransposeLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(transposeLayer->GetInputSlot(0));
    transposeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    transposeLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    TransposeLayerVerifier verifier(layerName, {inputTensorInfo}, {outputTensorInfo}, descriptor);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeTransposeConvolution2d)
{
    using Descriptor = armnn::TransposeConvolution2dDescriptor;
    class TransposeConvolution2dLayerVerifier : public LayerVerifierBaseWithDescriptor<Descriptor>
    {
    public:
        TransposeConvolution2dLayerVerifier(const std::string& layerName,
                                            const std::vector<armnn::TensorInfo>& inputInfos,
                                            const std::vector<armnn::TensorInfo>& outputInfos,
                                            const Descriptor& descriptor,
                                            const armnn::ConstTensor& weights,
                                            const armnn::Optional<armnn::ConstTensor>& biases)
            : LayerVerifierBaseWithDescriptor<Descriptor>(layerName, inputInfos, outputInfos, descriptor)
            , m_Weights(weights)
            , m_Biases(biases)
        {}

        void VisitTransposeConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                              const Descriptor& descriptor,
                                              const armnn::ConstTensor& weights,
                                              const armnn::Optional<armnn::ConstTensor>& biases,
                                              const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            VerifyDescriptor(descriptor);

            // check weights
            CompareConstTensor(weights, m_Weights);

            // check biases
            BOOST_CHECK(biases.has_value() == descriptor.m_BiasEnabled);
            BOOST_CHECK(biases.has_value() == m_Biases.has_value());

            if (biases.has_value() && m_Biases.has_value())
            {
                CompareConstTensor(biases.value(), m_Biases.value());
            }
        }

    private:
        armnn::ConstTensor                      m_Weights;
        armnn::Optional<armnn::ConstTensor>     m_Biases;
    };

    const std::string layerName("transposeConvolution2d");
    const armnn::TensorInfo inputInfo ({ 1, 7, 7, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 9, 9, 1 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32);

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);

    std::vector<float> biasesData = GenerateRandomData<float>(biasesInfo.GetNumElements());
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::TransposeConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 1;
    descriptor.m_PadRight    = 1;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 1;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer  = network->AddInputLayer(0);
    armnn::IConnectableLayer* const convLayer   =
            network->AddTransposeConvolution2dLayer(descriptor,
                                                    weights,
                                                    armnn::Optional<armnn::ConstTensor>(biases),
                                                    layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    TransposeConvolution2dLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, descriptor, weights, biases);
    deserializedNetwork->Accept(verifier);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeNonLinearNetwork)
{
    class ConstantLayerVerifier : public LayerVerifierBase
    {
    public:
        ConstantLayerVerifier(const std::string& layerName,
                              const std::vector<armnn::TensorInfo>& inputInfos,
                              const std::vector<armnn::TensorInfo>& outputInfos,
                              const armnn::ConstTensor& layerInput)
            : LayerVerifierBase(layerName, inputInfos, outputInfos)
            , m_LayerInput(layerInput) {}

        void VisitConstantLayer(const armnn::IConnectableLayer* layer,
                                const armnn::ConstTensor& input,
                                const char* name) override
        {
            VerifyNameAndConnections(layer, name);
            CompareConstTensor(input, m_LayerInput);
        }

        void VisitAdditionLayer(const armnn::IConnectableLayer*, const char*) override {}

    private:
        armnn::ConstTensor m_LayerInput;
    };

    const std::string layerName("constant");
    const armnn::TensorInfo info({ 2, 3 }, armnn::DataType::Float32);

    std::vector<float> constantData = GenerateRandomData<float>(info.GetNumElements());
    armnn::ConstTensor constTensor(info, constantData);

    armnn::INetworkPtr network(armnn::INetwork::Create());
    armnn::IConnectableLayer* input = network->AddInputLayer(0);
    armnn::IConnectableLayer* add = network->AddAdditionLayer();
    armnn::IConnectableLayer* constant = network->AddConstantLayer(constTensor, layerName.c_str());
    armnn::IConnectableLayer* output = network->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(info);
    constant->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    ConstantLayerVerifier verifier(layerName, {}, {info}, constTensor);
    deserializedNetwork->Accept(verifier);
}

class VerifyLstmLayer : public LayerVerifierBaseWithDescriptor<armnn::LstmDescriptor>
{
public:
    VerifyLstmLayer(const std::string& layerName,
                    const std::vector<armnn::TensorInfo>& inputInfos,
                    const std::vector<armnn::TensorInfo>& outputInfos,
                    const armnn::LstmDescriptor& descriptor,
                    const armnn::LstmInputParams& inputParams)
        : LayerVerifierBaseWithDescriptor<armnn::LstmDescriptor>(layerName, inputInfos, outputInfos, descriptor)
        , m_InputParams(inputParams) {}

    void VisitLstmLayer(const armnn::IConnectableLayer* layer,
                        const armnn::LstmDescriptor& descriptor,
                        const armnn::LstmInputParams& params,
                        const char* name)
    {
        VerifyNameAndConnections(layer, name);
        VerifyDescriptor(descriptor);
        VerifyInputParameters(params);
    }

protected:
    void VerifyInputParameters(const armnn::LstmInputParams& params)
    {
        VerifyConstTensors(
            "m_InputToInputWeights", m_InputParams.m_InputToInputWeights, params.m_InputToInputWeights);
        VerifyConstTensors(
            "m_InputToForgetWeights", m_InputParams.m_InputToForgetWeights, params.m_InputToForgetWeights);
        VerifyConstTensors(
            "m_InputToCellWeights", m_InputParams.m_InputToCellWeights, params.m_InputToCellWeights);
        VerifyConstTensors(
            "m_InputToOutputWeights", m_InputParams.m_InputToOutputWeights, params.m_InputToOutputWeights);
        VerifyConstTensors(
            "m_RecurrentToInputWeights", m_InputParams.m_RecurrentToInputWeights, params.m_RecurrentToInputWeights);
        VerifyConstTensors(
            "m_RecurrentToForgetWeights", m_InputParams.m_RecurrentToForgetWeights, params.m_RecurrentToForgetWeights);
        VerifyConstTensors(
            "m_RecurrentToCellWeights", m_InputParams.m_RecurrentToCellWeights, params.m_RecurrentToCellWeights);
        VerifyConstTensors(
            "m_RecurrentToOutputWeights", m_InputParams.m_RecurrentToOutputWeights, params.m_RecurrentToOutputWeights);
        VerifyConstTensors(
            "m_CellToInputWeights", m_InputParams.m_CellToInputWeights, params.m_CellToInputWeights);
        VerifyConstTensors(
            "m_CellToForgetWeights", m_InputParams.m_CellToForgetWeights, params.m_CellToForgetWeights);
        VerifyConstTensors(
            "m_CellToOutputWeights", m_InputParams.m_CellToOutputWeights, params.m_CellToOutputWeights);
        VerifyConstTensors(
            "m_InputGateBias", m_InputParams.m_InputGateBias, params.m_InputGateBias);
        VerifyConstTensors(
            "m_ForgetGateBias", m_InputParams.m_ForgetGateBias, params.m_ForgetGateBias);
        VerifyConstTensors(
            "m_CellBias", m_InputParams.m_CellBias, params.m_CellBias);
        VerifyConstTensors(
            "m_OutputGateBias", m_InputParams.m_OutputGateBias, params.m_OutputGateBias);
        VerifyConstTensors(
            "m_ProjectionWeights", m_InputParams.m_ProjectionWeights, params.m_ProjectionWeights);
        VerifyConstTensors(
            "m_ProjectionBias", m_InputParams.m_ProjectionBias, params.m_ProjectionBias);
        VerifyConstTensors(
            "m_InputLayerNormWeights", m_InputParams.m_InputLayerNormWeights, params.m_InputLayerNormWeights);
        VerifyConstTensors(
            "m_ForgetLayerNormWeights", m_InputParams.m_ForgetLayerNormWeights, params.m_ForgetLayerNormWeights);
        VerifyConstTensors(
            "m_CellLayerNormWeights", m_InputParams.m_CellLayerNormWeights, params.m_CellLayerNormWeights);
        VerifyConstTensors(
            "m_OutputLayerNormWeights", m_InputParams.m_OutputLayerNormWeights, params.m_OutputLayerNormWeights);
    }

private:
    armnn::LstmInputParams m_InputParams;
};

BOOST_AUTO_TEST_CASE(SerializeDeserializeLstmCifgPeepholeNoProjection)
{
    armnn::LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 4;
    descriptor.m_ClippingThresProj = 0.0f;
    descriptor.m_ClippingThresCell = 0.0f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_ProjectionEnabled = false;
    descriptor.m_PeepholeEnabled = true;

    const uint32_t batchSize = 1;
    const uint32_t inputSize = 2;
    const uint32_t numUnits = 4;
    const uint32_t outputSize = numUnits;

    armnn::TensorInfo inputWeightsInfo1({numUnits, inputSize}, armnn::DataType::Float32);
    std::vector<float> inputToForgetWeightsData = GenerateRandomData<float>(inputWeightsInfo1.GetNumElements());
    armnn::ConstTensor inputToForgetWeights(inputWeightsInfo1, inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = GenerateRandomData<float>(inputWeightsInfo1.GetNumElements());
    armnn::ConstTensor inputToCellWeights(inputWeightsInfo1, inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = GenerateRandomData<float>(inputWeightsInfo1.GetNumElements());
    armnn::ConstTensor inputToOutputWeights(inputWeightsInfo1, inputToOutputWeightsData);

    armnn::TensorInfo inputWeightsInfo2({numUnits, outputSize}, armnn::DataType::Float32);
    std::vector<float> recurrentToForgetWeightsData = GenerateRandomData<float>(inputWeightsInfo2.GetNumElements());
    armnn::ConstTensor recurrentToForgetWeights(inputWeightsInfo2, recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = GenerateRandomData<float>(inputWeightsInfo2.GetNumElements());
    armnn::ConstTensor recurrentToCellWeights(inputWeightsInfo2, recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = GenerateRandomData<float>(inputWeightsInfo2.GetNumElements());
    armnn::ConstTensor recurrentToOutputWeights(inputWeightsInfo2, recurrentToOutputWeightsData);

    armnn::TensorInfo inputWeightsInfo3({numUnits}, armnn::DataType::Float32);
    std::vector<float> cellToForgetWeightsData = GenerateRandomData<float>(inputWeightsInfo3.GetNumElements());
    armnn::ConstTensor cellToForgetWeights(inputWeightsInfo3, cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData = GenerateRandomData<float>(inputWeightsInfo3.GetNumElements());
    armnn::ConstTensor cellToOutputWeights(inputWeightsInfo3, cellToOutputWeightsData);

    std::vector<float> forgetGateBiasData(numUnits, 1.0f);
    armnn::ConstTensor forgetGateBias(inputWeightsInfo3, forgetGateBiasData);

    std::vector<float> cellBiasData(numUnits, 0.0f);
    armnn::ConstTensor cellBias(inputWeightsInfo3, cellBiasData);

    std::vector<float> outputGateBiasData(numUnits, 0.0f);
    armnn::ConstTensor outputGateBias(inputWeightsInfo3, outputGateBiasData);

    armnn::LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;
    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const cellStateIn = network->AddInputLayer(1);
    armnn::IConnectableLayer* const outputStateIn = network->AddInputLayer(2);
    const std::string layerName("lstm");
    armnn::IConnectableLayer* const lstmLayer = network->AddLstmLayer(descriptor, params, layerName.c_str());
    armnn::IConnectableLayer* const scratchBuffer  = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const outputStateOut  = network->AddOutputLayer(1);
    armnn::IConnectableLayer* const cellStateOut  = network->AddOutputLayer(2);
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(3);

    // connect up
    armnn::TensorInfo inputTensorInfo({ batchSize, inputSize }, armnn::DataType::Float32);
    armnn::TensorInfo cellStateTensorInfo({ batchSize, numUnits}, armnn::DataType::Float32);
    armnn::TensorInfo outputStateTensorInfo({ batchSize, outputSize }, armnn::DataType::Float32);
    armnn::TensorInfo lstmTensorInfoScratchBuff({ batchSize, numUnits * 3 }, armnn::DataType::Float32);

    inputLayer->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    outputStateIn->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(1));
    outputStateIn->GetOutputSlot(0).SetTensorInfo(outputStateTensorInfo);

    cellStateIn->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(2));
    cellStateIn->GetOutputSlot(0).SetTensorInfo(cellStateTensorInfo);

    lstmLayer->GetOutputSlot(0).Connect(scratchBuffer->GetInputSlot(0));
    lstmLayer->GetOutputSlot(0).SetTensorInfo(lstmTensorInfoScratchBuff);

    lstmLayer->GetOutputSlot(1).Connect(outputStateOut->GetInputSlot(0));
    lstmLayer->GetOutputSlot(1).SetTensorInfo(outputStateTensorInfo);

    lstmLayer->GetOutputSlot(2).Connect(cellStateOut->GetInputSlot(0));
    lstmLayer->GetOutputSlot(2).SetTensorInfo(cellStateTensorInfo);

    lstmLayer->GetOutputSlot(3).Connect(outputLayer->GetInputSlot(0));
    lstmLayer->GetOutputSlot(3).SetTensorInfo(outputStateTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyLstmLayer checker(
        layerName,
        {inputTensorInfo, outputStateTensorInfo, cellStateTensorInfo},
        {lstmTensorInfoScratchBuff, outputStateTensorInfo, cellStateTensorInfo, outputStateTensorInfo},
        descriptor,
        params);
    deserializedNetwork->Accept(checker);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeLstmNoCifgWithPeepholeAndProjection)
{
    armnn::LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 4;
    descriptor.m_ClippingThresProj = 0.0f;
    descriptor.m_ClippingThresCell = 0.0f;
    descriptor.m_CifgEnabled = false; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_ProjectionEnabled = true;
    descriptor.m_PeepholeEnabled = true;

    const uint32_t batchSize = 2;
    const uint32_t inputSize = 5;
    const uint32_t numUnits = 20;
    const uint32_t outputSize = 16;

    armnn::TensorInfo tensorInfo20x5({numUnits, inputSize}, armnn::DataType::Float32);
    std::vector<float> inputToInputWeightsData = GenerateRandomData<float>(tensorInfo20x5.GetNumElements());
    armnn::ConstTensor inputToInputWeights(tensorInfo20x5, inputToInputWeightsData);

    std::vector<float> inputToForgetWeightsData = GenerateRandomData<float>(tensorInfo20x5.GetNumElements());
    armnn::ConstTensor inputToForgetWeights(tensorInfo20x5, inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = GenerateRandomData<float>(tensorInfo20x5.GetNumElements());
    armnn::ConstTensor inputToCellWeights(tensorInfo20x5, inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = GenerateRandomData<float>(tensorInfo20x5.GetNumElements());
    armnn::ConstTensor inputToOutputWeights(tensorInfo20x5, inputToOutputWeightsData);

    armnn::TensorInfo tensorInfo20({numUnits}, armnn::DataType::Float32);
    std::vector<float> inputGateBiasData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor inputGateBias(tensorInfo20, inputGateBiasData);

    std::vector<float> forgetGateBiasData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor forgetGateBias(tensorInfo20, forgetGateBiasData);

    std::vector<float> cellBiasData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellBias(tensorInfo20, cellBiasData);

    std::vector<float> outputGateBiasData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor outputGateBias(tensorInfo20, outputGateBiasData);

    armnn::TensorInfo tensorInfo20x16({numUnits, outputSize}, armnn::DataType::Float32);
    std::vector<float> recurrentToInputWeightsData = GenerateRandomData<float>(tensorInfo20x16.GetNumElements());
    armnn::ConstTensor recurrentToInputWeights(tensorInfo20x16, recurrentToInputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = GenerateRandomData<float>(tensorInfo20x16.GetNumElements());
    armnn::ConstTensor recurrentToForgetWeights(tensorInfo20x16, recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = GenerateRandomData<float>(tensorInfo20x16.GetNumElements());
    armnn::ConstTensor recurrentToCellWeights(tensorInfo20x16, recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = GenerateRandomData<float>(tensorInfo20x16.GetNumElements());
    armnn::ConstTensor recurrentToOutputWeights(tensorInfo20x16, recurrentToOutputWeightsData);

    std::vector<float> cellToInputWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellToInputWeights(tensorInfo20, cellToInputWeightsData);

    std::vector<float> cellToForgetWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellToForgetWeights(tensorInfo20, cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellToOutputWeights(tensorInfo20,  cellToOutputWeightsData);

    armnn::TensorInfo tensorInfo16x20({outputSize, numUnits}, armnn::DataType::Float32);
    std::vector<float> projectionWeightsData = GenerateRandomData<float>(tensorInfo16x20.GetNumElements());
    armnn::ConstTensor projectionWeights(tensorInfo16x20, projectionWeightsData);

    armnn::TensorInfo tensorInfo16({outputSize}, armnn::DataType::Float32);
    std::vector<float> projectionBiasData(outputSize, 0.f);
    armnn::ConstTensor projectionBias(tensorInfo16, projectionBiasData);

    armnn::LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    // additional params because: descriptor.m_CifgEnabled = false
    params.m_InputToInputWeights = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_CellToInputWeights = &cellToInputWeights;
    params.m_InputGateBias = &inputGateBias;

    // additional params because: descriptor.m_ProjectionEnabled = true
    params.m_ProjectionWeights = &projectionWeights;
    params.m_ProjectionBias = &projectionBias;

    // additional params because: descriptor.m_PeepholeEnabled = true
    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const cellStateIn = network->AddInputLayer(1);
    armnn::IConnectableLayer* const outputStateIn = network->AddInputLayer(2);
    const std::string layerName("lstm");
    armnn::IConnectableLayer* const lstmLayer = network->AddLstmLayer(descriptor, params, layerName.c_str());
    armnn::IConnectableLayer* const scratchBuffer  = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const outputStateOut  = network->AddOutputLayer(1);
    armnn::IConnectableLayer* const cellStateOut  = network->AddOutputLayer(2);
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(3);

    // connect up
    armnn::TensorInfo inputTensorInfo({ batchSize, inputSize }, armnn::DataType::Float32);
    armnn::TensorInfo cellStateTensorInfo({ batchSize, numUnits}, armnn::DataType::Float32);
    armnn::TensorInfo outputStateTensorInfo({ batchSize, outputSize }, armnn::DataType::Float32);
    armnn::TensorInfo lstmTensorInfoScratchBuff({ batchSize, numUnits * 4 }, armnn::DataType::Float32);

    inputLayer->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    outputStateIn->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(1));
    outputStateIn->GetOutputSlot(0).SetTensorInfo(outputStateTensorInfo);

    cellStateIn->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(2));
    cellStateIn->GetOutputSlot(0).SetTensorInfo(cellStateTensorInfo);

    lstmLayer->GetOutputSlot(0).Connect(scratchBuffer->GetInputSlot(0));
    lstmLayer->GetOutputSlot(0).SetTensorInfo(lstmTensorInfoScratchBuff);

    lstmLayer->GetOutputSlot(1).Connect(outputStateOut->GetInputSlot(0));
    lstmLayer->GetOutputSlot(1).SetTensorInfo(outputStateTensorInfo);

    lstmLayer->GetOutputSlot(2).Connect(cellStateOut->GetInputSlot(0));
    lstmLayer->GetOutputSlot(2).SetTensorInfo(cellStateTensorInfo);

    lstmLayer->GetOutputSlot(3).Connect(outputLayer->GetInputSlot(0));
    lstmLayer->GetOutputSlot(3).SetTensorInfo(outputStateTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyLstmLayer checker(
        layerName,
        {inputTensorInfo, outputStateTensorInfo, cellStateTensorInfo},
        {lstmTensorInfoScratchBuff, outputStateTensorInfo, cellStateTensorInfo, outputStateTensorInfo},
        descriptor,
        params);
    deserializedNetwork->Accept(checker);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeLstmNoCifgWithPeepholeWithProjectionWithLayerNorm)
{
    armnn::LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 4;
    descriptor.m_ClippingThresProj = 0.0f;
    descriptor.m_ClippingThresCell = 0.0f;
    descriptor.m_CifgEnabled = false; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_ProjectionEnabled = true;
    descriptor.m_PeepholeEnabled = true;
    descriptor.m_LayerNormEnabled = true;

    const uint32_t batchSize = 2;
    const uint32_t inputSize = 5;
    const uint32_t numUnits = 20;
    const uint32_t outputSize = 16;

    armnn::TensorInfo tensorInfo20x5({numUnits, inputSize}, armnn::DataType::Float32);
    std::vector<float> inputToInputWeightsData = GenerateRandomData<float>(tensorInfo20x5.GetNumElements());
    armnn::ConstTensor inputToInputWeights(tensorInfo20x5, inputToInputWeightsData);

    std::vector<float> inputToForgetWeightsData = GenerateRandomData<float>(tensorInfo20x5.GetNumElements());
    armnn::ConstTensor inputToForgetWeights(tensorInfo20x5, inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = GenerateRandomData<float>(tensorInfo20x5.GetNumElements());
    armnn::ConstTensor inputToCellWeights(tensorInfo20x5, inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = GenerateRandomData<float>(tensorInfo20x5.GetNumElements());
    armnn::ConstTensor inputToOutputWeights(tensorInfo20x5, inputToOutputWeightsData);

    armnn::TensorInfo tensorInfo20({numUnits}, armnn::DataType::Float32);
    std::vector<float> inputGateBiasData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor inputGateBias(tensorInfo20, inputGateBiasData);

    std::vector<float> forgetGateBiasData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor forgetGateBias(tensorInfo20, forgetGateBiasData);

    std::vector<float> cellBiasData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellBias(tensorInfo20, cellBiasData);

    std::vector<float> outputGateBiasData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor outputGateBias(tensorInfo20, outputGateBiasData);

    armnn::TensorInfo tensorInfo20x16({numUnits, outputSize}, armnn::DataType::Float32);
    std::vector<float> recurrentToInputWeightsData = GenerateRandomData<float>(tensorInfo20x16.GetNumElements());
    armnn::ConstTensor recurrentToInputWeights(tensorInfo20x16, recurrentToInputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = GenerateRandomData<float>(tensorInfo20x16.GetNumElements());
    armnn::ConstTensor recurrentToForgetWeights(tensorInfo20x16, recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = GenerateRandomData<float>(tensorInfo20x16.GetNumElements());
    armnn::ConstTensor recurrentToCellWeights(tensorInfo20x16, recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = GenerateRandomData<float>(tensorInfo20x16.GetNumElements());
    armnn::ConstTensor recurrentToOutputWeights(tensorInfo20x16, recurrentToOutputWeightsData);

    std::vector<float> cellToInputWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellToInputWeights(tensorInfo20, cellToInputWeightsData);

    std::vector<float> cellToForgetWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellToForgetWeights(tensorInfo20, cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellToOutputWeights(tensorInfo20,  cellToOutputWeightsData);

    armnn::TensorInfo tensorInfo16x20({outputSize, numUnits}, armnn::DataType::Float32);
    std::vector<float> projectionWeightsData = GenerateRandomData<float>(tensorInfo16x20.GetNumElements());
    armnn::ConstTensor projectionWeights(tensorInfo16x20, projectionWeightsData);

    armnn::TensorInfo tensorInfo16({outputSize}, armnn::DataType::Float32);
    std::vector<float> projectionBiasData(outputSize, 0.f);
    armnn::ConstTensor projectionBias(tensorInfo16, projectionBiasData);

    std::vector<float> inputLayerNormWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor inputLayerNormWeights(tensorInfo20, forgetGateBiasData);

    std::vector<float> forgetLayerNormWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor forgetLayerNormWeights(tensorInfo20, forgetGateBiasData);

    std::vector<float> cellLayerNormWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor cellLayerNormWeights(tensorInfo20, forgetGateBiasData);

    std::vector<float> outLayerNormWeightsData = GenerateRandomData<float>(tensorInfo20.GetNumElements());
    armnn::ConstTensor outLayerNormWeights(tensorInfo20, forgetGateBiasData);

    armnn::LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    // additional params because: descriptor.m_CifgEnabled = false
    params.m_InputToInputWeights = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_CellToInputWeights = &cellToInputWeights;
    params.m_InputGateBias = &inputGateBias;

    // additional params because: descriptor.m_ProjectionEnabled = true
    params.m_ProjectionWeights = &projectionWeights;
    params.m_ProjectionBias = &projectionBias;

    // additional params because: descriptor.m_PeepholeEnabled = true
    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    // additional params because: despriptor.m_LayerNormEnabled = true
    params.m_InputLayerNormWeights = &inputLayerNormWeights;
    params.m_ForgetLayerNormWeights = &forgetLayerNormWeights;
    params.m_CellLayerNormWeights = &cellLayerNormWeights;
    params.m_OutputLayerNormWeights = &outLayerNormWeights;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const cellStateIn = network->AddInputLayer(1);
    armnn::IConnectableLayer* const outputStateIn = network->AddInputLayer(2);
    const std::string layerName("lstm");
    armnn::IConnectableLayer* const lstmLayer = network->AddLstmLayer(descriptor, params, layerName.c_str());
    armnn::IConnectableLayer* const scratchBuffer  = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const outputStateOut  = network->AddOutputLayer(1);
    armnn::IConnectableLayer* const cellStateOut  = network->AddOutputLayer(2);
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(3);

    // connect up
    armnn::TensorInfo inputTensorInfo({ batchSize, inputSize }, armnn::DataType::Float32);
    armnn::TensorInfo cellStateTensorInfo({ batchSize, numUnits}, armnn::DataType::Float32);
    armnn::TensorInfo outputStateTensorInfo({ batchSize, outputSize }, armnn::DataType::Float32);
    armnn::TensorInfo lstmTensorInfoScratchBuff({ batchSize, numUnits * 4 }, armnn::DataType::Float32);

    inputLayer->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    outputStateIn->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(1));
    outputStateIn->GetOutputSlot(0).SetTensorInfo(outputStateTensorInfo);

    cellStateIn->GetOutputSlot(0).Connect(lstmLayer->GetInputSlot(2));
    cellStateIn->GetOutputSlot(0).SetTensorInfo(cellStateTensorInfo);

    lstmLayer->GetOutputSlot(0).Connect(scratchBuffer->GetInputSlot(0));
    lstmLayer->GetOutputSlot(0).SetTensorInfo(lstmTensorInfoScratchBuff);

    lstmLayer->GetOutputSlot(1).Connect(outputStateOut->GetInputSlot(0));
    lstmLayer->GetOutputSlot(1).SetTensorInfo(outputStateTensorInfo);

    lstmLayer->GetOutputSlot(2).Connect(cellStateOut->GetInputSlot(0));
    lstmLayer->GetOutputSlot(2).SetTensorInfo(cellStateTensorInfo);

    lstmLayer->GetOutputSlot(3).Connect(outputLayer->GetInputSlot(0));
    lstmLayer->GetOutputSlot(3).SetTensorInfo(outputStateTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyLstmLayer checker(
            layerName,
            {inputTensorInfo, outputStateTensorInfo, cellStateTensorInfo},
            {lstmTensorInfoScratchBuff, outputStateTensorInfo, cellStateTensorInfo, outputStateTensorInfo},
            descriptor,
            params);
    deserializedNetwork->Accept(checker);
}

BOOST_AUTO_TEST_CASE(EnsureLstmLayersBackwardCompatibility)
{
    // The hex data below is a flat buffer containing a lstm layer with no Cifg, with peephole and projection
    // enabled. That data was obtained before additional layer normalization parameters where added to the
    // lstm serializer. That way it can be tested if a lstm model with the old parameter configuration can
    // still be loaded
    const std::vector<uint8_t> lstmNoCifgWithPeepholeAndProjectionModel =
    {
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x0A, 0x00,
        0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x2C, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
        0xDC, 0x29, 0x00, 0x00, 0x38, 0x29, 0x00, 0x00, 0xB4, 0x28, 0x00, 0x00, 0x94, 0x01, 0x00, 0x00, 0x3C, 0x01,
        0x00, 0x00, 0xE0, 0x00, 0x00, 0x00, 0x84, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x70, 0xD6, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x0B, 0x04, 0x00, 0x00, 0x00, 0x06, 0xD7, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x88, 0xD7,
        0xFF, 0xFF, 0x08, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xF6, 0xD6, 0xFF, 0xFF, 0x07, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xE8, 0xD7, 0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xC8, 0xD6, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x0B, 0x04, 0x00, 0x00, 0x00, 0x5E, 0xD7, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0xE0, 0xD7, 0xFF, 0xFF,
        0x08, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x4E, 0xD7, 0xFF, 0xFF, 0x06, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0xD8,
        0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x20, 0xD7, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0B,
        0x04, 0x00, 0x00, 0x00, 0xB6, 0xD7, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x38, 0xD8, 0xFF, 0xFF, 0x08, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xA6, 0xD7, 0xFF, 0xFF, 0x05, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x98, 0xD8, 0xFF, 0xFF,
        0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x78, 0xD7, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0B, 0x04, 0x00,
        0x00, 0x00, 0x0E, 0xD8, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x16, 0xD8, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00,
        0xFA, 0xD7, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xEC, 0xD8, 0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x6C, 0xD8, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x23, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00,
        0x12, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x0A, 0x00, 0x00, 0x00, 0xE0, 0x25, 0x00, 0x00, 0xD0, 0x25,
        0x00, 0x00, 0x2C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x26, 0x00, 0x48, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00,
        0x10, 0x00, 0x14, 0x00, 0x18, 0x00, 0x1C, 0x00, 0x20, 0x00, 0x24, 0x00, 0x28, 0x00, 0x2C, 0x00, 0x30, 0x00,
        0x34, 0x00, 0x38, 0x00, 0x3C, 0x00, 0x40, 0x00, 0x44, 0x00, 0x26, 0x00, 0x00, 0x00, 0xC4, 0x23, 0x00, 0x00,
        0xF8, 0x21, 0x00, 0x00, 0x2C, 0x20, 0x00, 0x00, 0xF0, 0x1A, 0x00, 0x00, 0xB4, 0x15, 0x00, 0x00, 0x78, 0x10,
        0x00, 0x00, 0xF0, 0x0F, 0x00, 0x00, 0x68, 0x0F, 0x00, 0x00, 0xE0, 0x0E, 0x00, 0x00, 0x14, 0x0D, 0x00, 0x00,
        0xD8, 0x07, 0x00, 0x00, 0x50, 0x07, 0x00, 0x00, 0xC8, 0x06, 0x00, 0x00, 0x8C, 0x01, 0x00, 0x00, 0x14, 0x01,
        0x00, 0x00, 0x8C, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xEE, 0xD7, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03,
        0x64, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFE, 0xD8, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x14, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5A, 0xD8, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x72, 0xD8,
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0x64, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x82, 0xD9, 0xFF, 0xFF,
        0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xDE, 0xD8,
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x14, 0x00, 0x00, 0x00, 0xF6, 0xD8, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0x54, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x06, 0xDA, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x52, 0xD9, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x6A, 0xD9, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x03, 0x14, 0x05, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x7A, 0xDA, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00,
        0x40, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x86, 0xDE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0xA2, 0xDE,
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0x64, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xB2, 0xDF, 0xFF, 0xFF,
        0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0E, 0xDF,
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x14, 0x00, 0x00, 0x00, 0x26, 0xDF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0x64, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x36, 0xE0, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x92, 0xDF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0xAA, 0xDF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03,
        0x14, 0x05, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xBA, 0xE0, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x40, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xC6, 0xE4, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xE2, 0xE4, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x03, 0xA4, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xF2, 0xE5, 0xFF, 0xFF, 0x04, 0x00,
        0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8E, 0xE6, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x05, 0x00,
        0x00, 0x00, 0xAA, 0xE6, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0x64, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0xBA, 0xE7, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x16, 0xE7, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x2E, 0xE7, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0x64, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x3E, 0xE8, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9A, 0xE7, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0xB2, 0xE7, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x03, 0x64, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xC2, 0xE8, 0xFF, 0xFF, 0x04, 0x00,
        0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1E, 0xE8, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00,
        0x00, 0x00, 0x36, 0xE8, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0x14, 0x05, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x46, 0xE9, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x52, 0xED, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x14, 0x00,
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x6E, 0xED, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0x14, 0x05, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00, 0x7E, 0xEE, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x8A, 0xF2, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xA6, 0xF2, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03,
        0x14, 0x05, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xB6, 0xF3, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x40, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xC2, 0xF7, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xDE, 0xF7, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x03, 0xA4, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xEE, 0xF8, 0xFF, 0xFF, 0x04, 0x00,
        0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8A, 0xF9, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x05, 0x00,
        0x00, 0x00, 0xA6, 0xF9, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0xA4, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0xB6, 0xFA, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x52, 0xFB,
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x14, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x6E, 0xFB, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x03, 0xA4, 0x01,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x7E, 0xFC, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x1A, 0xFD, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x10, 0x00, 0x0C, 0x00,
        0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x01, 0x04, 0x00, 0x00, 0x00, 0x2E, 0xFE, 0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x22, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x6C, 0x73,
        0x74, 0x6D, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xEC, 0x00, 0x00, 0x00, 0xD0, 0x00, 0x00, 0x00,
        0xB4, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x88, 0x00, 0x00, 0x00, 0x5C, 0x00, 0x00, 0x00, 0x30, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x14, 0xFF, 0xFF, 0xFF, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0xA6, 0xFD, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x3C, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00, 0xCE, 0xFD, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x64, 0xFF, 0xFF, 0xFF,
        0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xF6, 0xFD, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0xB4, 0xFE, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0x1A, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
        0xF0, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00,
        0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE8, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x09, 0x04, 0x00, 0x00, 0x00,
        0x7E, 0xFF, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x76, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x68, 0xFF, 0xFF, 0xFF, 0x04, 0x00, 0x00, 0x00, 0xCE, 0xFE, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x0E, 0x00, 0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x0C, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x0E, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0C, 0x00, 0x10, 0x00, 0x14, 0x00,
        0x0E, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x6E, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00,
        0x0C, 0x00, 0x07, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x04, 0x00, 0x00, 0x00,
        0xF6, 0xFF, 0xFF, 0xFF, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x0A, 0x00, 0x04, 0x00, 0x06, 0x00,
        0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00,
        0x0C, 0x00, 0x10, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00,
        0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x08, 0x00, 0x07, 0x00, 0x0C, 0x00,
        0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00
    };

    armnn::INetworkPtr deserializedNetwork =
        DeserializeNetwork(std::string(lstmNoCifgWithPeepholeAndProjectionModel.begin(),
                                       lstmNoCifgWithPeepholeAndProjectionModel.end()));

    BOOST_CHECK(deserializedNetwork);

    // generating the same model parameters which where used to serialize the model (Layer norm is not specified)
    armnn::LstmDescriptor descriptor;
    descriptor.m_ActivationFunc    = 4;
    descriptor.m_ClippingThresProj = 0.0f;
    descriptor.m_ClippingThresCell = 0.0f;
    descriptor.m_CifgEnabled       = false;
    descriptor.m_ProjectionEnabled = true;
    descriptor.m_PeepholeEnabled   = true;

    const uint32_t batchSize  = 2u;
    const uint32_t inputSize  = 5u;
    const uint32_t numUnits   = 20u;
    const uint32_t outputSize = 16u;

    armnn::TensorInfo tensorInfo20x5({numUnits, inputSize}, armnn::DataType::Float32);
    std::vector<float> inputToInputWeightsData(tensorInfo20x5.GetNumElements(), 0.0f);
    armnn::ConstTensor inputToInputWeights(tensorInfo20x5, inputToInputWeightsData);

    std::vector<float> inputToForgetWeightsData(tensorInfo20x5.GetNumElements(), 0.0f);
    armnn::ConstTensor inputToForgetWeights(tensorInfo20x5, inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData(tensorInfo20x5.GetNumElements(), 0.0f);
    armnn::ConstTensor inputToCellWeights(tensorInfo20x5, inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData(tensorInfo20x5.GetNumElements(), 0.0f);
    armnn::ConstTensor inputToOutputWeights(tensorInfo20x5, inputToOutputWeightsData);

    armnn::TensorInfo tensorInfo20({numUnits}, armnn::DataType::Float32);
    std::vector<float> inputGateBiasData(tensorInfo20.GetNumElements(), 0.0f);
    armnn::ConstTensor inputGateBias(tensorInfo20, inputGateBiasData);

    std::vector<float> forgetGateBiasData(tensorInfo20.GetNumElements(), 0.0f);
    armnn::ConstTensor forgetGateBias(tensorInfo20, forgetGateBiasData);

    std::vector<float> cellBiasData(tensorInfo20.GetNumElements(), 0.0f);
    armnn::ConstTensor cellBias(tensorInfo20, cellBiasData);

    std::vector<float> outputGateBiasData(tensorInfo20.GetNumElements(), 0.0f);
    armnn::ConstTensor outputGateBias(tensorInfo20, outputGateBiasData);

    armnn::TensorInfo tensorInfo20x16({numUnits, outputSize}, armnn::DataType::Float32);
    std::vector<float> recurrentToInputWeightsData(tensorInfo20x16.GetNumElements(), 0.0f);
    armnn::ConstTensor recurrentToInputWeights(tensorInfo20x16, recurrentToInputWeightsData);

    std::vector<float> recurrentToForgetWeightsData(tensorInfo20x16.GetNumElements(), 0.0f);
    armnn::ConstTensor recurrentToForgetWeights(tensorInfo20x16, recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData(tensorInfo20x16.GetNumElements(), 0.0f);
    armnn::ConstTensor recurrentToCellWeights(tensorInfo20x16, recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData(tensorInfo20x16.GetNumElements(), 0.0f);
    armnn::ConstTensor recurrentToOutputWeights(tensorInfo20x16, recurrentToOutputWeightsData);

    std::vector<float> cellToInputWeightsData(tensorInfo20.GetNumElements(), 0.0f);
    armnn::ConstTensor cellToInputWeights(tensorInfo20, cellToInputWeightsData);

    std::vector<float> cellToForgetWeightsData(tensorInfo20.GetNumElements(), 0.0f);
    armnn::ConstTensor cellToForgetWeights(tensorInfo20, cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData(tensorInfo20.GetNumElements(), 0.0f);
    armnn::ConstTensor cellToOutputWeights(tensorInfo20,  cellToOutputWeightsData);

    armnn::TensorInfo tensorInfo16x20({outputSize, numUnits}, armnn::DataType::Float32);
    std::vector<float> projectionWeightsData(tensorInfo16x20.GetNumElements(), 0.0f);
    armnn::ConstTensor projectionWeights(tensorInfo16x20, projectionWeightsData);

    armnn::TensorInfo tensorInfo16({outputSize}, armnn::DataType::Float32);
    std::vector<float> projectionBiasData(outputSize, 0.0f);
    armnn::ConstTensor projectionBias(tensorInfo16, projectionBiasData);

    armnn::LstmInputParams params;
    params.m_InputToForgetWeights     = &inputToForgetWeights;
    params.m_InputToCellWeights       = &inputToCellWeights;
    params.m_InputToOutputWeights     = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias           = &forgetGateBias;
    params.m_CellBias                 = &cellBias;
    params.m_OutputGateBias           = &outputGateBias;

    // additional params because: descriptor.m_CifgEnabled = false
    params.m_InputToInputWeights      = &inputToInputWeights;
    params.m_RecurrentToInputWeights  = &recurrentToInputWeights;
    params.m_CellToInputWeights       = &cellToInputWeights;
    params.m_InputGateBias            = &inputGateBias;

    // additional params because: descriptor.m_ProjectionEnabled = true
    params.m_ProjectionWeights        = &projectionWeights;
    params.m_ProjectionBias           = &projectionBias;

    // additional params because: descriptor.m_PeepholeEnabled = true
    params.m_CellToForgetWeights      = &cellToForgetWeights;
    params.m_CellToOutputWeights      = &cellToOutputWeights;

    const std::string layerName("lstm");
    armnn::TensorInfo inputTensorInfo({ batchSize, inputSize }, armnn::DataType::Float32);
    armnn::TensorInfo cellStateTensorInfo({ batchSize, numUnits}, armnn::DataType::Float32);
    armnn::TensorInfo outputStateTensorInfo({ batchSize, outputSize }, armnn::DataType::Float32);
    armnn::TensorInfo lstmTensorInfoScratchBuff({ batchSize, numUnits * 4 }, armnn::DataType::Float32);

    VerifyLstmLayer checker(
            layerName,
            {inputTensorInfo, outputStateTensorInfo, cellStateTensorInfo},
            {lstmTensorInfoScratchBuff, outputStateTensorInfo, cellStateTensorInfo, outputStateTensorInfo},
            descriptor,
            params);
    deserializedNetwork->Accept(checker);
}
class VerifyQuantizedLstmLayer : public LayerVerifierBase
{

public:
    VerifyQuantizedLstmLayer(const std::string& layerName,
                             const std::vector<armnn::TensorInfo>& inputInfos,
                             const std::vector<armnn::TensorInfo>& outputInfos,
                             const armnn::QuantizedLstmInputParams& inputParams)
        : LayerVerifierBase(layerName, inputInfos, outputInfos), m_InputParams(inputParams) {}

    void VisitQuantizedLstmLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::QuantizedLstmInputParams& params,
                                 const char* name)
    {
        VerifyNameAndConnections(layer, name);
        VerifyInputParameters(params);
    }

protected:
    void VerifyInputParameters(const armnn::QuantizedLstmInputParams& params)
    {
        VerifyConstTensors("m_InputToInputWeights",
                           m_InputParams.m_InputToInputWeights, params.m_InputToInputWeights);
        VerifyConstTensors("m_InputToForgetWeights",
                           m_InputParams.m_InputToForgetWeights, params.m_InputToForgetWeights);
        VerifyConstTensors("m_InputToCellWeights",
                           m_InputParams.m_InputToCellWeights, params.m_InputToCellWeights);
        VerifyConstTensors("m_InputToOutputWeights",
                           m_InputParams.m_InputToOutputWeights, params.m_InputToOutputWeights);
        VerifyConstTensors("m_RecurrentToInputWeights",
                           m_InputParams.m_RecurrentToInputWeights, params.m_RecurrentToInputWeights);
        VerifyConstTensors("m_RecurrentToForgetWeights",
                           m_InputParams.m_RecurrentToForgetWeights, params.m_RecurrentToForgetWeights);
        VerifyConstTensors("m_RecurrentToCellWeights",
                           m_InputParams.m_RecurrentToCellWeights, params.m_RecurrentToCellWeights);
        VerifyConstTensors("m_RecurrentToOutputWeights",
                           m_InputParams.m_RecurrentToOutputWeights, params.m_RecurrentToOutputWeights);
        VerifyConstTensors("m_InputGateBias",
                           m_InputParams.m_InputGateBias, params.m_InputGateBias);
        VerifyConstTensors("m_ForgetGateBias",
                           m_InputParams.m_ForgetGateBias, params.m_ForgetGateBias);
        VerifyConstTensors("m_CellBias",
                           m_InputParams.m_CellBias, params.m_CellBias);
        VerifyConstTensors("m_OutputGateBias",
                           m_InputParams.m_OutputGateBias, params.m_OutputGateBias);
    }

private:
    armnn::QuantizedLstmInputParams m_InputParams;
};

BOOST_AUTO_TEST_CASE(SerializeDeserializeQuantizedLstm)
{
    const uint32_t batchSize = 1;
    const uint32_t inputSize = 2;
    const uint32_t numUnits = 4;
    const uint32_t outputSize = numUnits;

    // Scale/Offset for input/output, cellState In/Out, weights, bias
    float inputOutputScale = 0.0078125f;
    int32_t inputOutputOffset = 128;

    float cellStateScale = 0.00048828125f;
    int32_t cellStateOffset = 0;

    float weightsScale = 0.00408021f;
    int32_t weightsOffset = 100;

    float biasScale = 3.1876640625e-05f;
    int32_t biasOffset = 0;

    // The shape of weight data is {outputSize, inputSize} = {4, 2}
    armnn::TensorShape inputToInputWeightsShape = {4, 2};
    std::vector<uint8_t> inputToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8};
    armnn::TensorInfo inputToInputWeightsInfo(inputToInputWeightsShape,
                                              armnn::DataType::QAsymmU8,
                                              weightsScale,
                                              weightsOffset);
    armnn::ConstTensor inputToInputWeights(inputToInputWeightsInfo, inputToInputWeightsData);

    armnn::TensorShape inputToForgetWeightsShape = {4, 2};
    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8};
    armnn::TensorInfo inputToForgetWeightsInfo(inputToForgetWeightsShape,
                                               armnn::DataType::QAsymmU8,
                                               weightsScale,
                                               weightsOffset);
    armnn::ConstTensor inputToForgetWeights(inputToForgetWeightsInfo, inputToForgetWeightsData);

    armnn::TensorShape inputToCellWeightsShape = {4, 2};
    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8};
    armnn::TensorInfo inputToCellWeightsInfo(inputToCellWeightsShape,
                                             armnn::DataType::QAsymmU8,
                                             weightsScale,
                                             weightsOffset);
    armnn::ConstTensor inputToCellWeights(inputToCellWeightsInfo, inputToCellWeightsData);

    armnn::TensorShape inputToOutputWeightsShape = {4, 2};
    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8};
    armnn::TensorInfo inputToOutputWeightsInfo(inputToOutputWeightsShape,
                                               armnn::DataType::QAsymmU8,
                                               weightsScale,
                                               weightsOffset);
    armnn::ConstTensor inputToOutputWeights(inputToOutputWeightsInfo, inputToOutputWeightsData);

    // The shape of recurrent weight data is {outputSize, outputSize} = {4, 4}
    armnn::TensorShape recurrentToInputWeightsShape = {4, 4};
    std::vector<uint8_t> recurrentToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    armnn::TensorInfo recurrentToInputWeightsInfo(recurrentToInputWeightsShape,
                                                  armnn::DataType::QAsymmU8,
                                                  weightsScale,
                                                  weightsOffset);
    armnn::ConstTensor recurrentToInputWeights(recurrentToInputWeightsInfo, recurrentToInputWeightsData);

    armnn::TensorShape recurrentToForgetWeightsShape = {4, 4};
    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    armnn::TensorInfo recurrentToForgetWeightsInfo(recurrentToForgetWeightsShape,
                                                   armnn::DataType::QAsymmU8,
                                                   weightsScale,
                                                   weightsOffset);
    armnn::ConstTensor recurrentToForgetWeights(recurrentToForgetWeightsInfo, recurrentToForgetWeightsData);

    armnn::TensorShape recurrentToCellWeightsShape = {4, 4};
    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    armnn::TensorInfo recurrentToCellWeightsInfo(recurrentToCellWeightsShape,
                                                 armnn::DataType::QAsymmU8,
                                                 weightsScale,
                                                 weightsOffset);
    armnn::ConstTensor recurrentToCellWeights(recurrentToCellWeightsInfo, recurrentToCellWeightsData);

    armnn::TensorShape recurrentToOutputWeightsShape = {4, 4};
    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    armnn::TensorInfo recurrentToOutputWeightsInfo(recurrentToOutputWeightsShape,
                                                   armnn::DataType::QAsymmU8,
                                                   weightsScale,
                                                   weightsOffset);
    armnn::ConstTensor recurrentToOutputWeights(recurrentToOutputWeightsInfo, recurrentToOutputWeightsData);

    // The shape of bias data is {outputSize} = {4}
    armnn::TensorShape inputGateBiasShape = {4};
    std::vector<int32_t> inputGateBiasData = {1, 2, 3, 4};
    armnn::TensorInfo inputGateBiasInfo(inputGateBiasShape,
                                        armnn::DataType::Signed32,
                                        biasScale,
                                        biasOffset);
    armnn::ConstTensor inputGateBias(inputGateBiasInfo, inputGateBiasData);

    armnn::TensorShape forgetGateBiasShape = {4};
    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4};
    armnn::TensorInfo forgetGateBiasInfo(forgetGateBiasShape,
                                         armnn::DataType::Signed32,
                                         biasScale,
                                         biasOffset);
    armnn::ConstTensor forgetGateBias(forgetGateBiasInfo, forgetGateBiasData);

    armnn::TensorShape cellBiasShape = {4};
    std::vector<int32_t> cellBiasData = {1, 2, 3, 4};
    armnn::TensorInfo cellBiasInfo(cellBiasShape,
                                   armnn::DataType::Signed32,
                                   biasScale,
                                   biasOffset);
    armnn::ConstTensor cellBias(cellBiasInfo, cellBiasData);

    armnn::TensorShape outputGateBiasShape = {4};
    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4};
    armnn::TensorInfo outputGateBiasInfo(outputGateBiasShape,
                                         armnn::DataType::Signed32,
                                         biasScale,
                                         biasOffset);
    armnn::ConstTensor outputGateBias(outputGateBiasInfo, outputGateBiasData);

    armnn::QuantizedLstmInputParams params;
    params.m_InputToInputWeights = &inputToInputWeights;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_InputGateBias = &inputGateBias;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const cellStateIn = network->AddInputLayer(1);
    armnn::IConnectableLayer* const outputStateIn = network->AddInputLayer(2);
    const std::string layerName("QuantizedLstm");
    armnn::IConnectableLayer* const quantizedLstmLayer = network->AddQuantizedLstmLayer(params, layerName.c_str());
    armnn::IConnectableLayer* const cellStateOut = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(1);

    // Connect up
    armnn::TensorInfo inputTensorInfo({ batchSize, inputSize },
                                      armnn::DataType::QAsymmU8,
                                      inputOutputScale,
                                      inputOutputOffset);
    armnn::TensorInfo cellStateTensorInfo({ batchSize, numUnits },
                                          armnn::DataType::QSymmS16,
                                          cellStateScale,
                                          cellStateOffset);
    armnn::TensorInfo outputStateTensorInfo({ batchSize, outputSize },
                                            armnn::DataType::QAsymmU8,
                                            inputOutputScale,
                                            inputOutputOffset);

    inputLayer->GetOutputSlot(0).Connect(quantizedLstmLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    cellStateIn->GetOutputSlot(0).Connect(quantizedLstmLayer->GetInputSlot(1));
    cellStateIn->GetOutputSlot(0).SetTensorInfo(cellStateTensorInfo);

    outputStateIn->GetOutputSlot(0).Connect(quantizedLstmLayer->GetInputSlot(2));
    outputStateIn->GetOutputSlot(0).SetTensorInfo(outputStateTensorInfo);

    quantizedLstmLayer->GetOutputSlot(0).Connect(cellStateOut->GetInputSlot(0));
    quantizedLstmLayer->GetOutputSlot(0).SetTensorInfo(cellStateTensorInfo);

    quantizedLstmLayer->GetOutputSlot(1).Connect(outputLayer->GetInputSlot(0));
    quantizedLstmLayer->GetOutputSlot(1).SetTensorInfo(outputStateTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyQuantizedLstmLayer checker(layerName,
                                     {inputTensorInfo, cellStateTensorInfo, outputStateTensorInfo},
                                     {cellStateTensorInfo, outputStateTensorInfo},
                                     params);

    deserializedNetwork->Accept(checker);
}

class VerifyQLstmLayer : public LayerVerifierBaseWithDescriptor<armnn::QLstmDescriptor>
{
public:
    VerifyQLstmLayer(const std::string& layerName,
                     const std::vector<armnn::TensorInfo>& inputInfos,
                     const std::vector<armnn::TensorInfo>& outputInfos,
                     const armnn::QLstmDescriptor& descriptor,
                     const armnn::LstmInputParams& inputParams)
        : LayerVerifierBaseWithDescriptor<armnn::QLstmDescriptor>(layerName, inputInfos, outputInfos, descriptor)
        , m_InputParams(inputParams) {}

    void VisitQLstmLayer(const armnn::IConnectableLayer* layer,
                         const armnn::QLstmDescriptor& descriptor,
                         const armnn::LstmInputParams& params,
                         const char* name)
    {
        VerifyNameAndConnections(layer, name);
        VerifyDescriptor(descriptor);
        VerifyInputParameters(params);
    }

protected:
    void VerifyInputParameters(const armnn::LstmInputParams& params)
    {
        VerifyConstTensors(
            "m_InputToInputWeights", m_InputParams.m_InputToInputWeights, params.m_InputToInputWeights);
        VerifyConstTensors(
            "m_InputToForgetWeights", m_InputParams.m_InputToForgetWeights, params.m_InputToForgetWeights);
        VerifyConstTensors(
            "m_InputToCellWeights", m_InputParams.m_InputToCellWeights, params.m_InputToCellWeights);
        VerifyConstTensors(
            "m_InputToOutputWeights", m_InputParams.m_InputToOutputWeights, params.m_InputToOutputWeights);
        VerifyConstTensors(
            "m_RecurrentToInputWeights", m_InputParams.m_RecurrentToInputWeights, params.m_RecurrentToInputWeights);
        VerifyConstTensors(
            "m_RecurrentToForgetWeights", m_InputParams.m_RecurrentToForgetWeights, params.m_RecurrentToForgetWeights);
        VerifyConstTensors(
            "m_RecurrentToCellWeights", m_InputParams.m_RecurrentToCellWeights, params.m_RecurrentToCellWeights);
        VerifyConstTensors(
            "m_RecurrentToOutputWeights", m_InputParams.m_RecurrentToOutputWeights, params.m_RecurrentToOutputWeights);
        VerifyConstTensors(
            "m_CellToInputWeights", m_InputParams.m_CellToInputWeights, params.m_CellToInputWeights);
        VerifyConstTensors(
            "m_CellToForgetWeights", m_InputParams.m_CellToForgetWeights, params.m_CellToForgetWeights);
        VerifyConstTensors(
            "m_CellToOutputWeights", m_InputParams.m_CellToOutputWeights, params.m_CellToOutputWeights);
        VerifyConstTensors(
            "m_InputGateBias", m_InputParams.m_InputGateBias, params.m_InputGateBias);
        VerifyConstTensors(
            "m_ForgetGateBias", m_InputParams.m_ForgetGateBias, params.m_ForgetGateBias);
        VerifyConstTensors(
            "m_CellBias", m_InputParams.m_CellBias, params.m_CellBias);
        VerifyConstTensors(
            "m_OutputGateBias", m_InputParams.m_OutputGateBias, params.m_OutputGateBias);
        VerifyConstTensors(
            "m_ProjectionWeights", m_InputParams.m_ProjectionWeights, params.m_ProjectionWeights);
        VerifyConstTensors(
            "m_ProjectionBias", m_InputParams.m_ProjectionBias, params.m_ProjectionBias);
        VerifyConstTensors(
            "m_InputLayerNormWeights", m_InputParams.m_InputLayerNormWeights, params.m_InputLayerNormWeights);
        VerifyConstTensors(
            "m_ForgetLayerNormWeights", m_InputParams.m_ForgetLayerNormWeights, params.m_ForgetLayerNormWeights);
        VerifyConstTensors(
            "m_CellLayerNormWeights", m_InputParams.m_CellLayerNormWeights, params.m_CellLayerNormWeights);
        VerifyConstTensors(
            "m_OutputLayerNormWeights", m_InputParams.m_OutputLayerNormWeights, params.m_OutputLayerNormWeights);
    }

private:
    armnn::LstmInputParams m_InputParams;
};

BOOST_AUTO_TEST_CASE(SerializeDeserializeQLstmBasic)
{
    armnn::QLstmDescriptor descriptor;

    descriptor.m_CifgEnabled       = true;
    descriptor.m_ProjectionEnabled = false;
    descriptor.m_PeepholeEnabled   = false;
    descriptor.m_LayerNormEnabled  = false;

    descriptor.m_CellClip       = 0.0f;
    descriptor.m_ProjectionClip = 0.0f;

    descriptor.m_InputIntermediateScale  = 0.00001f;
    descriptor.m_ForgetIntermediateScale = 0.00001f;
    descriptor.m_CellIntermediateScale   = 0.00001f;
    descriptor.m_OutputIntermediateScale = 0.00001f;

    descriptor.m_HiddenStateScale     = 0.07f;
    descriptor.m_HiddenStateZeroPoint = 0;

    const unsigned int numBatches = 2;
    const unsigned int inputSize  = 5;
    const unsigned int outputSize = 4;
    const unsigned int numUnits   = 4;

    // Scale/Offset quantization info
    float inputScale    = 0.0078f;
    int32_t inputOffset = 0;

    float outputScale    = 0.0078f;
    int32_t outputOffset = 0;

    float cellStateScale    = 3.5002e-05f;
    int32_t cellStateOffset = 0;

    float weightsScale    = 0.007f;
    int32_t weightsOffset = 0;

    float biasScale    = 3.5002e-05f / 1024;
    int32_t biasOffset = 0;

    // Weights and bias tensor and quantization info
    armnn::TensorInfo inputWeightsInfo({numUnits, inputSize},
                                       armnn::DataType::QSymmS8,
                                       weightsScale,
                                       weightsOffset);

    armnn::TensorInfo recurrentWeightsInfo({numUnits, outputSize},
                                           armnn::DataType::QSymmS8,
                                           weightsScale,
                                           weightsOffset);

    armnn::TensorInfo biasInfo({numUnits}, armnn::DataType::Signed32, biasScale, biasOffset);

    std::vector<int8_t> inputToForgetWeightsData = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());
    std::vector<int8_t> inputToCellWeightsData   = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());
    std::vector<int8_t> inputToOutputWeightsData = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());

    armnn::ConstTensor inputToForgetWeights(inputWeightsInfo, inputToForgetWeightsData);
    armnn::ConstTensor inputToCellWeights(inputWeightsInfo, inputToCellWeightsData);
    armnn::ConstTensor inputToOutputWeights(inputWeightsInfo, inputToOutputWeightsData);

    std::vector<int8_t> recurrentToForgetWeightsData =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());
    std::vector<int8_t> recurrentToCellWeightsData   =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());
    std::vector<int8_t> recurrentToOutputWeightsData =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());

    armnn::ConstTensor recurrentToForgetWeights(recurrentWeightsInfo, recurrentToForgetWeightsData);
    armnn::ConstTensor recurrentToCellWeights(recurrentWeightsInfo, recurrentToCellWeightsData);
    armnn::ConstTensor recurrentToOutputWeights(recurrentWeightsInfo, recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData(numUnits, 1);
    std::vector<int32_t> cellBiasData(numUnits, 0);
    std::vector<int32_t> outputGateBiasData(numUnits, 0);

    armnn::ConstTensor forgetGateBias(biasInfo, forgetGateBiasData);
    armnn::ConstTensor cellBias(biasInfo, cellBiasData);
    armnn::ConstTensor outputGateBias(biasInfo, outputGateBiasData);

    // Set up params
    armnn::LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights   = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;

    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;

    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias       = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    // Create network
    armnn::INetworkPtr network = armnn::INetwork::Create();
    const std::string layerName("qLstm");

    armnn::IConnectableLayer* const input         = network->AddInputLayer(0);
    armnn::IConnectableLayer* const outputStateIn = network->AddInputLayer(1);
    armnn::IConnectableLayer* const cellStateIn   = network->AddInputLayer(2);

    armnn::IConnectableLayer* const qLstmLayer = network->AddQLstmLayer(descriptor, params, layerName.c_str());

    armnn::IConnectableLayer* const outputStateOut = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const cellStateOut   = network->AddOutputLayer(1);
    armnn::IConnectableLayer* const outputLayer    = network->AddOutputLayer(2);

    // Input/Output tensor info
    armnn::TensorInfo inputInfo({numBatches , inputSize},
                                armnn::DataType::QAsymmS8,
                                inputScale,
                                inputOffset);

    armnn::TensorInfo cellStateInfo({numBatches , numUnits},
                                    armnn::DataType::QSymmS16,
                                    cellStateScale,
                                    cellStateOffset);

    armnn::TensorInfo outputStateInfo({numBatches , outputSize},
                                      armnn::DataType::QAsymmS8,
                                      outputScale,
                                      outputOffset);

    // Connect input/output slots
    input->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);

    outputStateIn->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(1));
    outputStateIn->GetOutputSlot(0).SetTensorInfo(cellStateInfo);

    cellStateIn->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(2));
    cellStateIn->GetOutputSlot(0).SetTensorInfo(outputStateInfo);

    qLstmLayer->GetOutputSlot(0).Connect(outputStateOut->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(0).SetTensorInfo(outputStateInfo);

    qLstmLayer->GetOutputSlot(1).Connect(cellStateOut->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(1).SetTensorInfo(cellStateInfo);

    qLstmLayer->GetOutputSlot(2).Connect(outputLayer->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(2).SetTensorInfo(outputStateInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyQLstmLayer checker(layerName,
                             {inputInfo, cellStateInfo, outputStateInfo},
                             {outputStateInfo, cellStateInfo, outputStateInfo},
                             descriptor,
                             params);

    deserializedNetwork->Accept(checker);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeQLstmCifgLayerNorm)
{
    armnn::QLstmDescriptor descriptor;

    // CIFG params are used when CIFG is disabled
    descriptor.m_CifgEnabled       = true;
    descriptor.m_ProjectionEnabled = false;
    descriptor.m_PeepholeEnabled   = false;
    descriptor.m_LayerNormEnabled  = true;

    descriptor.m_CellClip       = 0.0f;
    descriptor.m_ProjectionClip = 0.0f;

    descriptor.m_InputIntermediateScale  = 0.00001f;
    descriptor.m_ForgetIntermediateScale = 0.00001f;
    descriptor.m_CellIntermediateScale   = 0.00001f;
    descriptor.m_OutputIntermediateScale = 0.00001f;

    descriptor.m_HiddenStateScale     = 0.07f;
    descriptor.m_HiddenStateZeroPoint = 0;

    const unsigned int numBatches = 2;
    const unsigned int inputSize  = 5;
    const unsigned int outputSize = 4;
    const unsigned int numUnits   = 4;

    // Scale/Offset quantization info
    float inputScale    = 0.0078f;
    int32_t inputOffset = 0;

    float outputScale    = 0.0078f;
    int32_t outputOffset = 0;

    float cellStateScale    = 3.5002e-05f;
    int32_t cellStateOffset = 0;

    float weightsScale    = 0.007f;
    int32_t weightsOffset = 0;

    float layerNormScale    = 3.5002e-05f;
    int32_t layerNormOffset = 0;

    float biasScale    = layerNormScale / 1024;
    int32_t biasOffset = 0;

    // Weights and bias tensor and quantization info
    armnn::TensorInfo inputWeightsInfo({numUnits, inputSize},
                                       armnn::DataType::QSymmS8,
                                       weightsScale,
                                       weightsOffset);

    armnn::TensorInfo recurrentWeightsInfo({numUnits, outputSize},
                                           armnn::DataType::QSymmS8,
                                           weightsScale,
                                           weightsOffset);

    armnn::TensorInfo biasInfo({numUnits},
                               armnn::DataType::Signed32,
                               biasScale,
                               biasOffset);

    armnn::TensorInfo layerNormWeightsInfo({numUnits},
                                           armnn::DataType::QSymmS16,
                                           layerNormScale,
                                           layerNormOffset);

    // Mandatory params
    std::vector<int8_t> inputToForgetWeightsData = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());
    std::vector<int8_t> inputToCellWeightsData   = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());
    std::vector<int8_t> inputToOutputWeightsData = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());

    armnn::ConstTensor inputToForgetWeights(inputWeightsInfo, inputToForgetWeightsData);
    armnn::ConstTensor inputToCellWeights(inputWeightsInfo, inputToCellWeightsData);
    armnn::ConstTensor inputToOutputWeights(inputWeightsInfo, inputToOutputWeightsData);

    std::vector<int8_t> recurrentToForgetWeightsData =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());
    std::vector<int8_t> recurrentToCellWeightsData   =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());
    std::vector<int8_t> recurrentToOutputWeightsData =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());

    armnn::ConstTensor recurrentToForgetWeights(recurrentWeightsInfo, recurrentToForgetWeightsData);
    armnn::ConstTensor recurrentToCellWeights(recurrentWeightsInfo, recurrentToCellWeightsData);
    armnn::ConstTensor recurrentToOutputWeights(recurrentWeightsInfo, recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData(numUnits, 1);
    std::vector<int32_t> cellBiasData(numUnits, 0);
    std::vector<int32_t> outputGateBiasData(numUnits, 0);

    armnn::ConstTensor forgetGateBias(biasInfo, forgetGateBiasData);
    armnn::ConstTensor cellBias(biasInfo, cellBiasData);
    armnn::ConstTensor outputGateBias(biasInfo, outputGateBiasData);

    // Layer Norm
    std::vector<int16_t> forgetLayerNormWeightsData =
            GenerateRandomData<int16_t>(layerNormWeightsInfo.GetNumElements());
    std::vector<int16_t> cellLayerNormWeightsData =
            GenerateRandomData<int16_t>(layerNormWeightsInfo.GetNumElements());
    std::vector<int16_t> outputLayerNormWeightsData =
            GenerateRandomData<int16_t>(layerNormWeightsInfo.GetNumElements());

    armnn::ConstTensor forgetLayerNormWeights(layerNormWeightsInfo, forgetLayerNormWeightsData);
    armnn::ConstTensor cellLayerNormWeights(layerNormWeightsInfo, cellLayerNormWeightsData);
    armnn::ConstTensor outputLayerNormWeights(layerNormWeightsInfo, outputLayerNormWeightsData);

    // Set up params
    armnn::LstmInputParams params;

    // Mandatory params
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights   = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;

    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;

    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias       = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    // Layer Norm
    params.m_ForgetLayerNormWeights = &forgetLayerNormWeights;
    params.m_CellLayerNormWeights   = &cellLayerNormWeights;
    params.m_OutputLayerNormWeights = &outputLayerNormWeights;

    // Create network
    armnn::INetworkPtr network = armnn::INetwork::Create();
    const std::string layerName("qLstm");

    armnn::IConnectableLayer* const input         = network->AddInputLayer(0);
    armnn::IConnectableLayer* const outputStateIn = network->AddInputLayer(1);
    armnn::IConnectableLayer* const cellStateIn   = network->AddInputLayer(2);

    armnn::IConnectableLayer* const qLstmLayer = network->AddQLstmLayer(descriptor, params, layerName.c_str());

    armnn::IConnectableLayer* const outputStateOut  = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const cellStateOut  = network->AddOutputLayer(1);
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(2);

    // Input/Output tensor info
    armnn::TensorInfo inputInfo({numBatches , inputSize},
                                armnn::DataType::QAsymmS8,
                                inputScale,
                                inputOffset);

    armnn::TensorInfo cellStateInfo({numBatches , numUnits},
                                    armnn::DataType::QSymmS16,
                                    cellStateScale,
                                    cellStateOffset);

    armnn::TensorInfo outputStateInfo({numBatches , outputSize},
                                      armnn::DataType::QAsymmS8,
                                      outputScale,
                                      outputOffset);

    // Connect input/output slots
    input->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);

    outputStateIn->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(1));
    outputStateIn->GetOutputSlot(0).SetTensorInfo(cellStateInfo);

    cellStateIn->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(2));
    cellStateIn->GetOutputSlot(0).SetTensorInfo(outputStateInfo);

    qLstmLayer->GetOutputSlot(0).Connect(outputStateOut->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(0).SetTensorInfo(outputStateInfo);

    qLstmLayer->GetOutputSlot(1).Connect(cellStateOut->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(1).SetTensorInfo(cellStateInfo);

    qLstmLayer->GetOutputSlot(2).Connect(outputLayer->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(2).SetTensorInfo(outputStateInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyQLstmLayer checker(layerName,
                             {inputInfo, cellStateInfo, outputStateInfo},
                             {outputStateInfo, cellStateInfo, outputStateInfo},
                             descriptor,
                             params);

    deserializedNetwork->Accept(checker);
}

BOOST_AUTO_TEST_CASE(SerializeDeserializeQLstmAdvanced)
{
    armnn::QLstmDescriptor descriptor;

    descriptor.m_CifgEnabled       = false;
    descriptor.m_ProjectionEnabled = true;
    descriptor.m_PeepholeEnabled   = true;
    descriptor.m_LayerNormEnabled  = true;

    descriptor.m_CellClip       = 0.1f;
    descriptor.m_ProjectionClip = 0.1f;

    descriptor.m_InputIntermediateScale  = 0.00001f;
    descriptor.m_ForgetIntermediateScale = 0.00001f;
    descriptor.m_CellIntermediateScale   = 0.00001f;
    descriptor.m_OutputIntermediateScale = 0.00001f;

    descriptor.m_HiddenStateScale     = 0.07f;
    descriptor.m_HiddenStateZeroPoint = 0;

    const unsigned int numBatches = 2;
    const unsigned int inputSize  = 5;
    const unsigned int outputSize = 4;
    const unsigned int numUnits   = 4;

    // Scale/Offset quantization info
    float inputScale    = 0.0078f;
    int32_t inputOffset = 0;

    float outputScale    = 0.0078f;
    int32_t outputOffset = 0;

    float cellStateScale    = 3.5002e-05f;
    int32_t cellStateOffset = 0;

    float weightsScale    = 0.007f;
    int32_t weightsOffset = 0;

    float layerNormScale    = 3.5002e-05f;
    int32_t layerNormOffset = 0;

    float biasScale    = layerNormScale / 1024;
    int32_t biasOffset = 0;

    // Weights and bias tensor and quantization info
    armnn::TensorInfo inputWeightsInfo({numUnits, inputSize},
                                       armnn::DataType::QSymmS8,
                                       weightsScale,
                                       weightsOffset);

    armnn::TensorInfo recurrentWeightsInfo({numUnits, outputSize},
                                           armnn::DataType::QSymmS8,
                                           weightsScale,
                                           weightsOffset);

    armnn::TensorInfo biasInfo({numUnits},
                               armnn::DataType::Signed32,
                               biasScale,
                               biasOffset);

    armnn::TensorInfo peepholeWeightsInfo({numUnits},
                                          armnn::DataType::QSymmS16,
                                          weightsScale,
                                          weightsOffset);

    armnn::TensorInfo layerNormWeightsInfo({numUnits},
                                           armnn::DataType::QSymmS16,
                                           layerNormScale,
                                           layerNormOffset);

    armnn::TensorInfo projectionWeightsInfo({outputSize, numUnits},
                                             armnn::DataType::QSymmS8,
                                             weightsScale,
                                             weightsOffset);

    // Mandatory params
    std::vector<int8_t> inputToForgetWeightsData = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());
    std::vector<int8_t> inputToCellWeightsData   = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());
    std::vector<int8_t> inputToOutputWeightsData = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());

    armnn::ConstTensor inputToForgetWeights(inputWeightsInfo, inputToForgetWeightsData);
    armnn::ConstTensor inputToCellWeights(inputWeightsInfo, inputToCellWeightsData);
    armnn::ConstTensor inputToOutputWeights(inputWeightsInfo, inputToOutputWeightsData);

    std::vector<int8_t> recurrentToForgetWeightsData =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());
    std::vector<int8_t> recurrentToCellWeightsData   =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());
    std::vector<int8_t> recurrentToOutputWeightsData =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());

    armnn::ConstTensor recurrentToForgetWeights(recurrentWeightsInfo, recurrentToForgetWeightsData);
    armnn::ConstTensor recurrentToCellWeights(recurrentWeightsInfo, recurrentToCellWeightsData);
    armnn::ConstTensor recurrentToOutputWeights(recurrentWeightsInfo, recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData(numUnits, 1);
    std::vector<int32_t> cellBiasData(numUnits, 0);
    std::vector<int32_t> outputGateBiasData(numUnits, 0);

    armnn::ConstTensor forgetGateBias(biasInfo, forgetGateBiasData);
    armnn::ConstTensor cellBias(biasInfo, cellBiasData);
    armnn::ConstTensor outputGateBias(biasInfo, outputGateBiasData);

    // CIFG
    std::vector<int8_t> inputToInputWeightsData = GenerateRandomData<int8_t>(inputWeightsInfo.GetNumElements());
    std::vector<int8_t> recurrentToInputWeightsData =
            GenerateRandomData<int8_t>(recurrentWeightsInfo.GetNumElements());
    std::vector<int32_t> inputGateBiasData(numUnits, 1);

    armnn::ConstTensor inputToInputWeights(inputWeightsInfo, inputToInputWeightsData);
    armnn::ConstTensor recurrentToInputWeights(recurrentWeightsInfo, recurrentToInputWeightsData);
    armnn::ConstTensor inputGateBias(biasInfo, inputGateBiasData);

    // Peephole
    std::vector<int16_t> cellToInputWeightsData  = GenerateRandomData<int16_t>(peepholeWeightsInfo.GetNumElements());
    std::vector<int16_t> cellToForgetWeightsData = GenerateRandomData<int16_t>(peepholeWeightsInfo.GetNumElements());
    std::vector<int16_t> cellToOutputWeightsData = GenerateRandomData<int16_t>(peepholeWeightsInfo.GetNumElements());

    armnn::ConstTensor cellToInputWeights(peepholeWeightsInfo, cellToInputWeightsData);
    armnn::ConstTensor cellToForgetWeights(peepholeWeightsInfo, cellToForgetWeightsData);
    armnn::ConstTensor cellToOutputWeights(peepholeWeightsInfo, cellToOutputWeightsData);

    // Projection
    std::vector<int8_t> projectionWeightsData = GenerateRandomData<int8_t>(projectionWeightsInfo.GetNumElements());
    std::vector<int32_t> projectionBiasData(outputSize, 1);

    armnn::ConstTensor projectionWeights(projectionWeightsInfo, projectionWeightsData);
    armnn::ConstTensor projectionBias(biasInfo, projectionBiasData);

    // Layer Norm
    std::vector<int16_t> inputLayerNormWeightsData =
            GenerateRandomData<int16_t>(layerNormWeightsInfo.GetNumElements());
    std::vector<int16_t> forgetLayerNormWeightsData =
            GenerateRandomData<int16_t>(layerNormWeightsInfo.GetNumElements());
    std::vector<int16_t> cellLayerNormWeightsData =
            GenerateRandomData<int16_t>(layerNormWeightsInfo.GetNumElements());
    std::vector<int16_t> outputLayerNormWeightsData =
            GenerateRandomData<int16_t>(layerNormWeightsInfo.GetNumElements());

    armnn::ConstTensor inputLayerNormWeights(layerNormWeightsInfo, inputLayerNormWeightsData);
    armnn::ConstTensor forgetLayerNormWeights(layerNormWeightsInfo, forgetLayerNormWeightsData);
    armnn::ConstTensor cellLayerNormWeights(layerNormWeightsInfo, cellLayerNormWeightsData);
    armnn::ConstTensor outputLayerNormWeights(layerNormWeightsInfo, outputLayerNormWeightsData);

    // Set up params
    armnn::LstmInputParams params;

    // Mandatory params
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights   = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;

    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;

    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias       = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    // CIFG
    params.m_InputToInputWeights     = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_InputGateBias           = &inputGateBias;

    // Peephole
    params.m_CellToInputWeights  = &cellToInputWeights;
    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    // Projection
    params.m_ProjectionWeights = &projectionWeights;
    params.m_ProjectionBias    = &projectionBias;

    // Layer Norm
    params.m_InputLayerNormWeights  = &inputLayerNormWeights;
    params.m_ForgetLayerNormWeights = &forgetLayerNormWeights;
    params.m_CellLayerNormWeights   = &cellLayerNormWeights;
    params.m_OutputLayerNormWeights = &outputLayerNormWeights;

    // Create network
    armnn::INetworkPtr network = armnn::INetwork::Create();
    const std::string layerName("qLstm");

    armnn::IConnectableLayer* const input         = network->AddInputLayer(0);
    armnn::IConnectableLayer* const outputStateIn = network->AddInputLayer(1);
    armnn::IConnectableLayer* const cellStateIn   = network->AddInputLayer(2);

    armnn::IConnectableLayer* const qLstmLayer = network->AddQLstmLayer(descriptor, params, layerName.c_str());

    armnn::IConnectableLayer* const outputStateOut = network->AddOutputLayer(0);
    armnn::IConnectableLayer* const cellStateOut   = network->AddOutputLayer(1);
    armnn::IConnectableLayer* const outputLayer    = network->AddOutputLayer(2);

    // Input/Output tensor info
    armnn::TensorInfo inputInfo({numBatches , inputSize},
                                armnn::DataType::QAsymmS8,
                                inputScale,
                                inputOffset);

    armnn::TensorInfo cellStateInfo({numBatches , numUnits},
                                    armnn::DataType::QSymmS16,
                                    cellStateScale,
                                    cellStateOffset);

    armnn::TensorInfo outputStateInfo({numBatches , outputSize},
                                      armnn::DataType::QAsymmS8,
                                      outputScale,
                                      outputOffset);

    // Connect input/output slots
    input->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);

    outputStateIn->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(1));
    outputStateIn->GetOutputSlot(0).SetTensorInfo(cellStateInfo);

    cellStateIn->GetOutputSlot(0).Connect(qLstmLayer->GetInputSlot(2));
    cellStateIn->GetOutputSlot(0).SetTensorInfo(outputStateInfo);

    qLstmLayer->GetOutputSlot(0).Connect(outputStateOut->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(0).SetTensorInfo(outputStateInfo);

    qLstmLayer->GetOutputSlot(1).Connect(cellStateOut->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(1).SetTensorInfo(cellStateInfo);

    qLstmLayer->GetOutputSlot(2).Connect(outputLayer->GetInputSlot(0));
    qLstmLayer->GetOutputSlot(2).SetTensorInfo(outputStateInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    BOOST_CHECK(deserializedNetwork);

    VerifyQLstmLayer checker(layerName,
                             {inputInfo, cellStateInfo, outputStateInfo},
                             {outputStateInfo, cellStateInfo, outputStateInfo},
                             descriptor,
                             params);

    deserializedNetwork->Accept(checker);
}

BOOST_AUTO_TEST_SUITE_END()
