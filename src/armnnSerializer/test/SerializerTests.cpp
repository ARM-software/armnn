//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Serializer.hpp"
#include "SerializerTestUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/QuantizedLstmParams.hpp>
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <random>
#include <vector>

#include <doctest/doctest.h>

using armnnDeserializer::IDeserializer;

TEST_SUITE("SerializerTests")
{

TEST_CASE("SerializeAddition")
{
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

    std::string serializedNetwork = SerializeNetwork(*network);
    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(serializedNetwork);
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {tensorInfo, tensorInfo}, {tensorInfo});
    deserializedNetwork->ExecuteStrategy(verifier);
}

void SerializeArgMinMaxTest(armnn::DataType dataType)
{
    const std::string layerName("argminmax");
    const armnn::TensorInfo inputInfo({1, 2, 3}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1, 3}, dataType);

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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::ArgMinMaxDescriptor> verifier(layerName,
                                                                         {inputInfo},
                                                                         {outputInfo},
                                                                         descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeArgMinMaxSigned32")
{
    SerializeArgMinMaxTest(armnn::DataType::Signed32);
}

TEST_CASE("SerializeArgMinMaxSigned64")
{
    SerializeArgMinMaxTest(armnn::DataType::Signed64);
}

TEST_CASE("SerializeBatchNormalization")
{
    const std::string layerName("batchNormalization");
    const armnn::TensorInfo inputInfo ({ 1, 3, 3, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);

    const armnn::TensorInfo meanInfo({1}, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo varianceInfo({1}, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo betaInfo({1}, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo gammaInfo({1}, armnn::DataType::Float32, 0.0f, 0, true);

    armnn::BatchNormalizationDescriptor descriptor;
    descriptor.m_Eps = 0.0010000000475f;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    std::vector<float> meanData({5.0});
    std::vector<float> varianceData({2.0});
    std::vector<float> betaData({1.0});
    std::vector<float> gammaData({0.0});

    std::vector<armnn::ConstTensor> constants;
    constants.emplace_back(armnn::ConstTensor(meanInfo, meanData));
    constants.emplace_back(armnn::ConstTensor(varianceInfo, varianceData));
    constants.emplace_back(armnn::ConstTensor(betaInfo, betaData));
    constants.emplace_back(armnn::ConstTensor(gammaInfo, gammaData));

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const batchNormalizationLayer =
        network->AddBatchNormalizationLayer(descriptor,
                                            constants[0],
                                            constants[1],
                                            constants[2],
                                            constants[3],
                                            layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(batchNormalizationLayer->GetInputSlot(0));
    batchNormalizationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    batchNormalizationLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptorAndConstants<armnn::BatchNormalizationDescriptor> verifier(
        layerName, {inputInfo}, {outputInfo}, descriptor, constants);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeBatchToSpaceNd")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::BatchToSpaceNdDescriptor> verifier(layerName,
                                                                              {inputInfo},
                                                                              {outputInfo},
                                                                              desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeCast")
{
        const std::string layerName("cast");

        const armnn::TensorShape shape{1, 5, 2, 3};

        const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Signed32);
        const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Float32);

        armnn::INetworkPtr network = armnn::INetwork::Create();
        armnn::IConnectableLayer* inputLayer      = network->AddInputLayer(0);
        armnn::IConnectableLayer* castLayer       = network->AddCastLayer(layerName.c_str());
        armnn::IConnectableLayer* outputLayer     = network->AddOutputLayer(0);

        inputLayer->GetOutputSlot(0).Connect(castLayer->GetInputSlot(0));
        castLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
        castLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
        CHECK(deserializedNetwork);

        LayerVerifierBase verifier(layerName, {inputInfo}, {outputInfo});
        deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeChannelShuffle")
{
    const std::string layerName("channelShuffle");
    const armnn::TensorInfo inputInfo({1, 9}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1, 9}, armnn::DataType::Float32);

    armnn::ChannelShuffleDescriptor descriptor({3, 1});

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const ChannelShuffleLayer =
            network->AddChannelShuffleLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(ChannelShuffleLayer->GetInputSlot(0));
    ChannelShuffleLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    ChannelShuffleLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::ChannelShuffleDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeComparison")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::ComparisonDescriptor> verifier(layerName,
                                                                          { inputInfo, inputInfo },
                                                                          { outputInfo },
                                                                          descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeConstant")
{
    class ConstantLayerVerifier : public LayerVerifierBase
    {
    public:
        ConstantLayerVerifier(const std::string& layerName,
                              const std::vector<armnn::TensorInfo>& inputInfos,
                              const std::vector<armnn::TensorInfo>& outputInfos,
                              const std::vector<armnn::ConstTensor>& constants)
            : LayerVerifierBase(layerName, inputInfos, outputInfos)
            , m_Constants(constants) {}

        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>& constants,
                             const char* name,
                             const armnn::LayerBindingId id = 0) override
        {
            armnn::IgnoreUnused(descriptor, id);

            switch (layer->GetType())
            {
                case armnn::LayerType::Input: break;
                case armnn::LayerType::Output: break;
                case armnn::LayerType::Addition: break;
                default:
                {
                    this->VerifyNameAndConnections(layer, name);

                    for (std::size_t i = 0; i < constants.size(); i++)
                    {
                        CompareConstTensor(constants[i], m_Constants[i]);
                    }
                }
            }
        }

    private:
        const std::vector<armnn::ConstTensor> m_Constants;
    };

    const std::string layerName("constant");
    const armnn::TensorInfo info({ 2, 3 }, armnn::DataType::Float32, 0.0f, 0, true);

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
    CHECK(deserializedNetwork);

    ConstantLayerVerifier verifier(layerName, {}, {info}, {constTensor});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeConvolution2d")
{
    const std::string layerName("convolution2d");
    const armnn::TensorInfo inputInfo ({ 1, 5, 5, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32, 0.0f, 0, true);

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
    CHECK(deserializedNetwork);

    const std::vector<armnn::ConstTensor>& constants {weights, biases};
    LayerVerifierBaseWithDescriptorAndConstants<armnn::Convolution2dDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, descriptor, constants);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeConvolution2dWithPerAxisParams")
{
    using namespace armnn;

    const std::string layerName("convolution2dWithPerAxis");
    const TensorInfo inputInfo ({ 1, 3, 1, 2 }, DataType::QAsymmU8, 0.55f, 128);
    const TensorInfo outputInfo({ 1, 3, 1, 3 }, DataType::QAsymmU8, 0.75f, 128);

    const std::vector<float> quantScales{ 0.75f, 0.65f, 0.85f };
    constexpr unsigned int quantDimension = 0;

    const TensorInfo kernelInfo({ 3, 1, 1, 2 }, DataType::QSymmS8, quantScales, quantDimension, true);

    const std::vector<float> biasQuantScales{ 0.25f, 0.50f, 0.75f };
    const TensorInfo biasInfo({ 3 }, DataType::Signed32, biasQuantScales, quantDimension, true);

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
    CHECK(deserializedNetwork);

    const std::vector<armnn::ConstTensor>& constants {weights, biases};
    LayerVerifierBaseWithDescriptorAndConstants<Convolution2dDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, descriptor, constants);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeConvolution3d")
{
    const std::string layerName("convolution3d");
    const armnn::TensorInfo inputInfo ({ 1, 5, 5, 5, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 2, 2, 2, 1 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 3, 3, 3, 1, 1 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32, 0.0f, 0, true);

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);

    std::vector<float> biasesData = GenerateRandomData<float>(biasesInfo.GetNumElements());
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::Convolution3dDescriptor descriptor;
    descriptor.m_PadLeft     = 0;
    descriptor.m_PadRight    = 0;
    descriptor.m_PadTop      = 0;
    descriptor.m_PadBottom   = 0;
    descriptor.m_PadFront    = 0;
    descriptor.m_PadBack     = 0;
    descriptor.m_DilationX   = 1;
    descriptor.m_DilationY   = 1;
    descriptor.m_DilationZ   = 1;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_StrideZ     = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NDHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer  = network->AddInputLayer(0);
    armnn::IConnectableLayer* const weightsLayer = network->AddConstantLayer(weights, "Weights");
    armnn::IConnectableLayer* const biasesLayer = network->AddConstantLayer(biases, "Biases");
    armnn::IConnectableLayer* const convLayer   = network->AddConvolution3dLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1));
    biasesLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);
    biasesLayer->GetOutputSlot(0).SetTensorInfo(biasesInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::Convolution3dDescriptor> verifier(
            layerName, {inputInfo, weightsInfo, biasesInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeDepthToSpace")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::DepthToSpaceDescriptor> verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeDepthwiseConvolution2d")
{
    const std::string layerName("depwiseConvolution2d");
    const armnn::TensorInfo inputInfo ({ 1, 5, 5, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 3, 3, 3 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 1, 3, 3, 3 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 3 }, armnn::DataType::Float32, 0.0f, 0, true);

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
    CHECK(deserializedNetwork);

    const std::vector<armnn::ConstTensor>& constants {weights, biases};
    LayerVerifierBaseWithDescriptorAndConstants<armnn::DepthwiseConvolution2dDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, descriptor, constants);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeDepthwiseConvolution2dWithPerAxisParams")
{
    using namespace armnn;

    const std::string layerName("depwiseConvolution2dWithPerAxis");
    const TensorInfo inputInfo ({ 1, 3, 3, 2 }, DataType::QAsymmU8, 0.55f, 128);
    const TensorInfo outputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 0.75f, 128);

    const std::vector<float> quantScales{ 0.75f, 0.80f, 0.90f, 0.95f };
    const unsigned int quantDimension = 0;
    TensorInfo kernelInfo({ 2, 2, 2, 2 }, DataType::QSymmS8, quantScales, quantDimension, true);

    const std::vector<float> biasQuantScales{ 0.25f, 0.35f, 0.45f, 0.55f };
    constexpr unsigned int biasQuantDimension = 0;
    TensorInfo biasInfo({ 4 }, DataType::Signed32, biasQuantScales, biasQuantDimension, true);

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
    CHECK(deserializedNetwork);

    const std::vector<armnn::ConstTensor>& constants {weights, biases};
    LayerVerifierBaseWithDescriptorAndConstants<armnn::DepthwiseConvolution2dDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, descriptor, constants);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeDequantize")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {inputInfo}, {outputInfo});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeDeserializeDetectionPostProcess")
{
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

    const armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32, 0.0f, 0, true);
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
    CHECK(deserializedNetwork);

    const std::vector<armnn::ConstTensor>& constants {anchors};
    LayerVerifierBaseWithDescriptorAndConstants<armnn::DetectionPostProcessDescriptor> verifier(
            layerName, inputInfos, outputInfos, descriptor, constants);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeDivision")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {info, info}, {info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeDeserializeComparisonEqual")
{
    const std::string layerName("EqualLayer");
    const armnn::TensorInfo inputTensorInfo1 = armnn::TensorInfo({2, 1, 2, 4}, armnn::DataType::Float32);
    const armnn::TensorInfo inputTensorInfo2 = armnn::TensorInfo({2, 1, 2, 4}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({2, 1, 2, 4}, armnn::DataType::Boolean);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer2 = network->AddInputLayer(1);
    armnn::ComparisonDescriptor equalDescriptor(armnn::ComparisonOperation::Equal);
    armnn::IConnectableLayer* const equalLayer = network->AddComparisonLayer(equalDescriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer1->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputTensorInfo1);
    inputLayer2->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(1));
    inputLayer2->GetOutputSlot(0).SetTensorInfo(inputTensorInfo2);
    equalLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    equalLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {inputTensorInfo1, inputTensorInfo2}, {outputTensorInfo});
    deserializedNetwork->ExecuteStrategy(verifier);
}

void SerializeElementwiseUnaryTest(armnn::UnaryOperation unaryOperation)
{
    auto layerName = GetUnaryOperationAsCString(unaryOperation);

    const armnn::TensorShape shape{2, 1, 2, 2};

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Float32);

    armnn::ElementwiseUnaryDescriptor descriptor(unaryOperation);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const elementwiseUnaryLayer =
                                network->AddElementwiseUnaryLayer(descriptor, layerName);
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(elementwiseUnaryLayer->GetInputSlot(0));
    elementwiseUnaryLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    elementwiseUnaryLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));

    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::ElementwiseUnaryDescriptor>
        verifier(layerName, { inputInfo }, { outputInfo }, descriptor);

    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeElementwiseUnary")
{
    using op = armnn::UnaryOperation;
    std::initializer_list<op> allUnaryOperations = {op::Abs, op::Exp, op::Sqrt, op::Rsqrt, op::Neg,
                                                    op::LogicalNot, op::Log, op::Sin};

    for (auto unaryOperation : allUnaryOperations)
    {
        SerializeElementwiseUnaryTest(unaryOperation);
    }
}

TEST_CASE("SerializeFill")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::FillDescriptor> verifier(layerName, {inputInfo}, {outputInfo}, descriptor);

    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeFloor")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {info}, {info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

using FullyConnectedDescriptor = armnn::FullyConnectedDescriptor;
class FullyConnectedLayerVerifier : public LayerVerifierBaseWithDescriptor<FullyConnectedDescriptor>
{
public:
    FullyConnectedLayerVerifier(const std::string& layerName,
                        const std::vector<armnn::TensorInfo>& inputInfos,
                        const std::vector<armnn::TensorInfo>& outputInfos,
                        const FullyConnectedDescriptor& descriptor)
        : LayerVerifierBaseWithDescriptor<FullyConnectedDescriptor>(layerName, inputInfos, outputInfos, descriptor) {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Input: break;
            case armnn::LayerType::Output: break;
            case armnn::LayerType::Constant: break;
            default:
            {
                VerifyNameAndConnections(layer, name);
                const FullyConnectedDescriptor& layerDescriptor =
                        static_cast<const FullyConnectedDescriptor&>(descriptor);
                CHECK(layerDescriptor.m_ConstantWeights == m_Descriptor.m_ConstantWeights);
                CHECK(layerDescriptor.m_BiasEnabled == m_Descriptor.m_BiasEnabled);
                CHECK(layerDescriptor.m_TransposeWeightMatrix == m_Descriptor.m_TransposeWeightMatrix);
            }
        }
    }
};

TEST_CASE("SerializeFullyConnected")
{
    const std::string layerName("fullyConnected");
    const armnn::TensorInfo inputInfo ({ 2, 5, 1, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 2, 3 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 5, 3 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 3 }, armnn::DataType::Float32, 0.0f, 0, true);
    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    std::vector<float> biasesData  = GenerateRandomData<float>(biasesInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = true;
    descriptor.m_TransposeWeightMatrix = false;
    descriptor.m_ConstantWeights = true;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);

    // Old way of handling constant tensors.
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const fullyConnectedLayer =
        network->AddFullyConnectedLayer(descriptor,
                                        weights,
                                        armnn::Optional<armnn::ConstTensor>(biases),
                                        layerName.c_str());
    ARMNN_NO_DEPRECATE_WARN_END

    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    FullyConnectedLayerVerifier verifier(layerName, {inputInfo, weightsInfo, biasesInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeFullyConnectedWeightsAndBiasesAsInputs")
{
    const std::string layerName("fullyConnected_weights_as_inputs");
    const armnn::TensorInfo inputInfo ({ 2, 5, 1, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 2, 3 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 5, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo biasesInfo ({ 3 }, armnn::DataType::Float32);

    armnn::Optional<armnn::ConstTensor> weights = armnn::EmptyOptional();
    armnn::Optional<armnn::ConstTensor> bias = armnn::EmptyOptional();

    armnn::FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = true;
    descriptor.m_TransposeWeightMatrix = false;
    descriptor.m_ConstantWeights = false;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const weightsInputLayer = network->AddInputLayer(1);
    armnn::IConnectableLayer* const biasInputLayer = network->AddInputLayer(2);
    armnn::IConnectableLayer* const fullyConnectedLayer =
        network->AddFullyConnectedLayer(descriptor,
                                        layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    weightsInputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    biasInputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(2));
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    weightsInputLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);
    biasInputLayer->GetOutputSlot(0).SetTensorInfo(biasesInfo);
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    const std::vector<armnn::ConstTensor> constants {};
    LayerVerifierBaseWithDescriptorAndConstants<armnn::FullyConnectedDescriptor> verifier(
        layerName, {inputInfo, weightsInfo, biasesInfo}, {outputInfo}, descriptor, constants);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeFullyConnectedWeightsAndBiasesAsConstantLayers")
{
    const std::string layerName("fullyConnected_weights_as_inputs");
    const armnn::TensorInfo inputInfo ({ 2, 5, 1, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 2, 3 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 5, 3 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 3 }, armnn::DataType::Float32, 0.0f, 0, true);

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    std::vector<float> biasesData  = GenerateRandomData<float>(biasesInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = true;
    descriptor.m_TransposeWeightMatrix = false;
    descriptor.m_ConstantWeights = true;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const weightsLayer = network->AddConstantLayer(weights, "Weights");
    armnn::IConnectableLayer* const biasesLayer = network->AddConstantLayer(biases, "Biases");
    armnn::IConnectableLayer* const fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor,layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    biasesLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(2));
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);
    biasesLayer->GetOutputSlot(0).SetTensorInfo(biasesInfo);
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    FullyConnectedLayerVerifier verifier(layerName, {inputInfo, weightsInfo, biasesInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeGather")
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

        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>& constants,
                             const char* name,
                             const armnn::LayerBindingId id = 0) override
        {
            armnn::IgnoreUnused(constants, id);
            switch (layer->GetType())
            {
                case armnn::LayerType::Input: break;
                case armnn::LayerType::Output: break;
                case armnn::LayerType::Constant: break;
                default:
                {
                    VerifyNameAndConnections(layer, name);
                    const GatherDescriptor& layerDescriptor = static_cast<const GatherDescriptor&>(descriptor);
                    CHECK(layerDescriptor.m_Axis == m_Descriptor.m_Axis);
                }
            }
        }
    };

    const std::string layerName("gather");
    armnn::TensorInfo paramsInfo({ 8 }, armnn::DataType::QAsymmU8);
    armnn::TensorInfo outputInfo({ 3 }, armnn::DataType::QAsymmU8);
    const armnn::TensorInfo indicesInfo({ 3 }, armnn::DataType::Signed32, 0.0f, 0, true);
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
    CHECK(deserializedNetwork);

    GatherLayerVerifier verifier(layerName, {paramsInfo, indicesInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}


TEST_CASE("SerializeComparisonGreater")
{
    const std::string layerName("greater");

    const armnn::TensorShape shape{2, 1, 2, 4};

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo(shape, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo(shape, armnn::DataType::Boolean);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* const inputLayer1 = network->AddInputLayer(1);
    armnn::ComparisonDescriptor greaterDescriptor(armnn::ComparisonOperation::Greater);
    armnn::IConnectableLayer* const equalLayer = network->AddComparisonLayer(greaterDescriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(equalLayer->GetInputSlot(1));
    equalLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(inputInfo);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputInfo);
    equalLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, { inputInfo, inputInfo }, { outputInfo });
    deserializedNetwork->ExecuteStrategy(verifier);
}


TEST_CASE("SerializeInstanceNormalization")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::InstanceNormalizationDescriptor> verifier(
            layerName, {info}, {info}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeL2Normalization")
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::L2NormalizationDescriptor> verifier(
            l2NormLayerName, {info}, {info}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("EnsureL2NormalizationBackwardCompatibility")
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
    CHECK(deserializedNetwork);

    const std::string layerName("l2Normalization");
    const armnn::TensorInfo inputInfo = armnn::TensorInfo({1, 2, 1, 5}, armnn::DataType::Float32);

    armnn::L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NCHW;
    // Since this variable does not exist in the l2NormalizationModel dump, the default value will be loaded
    desc.m_Eps = 1e-12f;

    LayerVerifierBaseWithDescriptor<armnn::L2NormalizationDescriptor> verifier(
            layerName, {inputInfo}, {inputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeLogicalBinary")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::LogicalBinaryDescriptor> verifier(
            layerName, { inputInfo, inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeLogSoftmax")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::LogSoftmaxDescriptor> verifier(layerName, {info}, {info}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeMaximum")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {info, info}, {info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeMean")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::MeanDescriptor> verifier(layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeMerge")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {info, info}, {info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

class MergerLayerVerifier : public LayerVerifierBaseWithDescriptor<armnn::OriginsDescriptor>
{
public:
    MergerLayerVerifier(const std::string& layerName,
                        const std::vector<armnn::TensorInfo>& inputInfos,
                        const std::vector<armnn::TensorInfo>& outputInfos,
                        const armnn::OriginsDescriptor& descriptor)
        : LayerVerifierBaseWithDescriptor<armnn::OriginsDescriptor>(layerName, inputInfos, outputInfos, descriptor) {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Input: break;
            case armnn::LayerType::Output: break;
            case armnn::LayerType::Merge:
            {
                throw armnn::Exception("MergerLayer should have translated to ConcatLayer");
                break;
            }
            case armnn::LayerType::Concat:
            {
                VerifyNameAndConnections(layer, name);
                const armnn::MergerDescriptor& layerDescriptor =
                        static_cast<const armnn::MergerDescriptor&>(descriptor);
                VerifyDescriptor(layerDescriptor);
                break;
            }
            default:
            {
                throw armnn::Exception("Unexpected layer type in Merge test model");
            }
        }
    }
};

TEST_CASE("EnsureMergerLayerBackwardCompatibility")
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
    CHECK(deserializedNetwork);

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({ 2, 3, 2, 2 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({ 4, 3, 2, 2 }, armnn::DataType::Float32);

    const std::vector<armnn::TensorShape> shapes({inputInfo.GetShape(), inputInfo.GetShape()});

    armnn::OriginsDescriptor descriptor =
            armnn::CreateDescriptorForConcatenation(shapes.begin(), shapes.end(), 0);

    MergerLayerVerifier verifier("merger", { inputInfo, inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeConcat")
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
    CHECK(deserializedNetwork);

    // NOTE: using the MergerLayerVerifier to ensure that it is a concat layer and not a
    //       merger layer that gets placed into the graph.
    MergerLayerVerifier verifier(layerName, {inputInfo, inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeMinimum")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {info, info}, {info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeMultiplication")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {info, info}, {info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializePrelu")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {inputTensorInfo, alphaTensorInfo}, {outputTensorInfo});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeNormalization")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::NormalizationDescriptor> verifier(layerName, {info}, {info}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializePad")
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::PadDescriptor> verifier(layerName,
                                                                   {inputTensorInfo},
                                                                   {outputTensorInfo},
                                                                   desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializePadReflect")
{
    const std::string layerName("padReflect");
    const armnn::TensorInfo inputTensorInfo = armnn::TensorInfo({1, 2, 3, 4}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo = armnn::TensorInfo({1, 3, 5, 7}, armnn::DataType::Float32);

    armnn::PadDescriptor desc({{0, 0}, {1, 0}, {1, 1}, {1, 2}});
    desc.m_PaddingMode = armnn::PaddingMode::Reflect;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const padLayer = network->AddPadLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    padLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::PadDescriptor> verifier(layerName,
                                                                   {inputTensorInfo},
                                                                   {outputTensorInfo},
                                                                   desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("EnsurePadBackwardCompatibility")
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
    CHECK(deserializedNetwork);

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({ 1, 2, 3, 4 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({ 1, 3, 5, 7 }, armnn::DataType::Float32);

    armnn::PadDescriptor descriptor({{ 0, 0 }, { 1, 0 }, { 1, 1 }, { 1, 2 }});

    LayerVerifierBaseWithDescriptor<armnn::PadDescriptor> verifier("pad", { inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializePermute")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::PermuteDescriptor> verifier(
            layerName, {inputTensorInfo}, {outputTensorInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializePooling2d")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::Pooling2dDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializePooling3d")
{
    const std::string layerName("pooling3d");
    const armnn::TensorInfo inputInfo({1, 1, 2, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1, 1, 1, 1, 1}, armnn::DataType::Float32);

    armnn::Pooling3dDescriptor desc;
    desc.m_DataLayout          = armnn::DataLayout::NDHWC;
    desc.m_PadFront            = 0;
    desc.m_PadBack             = 0;
    desc.m_PadTop              = 0;
    desc.m_PadBottom           = 0;
    desc.m_PadLeft             = 0;
    desc.m_PadRight            = 0;
    desc.m_PoolType            = armnn::PoolingAlgorithm::Average;
    desc.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    desc.m_PaddingMethod       = armnn::PaddingMethod::Exclude;
    desc.m_PoolHeight          = 2;
    desc.m_PoolWidth           = 2;
    desc.m_PoolDepth           = 2;
    desc.m_StrideX             = 2;
    desc.m_StrideY             = 2;
    desc.m_StrideZ             = 2;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const pooling3dLayer = network->AddPooling3dLayer(desc, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(pooling3dLayer->GetInputSlot(0));
    pooling3dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    pooling3dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::Pooling3dDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeQuantize")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {info}, {info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeRank")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {inputInfo}, {outputInfo});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeReduceSum")
{
    const std::string layerName("Reduce_Sum");
    const armnn::TensorInfo inputInfo({1, 1, 3, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({1, 1, 1, 2}, armnn::DataType::Float32);

    armnn::ReduceDescriptor descriptor;
    descriptor.m_vAxis = { 2 };
    descriptor.m_ReduceOperation = armnn::ReduceOperation::Sum;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer   = network->AddInputLayer(0);
    armnn::IConnectableLayer* const reduceSumLayer = network->AddReduceLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer  = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(reduceSumLayer->GetInputSlot(0));
    reduceSumLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    reduceSumLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::ReduceDescriptor> verifier(layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeReshape")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::ReshapeDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeResize")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::ResizeDescriptor> verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

class ResizeBilinearLayerVerifier : public LayerVerifierBaseWithDescriptor<armnn::ResizeDescriptor>
{
public:
    ResizeBilinearLayerVerifier(const std::string& layerName,
                                const std::vector<armnn::TensorInfo>& inputInfos,
                                const std::vector<armnn::TensorInfo>& outputInfos,
                                const armnn::ResizeDescriptor& descriptor)
        : LayerVerifierBaseWithDescriptor<armnn::ResizeDescriptor>(
            layerName, inputInfos, outputInfos, descriptor) {}

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Input: break;
            case armnn::LayerType::Output: break;
            case armnn::LayerType::Resize:
            {
                VerifyNameAndConnections(layer, name);
                const armnn::ResizeDescriptor& layerDescriptor =
                        static_cast<const armnn::ResizeDescriptor&>(descriptor);
                CHECK(layerDescriptor.m_Method             == armnn::ResizeMethod::Bilinear);
                CHECK(layerDescriptor.m_TargetWidth        == m_Descriptor.m_TargetWidth);
                CHECK(layerDescriptor.m_TargetHeight       == m_Descriptor.m_TargetHeight);
                CHECK(layerDescriptor.m_DataLayout         == m_Descriptor.m_DataLayout);
                CHECK(layerDescriptor.m_AlignCorners       == m_Descriptor.m_AlignCorners);
                CHECK(layerDescriptor.m_HalfPixelCenters   == m_Descriptor.m_HalfPixelCenters);
                break;
            }
            default:
            {
                throw armnn::Exception("Unexpected layer type in test model. ResizeBiliniar "
                                       "should have translated to Resize");
            }
        }
    }
};

TEST_CASE("SerializeResizeBilinear")
{
    const std::string layerName("resizeBilinear");
    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({1, 3, 5, 5}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({1, 3, 2, 4}, armnn::DataType::Float32);

    armnn::ResizeDescriptor desc;
    desc.m_Method = armnn::ResizeMethod::Bilinear;
    desc.m_TargetWidth  = 4u;
    desc.m_TargetHeight = 2u;
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
    CHECK(deserializedNetwork);

    ResizeBilinearLayerVerifier verifier(layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("EnsureResizeBilinearBackwardCompatibility")
{
    // The hex data below is a flat buffer containing a simple network with an input,
    // a ResizeBilinearLayer (now deprecated and removed) and an output
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
    CHECK(deserializedNetwork);

    const armnn::TensorInfo inputInfo  = armnn::TensorInfo({1, 3, 5, 5}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({1, 3, 2, 4}, armnn::DataType::Float32);

    armnn::ResizeDescriptor descriptor;
    descriptor.m_TargetWidth  = 4u;
    descriptor.m_TargetHeight = 2u;

    ResizeBilinearLayerVerifier verifier("resizeBilinear", { inputInfo }, { outputInfo }, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeShape")
{
    const std::string layerName("shape");
    const armnn::TensorInfo inputInfo({1, 3, 3, 1}, armnn::DataType::Signed32);
    const armnn::TensorInfo outputInfo({ 4 }, armnn::DataType::Signed32);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const shapeLayer = network->AddShapeLayer(layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(shapeLayer->GetInputSlot(0));
    shapeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    shapeLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    armnn::INetworkPtr deserializedNetwork = DeserializeNetwork(SerializeNetwork(*network));
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {inputInfo}, {outputInfo});

    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeSlice")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::SliceDescriptor> verifier(layerName, {inputInfo}, {outputInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeSoftmax")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::SoftmaxDescriptor> verifier(layerName, {info}, {info}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeSpaceToBatchNd")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::SpaceToBatchNdDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeSpaceToDepth")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::SpaceToDepthDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeSplitter")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::ViewsDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo, outputInfo, outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeStack")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::StackDescriptor> verifier(
            layerName, {inputTensorInfo, inputTensorInfo}, {outputTensorInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeStandIn")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::StandInDescriptor> verifier(
            layerName, { tensorInfo, tensorInfo }, { tensorInfo, tensorInfo }, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeStridedSlice")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::StridedSliceDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, desc);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeSubtraction")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBase verifier(layerName, {info, info}, {info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeSwitch")
{
    class SwitchLayerVerifier : public LayerVerifierBase
    {
    public:
        SwitchLayerVerifier(const std::string& layerName,
                            const std::vector<armnn::TensorInfo>& inputInfos,
                            const std::vector<armnn::TensorInfo>& outputInfos)
                : LayerVerifierBase(layerName, inputInfos, outputInfos) {}

        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>& constants,
                             const char* name,
                             const armnn::LayerBindingId id = 0) override
        {
            armnn::IgnoreUnused(descriptor, constants, id);
            switch (layer->GetType())
            {
                case armnn::LayerType::Input: break;
                case armnn::LayerType::Output: break;
                case armnn::LayerType::Constant: break;
                case armnn::LayerType::Switch:
                {
                    VerifyNameAndConnections(layer, name);
                    break;
                }
                default:
                {
                    throw armnn::Exception("Unexpected layer type in Switch test model");
                }
            }
        }
    };

    const std::string layerName("switch");
    const armnn::TensorInfo info({ 1, 4 }, armnn::DataType::Float32, 0.0f, 0, true);

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
    CHECK(deserializedNetwork);

    SwitchLayerVerifier verifier(layerName, {info, info}, {info, info});
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeTranspose")
{
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
    CHECK(deserializedNetwork);

    LayerVerifierBaseWithDescriptor<armnn::TransposeDescriptor> verifier(
            layerName, {inputTensorInfo}, {outputTensorInfo}, descriptor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeTransposeConvolution2d")
{
    const std::string layerName("transposeConvolution2d");
    const armnn::TensorInfo inputInfo ({ 1, 7, 7, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 9, 9, 1 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32, 0.0f, 0, true);

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
    CHECK(deserializedNetwork);

    const std::vector<armnn::ConstTensor> constants {weights, biases};
    LayerVerifierBaseWithDescriptorAndConstants<armnn::TransposeConvolution2dDescriptor> verifier(
            layerName, {inputInfo}, {outputInfo}, descriptor, constants);
    deserializedNetwork->ExecuteStrategy(verifier);
}

TEST_CASE("SerializeDeserializeNonLinearNetwork")
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

        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>& constants,
                             const char* name,
                             const armnn::LayerBindingId id = 0) override
        {
            armnn::IgnoreUnused(descriptor, constants, id);
            switch (layer->GetType())
            {
                case armnn::LayerType::Input: break;
                case armnn::LayerType::Output: break;
                case armnn::LayerType::Addition: break;
                case armnn::LayerType::Constant:
                {
                    VerifyNameAndConnections(layer, name);
                    CompareConstTensor(constants.at(0), m_LayerInput);
                    break;
                }
                default:
                {
                    throw armnn::Exception("Unexpected layer type in test model");
                }
            }
        }

    private:
        armnn::ConstTensor m_LayerInput;
    };

    const std::string layerName("constant");
    const armnn::TensorInfo info({ 2, 3 }, armnn::DataType::Float32, 0.0f, 0, true);

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
    CHECK(deserializedNetwork);

    ConstantLayerVerifier verifier(layerName, {}, {info}, constTensor);
    deserializedNetwork->ExecuteStrategy(verifier);
}

}