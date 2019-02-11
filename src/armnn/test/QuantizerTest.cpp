//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/INetworkQuantizer.hpp>
#include <armnn/Types.hpp>

#include "armnn/LayerVisitorBase.hpp"
#include "../Network.hpp"
#include "../Graph.hpp"
#include "../NetworkQuantizerUtils.hpp"
#include "../OverrideInputRangeVisitor.hpp"

#include <boost/test/unit_test.hpp>

#include <unordered_map>

namespace armnn
{
using MinMaxRange = std::pair<float, float>;
using MinMaxRanges = std::vector<MinMaxRange>;
using MinMaxRangeMap = std::unordered_map<LayerGuid, MinMaxRanges>;

BOOST_AUTO_TEST_SUITE(Quantizer)

class TestQuantization : public LayerVisitorBase<VisitorThrowingPolicy>
{
public:
    virtual void VisitInputLayer(const IConnectableLayer* layer,
                                 LayerBindingId id,
                                 const char* name = nullptr)
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current default [-15.0f, 15.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f/255.0f, 0.000001f);
    }

    virtual void VisitOutputLayer(const IConnectableLayer* layer,
                                  LayerBindingId id,
                                  const char* name = nullptr)
    {}
};

void VisitLayersTopologically(const INetwork* inputNetwork, ILayerVisitor& visitor)
{
    auto network = boost::polymorphic_downcast<const Network*>(inputNetwork);
    auto graph = network->GetGraph().TopologicalSort();

    VisitLayers(graph, visitor);
}

BOOST_AUTO_TEST_CASE(QuantizeAddition)
{
    class TestAdditionQuantization : public TestQuantization
    {
    public:
        virtual void VisitAdditionLayer(const IConnectableLayer* layer,
                                        const char* name = nullptr)
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

            BOOST_TEST((info.GetQuantizationOffset() == 128));

            // Based off current static value [-20.0f, 20.0f]
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 40.0f/255.0f, 0.000001f);
        }
    };

    auto network = INetwork::Create();

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* input1 = network->AddInputLayer(1);
    IConnectableLayer* addition = network->AddAdditionLayer();
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input0->GetOutputSlot(0).Connect(addition->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(addition->GetInputSlot(1));
    addition->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    addition->GetOutputSlot(0).SetTensorInfo(info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestAdditionQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

class TestActivationQuantization : public TestQuantization
{
public:
    virtual void VisitActivationLayer(const IConnectableLayer* layer,
                                      const ActivationDescriptor& descriptor,
                                      const char* name = nullptr)
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 0));

        // Based off current static value [-20.0f, 20.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 15.0f/255.0f, 0.000001f);
    }
};

INetworkPtr CreateNetworkWithActivationLayer(const ActivationDescriptor& descriptor)
{
    auto network = INetwork::Create();
    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* activation = network->AddActivationLayer(descriptor);
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input0->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);

    return network;
}

BOOST_AUTO_TEST_CASE(QuantizeAbsActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Abs;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeLinearActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Linear;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::ReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeSoftReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::SoftReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeBoundedReluActivation)
{
    class TestBoundedReluActivationQuantization : public TestQuantization
    {
    public:
        virtual void VisitActivationLayer(const IConnectableLayer* layer,
                                          const ActivationDescriptor& descriptor,
                                          const char* name = nullptr)
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

            BOOST_TEST((info.GetQuantizationOffset() == 0));

            // Based off current static value [0.0f, 3.5f(<-layer upper bound)]
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 3.5f/255.0f, 0.000001f);
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::BoundedReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBoundedReluActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeTanHActivation)
{
    class TestTanHActivationQuantization : public TestQuantization
    {
    public:
        virtual void VisitActivationLayer(const IConnectableLayer* layer,
                                          const ActivationDescriptor& descriptor,
                                          const char* name = nullptr)
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

            BOOST_TEST((info.GetQuantizationOffset() == 128));

            // Based off current static value [-1.0f, 1.0f]
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 2.0f/255.0f, 0.000001f);
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::TanH;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestTanHActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeLeakyReLuActivation)
{
    class TestLeakyReLuActivationQuantization : public TestQuantization
    {
    public:
        virtual void VisitActivationLayer(const IConnectableLayer* layer,
                                          const ActivationDescriptor& descriptor,
                                          const char* name = nullptr)
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

            BOOST_TEST((info.GetQuantizationOffset() == 64));

            // Based off current static value [-5.0f, 15.0f]
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 20.0f/255.0f, 0.000001f);
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::LeakyReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestLeakyReLuActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeBatchNorm)
{

    class TestQuantization : public LayerVisitorBase<VisitorThrowingPolicy>
    {
    public:
        virtual void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                                  const BatchNormalizationDescriptor& desc,
                                                  const ConstTensor& mean,
                                                  const ConstTensor& variance,
                                                  const ConstTensor& beta,
                                                  const ConstTensor& gamma,
                                                  const char* name = nullptr)
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

            BOOST_TEST((info.GetQuantizationOffset() == 128));

            // Based off current static value [-15.0f, 15.0f]
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f/255.0f, 0.000001f);

            //Test constants
            BOOST_TEST((mean.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
            BOOST_TEST((variance.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
            BOOST_TEST((beta.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
            BOOST_TEST((gamma.GetInfo().GetDataType() == DataType::QuantisedAsymm8));

            BOOST_CHECK_CLOSE(mean.GetInfo().GetQuantizationScale(), 3.0f/255.0f, 0.000001f);
            BOOST_CHECK_CLOSE(variance.GetInfo().GetQuantizationScale(), 3.0f/255.0f, 0.000001f);
            BOOST_CHECK_CLOSE(beta.GetInfo().GetQuantizationScale(), 3.0f/255.0f, 0.000001f);
            BOOST_CHECK_CLOSE(gamma.GetInfo().GetQuantizationScale(), 3.0f/255.0f, 0.000001f);

            BOOST_TEST((mean.GetInfo().GetQuantizationOffset() == 85));
        }

        virtual void VisitInputLayer(const IConnectableLayer* layer,
                                     LayerBindingId id,
                                     const char* name = nullptr)
        {}

        virtual void VisitOutputLayer(const IConnectableLayer* layer,
                                      LayerBindingId id,
                                      const char* name = nullptr)
        {}
    };

    auto network = INetwork::Create();

    TensorShape shape{3U};
    TensorInfo info(shape, DataType::Float32);

    std::vector<float> meanData{-1.0f, 1.5f, 2.0f};
    std::vector<float> varData{-1.0f, 1.5f, 2.0f};
    std::vector<float> betaData{-1.0f, 1.5f, 2.0f};
    std::vector<float> gammaData{-1.0f, 1.5f, 2.0f};

    ConstTensor mean(info, meanData);
    ConstTensor var(info, varData);
    ConstTensor beta(info, betaData);
    ConstTensor gamma(info, gammaData);

    BatchNormalizationDescriptor desc;

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* batchNorm = network->AddBatchNormalizationLayer(desc, mean, var, beta, gamma);
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    input0->GetOutputSlot(0).Connect(batchNorm->GetInputSlot(0));
    batchNorm->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    batchNorm->GetOutputSlot(0).SetTensorInfo(info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(OverrideInputRangeEmptyNetwork)
{
    MinMaxRangeMap guidToRangesMap; // Empty map of ranges
    MinMaxRange minMaxRange(-12.3f, 45.6f); // Range to use for the override

    Network network; // Empty network
    auto inputLayers = network.GetGraph().GetInputLayers(); // Empty list of input layers

    OverrideInputRangeVisitor overrideInputRangeVisitor(guidToRangesMap, 0, minMaxRange);
    VisitLayers(inputLayers, overrideInputRangeVisitor);

    BOOST_CHECK(guidToRangesMap.empty()); // Check that the map of ranges remained untouched
}

BOOST_AUTO_TEST_CASE(OverrideInputRangeNoInputLayers)
{
    MinMaxRangeMap guidToRangesMap; // Empty map of ranges
    MinMaxRange minMaxRange(-12.3f, 45.6f); // Range to use for the override

    Network network;
    network.AddAdditionLayer(); // Network with no input layers
    auto inputLayers = network.GetGraph().GetInputLayers(); // Empty list of input layers

    OverrideInputRangeVisitor overrideInputRangeVisitor(guidToRangesMap, 0, minMaxRange);
    VisitLayers(inputLayers, overrideInputRangeVisitor);

    BOOST_CHECK(guidToRangesMap.empty()); // Check that the map of ranges remained untouched
}

BOOST_AUTO_TEST_CASE(OverrideInputRangeInputLayers)
{
    MinMaxRangeMap guidToRangesMap; // Empty map of ranges
    MinMaxRange minMaxRange(-12.3f, 45.6f); // Range to use for the override

    Network network;

    // Adding the layers
    IConnectableLayer* input0 = network.AddInputLayer(0);
    IConnectableLayer* input1 = network.AddInputLayer(1);
    IConnectableLayer* addition = network.AddAdditionLayer();
    IConnectableLayer* output = network.AddOutputLayer(2);

    // Connecting the layer
    input0->GetOutputSlot(0).Connect(addition->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(addition->GetInputSlot(1));
    addition->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Setting the TensorInfos
    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    addition->GetOutputSlot(0).SetTensorInfo(info);

    auto inputLayers = network.GetGraph().GetInputLayers(); // List of input layers

    // Trying to override the input range for the input layer with binding id 3 (does not exist in the network)
    OverrideInputRangeVisitor overrideInputRangeVisitorLayer3(guidToRangesMap, 3, minMaxRange);
    VisitLayers(inputLayers, overrideInputRangeVisitorLayer3);

    // Check that the map of ranges remained untouched
    BOOST_CHECK(guidToRangesMap.empty());

    // Override the input range for the input layer with binding id 1
    OverrideInputRangeVisitor overrideInputRangeVisitorLayer1(guidToRangesMap, 1, minMaxRange);
    VisitLayers(inputLayers, overrideInputRangeVisitorLayer1);

    // Check that the map of ranges has been populated
    BOOST_CHECK(!guidToRangesMap.empty());

    // Check that an entry for the input layer with binding id 0 does not exist
    BOOST_CHECK(guidToRangesMap.find(input0->GetGuid()) == guidToRangesMap.end());

    // Check that an entry for the input layer with binding id 1 exists
    BOOST_CHECK(guidToRangesMap.find(input1->GetGuid()) != guidToRangesMap.end());

    // Check that at least a value has been added for the input layer with binding id 1
    BOOST_CHECK(!guidToRangesMap[input1->GetGuid()].empty());

    // Check the the overridden values are what we intended to set
    BOOST_CHECK(guidToRangesMap[input1->GetGuid()].at(0).first  == minMaxRange.first);
    BOOST_CHECK(guidToRangesMap[input1->GetGuid()].at(0).second == minMaxRange.second);
}

INetworkPtr CreateNetworkWithFullyConnectedLayer(const bool biasEnabled)
{
    FullyConnectedDescriptor desc;
    desc.m_BiasEnabled = biasEnabled;
    auto network = INetwork::Create();

    TensorShape shape{3U};
    TensorInfo info(shape, DataType::Float32);

    std::vector<float> weightsData{-1.0f, 1.5f, 2.0f};
    ConstTensor weights(info, weightsData);

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* fullyConnected;
    if (desc.m_BiasEnabled)
    {
        std::vector<float> biasData{10.0f, 20.0f, 30.0f};
        ConstTensor bias(info, biasData);
        fullyConnected = network->AddFullyConnectedLayer(desc, weights, bias);
    }
    else
    {
        fullyConnected = network->AddFullyConnectedLayer(desc, weights);
    }
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    input0->GetOutputSlot(0).Connect(fullyConnected->GetInputSlot(0));
    fullyConnected->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    fullyConnected->GetOutputSlot(0).SetTensorInfo(info);

    return network;
}

class TestFullyConnectedQuantization : public TestQuantization
{
public:
    void VisitFullyConnectedLayer(const IConnectableLayer* layer,
                                  const FullyConnectedDescriptor& desc,
                                  const ConstTensor& weights,
                                  const Optional<ConstTensor>& biases,
                                  const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current static value [-15.0f, 15.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f/255.0f, 0.000001f );

        //Test weights
        BOOST_TEST((weights.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
        BOOST_CHECK_CLOSE(weights.GetInfo().GetQuantizationScale(), 3.0f/255.0f, 0.000001f);
        BOOST_TEST((weights.GetInfo().GetQuantizationOffset() == 85));

        // Test biases
        if (biases.has_value())
        {
            BOOST_TEST((biases.value().GetInfo().GetDataType() == DataType::QuantisedAsymm8));
            BOOST_CHECK_CLOSE(biases.value().GetInfo().GetQuantizationScale(), 30.0f/255.0f, 0.000001f);
        }
    }
};

void ValidateFullyConnectedLayer(const bool biasEnabled)
{
    auto network = CreateNetworkWithFullyConnectedLayer(biasEnabled);
    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestFullyConnectedQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeFullyConnected)
{
    ValidateFullyConnectedLayer(false);
}

BOOST_AUTO_TEST_CASE(QuantizeFullyConnectedBiasEnabled)
{
    ValidateFullyConnectedLayer(true);
}

class TestConv2dQuantization : public TestQuantization
{
public:
    virtual void VisitConvolution2dLayer(const IConnectableLayer *layer,
                                         const Convolution2dDescriptor& convolution2dDescriptor,
                                         const ConstTensor& weights,
                                         const Optional<ConstTensor>& biases,
                                         const char *name = nullptr)
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();
        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));
        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current static value [-15.0f, 15.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f / 255.0f, 0.000001f);

        // Test weitghs
        BOOST_TEST((weights.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
        BOOST_CHECK_CLOSE(weights.GetInfo().GetQuantizationScale(), 3.0f / 255.0f, 0.000001f);
        BOOST_TEST((weights.GetInfo().GetQuantizationOffset() == 85));

        // Test biases
        if (biases.has_value())
        {
            BOOST_TEST((biases.value().GetInfo().GetDataType() == DataType::QuantisedAsymm8));
            BOOST_CHECK_CLOSE(biases.value().GetInfo().GetQuantizationScale(), 3.0f / 255.0f, 0.000001f);
            BOOST_TEST((biases.value().GetInfo().GetQuantizationOffset() == 85));
        }
    }
};

void TestQuantizeConvolution2d(bool useBiases)
{
    auto network = INetwork::Create();

    TensorShape shape{3U};
    TensorInfo info(shape, DataType::Float32);

    std::vector<float> weightsData{-1.0f, 1.5f, 2.0f};
    ConstTensor weights(info, weightsData);

    Convolution2dDescriptor descriptor;
    descriptor.m_BiasEnabled = useBiases;

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* conv2d;
    if (useBiases)
    {
        std::vector<float> biasesData{-1.0f, 1.5f, 2.0f};
        ConstTensor biases(info, biasesData);
        conv2d = network->AddConvolution2dLayer(descriptor, weights, biases);
    }
    else
    {
        conv2d = network->AddConvolution2dLayer(descriptor, weights);
    }
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    input0->GetOutputSlot(0).Connect(conv2d->GetInputSlot(0));
    conv2d->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    conv2d->GetOutputSlot(0).SetTensorInfo(info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestConv2dQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeConvolution2d)
{
    TestQuantizeConvolution2d(false);
}

BOOST_AUTO_TEST_CASE(QuantizeConvolution2dWithBiases)
{
    TestQuantizeConvolution2d(true);
}

class TestSoftmaxQuantization : public TestQuantization
{
public:
    virtual void VisitSoftmaxLayer(const IConnectableLayer* layer,
                                   const SoftmaxDescriptor& descriptor,
                                   const char* name = nullptr)
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 0));

        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 1.0f/255.0f, 0.000001f );
    }
};

INetworkPtr CreateNetworkWithSoftmaxLayer(const SoftmaxDescriptor& descriptor)
{
    auto network = INetwork::Create();
    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* softmax = network->AddSoftmaxLayer(descriptor);
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input0->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(info);
    softmax->GetOutputSlot(0).SetTensorInfo(info);

    return network;
}

BOOST_AUTO_TEST_CASE(QuantizeSoftmax)
{
    SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;

    auto network = CreateNetworkWithSoftmaxLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSoftmaxQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_SUITE_END()
} // namespace armnn
