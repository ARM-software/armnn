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
#include "../RangeTracker.hpp"

#include <boost/test/unit_test.hpp>

#include <unordered_map>

namespace armnn
{
using MinMaxRange = std::pair<float, float>;
using MinMaxRanges = std::vector<MinMaxRange>;
using MinMaxRangeMap = std::unordered_map<LayerGuid, MinMaxRanges>;

const float g_QuantizationBase = 255.0f;
const float g_TestTolerance = 0.000001f;

BOOST_AUTO_TEST_SUITE(Quantizer)

class TestQuantization : public LayerVisitorBase<VisitorThrowingPolicy>
{
public:
    void VisitInputLayer(const IConnectableLayer* layer,
                         LayerBindingId id,
                         const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current default [-15.0f, 15.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f / g_QuantizationBase, g_TestTolerance);
    }

    void VisitOutputLayer(const IConnectableLayer* layer,
                          LayerBindingId id,
                          const char* name = nullptr) override
    {}
};

void VisitLayersTopologically(const INetwork* inputNetwork, ILayerVisitor& visitor)
{
    auto network = boost::polymorphic_downcast<const Network*>(inputNetwork);
    auto graph = network->GetGraph().TopologicalSort();

    VisitLayers(graph, visitor);
}

class TestAdditionQuantization : public TestQuantization
{
public:
    void VisitAdditionLayer(const IConnectableLayer* layer,
                            const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current static value [-20.0f, 20.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 40.0f / g_QuantizationBase, g_TestTolerance);
    }
};

BOOST_AUTO_TEST_CASE(QuantizeAddition)
{
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

    // Set TensorInfo
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
    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& descriptor,
                              const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 0));

        // Based off current static value [-20.0f, 20.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 15.0f / g_QuantizationBase, g_TestTolerance);
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

    // Set TensorInfo
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

class TestBoundedReluActivationQuantization : public TestQuantization
{
public:
    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& descriptor,
                              const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 0));

        // Based off current static value [0.0f, 3.5f(<-layer upper bound)]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 3.5f / g_QuantizationBase, g_TestTolerance);
    }
};

BOOST_AUTO_TEST_CASE(QuantizeBoundedReluActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::BoundedReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBoundedReluActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

class TestTanHActivationQuantization : public TestQuantization
{
public:
    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& descriptor,
                              const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current static value [-1.0f, 1.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 2.0f / g_QuantizationBase, g_TestTolerance);
    }
};

BOOST_AUTO_TEST_CASE(QuantizeTanHActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::TanH;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestTanHActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

class TestLeakyReLuActivationQuantization : public TestQuantization
{
public:
    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& descriptor,
                              const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 64));

        // Based off current static value [-5.0f, 15.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 20.0f / g_QuantizationBase, g_TestTolerance);
    }
protected:
    // used by the descendant classes which test layers
    // that are forwarding their parent layer settings
    void CheckForwardedQuantizationSettings(const IConnectableLayer* layer)
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 64));

        // Based off parent LeakyReLu [-5.f, 15.f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 20.0f/g_QuantizationBase, g_TestTolerance);
    }
};

BOOST_AUTO_TEST_CASE(QuantizeLeakyReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::LeakyReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    auto network = CreateNetworkWithActivationLayer(descriptor);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestLeakyReLuActivationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

class TestBatchNormalizationQuantization : public TestQuantization
{
public:
    void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                      const BatchNormalizationDescriptor& desc,
                                      const ConstTensor& mean,
                                      const ConstTensor& variance,
                                      const ConstTensor& beta,
                                      const ConstTensor& gamma,
                                      const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current static value [-15.0f, 15.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f / g_QuantizationBase, g_TestTolerance);

        // Test constants
        BOOST_TEST((mean.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
        BOOST_TEST((variance.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
        BOOST_TEST((beta.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
        BOOST_TEST((gamma.GetInfo().GetDataType() == DataType::QuantisedAsymm8));

        float expectedQuantizationScale = 3.0f / g_QuantizationBase;
        BOOST_CHECK_CLOSE(mean.GetInfo().GetQuantizationScale(),     expectedQuantizationScale, g_TestTolerance);
        BOOST_CHECK_CLOSE(variance.GetInfo().GetQuantizationScale(), expectedQuantizationScale, g_TestTolerance);
        BOOST_CHECK_CLOSE(beta.GetInfo().GetQuantizationScale(),     expectedQuantizationScale, g_TestTolerance);
        BOOST_CHECK_CLOSE(gamma.GetInfo().GetQuantizationScale(),    expectedQuantizationScale, g_TestTolerance);

        BOOST_TEST((mean.GetInfo().GetQuantizationOffset() == 85));
    }
};

BOOST_AUTO_TEST_CASE(QuantizeBatchNorm)
{
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

    // Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    batchNorm->GetOutputSlot(0).SetTensorInfo(info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBatchNormalizationQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(OverrideInputRangeEmptyNetwork)
{
    RangeTracker ranges;
    RangeTracker::MinMaxRange minMaxRange(-12.3f, 45.6f); // Range to use for the override

    Network network; // Empty network
    auto inputLayers = network.GetGraph().GetInputLayers(); // Empty list of input layers

    OverrideInputRangeVisitor overrideInputRangeVisitor(ranges, 0, minMaxRange);
    VisitLayers(inputLayers, overrideInputRangeVisitor);

    BOOST_CHECK(ranges.IsEmpty()); // Check that the map of ranges remained untouched
}

BOOST_AUTO_TEST_CASE(OverrideInputRangeNoInputLayers)
{
    RangeTracker ranges;
    MinMaxRange minMaxRange(-12.3f, 45.6f); // Range to use for the override

    Network network;
    network.AddAdditionLayer(); // Network with no input layers
    auto inputLayers = network.GetGraph().GetInputLayers(); // Empty list of input layers

    OverrideInputRangeVisitor overrideInputRangeVisitor(ranges, 0, minMaxRange);
    VisitLayers(inputLayers, overrideInputRangeVisitor);

    BOOST_CHECK(ranges.IsEmpty()); // Check that the map of ranges remained untouched
}

BOOST_AUTO_TEST_CASE(OverrideInputRangeInputLayers)
{
    RangeTracker ranges;
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
    OverrideInputRangeVisitor overrideInputRangeVisitorLayer3(ranges, 3, minMaxRange);
    VisitLayers(inputLayers, overrideInputRangeVisitorLayer3);

    // Check that the map of ranges remained untouched
    BOOST_CHECK(ranges.IsEmpty());

    // Override the input range for the input layer with binding id 1
    OverrideInputRangeVisitor overrideInputRangeVisitorLayer1(ranges, 1, minMaxRange);
    VisitLayers(inputLayers, overrideInputRangeVisitorLayer1);

    // Check that the map of ranges has been populated
    BOOST_CHECK(!ranges.IsEmpty());

    // Check that an entry for the input layer with binding id 0 does not exist
    BOOST_CHECK(!ranges.HasRanges(input0->GetGuid()));

    // Check that an entry for the input layer with binding id 1 exists
    BOOST_CHECK(ranges.HasRanges(input1->GetGuid()));

    // Check the the overridden values are what we intended to set
    BOOST_CHECK(ranges.GetRange(input1->GetGuid(), 0) == minMaxRange);
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

    // Set TensorInfo
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
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f / g_QuantizationBase, g_TestTolerance );

        // Test weights
        BOOST_TEST((weights.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
        BOOST_CHECK_CLOSE(weights.GetInfo().GetQuantizationScale(), 3.0f / g_QuantizationBase, g_TestTolerance);
        BOOST_TEST((weights.GetInfo().GetQuantizationOffset() == 85));

        // Test biases
        if (biases.has_value())
        {
            BOOST_TEST((biases.value().GetInfo().GetDataType() == DataType::QuantisedAsymm8));
            BOOST_CHECK_CLOSE(biases.value().GetInfo().GetQuantizationScale(),
                              30.0f / g_QuantizationBase,
                              g_TestTolerance);
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
    void VisitConvolution2dLayer(const IConnectableLayer *layer,
                                 const Convolution2dDescriptor& convolution2dDescriptor,
                                 const ConstTensor& weights,
                                 const Optional<ConstTensor>& biases,
                                 const char *name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();
        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));
        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current static value [-15.0f, 15.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f / g_QuantizationBase, g_TestTolerance);

        // Test weights
        // Instantiate expected values
        const float quantizationScale = 3.0f / g_QuantizationBase;
        const float tolerance = 3.0f / g_QuantizationBase;
        const int quantizationOffset = 85;
        BOOST_TEST((weights.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
        BOOST_CHECK_CLOSE(weights.GetInfo().GetQuantizationScale(), quantizationScale, tolerance);
        BOOST_TEST((weights.GetInfo().GetQuantizationOffset() == quantizationOffset));

        // Test biases
        if (biases.has_value())
        {
            BOOST_TEST((biases.value().GetInfo().GetDataType() == DataType::QuantisedAsymm8));
            BOOST_CHECK_CLOSE(biases.value().GetInfo().GetQuantizationScale(), quantizationScale, tolerance);
            BOOST_TEST((biases.value().GetInfo().GetQuantizationOffset() == quantizationOffset));
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

    // Set TensorInfo
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

class TestDepthwiseConv2dQuantization : public TestQuantization
{
public:
    void VisitDepthwiseConvolution2dLayer(const IConnectableLayer *layer,
                                          const DepthwiseConvolution2dDescriptor& desc,
                                          const ConstTensor& weights,
                                          const Optional<ConstTensor>& biases,
                                          const char *name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();
        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));
        BOOST_TEST((info.GetQuantizationOffset() == 128));

        // Based off current static value [-15.0f, 15.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f / g_QuantizationBase, g_TestTolerance);

        // Test weights
        // Instantiate expected values
        const float quantizationScale = 3.0f / g_QuantizationBase;
        const float tolerance = 3.0f / g_QuantizationBase;
        const int quantizationOffset = 85;
        BOOST_TEST((weights.GetInfo().GetDataType() == DataType::QuantisedAsymm8));
        BOOST_CHECK_CLOSE(weights.GetInfo().GetQuantizationScale(), quantizationScale, tolerance);
        BOOST_TEST((weights.GetInfo().GetQuantizationOffset() == quantizationOffset));

        // Test biases
        if (biases.has_value())
        {
            BOOST_TEST((biases.value().GetInfo().GetDataType() == DataType::QuantisedAsymm8));
            BOOST_CHECK_CLOSE(biases.value().GetInfo().GetQuantizationScale(), quantizationScale, tolerance);
            BOOST_TEST((biases.value().GetInfo().GetQuantizationOffset() == quantizationOffset));
        }
    }
};

void TestQuantizeDepthwiseConvolution2d(bool useBiases)
{
    auto network = INetwork::Create();

    TensorShape shape{3U};
    TensorInfo info(shape, DataType::Float32);

    std::vector<float> weightsData{-1.0f, 1.5f, 2.0f};
    ConstTensor weights(info, weightsData);

    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_BiasEnabled = useBiases;

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* depthwiseConv2d;
    if (useBiases)
    {
        std::vector<float> biasesData{-1.0f, 1.5f, 2.0f};
        ConstTensor biases(info, biasesData);
        depthwiseConv2d = network->AddDepthwiseConvolution2dLayer(descriptor, weights, biases);
    }
    else
    {
        depthwiseConv2d = network->AddDepthwiseConvolution2dLayer(descriptor, weights);
    }
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    input0->GetOutputSlot(0).Connect(depthwiseConv2d->GetInputSlot(0));
    depthwiseConv2d->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    depthwiseConv2d->GetOutputSlot(0).SetTensorInfo(info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestDepthwiseConv2dQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeDepthwiseConvolution2d)
{
    TestQuantizeDepthwiseConvolution2d(false);
}

BOOST_AUTO_TEST_CASE(QuantizeDepthwiseConvolution2dWithBiases)
{
    TestQuantizeDepthwiseConvolution2d(true);
}

class TestSoftmaxQuantization : public TestQuantization
{
public:
    void VisitSoftmaxLayer(const IConnectableLayer* layer,
                           const SoftmaxDescriptor& descriptor,
                           const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() ==  DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 0));

        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 1.0f / g_QuantizationBase, g_TestTolerance );
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

    // Set TensorInfo
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

IConnectableLayer* CreateStartOfLeakyReluNetwork(INetwork* network, const TensorInfo& info)
{
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::LeakyReLu;
    activationDescriptor.m_A        = 3.5f;
    activationDescriptor.m_B        = -10.0f;

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* activation = network->AddActivationLayer(activationDescriptor);

    // Establish connections
    input0->GetOutputSlot(0).Connect(activation->GetInputSlot(0));

    //Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);

    return activation;
}

void  CompleteLeakyReluNetwork(INetwork* network,
                               IConnectableLayer* activation,
                               IConnectableLayer* layerUnderTest,
                               const TensorInfo& info)
{
    // Add the output Layer
    IConnectableLayer* output = network->AddOutputLayer(3);

    // Establish connections
    activation->GetOutputSlot(0).Connect(layerUnderTest->GetInputSlot(0));
    layerUnderTest->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    layerUnderTest->GetOutputSlot(0).SetTensorInfo(info);
}

BOOST_AUTO_TEST_CASE(QuantizePermute)
{
    class TestPermuteQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        void VisitPermuteLayer(const IConnectableLayer* layer,
                               const PermuteDescriptor& desc,
                               const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    PermuteDescriptor desc;
    IConnectableLayer* permute = network->AddPermuteLayer(desc);

    CompleteLeakyReluNetwork(network.get(), activation, permute, info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPermuteQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeSpaceToBatch)
{
    class TestSpaceToBatchQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        void VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                      const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                      const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    SpaceToBatchNdDescriptor descriptor;
    IConnectableLayer* spaceToBatch = network->AddSpaceToBatchNdLayer(descriptor);

    CompleteLeakyReluNetwork(network.get(), activation, spaceToBatch, info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSpaceToBatchQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

class TestPooling2dQuantization : public TestLeakyReLuActivationQuantization
{
public:
    void VisitPooling2dLayer(const IConnectableLayer* layer,
                             const Pooling2dDescriptor& desc,
                             const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

        BOOST_TEST((info.GetQuantizationOffset() == 64));

        // Based off parent LeakyReLu [-5.f, 15.f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 20.0f / g_QuantizationBase, g_TestTolerance);
    }
};

BOOST_AUTO_TEST_CASE(QuantizePooling2d)
{
    auto network = INetwork::Create();

    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    Pooling2dDescriptor desc;
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::LeakyReLu;
    activationDescriptor.m_A        = 3.5f;
    activationDescriptor.m_B        = -10.0f;

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* activation = network->AddActivationLayer(activationDescriptor);
    IConnectableLayer* pooling2d = network->AddPooling2dLayer(desc);
    IConnectableLayer* output = network->AddOutputLayer(3);

    // Establish connections
    input0->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(pooling2d->GetInputSlot(0));
    pooling2d->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);
    pooling2d->GetOutputSlot(0).SetTensorInfo(info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPooling2dQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

class TestConstantQuantization : public TestAdditionQuantization
{
public:
    void VisitConstantLayer(const IConnectableLayer* layer,
                            const ConstTensor& input,
                            const char* name = nullptr) override
    {
        BOOST_CHECK(std::string(name) == "ConstantLayer");

        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();
        BOOST_CHECK(info.GetDataType() == DataType::QuantisedAsymm8);
        BOOST_CHECK(info.GetQuantizationOffset() == 64);

        // Based off the range of values in the const tensor used for the test: [-2.0f, 6.0f]
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 8.0f / g_QuantizationBase, g_TestTolerance);
    }
};

BOOST_AUTO_TEST_CASE(QuantizeConstant)
{
    auto network = INetwork::Create();

    // Constant layer data
    const char* name = "ConstantLayer";
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    TensorInfo tensorInfo(4, dimensions.data(), DataType::Float32);
    ConstTensor constantTensor(tensorInfo, data);

    // Add the layers
    IConnectableLayer* input    = network->AddInputLayer(0);
    IConnectableLayer* constant = network->AddConstantLayer(constantTensor, name);
    IConnectableLayer* addition = network->AddAdditionLayer();
    IConnectableLayer* output   = network->AddOutputLayer(1);

    // Establish connections
    input->GetOutputSlot(0).Connect(addition->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(addition->GetInputSlot(1));
    addition->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Set TensorInfo in the remaining layers
    input->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    addition->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestConstantQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeMerger)
{
    class TestMergerVisitor : public LayerVisitorBase<VisitorThrowingPolicy>
    {
    public:
        TestMergerVisitor(float min, float max) : m_Min(min), m_Max(max) {}

        virtual void VisitInputLayer(const IConnectableLayer* layer,
                                     LayerBindingId id,
                                     const char* name = nullptr)
        {}
        virtual void VisitOutputLayer(const IConnectableLayer* layer,
                                      LayerBindingId id,
                                      const char* name = nullptr)
        {}
        virtual void VisitMergerLayer(const IConnectableLayer* layer,
                                      const OriginsDescriptor& mergerDescriptor,
                                      const char* name = nullptr)
        {
            std::pair<int, float> expectedValues = ComputeQAsymmParams(8, m_Min, m_Max);

            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            BOOST_TEST((info.GetDataType() == DataType::QuantisedAsymm8));

            BOOST_TEST((info.GetQuantizationOffset() == expectedValues.first));

            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), expectedValues.second, 0.000001f);
        }

    private:
        float m_Min;
        float m_Max;
    };

    INetworkPtr network = INetwork::Create();

    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* input1 = network->AddInputLayer(1);
    IConnectableLayer* input2 = network->AddInputLayer(2);

    OriginsDescriptor descriptor(3, 1);
    IConnectableLayer* merger = network->AddMergerLayer(descriptor);

    IConnectableLayer* output0 = network->AddOutputLayer(3);

    // Establish connections
    input0->GetOutputSlot(0).Connect(merger->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(merger->GetInputSlot(1));
    input2->GetOutputSlot(0).Connect(merger->GetInputSlot(2));
    merger->GetOutputSlot(0).Connect(output0->GetInputSlot(0));

    // Set TensorInfo
    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    input2->GetOutputSlot(0).SetTensorInfo(info);
    merger->GetOutputSlot(0).SetTensorInfo(info);

    INetworkQuantizerPtr quantizerPtr =  INetworkQuantizer::Create(network.get());
    // Override the input ranges
    float min = -15.5f;
    float max = 45.3f;

    quantizerPtr->OverrideInputRange(0, (min + 2.1f), (max - 3.2f));
    quantizerPtr->OverrideInputRange(1, (min + 6.7f), max);
    quantizerPtr->OverrideInputRange(2, min, (max - 7.8f));

    auto quantizedNetwork = quantizerPtr->ExportNetwork();
    TestMergerVisitor validator(min, max);
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_CASE(QuantizeReshape)
{
    class TestReshapeQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        virtual void VisitReshapeLayer(const IConnectableLayer* layer,
                                       const ReshapeDescriptor& reshapeDescriptor,
                                       const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    ReshapeDescriptor descriptor({1, 2, 3, 4});
    IConnectableLayer* reshape = network->AddReshapeLayer(descriptor);

    CompleteLeakyReluNetwork(network.get(), activation, reshape, info);

    auto quantizedNetwork = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestReshapeQuantization validator;
    VisitLayersTopologically(quantizedNetwork.get(), validator);
}

BOOST_AUTO_TEST_SUITE_END()
} // namespace armnn
