//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/INetworkQuantizer.hpp>
#include <armnn/Types.hpp>

#include "armnn/LayerVisitorBase.hpp"
#include "../Graph.hpp"
#include "../Network.hpp"
#include "../NetworkQuantizerUtils.hpp"
#include "../OverrideInputRangeVisitor.hpp"
#include "../RangeTracker.hpp"
#include "../backends/backendsCommon/test/QuantizeHelper.hpp"

#include <boost/test/unit_test.hpp>

#include <unordered_map>

namespace armnn
{
using MinMaxRange = std::pair<float, float>;
using MinMaxRanges = std::vector<MinMaxRange>;
using MinMaxRangeMap = std::unordered_map<LayerGuid, MinMaxRanges>;

const float g_Asymm8QuantizationBase = 255.0f;
const float g_Symm16QuantizationBase = 32767.0f;
const float g_TestTolerance = 0.000001f;

BOOST_AUTO_TEST_SUITE(Quantizer)

class TestQuantization : public LayerVisitorBase<VisitorThrowingPolicy>
{
public:
    TestQuantization()
    : LayerVisitorBase<VisitorThrowingPolicy>()
    , m_QuantizerOptions(QuantizerOptions()) {}

    TestQuantization(const QuantizerOptions& options)
    : LayerVisitorBase<VisitorThrowingPolicy>()
    , m_QuantizerOptions(options) {}

    void VisitInputLayer(const IConnectableLayer* layer,
                         LayerBindingId id,
                         const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        // Based off current default [-15.0f, 15.0f]
        TestQuantizationParams(info, {30.0f / g_Asymm8QuantizationBase, 128}, {15.0f / g_Symm16QuantizationBase, 0});
    }

    void VisitOutputLayer(const IConnectableLayer* layer,
                          LayerBindingId id,
                          const char* name = nullptr) override
    {}

protected:
    void TestQuantizationParams(const TensorInfo& info,
                                const OffsetScalePair& qAsymm8Params,
                                const OffsetScalePair& qSymm16Params)
    {
        switch (m_QuantizerOptions.m_ActivationFormat)
        {
            case DataType::QuantisedAsymm8:
                TestQuantizationParamsImpl(
                    info, DataType::QuantisedAsymm8, qAsymm8Params.first, qAsymm8Params.second);
                break;
            case DataType::QuantisedSymm16:
                TestQuantizationParamsImpl(
                    info, DataType::QuantisedSymm16, qSymm16Params.first, qSymm16Params.second);
                break;
            default:
                throw InvalidArgumentException("Unsupported quantization target");
        }
    }

    void TestConstantQuantizationParams(const TensorInfo& info, const OffsetScalePair& params)
    {
        TestQuantizationParamsImpl(info, DataType::QuantisedAsymm8, params.first, params.second);
    }

private:
    void TestQuantizationParamsImpl(const TensorInfo& info, DataType dataType, float scale, int32_t offset)
    {
        BOOST_TEST((info.GetDataType() == dataType));
        BOOST_TEST(info.GetQuantizationOffset() == offset);
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), scale, g_TestTolerance);
    }

    QuantizerOptions m_QuantizerOptions;
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
    TestAdditionQuantization()
    : TestQuantization() {}

    TestAdditionQuantization(const QuantizerOptions& options)
    : TestQuantization(options) {}

    void VisitAdditionLayer(const IConnectableLayer* layer,
                            const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        // Based off default static range [-20.0f, 20.0f]
        TestQuantizationParams(info, {40.0f / g_Asymm8QuantizationBase, 128}, {20.0f / g_Symm16QuantizationBase, 0});
    }
};

BOOST_AUTO_TEST_CASE(QuantizeAddition)
{
    INetworkPtr network = INetwork::Create();

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

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestAdditionQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestAdditionQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

class TestActivationQuantization : public TestQuantization
{
public:
    TestActivationQuantization()
    : TestQuantization() {}

    TestActivationQuantization(const QuantizerOptions& options)
    : TestQuantization(options) {}

    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& descriptor,
                              const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        // Based off default static range [0.0f, 15.0f]
        TestQuantizationParams(info, {15.0f / g_Asymm8QuantizationBase, 0}, {15.0f / g_Symm16QuantizationBase, 0});
    }
};

INetworkPtr CreateNetworkWithActivationLayer(const ActivationDescriptor& descriptor)
{
    INetworkPtr network = INetwork::Create();

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

    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestActivationQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeLinearActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Linear;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestActivationQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::ReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestActivationQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeSoftReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::SoftReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestActivationQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeBoundedReluActivation)
{
    class TestBoundedReluActivationQuantization : public TestQuantization
    {
    public:
        TestBoundedReluActivationQuantization()
        : TestQuantization() {}

        TestBoundedReluActivationQuantization(const QuantizerOptions& options)
        : TestQuantization(options) {}

        void VisitActivationLayer(const IConnectableLayer* layer,
                                  const ActivationDescriptor& descriptor,
                                  const char* name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [0.0f, 3.5f]
            TestQuantizationParams(info, {3.5f / g_Asymm8QuantizationBase, 0}, {3.5f / g_Symm16QuantizationBase, 0});
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::BoundedReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBoundedReluActivationQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestBoundedReluActivationQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeTanHActivation)
{
    class TestTanHActivationQuantization : public TestQuantization
    {
    public:
        TestTanHActivationQuantization()
        : TestQuantization() {}

        TestTanHActivationQuantization(const QuantizerOptions& options)
        : TestQuantization(options) {}

        void VisitActivationLayer(const IConnectableLayer* layer,
                                  const ActivationDescriptor& descriptor,
                                  const char* name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-1.0f, 1.0f]
            TestQuantizationParams(
                info, {2.0f / g_Asymm8QuantizationBase, 128}, {1.0f / g_Symm16QuantizationBase, 0});
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::TanH;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestTanHActivationQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestTanHActivationQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

class TestLeakyReLuActivationQuantization : public TestQuantization
{
public:
    TestLeakyReLuActivationQuantization()
    : TestQuantization() {}

    TestLeakyReLuActivationQuantization(const QuantizerOptions& options)
    : TestQuantization(options) {}

    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& descriptor,
                              const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        // Based off default static range [-5.0f, 15.0f]
        TestQuantizationParams(info, {20.0f / g_Asymm8QuantizationBase, 64}, {15.0f / g_Symm16QuantizationBase, 0});
    }

protected:
    // Used by the descendant classes which test layers
    // that are forwarding their parent layer settings
    void CheckForwardedQuantizationSettings(const IConnectableLayer* layer)
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();
        TestQuantizationParams(info, {20.0f / g_Asymm8QuantizationBase, 64}, {15.0f / g_Symm16QuantizationBase, 0});
    }
};

BOOST_AUTO_TEST_CASE(QuantizeLeakyReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::LeakyReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestLeakyReLuActivationQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestLeakyReLuActivationQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeBatchNorm)
{
    class TestBatchNormalizationQuantization : public TestQuantization
    {
    public:
        TestBatchNormalizationQuantization()
        : TestQuantization() {}

        TestBatchNormalizationQuantization(const QuantizerOptions& options)
        : TestQuantization(options) {}

        void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                          const BatchNormalizationDescriptor& desc,
                                          const ConstTensor& mean,
                                          const ConstTensor& variance,
                                          const ConstTensor& beta,
                                          const ConstTensor& gamma,
                                          const char* name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-15.0f, 15.0f]
            TestQuantizationParams(
                info, {30.0f / g_Asymm8QuantizationBase, 128}, {15.0f / g_Symm16QuantizationBase, 0});

            // Test constants
            TestConstantQuantizationParams(mean.GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});
            TestConstantQuantizationParams(variance.GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});
            TestConstantQuantizationParams(beta.GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});
            TestConstantQuantizationParams(gamma.GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});
        }
    };

    INetworkPtr network = INetwork::Create();

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

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBatchNormalizationQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestBatchNormalizationQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
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
    INetworkPtr network = INetwork::Create();

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

void ValidateFullyConnectedLayer(const bool biasEnabled)
{
    class TestFullyConnectedQuantization : public TestQuantization
    {
    public:
        TestFullyConnectedQuantization()
        : TestQuantization() {}

        TestFullyConnectedQuantization(const QuantizerOptions& options)
        : TestQuantization(options) {}

        void VisitFullyConnectedLayer(const IConnectableLayer* layer,
                                      const FullyConnectedDescriptor& desc,
                                      const ConstTensor& weights,
                                      const Optional<ConstTensor>& biases,
                                      const char* name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-15.0f, 15.0f]
            TestQuantizationParams(
                info, {30.0f / g_Asymm8QuantizationBase, 128}, {15.0f / g_Symm16QuantizationBase, 0});

            TestConstantQuantizationParams(weights.GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});

            if (biases.has_value())
            {
                TestConstantQuantizationParams(biases.value().GetInfo(), {30.0f / g_Asymm8QuantizationBase, 0});
            }
        }
    };

    INetworkPtr network = CreateNetworkWithFullyConnectedLayer(biasEnabled);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestFullyConnectedQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestFullyConnectedQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeFullyConnected)
{
    ValidateFullyConnectedLayer(false);
}

BOOST_AUTO_TEST_CASE(QuantizeFullyConnectedBiasEnabled)
{
    ValidateFullyConnectedLayer(true);
}

void TestQuantizeConvolution2d(bool useBiases)
{
    class TestConv2dQuantization : public TestQuantization
    {
    public:
        TestConv2dQuantization()
        : TestQuantization() {}

        TestConv2dQuantization(const QuantizerOptions& options)
        : TestQuantization(options) {}

        void VisitConvolution2dLayer(const IConnectableLayer *layer,
                                     const Convolution2dDescriptor& convolution2dDescriptor,
                                     const ConstTensor& weights,
                                     const Optional<ConstTensor>& biases,
                                     const char *name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-15.0f, 15.0f]
            TestQuantizationParams(
                info, {30.0f / g_Asymm8QuantizationBase, 128}, {15.0f / g_Symm16QuantizationBase, 0});

            TestConstantQuantizationParams(weights.GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});

            if (biases.has_value())
            {
                TestConstantQuantizationParams(biases.value().GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});
            }
        }
    };

    INetworkPtr network = INetwork::Create();

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

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestConv2dQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestConv2dQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeConvolution2d)
{
    TestQuantizeConvolution2d(false);
}

BOOST_AUTO_TEST_CASE(QuantizeConvolution2dWithBiases)
{
    TestQuantizeConvolution2d(true);
}

void TestQuantizeDepthwiseConvolution2d(bool useBiases)
{
    class TestDepthwiseConv2dQuantization : public TestQuantization
    {
    public:
        TestDepthwiseConv2dQuantization()
        : TestQuantization() {}

        TestDepthwiseConv2dQuantization(const QuantizerOptions& options)
        : TestQuantization(options) {}

        void VisitDepthwiseConvolution2dLayer(const IConnectableLayer *layer,
                                              const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
                                              const ConstTensor& weights,
                                              const Optional<ConstTensor>& biases,
                                              const char *name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-15.0f, 15.0f]
            TestQuantizationParams(
                info, {30.0f / g_Asymm8QuantizationBase, 128}, {15.0f / g_Symm16QuantizationBase, 0});

            TestConstantQuantizationParams(weights.GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});

            if (biases.has_value())
            {
                TestConstantQuantizationParams(biases.value().GetInfo(), {3.0f / g_Asymm8QuantizationBase, 85});
            }
        }
    };

    INetworkPtr network = INetwork::Create();

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

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestDepthwiseConv2dQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestDepthwiseConv2dQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeDepthwiseConvolution2d)
{
    TestQuantizeDepthwiseConvolution2d(false);
}

BOOST_AUTO_TEST_CASE(QuantizeDepthwiseConvolution2dWithBiases)
{
    TestQuantizeDepthwiseConvolution2d(true);
}

INetworkPtr CreateNetworkWithSoftmaxLayer(const SoftmaxDescriptor& descriptor)
{
    INetworkPtr network = INetwork::Create();

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
    class TestSoftmaxQuantization : public TestQuantization
    {
    public:
        TestSoftmaxQuantization()
        : TestQuantization() {}

        TestSoftmaxQuantization(const QuantizerOptions& options)
        : TestQuantization(options) {}

        void VisitSoftmaxLayer(const IConnectableLayer* layer,
                               const SoftmaxDescriptor& descriptor,
                               const char* name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [0.0f, 1.0f]
            TestQuantizationParams(info, {1.0f / g_Asymm8QuantizationBase, 0}, {1.0f / g_Symm16QuantizationBase, 0});
        }
    };

    SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;

    INetworkPtr network = CreateNetworkWithSoftmaxLayer(descriptor);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSoftmaxQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestSoftmaxQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
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

    // Set TensorInfo
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
        TestPermuteQuantization()
        : TestLeakyReLuActivationQuantization() {}

        TestPermuteQuantization(const QuantizerOptions& options)
        : TestLeakyReLuActivationQuantization(options) {}

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

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPermuteQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestPermuteQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeSpaceToBatch)
{
    class TestSpaceToBatchQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestSpaceToBatchQuantization()
        : TestLeakyReLuActivationQuantization() {}

        TestSpaceToBatchQuantization(const QuantizerOptions& options)
        : TestLeakyReLuActivationQuantization(options) {}

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

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSpaceToBatchQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestSpaceToBatchQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizePooling2d)
{
    class TestPooling2dQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestPooling2dQuantization()
        : TestLeakyReLuActivationQuantization() {}

        TestPooling2dQuantization(const QuantizerOptions& options)
        : TestLeakyReLuActivationQuantization(options) {}

        void VisitPooling2dLayer(const IConnectableLayer* layer,
                                 const Pooling2dDescriptor& desc,
                                 const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

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

    // Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);
    pooling2d->GetOutputSlot(0).SetTensorInfo(info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPooling2dQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestPooling2dQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeConstant)
{
    class TestConstantQuantization : public TestAdditionQuantization
    {
    public:
        TestConstantQuantization()
        : TestAdditionQuantization() {}

        TestConstantQuantization(const QuantizerOptions& options)
        : TestAdditionQuantization(options) {}

        void VisitConstantLayer(const IConnectableLayer* layer,
                                const ConstTensor& input,
                                const char* name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off the range of values in the const tensor used for the test: [-2.0f, 6.0f]
            TestQuantizationParams(info, {8.0f / g_Asymm8QuantizationBase, 64}, {6.0f / g_Symm16QuantizationBase, 0});
        }
    };

    INetworkPtr network = INetwork::Create();

    // Constant layer data
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    TensorInfo tensorInfo(4, dimensions.data(), DataType::Float32);
    ConstTensor constantTensor(tensorInfo, data);

    // Add the layers
    IConnectableLayer* input    = network->AddInputLayer(0);
    IConnectableLayer* constant = network->AddConstantLayer(constantTensor);
    IConnectableLayer* addition = network->AddAdditionLayer();
    IConnectableLayer* output   = network->AddOutputLayer(1);

    // Establish connections
    input->GetOutputSlot(0).Connect(addition->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(addition->GetInputSlot(1));
    addition->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Set TensorInfo in the remaining layers
    input->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    addition->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestConstantQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestConstantQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeMerger)
{
    class TestMergerQuantization : public TestQuantization
    {
    public:
        TestMergerQuantization()
        : TestQuantization() {}

        TestMergerQuantization(const QuantizerOptions& options)
        : TestQuantization(options) {}

        void VisitInputLayer(const IConnectableLayer* layer,
                             LayerBindingId id,
                             const char* name = nullptr) override
        {}
        void VisitOutputLayer(const IConnectableLayer* layer,
                              LayerBindingId id,
                              const char* name = nullptr) override
        {}
        void VisitMergerLayer(const IConnectableLayer* layer,
                              const OriginsDescriptor& mergerDescriptor,
                              const char* name = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            TestQuantizationParams(
                info, {60.8f / g_Asymm8QuantizationBase, 65}, {45.3f / g_Symm16QuantizationBase, 0});
        }
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

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkQuantizerPtr quantizerPtrQAsymm8 =  INetworkQuantizer::Create(network.get());
    INetworkQuantizerPtr quantizerPtrQSymm16 =  INetworkQuantizer::Create(network.get(), options);
    // Override the input ranges
    float min = -15.5f;
    float max = 45.3f;

    quantizerPtrQAsymm8->OverrideInputRange(0, (min + 2.1f), (max - 3.2f));
    quantizerPtrQAsymm8->OverrideInputRange(1, (min + 6.7f), max);
    quantizerPtrQAsymm8->OverrideInputRange(2, min, (max - 7.8f));

    quantizerPtrQSymm16->OverrideInputRange(0, (min + 2.1f), (max - 3.2f));
    quantizerPtrQSymm16->OverrideInputRange(1, (min + 6.7f), max);
    quantizerPtrQSymm16->OverrideInputRange(2, min, (max - 7.8f));

    INetworkPtr quantizedNetworkQAsymm8 = quantizerPtrQAsymm8->ExportNetwork();
    TestMergerQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    INetworkPtr quantizedNetworkQSymm16 = quantizerPtrQSymm16->ExportNetwork();
    TestMergerQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeReshape)
{
    class TestReshapeQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestReshapeQuantization()
        : TestLeakyReLuActivationQuantization() {}

        TestReshapeQuantization(const QuantizerOptions& options)
        : TestLeakyReLuActivationQuantization(options) {}

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

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestReshapeQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestReshapeQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeSplitter)
{
    class TestSplitterQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestSplitterQuantization()
        : TestLeakyReLuActivationQuantization() {}

        TestSplitterQuantization(const QuantizerOptions& options)
        : TestLeakyReLuActivationQuantization(options) {}

        virtual void VisitSplitterLayer(const IConnectableLayer* layer,
                                        const SplitterDescriptor& desc,
                                        const char* name = nullptr)
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    TensorShape shape{3U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    ViewsDescriptor splitterDesc(2,4);
    IConnectableLayer* splitter = network->AddSplitterLayer(splitterDesc);
    CompleteLeakyReluNetwork(network.get(), activation, splitter, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSplitterQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestSplitterQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeResizeBilinear)
{
    class TestResizeBilinearQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestResizeBilinearQuantization()
        : TestLeakyReLuActivationQuantization() {}

        TestResizeBilinearQuantization(const QuantizerOptions& options)
        : TestLeakyReLuActivationQuantization(options) {}

        void VisitResizeBilinearLayer(const IConnectableLayer* layer,
                                      const ResizeBilinearDescriptor& resizeDescriptor,
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
    ResizeBilinearDescriptor descriptor;
    descriptor.m_TargetHeight = 3;
    descriptor.m_TargetWidth = 3;
    IConnectableLayer* spaceToBatch = network->AddResizeBilinearLayer(descriptor);

    CompleteLeakyReluNetwork(network.get(), activation, spaceToBatch, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestResizeBilinearQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestResizeBilinearQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeStridedSlice)
{
    class TestStridedSliceQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestStridedSliceQuantization()
        : TestLeakyReLuActivationQuantization() {}

        TestStridedSliceQuantization(const QuantizerOptions& options)
        : TestLeakyReLuActivationQuantization(options) {}

        virtual void VisitStridedSliceLayer(const IConnectableLayer* layer,
                                            const StridedSliceDescriptor& desc,
                                            const char* name = nullptr)
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    TensorShape shape{3U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    StridedSliceDescriptor stridedSliceDesc;
    IConnectableLayer* stridedSlice = network->AddStridedSliceLayer(stridedSliceDesc);

    CompleteLeakyReluNetwork(network.get(), activation, stridedSlice, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestStridedSliceQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestStridedSliceQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeBatchToSpace)
{
    class TestBatchToSpaceQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestBatchToSpaceQuantization()
        : TestLeakyReLuActivationQuantization() {}

        TestBatchToSpaceQuantization(const QuantizerOptions& options)
        : TestLeakyReLuActivationQuantization(options) {}

        void VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                      const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
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
    BatchToSpaceNdDescriptor descriptor;
    IConnectableLayer* batchToSpace = network->AddBatchToSpaceNdLayer(descriptor);

    CompleteLeakyReluNetwork(network.get(), activation, batchToSpace, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBatchToSpaceQuantization validatorQAsymm8;
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestBatchToSpaceQuantization validatorQSymm16(options);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

std::vector<uint8_t> SetupQuantize(float value)
{
    armnn::TensorInfo inputInfo({ 1, 2, 2 }, armnn::DataType::Float32);
    inputInfo.SetQuantizationScale(1.0f);
    inputInfo.SetQuantizationOffset(1);
    std::vector<float> input({
                                     value, 0.0f,
                                     0.0f, 1.0f
                             });
    const std::vector<float> &inputRef = input;

    auto output = QuantizedVector<uint8_t>(inputInfo.GetQuantizationScale(),
                                           inputInfo.GetQuantizationOffset(),
                                           inputRef);

    return output;
}

BOOST_AUTO_TEST_CASE(QuantizeInf)
{
    BOOST_CHECK_EQUAL(SetupQuantize(std::numeric_limits<float>::infinity())[0], 255);
}

BOOST_AUTO_TEST_CASE(QuantizeNegativeInf)
{
    BOOST_CHECK_EQUAL(SetupQuantize(-1 * std::numeric_limits<float>::infinity())[0], 0);
}

BOOST_AUTO_TEST_SUITE_END()
} // namespace armnn
