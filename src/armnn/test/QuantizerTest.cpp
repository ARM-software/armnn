//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnnQuantizer/INetworkQuantizer.hpp>
#include <armnn/Types.hpp>

#include "armnn/LayerVisitorBase.hpp"
#include "../Graph.hpp"
#include "../Network.hpp"
#include "../NetworkQuantizerUtils.hpp"
#include "../OverrideInputRangeVisitor.hpp"
#include "../RangeTracker.hpp"
#include "../backends/backendsCommon/test/QuantizeHelper.hpp"
#include "../../armnnQuantizer/CommandLineProcessor.hpp"

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
    TestQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
    : LayerVisitorBase<VisitorThrowingPolicy>()
    , m_InputShape(inputShape)
    , m_OutputShape(outputShape)
    , m_QuantizerOptions(QuantizerOptions()) {}

    TestQuantization(const QuantizerOptions& options, const TensorShape& inputShape, const TensorShape& outputShape)
    : LayerVisitorBase<VisitorThrowingPolicy>()
    , m_InputShape(inputShape)
    , m_OutputShape(outputShape)
    , m_QuantizerOptions(options) {}

    void VisitInputLayer(const IConnectableLayer* layer,
                         LayerBindingId id,
                         const char* name = nullptr) override
    {
        const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();
        BOOST_TEST(m_InputShape == info.GetShape());
        // Based off current default [-15.0f, 15.0f]
        TestQuantizationParams(info, {30.0f / g_Asymm8QuantizationBase, 128}, {15.0f / g_Symm16QuantizationBase, 0});
    }

    void VisitOutputLayer(const IConnectableLayer* layer,
                          LayerBindingId id,
                          const char* name = nullptr) override
    {
        const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
        BOOST_TEST(m_OutputShape == info.GetShape());
    }

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

    void TestDifferentQuantizationScale(const TensorInfo& info0, const TensorInfo& info1)
    {
        BOOST_TEST(info0.GetQuantizationScale() != info1.GetQuantizationScale());
    }

    void TestConstantQuantizationParams(const TensorInfo& info,
                                        const OffsetScalePair& params,
                                        DataType dataType = DataType::QuantisedAsymm8)
    {
        TestQuantizationParamsImpl(info, DataType::QuantisedAsymm8, params.first, params.second);
    }

    void TestBiasQuantizationParams(const TensorInfo& info,
                                    const OffsetScalePair& qAsymm8Params,
                                    const OffsetScalePair& qSymm16Params,
                                    DataType dataType = DataType::QuantisedAsymm8)
    {
        switch (m_QuantizerOptions.m_ActivationFormat)
        {
            case DataType::QuantisedAsymm8:
                TestQuantizationParamsImpl(info, dataType, qAsymm8Params.first, qAsymm8Params.second);
                break;
            case DataType::QuantisedSymm16:
                TestQuantizationParamsImpl(info, dataType, qSymm16Params.first, qSymm16Params.second);
                break;
            default:
                throw InvalidArgumentException("Unsupported quantization target");
        }
    }

    void TestQuantizationOnLayersWithBiases(const IConnectableLayer* layer,
                                            const ConstTensor& weights,
                                            const Optional<ConstTensor>& biases)
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();
        float inputScaleQAsymm8 = 30.0f / g_Asymm8QuantizationBase;
        float inputScaleQSymm16 = 15.0f / g_Symm16QuantizationBase;
        float weightsScale = 3.0f / g_Asymm8QuantizationBase;

        // Based off default static range [-15.0f, 15.0f]
        TestQuantizationParams(info, {inputScaleQAsymm8, 128}, {inputScaleQSymm16, 0});

        TestConstantQuantizationParams(weights.GetInfo(), {weightsScale, 85});

        if (biases.has_value())
        {
            TestBiasQuantizationParams(biases.value().GetInfo(),
                                       {inputScaleQAsymm8 * weightsScale, 0},
                                       {inputScaleQSymm16 * weightsScale, 0},
                                       DataType::Signed32);
        }
    }

    TensorShape m_InputShape;
    TensorShape m_OutputShape;

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
    TestAdditionQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
    : TestQuantization(inputShape, outputShape) {}

    TestAdditionQuantization(const QuantizerOptions& options,
                             const TensorShape& inputShape,
                             const TensorShape& outputShape)
    : TestQuantization(options, inputShape, outputShape) {}

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
    const TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    addition->GetOutputSlot(0).SetTensorInfo(info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestAdditionQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestAdditionQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

class TestActivationQuantization : public TestQuantization
{
public:
    TestActivationQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
    : TestQuantization(inputShape, outputShape) {}

    TestActivationQuantization(const QuantizerOptions& options,
                               const TensorShape& inputShape,
                               const TensorShape& outputShape)
    : TestQuantization(options, inputShape, outputShape) {}

    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& descriptor,
                              const char* name = nullptr) override
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        // Based off default static range [0.0f, 15.0f]
        TestQuantizationParams(info, {15.0f / g_Asymm8QuantizationBase, 0}, {15.0f / g_Symm16QuantizationBase, 0});
    }
};

INetworkPtr CreateNetworkWithActivationLayer(const ActivationDescriptor& descriptor, const TensorShape& shape)
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
    TensorInfo info(shape, DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);

    return network;
}

INetworkPtr CreateNetworkWithInputOutputLayers()
{
    INetworkPtr network = INetwork::Create();

    // Add input/output layers
    IConnectableLayer* inputLayer = network->AddInputLayer(0);
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    inputLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Set TensorInfo
    TensorShape shape{8U};
    TensorInfo info(shape, DataType::Float32);
    inputLayer->GetOutputSlot(0).SetTensorInfo(info);

    return network;
}

TensorInfo GetInputTensorInfo(const Network* network)
{
    for (auto&& inputLayer : network->GetGraph().GetInputLayers())
    {
        BOOST_ASSERT_MSG(inputLayer->GetNumOutputSlots() == 1, "Input layer should have exactly 1 output slot");
        return inputLayer->GetOutputSlot(0).GetTensorInfo();
    }
    throw InvalidArgumentException("Network has no input layers");
}

BOOST_AUTO_TEST_CASE(InputOutputLayerDynamicQuant)
{
    INetworkPtr network = CreateNetworkWithInputOutputLayers();

    armnn::TensorInfo tensorInfo = GetInputTensorInfo(boost::polymorphic_downcast<const Network*>(network.get()));

    // Outliers -56 and 98
    std::vector<float> inputData({0, 0, 0, -56, 98, 0, 0, 0});
    armnn::ConstTensor inputTensor(tensorInfo, inputData.data());

    InputTensors inputTensors;
    inputTensors.push_back(std::make_pair(0, inputTensor));

    armnn::INetworkQuantizerPtr quantizer = armnn::INetworkQuantizer::Create(network.get());

    quantizer->Refine(inputTensors);

    // Outliers -77 and 65
    std::vector<float> inputData2({0, -77, 0, -56, 65, 0, 0, 0});
    armnn::ConstTensor inputTensor2(tensorInfo, inputData2.data());
    InputTensors inputTensors2;
    inputTensors2.push_back(std::make_pair(0, inputTensor2));

    quantizer->Refine(inputTensors2);

    INetworkPtr quantizedNetwork = quantizer->ExportNetwork();
    // Output Layer should be quantized for a min max of -77 and 98
    // according to QAsymm8 Quantization Scheme
    std::unique_ptr<IQuantizationScheme> quantizationScheme = std::make_unique<QAsymm8QuantizationScheme>();
    OffsetScalePair qParams = quantizationScheme->ComputeScheme(-77.0, 98.0);

    class TestOutputLayerVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
    {
    public:
        TestOutputLayerVisitor(const OffsetScalePair& offsetScalePair, const DataType& dataType) :
            m_OffsetScalePair(offsetScalePair), m_DataType(dataType) {}

        void VisitOutputLayer(const IConnectableLayer* layer,
                                      LayerBindingId id,
                                      const char* name = nullptr) override
        {
            const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
            BOOST_CHECK_MESSAGE(info.GetDataType() == m_DataType,
                                std::string(armnn::GetDataTypeName(info.GetDataType()))
                                        .append(" == ").append(armnn::GetDataTypeName(m_DataType)));
            // int_32t
            BOOST_CHECK(info.GetQuantizationOffset() == m_OffsetScalePair.second);
            // float
            BOOST_TEST(info.GetQuantizationScale() == m_OffsetScalePair.first, boost::test_tools::tolerance(0.001));
        }

    private:
        const OffsetScalePair m_OffsetScalePair;
        const DataType m_DataType;
    };

    TestOutputLayerVisitor visitor(qParams, quantizationScheme->GetDataType());
    quantizedNetwork->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(QuantizeAbsActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Abs;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestActivationQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeLinearActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Linear;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestActivationQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::ReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestActivationQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeSoftReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::SoftReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestActivationQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeBoundedReluActivation)
{
    class TestBoundedReluActivationQuantization : public TestQuantization
    {
    public:
        TestBoundedReluActivationQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestBoundedReluActivationQuantization(const QuantizerOptions& options,
                                              const TensorShape& inputShape,
                                              const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

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

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBoundedReluActivationQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestBoundedReluActivationQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeTanHActivation)
{
    class TestTanHActivationQuantization : public TestQuantization
    {
    public:
        TestTanHActivationQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestTanHActivationQuantization(const QuantizerOptions& options,
                                       const TensorShape& inputShape,
                                       const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

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

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestTanHActivationQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestTanHActivationQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

class TestLeakyReLuActivationQuantization : public TestQuantization
{
public:
    TestLeakyReLuActivationQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
    : TestQuantization(inputShape, outputShape) {}

    TestLeakyReLuActivationQuantization(const QuantizerOptions& options,
                                        const TensorShape& inputShape,
                                        const TensorShape& outputShape)
    : TestQuantization(options, inputShape, outputShape) {}

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

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestLeakyReLuActivationQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestLeakyReLuActivationQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeBatchNorm)
{
    class TestBatchNormalizationQuantization : public TestQuantization
    {
    public:
        TestBatchNormalizationQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestBatchNormalizationQuantization(const QuantizerOptions& options,
                                           const TensorShape& inputShape,
                                           const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

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

    const TensorShape shape{3U};
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
    TestBatchNormalizationQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestBatchNormalizationQuantization validatorQSymm16(options, shape, shape);
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

INetworkPtr CreateNetworkWithFullyConnectedLayer(const bool biasEnabled,
                                                 const TensorShape& inputShape,
                                                 const TensorShape& outputShape)
{
    FullyConnectedDescriptor desc;
    desc.m_BiasEnabled = biasEnabled;
    INetworkPtr network = INetwork::Create();

    const TensorInfo info(inputShape, DataType::Float32);
    const TensorInfo outputInfo(outputShape, DataType::Float32);

    std::vector<float> weightsData{-1.0f, 1.5f, 2.0f};
    ConstTensor weights(info, weightsData);

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* fullyConnected;
    Optional<ConstTensor> optionalBias;
    std::vector<float> biasData{10.0f, 20.0f, 30.0f};
    if (desc.m_BiasEnabled)
    {
        ConstTensor bias(info, biasData);
        optionalBias = Optional<ConstTensor>(bias);
    }
    fullyConnected = network->AddFullyConnectedLayer(desc, weights, optionalBias);
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    input0->GetOutputSlot(0).Connect(fullyConnected->GetInputSlot(0));
    fullyConnected->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    fullyConnected->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return network;
}

void ValidateFullyConnectedLayer(const bool biasEnabled)
{
    class TestFullyConnectedQuantization : public TestQuantization
    {
    public:
        TestFullyConnectedQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestFullyConnectedQuantization(const QuantizerOptions& options,
                                       const TensorShape& inputShape,
                                       const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

        void VisitFullyConnectedLayer(const IConnectableLayer* layer,
                                      const FullyConnectedDescriptor& desc,
                                      const ConstTensor& weights,
                                      const Optional<ConstTensor>& biases,
                                      const char* name = nullptr) override
        {
            TestQuantizationOnLayersWithBiases(layer, weights, biases);
        }
    };

    const TensorShape shape{3U};
    INetworkPtr network = CreateNetworkWithFullyConnectedLayer(biasEnabled, shape, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestFullyConnectedQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestFullyConnectedQuantization validatorQSymm16(options, shape, shape);
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
        TestConv2dQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestConv2dQuantization(const QuantizerOptions& options,
                               const TensorShape& inputShape,
                               const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

        void VisitConvolution2dLayer(const IConnectableLayer *layer,
                                     const Convolution2dDescriptor& convolution2dDescriptor,
                                     const ConstTensor& weights,
                                     const Optional<ConstTensor>& biases,
                                     const char *name = nullptr) override
        {
            TestQuantizationOnLayersWithBiases(layer, weights, biases);
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
    Optional<ConstTensor> optionalBiases;
    std::vector<float> biasesData{-1.0f, 1.5f, 2.0f};
    if (useBiases)
    {
        ConstTensor biases(info, biasesData);
        optionalBiases = Optional<ConstTensor>(biases);
    }
    conv2d = network->AddConvolution2dLayer(descriptor, weights, optionalBiases);
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    input0->GetOutputSlot(0).Connect(conv2d->GetInputSlot(0));
    conv2d->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    conv2d->GetOutputSlot(0).SetTensorInfo(info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestConv2dQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestConv2dQuantization validatorQSymm16(options, shape, shape);
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
        TestDepthwiseConv2dQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestDepthwiseConv2dQuantization(const QuantizerOptions& options,
                                        const TensorShape& inputShape,
                                        const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

        void VisitDepthwiseConvolution2dLayer(const IConnectableLayer *layer,
                                              const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
                                              const ConstTensor& weights,
                                              const Optional<ConstTensor>& biases,
                                              const char *name = nullptr) override
        {
            TestQuantizationOnLayersWithBiases(layer, weights, biases);
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
    Optional<ConstTensor> optionalBiases;
    std::vector<float> biasesData{-1.0f, 1.5f, 2.0f};
    if (useBiases)
    {
        ConstTensor biases(info, biasesData);
        optionalBiases = Optional<ConstTensor>(biases);
    }
    depthwiseConv2d = network->AddDepthwiseConvolution2dLayer(descriptor, weights, optionalBiases);
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    input0->GetOutputSlot(0).Connect(depthwiseConv2d->GetInputSlot(0));
    depthwiseConv2d->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Set TensorInfo
    input0->GetOutputSlot(0).SetTensorInfo(info);
    depthwiseConv2d->GetOutputSlot(0).SetTensorInfo(info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestDepthwiseConv2dQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestDepthwiseConv2dQuantization validatorQSymm16(options, shape, shape);
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

INetworkPtr CreateNetworkWithSoftmaxLayer(const SoftmaxDescriptor& descriptor, const TensorShape& shape)
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
        TestSoftmaxQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestSoftmaxQuantization(const QuantizerOptions& options,
                                const TensorShape& inputShape,
                                const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

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

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithSoftmaxLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSoftmaxQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestSoftmaxQuantization validatorQSymm16(options, shape, shape);
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
        TestPermuteQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(inputShape, outputShape) {}

        TestPermuteQuantization(const QuantizerOptions& options,
                                const TensorShape& inputShape,
                                const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(options, inputShape, outputShape) {}

        void VisitPermuteLayer(const IConnectableLayer* layer,
                               const PermuteDescriptor& desc,
                               const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    PermuteDescriptor desc;
    IConnectableLayer* permute = network->AddPermuteLayer(desc);

    CompleteLeakyReluNetwork(network.get(), activation, permute, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPermuteQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestPermuteQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeSpaceToBatch)
{
    class TestSpaceToBatchQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestSpaceToBatchQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(inputShape, outputShape) {}

        TestSpaceToBatchQuantization(const QuantizerOptions& options,
                                     const TensorShape& inputShape,
                                     const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(options, inputShape, outputShape) {}

        void VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                      const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                      const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    SpaceToBatchNdDescriptor descriptor;
    IConnectableLayer* spaceToBatch = network->AddSpaceToBatchNdLayer(descriptor);

    CompleteLeakyReluNetwork(network.get(), activation, spaceToBatch, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSpaceToBatchQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestSpaceToBatchQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeSpaceToDepth)
{
    class TestSpaceToDepthQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestSpaceToDepthQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
            : TestLeakyReLuActivationQuantization(inputShape, outputShape)
        {}

        TestSpaceToDepthQuantization(const QuantizerOptions& options,
                                     const TensorShape& inputShape,
                                     const TensorShape& outputShape)
            : TestLeakyReLuActivationQuantization(options, inputShape, outputShape)
        {}

        void VisitSpaceToDepthLayer(const IConnectableLayer* layer,
                                    const SpaceToDepthDescriptor&,
                                    const char* = nullptr) override
        {
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();
            TestQuantizationParams(info,
                                  { 30.0f / g_Asymm8QuantizationBase, 128 },
                                  { 15.0f / g_Symm16QuantizationBase, 0   });
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{ 1u };
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation   = CreateStartOfLeakyReluNetwork(network.get(), info);
    IConnectableLayer* spaceToDepth = network->AddSpaceToDepthLayer(SpaceToDepthDescriptor());

    CompleteLeakyReluNetwork(network.get(), activation, spaceToDepth, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSpaceToDepthQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestSpaceToDepthQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizePooling2d)
{
    class TestPooling2dQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestPooling2dQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(inputShape, outputShape) {}

        TestPooling2dQuantization(const QuantizerOptions& options,
                                  const TensorShape& inputShape,
                                  const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(options, inputShape, outputShape) {}

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
    TestPooling2dQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestPooling2dQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeConstant)
{
    class TestConstantQuantization : public TestAdditionQuantization
    {
    public:
        TestConstantQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestAdditionQuantization(inputShape, outputShape) {}

        TestConstantQuantization(const QuantizerOptions& options,
                                 const TensorShape& inputShape,
                                 const TensorShape& outputShape)
        : TestAdditionQuantization(options, inputShape, outputShape) {}

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
    const TensorShape shape{1U, 1U, 3U, 3U};
    TensorInfo tensorInfo(shape, DataType::Float32);
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
    constant->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestConstantQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestConstantQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeConcat)
{
    class TestConcatQuantization : public TestQuantization
    {
    public:
        TestConcatQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestConcatQuantization(const QuantizerOptions& options,
                               const TensorShape& inputShape,
                               const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

        void VisitInputLayer(const IConnectableLayer* layer,
                             LayerBindingId id,
                             const char* name = nullptr) override
        {}
        void VisitOutputLayer(const IConnectableLayer* layer,
                              LayerBindingId id,
                              const char* name = nullptr) override
        {}
        void VisitConcatLayer(const IConnectableLayer* layer,
                              const OriginsDescriptor& originsDescriptor,
                              const char* name = nullptr) override
        {
            TensorInfo outputInfo = layer->GetOutputSlot(0).GetTensorInfo();

            TestQuantizationParams(
                outputInfo, {60.8f / g_Asymm8QuantizationBase, 65}, {45.3f / g_Symm16QuantizationBase, 0});

            TensorInfo inputInfo0 = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
            TensorInfo inputInfo1 = layer->GetInputSlot(1).GetConnection()->GetTensorInfo();
            TensorInfo inputInfo2 = layer->GetInputSlot(2).GetConnection()->GetTensorInfo();

            TestDifferentQuantizationScale(inputInfo0, inputInfo1);
            TestDifferentQuantizationScale(inputInfo0, inputInfo2);
            TestDifferentQuantizationScale(inputInfo1, inputInfo2);
            TestDifferentQuantizationScale(inputInfo0, outputInfo);
        }
    };

    INetworkPtr network = INetwork::Create();

    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* input1 = network->AddInputLayer(1);
    IConnectableLayer* input2 = network->AddInputLayer(2);

    OriginsDescriptor descriptor(3, 1);
    IConnectableLayer* concatLayer = network->AddConcatLayer(descriptor);

    IConnectableLayer* output0 = network->AddOutputLayer(3);

    // Establish connections
    input0->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));
    input2->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(2));
    concatLayer->GetOutputSlot(0).Connect(output0->GetInputSlot(0));

    // Set TensorInfo
    const TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    input2->GetOutputSlot(0).SetTensorInfo(info);
    concatLayer->GetOutputSlot(0).SetTensorInfo(info);

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
    TestConcatQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    INetworkPtr quantizedNetworkQSymm16 = quantizerPtrQSymm16->ExportNetwork();
    TestConcatQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeReshape)
{
    class TestReshapeQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestReshapeQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(inputShape, outputShape) {}

        TestReshapeQuantization(const QuantizerOptions& options,
                                const TensorShape& inputShape,
                                const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(options, inputShape, outputShape) {}

        virtual void VisitReshapeLayer(const IConnectableLayer* layer,
                                       const ReshapeDescriptor& reshapeDescriptor,
                                       const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    ReshapeDescriptor descriptor({1, 2, 3, 4});
    IConnectableLayer* reshape = network->AddReshapeLayer(descriptor);

    CompleteLeakyReluNetwork(network.get(), activation, reshape, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestReshapeQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestReshapeQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeSplitter)
{
    class TestSplitterQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestSplitterQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(inputShape, outputShape) {}

        TestSplitterQuantization(const QuantizerOptions& options,
                                 const TensorShape& inputShape,
                                 const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(options, inputShape, outputShape) {}

        virtual void VisitSplitterLayer(const IConnectableLayer* layer,
                                        const SplitterDescriptor& desc,
                                        const char* name = nullptr)
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{3U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    ViewsDescriptor splitterDesc(2,4);
    IConnectableLayer* splitter = network->AddSplitterLayer(splitterDesc);
    CompleteLeakyReluNetwork(network.get(), activation, splitter, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSplitterQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestSplitterQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeResize)
{
    class TestResizeQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestResizeQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
                : TestLeakyReLuActivationQuantization(inputShape, outputShape)
        {}

        TestResizeQuantization(const QuantizerOptions& options,
                                       const TensorShape& inputShape,
                                       const TensorShape& outputShape)
                : TestLeakyReLuActivationQuantization(options, inputShape, outputShape)
        {}

        void VisitResizeLayer(const IConnectableLayer* layer,
                                      const ResizeDescriptor& resizeDescriptor,
                                      const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    ResizeDescriptor descriptor;
    descriptor.m_TargetHeight = 3;
    descriptor.m_TargetWidth  = 3;
    IConnectableLayer* resizeLayer = network->AddResizeLayer(descriptor);

    CompleteLeakyReluNetwork(network.get(), activation, resizeLayer, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestResizeQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestResizeQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeStridedSlice)
{
    class TestStridedSliceQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestStridedSliceQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(inputShape, outputShape) {}

        TestStridedSliceQuantization(const QuantizerOptions& options,
                                     const TensorShape& inputShape,
                                     const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(options, inputShape, outputShape) {}

        virtual void VisitStridedSliceLayer(const IConnectableLayer* layer,
                                            const StridedSliceDescriptor& desc,
                                            const char* name = nullptr)
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{3U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    StridedSliceDescriptor stridedSliceDesc;
    IConnectableLayer* stridedSlice = network->AddStridedSliceLayer(stridedSliceDesc);

    CompleteLeakyReluNetwork(network.get(), activation, stridedSlice, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestStridedSliceQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestStridedSliceQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeBatchToSpace)
{
    class TestBatchToSpaceQuantization : public TestLeakyReLuActivationQuantization
    {
    public:
        TestBatchToSpaceQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(inputShape, outputShape) {}

        TestBatchToSpaceQuantization(const QuantizerOptions& options,
                                     const TensorShape& inputShape,
                                     const TensorShape& outputShape)
        : TestLeakyReLuActivationQuantization(options, inputShape, outputShape) {}

        void VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                      const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                      const char* name = nullptr) override
        {
            CheckForwardedQuantizationSettings(layer);
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{1U};
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation = CreateStartOfLeakyReluNetwork(network.get(), info);

    // Add the layer under test
    BatchToSpaceNdDescriptor descriptor;
    IConnectableLayer* batchToSpace = network->AddBatchToSpaceNdLayer(descriptor);

    CompleteLeakyReluNetwork(network.get(), activation, batchToSpace, info);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBatchToSpaceQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestBatchToSpaceQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizePrelu)
{
    class TestPreluQuantization : public TestQuantization
    {
    public:
        TestPreluQuantization(const TensorShape& inputShape,
                              const TensorShape& alphaShape,
                              const TensorShape& outputShape)
            : TestQuantization(inputShape, outputShape)
            , m_AlphaShape(alphaShape)
        {}

        TestPreluQuantization(const QuantizerOptions& options,
                              const TensorShape& inputShape,
                              const TensorShape& alphaShape,
                              const TensorShape& outputShape)
            : TestQuantization(options, inputShape, outputShape)
            , m_AlphaShape(alphaShape)
        {}

        void VisitInputLayer(const IConnectableLayer* layer,
                             LayerBindingId id,
                             const char* name = nullptr) override
        {
            const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();

            switch (id)
            {
            case 0: // Input
                BOOST_TEST(m_InputShape == info.GetShape());
                break;
            case 1: // Alpha
                BOOST_TEST(m_AlphaShape == info.GetShape());
                break;
            default:
                throw InvalidArgumentException("Invalid layer binding id for PReLU layer");
            }

            // Based off current default [-15.0f, 15.0f]
            TestQuantizationParams(info,
                                   { 30.0f / g_Asymm8QuantizationBase, 128 }, // QASymm8
                                   { 15.0f / g_Symm16QuantizationBase, 0 });  // QSymm16
        }

        void VisitOutputLayer(const IConnectableLayer* layer,
                              LayerBindingId id,
                              const char* name = nullptr) override
        {
            const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
            BOOST_TEST(m_OutputShape == info.GetShape());
        }

        void VisitPreluLayer(const IConnectableLayer* layer,
                             const char* name = nullptr) override
        {
            const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();
            TestQuantizationParams(info,
                                   { 30.0f / g_Asymm8QuantizationBase, 128 }, // QASymm8
                                   { 15.0f / g_Symm16QuantizationBase, 0 });  // QSymm16
        }

    private:
        TensorShape m_AlphaShape;
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape inputShape{ 4, 1, 2 };
    const TensorShape alphaShape{ 5, 4, 3, 1 };
    const TensorShape outputShape{ 5, 4, 3, 2 };
    TensorInfo inputInfo(inputShape, DataType::Float32);
    TensorInfo alphaInfo(alphaShape, DataType::Float32);
    TensorInfo outputInfo(outputShape, DataType::Float32);

    // Add the input layers
    IConnectableLayer* input = network->AddInputLayer(0);
    IConnectableLayer* alpha = network->AddInputLayer(1);

    // Add the layer under test
    IConnectableLayer* prelu = network->AddPreluLayer("prelu");

    // Add the output layers
    IConnectableLayer* output = network->AddOutputLayer(0);

    // Establish connections
    input->GetOutputSlot(0).Connect(prelu->GetInputSlot(0));
    alpha->GetOutputSlot(0).Connect(prelu->GetInputSlot(1));
    prelu->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Set tensor info
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    alpha->GetOutputSlot(0).SetTensorInfo(alphaInfo);
    prelu->GetOutputSlot(0).SetTensorInfo(outputInfo);

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPreluQuantization validatorQAsymm8(inputShape, alphaShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestPreluQuantization validatorQSymm16(options, inputShape, alphaShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

void TestQuantizeTransposeConvolution2d(bool useBiases)
{
    class TestTransposeConvolution2dQuantization : public TestQuantization
    {
    public:
        TestTransposeConvolution2dQuantization(const TensorShape& inputShape, const TensorShape& outputShape) :
            TestQuantization(inputShape, outputShape)
        {}

        TestTransposeConvolution2dQuantization(const QuantizerOptions& options,
                                               const TensorShape& inputShape,
                                               const TensorShape& outputShape) :
            TestQuantization(options, inputShape, outputShape)
        {}

        void VisitTransposeConvolution2dLayer(const IConnectableLayer *layer,
                                              const TransposeConvolution2dDescriptor& descriptor,
                                              const ConstTensor& weights,
                                              const Optional<ConstTensor>& biases,
                                              const char *name = nullptr) override
        {
            TestQuantizationOnLayersWithBiases(layer, weights, biases);
        }
    };

    INetworkPtr network = INetwork::Create();

    TensorShape shape{ 3 };
    TensorInfo info(shape, DataType::Float32);

    std::initializer_list<float> floatData{ -1.0f, 1.5f, 2.0f };
    std::vector<float> weightsData(floatData);
    ConstTensor weights(info, weightsData);

    TransposeConvolution2dDescriptor descriptor;
    descriptor.m_BiasEnabled = useBiases;

    // construct network
    IConnectableLayer* input = network->AddInputLayer(0);
    Optional<ConstTensor> optionalBiases;
    std::vector<float> biasesData(floatData);
    if (useBiases)
    {
        ConstTensor biases(info, biasesData);
        optionalBiases = Optional<ConstTensor>(biases);
    }
    IConnectableLayer* transposeConv2d = network->AddTransposeConvolution2dLayer(descriptor, weights, optionalBiases);
    IConnectableLayer* output = network->AddOutputLayer(1);

    input->GetOutputSlot(0).Connect(transposeConv2d->GetInputSlot(0));
    transposeConv2d->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(info);
    transposeConv2d->GetOutputSlot(0).SetTensorInfo(info);

    // test QAsymm8 quantization
    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestTransposeConvolution2dQuantization validatorQAsymm8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    // test QSymm16 quantization
    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestTransposeConvolution2dQuantization validatorQSymm16(options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

BOOST_AUTO_TEST_CASE(QuantizeTransposeConvolution2d)
{
    TestQuantizeTransposeConvolution2d(false);
}

BOOST_AUTO_TEST_CASE(QuantizeTransposeConvolution2dWithBiases)
{
    TestQuantizeTransposeConvolution2d(true);
}

BOOST_AUTO_TEST_CASE(QuantizeStack)
{
    class TestStackQuantization : public TestQuantization
    {
    public:
        TestStackQuantization(const TensorShape& inputShape,
                              const TensorShape& outputShape)
            : TestQuantization(inputShape, outputShape) {}

        TestStackQuantization(const QuantizerOptions& options,
                              const TensorShape& inputShape,
                              const TensorShape& outputShape)
            : TestQuantization(options, inputShape, outputShape) {}

        void VisitInputLayer(const IConnectableLayer* layer,
                             LayerBindingId id,
                             const char* name = nullptr) override
        {}
        void VisitOutputLayer(const IConnectableLayer* layer,
                              LayerBindingId id,
                              const char* name = nullptr) override
        {}

        void VisitStackLayer(const IConnectableLayer* layer,
                             const StackDescriptor& descriptor,
                             const char* name = nullptr) override
        {
            TensorInfo outputInfo = layer->GetOutputSlot(0).GetTensorInfo();

            TestQuantizationParams(outputInfo,
                { 30.0f / g_Asymm8QuantizationBase, 128 },
                { 15.0f / g_Symm16QuantizationBase, 0 });
        }
    };

    INetworkPtr network = INetwork::Create();

    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* input1 = network->AddInputLayer(1);

    const TensorShape inputShape{ 3, 4, 5 };
    const TensorShape outputShape{ 3, 4, 2, 5 };

    StackDescriptor descriptor(2, 2, inputShape);
    IConnectableLayer* stackLayer = network->AddStackLayer(descriptor);

    IConnectableLayer* output = network->AddOutputLayer(0);

    input0->GetOutputSlot(0).Connect(stackLayer->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(stackLayer->GetInputSlot(1));
    stackLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestStackQuantization validatorQAsymm8(inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);

    const QuantizerOptions options(DataType::QuantisedSymm16);
    INetworkPtr quantizedNetworkQSymm16 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestStackQuantization validatorQSymm16(options, inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymm16.get(), validatorQSymm16);
}

std::vector<uint8_t> SetupQuantize(float value)
{
    armnn::TensorInfo inputInfo({ 1, 2, 2 }, armnn::DataType::Float32);
    inputInfo.SetQuantizationScale(1.0f);
    inputInfo.SetQuantizationOffset(1);
    std::vector<float> input({ value, 0.0f, 0.0f, 1.0f });
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

class TestPreserveType : public TestAdditionQuantization
{
public:
    TestPreserveType(const QuantizerOptions& options,
                     const DataType& dataType,
                     const TensorShape& inputShape,
                     const TensorShape& outputShape)
    : TestAdditionQuantization(options, inputShape, outputShape)
    , m_DataType(dataType)
    , m_VisitedQuantizeLayer(false)
    , m_VisitedDequantizeLayer(false) {}

    void VisitInputLayer(const IConnectableLayer* layer,
                         LayerBindingId id,
                         const char* name = nullptr) override
    {
        const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();
        BOOST_TEST(GetDataTypeName(info.GetDataType()) == GetDataTypeName(m_DataType));
        BOOST_TEST(m_InputShape == info.GetShape());
    }

    void VisitOutputLayer(const IConnectableLayer* layer,
                          LayerBindingId id,
                          const char* name = nullptr) override
    {
        const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
        BOOST_TEST(GetDataTypeName(info.GetDataType()) == GetDataTypeName(m_DataType));
        BOOST_TEST(m_OutputShape == info.GetShape());
    }

    void VisitQuantizeLayer(const IConnectableLayer* layer,
                            const char* name = nullptr) override
    {
        m_VisitedQuantizeLayer = true;
    }

    void VisitDequantizeLayer(const IConnectableLayer* layer,
                              const char* name = nullptr) override
    {
        m_VisitedDequantizeLayer = true;
    }

    void CheckQuantizeDequantizeLayerVisited(bool expected)
    {
        if (expected)
        {
            BOOST_CHECK(m_VisitedQuantizeLayer);
            BOOST_CHECK(m_VisitedDequantizeLayer);
        }
        else
        {
            BOOST_CHECK(!m_VisitedQuantizeLayer);
            BOOST_CHECK(!m_VisitedDequantizeLayer);
        }
    }
private:
    const DataType m_DataType;
    bool m_VisitedQuantizeLayer;
    bool m_VisitedDequantizeLayer;
};

void PreserveTypeTestImpl(const DataType& dataType)
{
    INetworkPtr network = INetwork::Create();

    // Add the layers
    IConnectableLayer* input0 = network->AddInputLayer(0);
    IConnectableLayer* input1 = network->AddInputLayer(1);
    IConnectableLayer* addition = network->AddAdditionLayer();
    IConnectableLayer* output = network->AddOutputLayer(2);

    input0->GetOutputSlot(0).Connect(addition->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(addition->GetInputSlot(1));
    addition->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    const TensorShape shape{1U, 2U, 3U};
    const TensorInfo info(shape, dataType);
    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    addition->GetOutputSlot(0).SetTensorInfo(info);

    const QuantizerOptions options(DataType::QuantisedAsymm8, true);
    INetworkPtr quantizedNetworkQAsymm8 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestPreserveType validatorQAsymm8(options, dataType, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymm8.get(), validatorQAsymm8);
    validatorQAsymm8.CheckQuantizeDequantizeLayerVisited(
        dataType == DataType::Float32 || dataType == DataType::Float16);
}

BOOST_AUTO_TEST_CASE(PreserveTypeFloat32)
{
    PreserveTypeTestImpl(DataType::Float32);
}

BOOST_AUTO_TEST_CASE(PreserveTypeQAsymm8)
{
    PreserveTypeTestImpl(DataType::QuantisedAsymm8);
}

BOOST_AUTO_TEST_SUITE_END()
} // namespace armnn
