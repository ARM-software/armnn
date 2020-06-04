//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../Graph.hpp"
#include "../Network.hpp"
#include "../NetworkQuantizerUtils.hpp"
#include "../OverrideInputRangeVisitor.hpp"
#include "../RangeTracker.hpp"
#include "../../armnnQuantizer/CommandLineProcessor.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/LayerVisitorBase.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnQuantizer/INetworkQuantizer.hpp>
#include <QuantizeHelper.hpp>

#include <boost/test/unit_test.hpp>

#include <unordered_map>

namespace armnn
{
using MinMaxRange = std::pair<float, float>;
using MinMaxRanges = std::vector<MinMaxRange>;
using MinMaxRangeMap = std::unordered_map<LayerGuid, MinMaxRanges>;

const float g_AsymmU8QuantizationBase = 255.0f;
// Coinciding with calcution which for AsymmS8 which calculates scale on an unsigned basis
const float g_AsymmS8QuantizationBase = 255.0f;
const float g_SymmS8QuantizationBase  = 127.0f;
const float g_SymmS16QuantizationBase = 32767.0f;
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
        IgnoreUnused(id, name);
        const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();
        BOOST_TEST(m_InputShape == info.GetShape());
        // Based off current default [-15.0f, 15.0f]
        TestQuantizationParams(info, {30.0f / g_AsymmU8QuantizationBase, 128},
                                     {30.0f / g_AsymmS8QuantizationBase, 0},
                                     {15.0f / g_SymmS8QuantizationBase , 0},
                                     {15.0f / g_SymmS16QuantizationBase, 0});
    }

    void VisitOutputLayer(const IConnectableLayer* layer,
                          LayerBindingId id,
                          const char* name = nullptr) override
    {
        IgnoreUnused(id, name);
        const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
        BOOST_TEST(m_OutputShape == info.GetShape());
    }

protected:
    void TestQuantizationParams(const TensorInfo& info,
                                const OffsetScalePair& qAsymmU8Params,
                                const OffsetScalePair& qAsymmS8Params,
                                const OffsetScalePair& qSymmS8Params,
                                const OffsetScalePair& qSymmS16Params)
    {
        switch (m_QuantizerOptions.m_ActivationFormat)
        {
            case DataType::QAsymmU8:
                TestQuantizationParamsImpl(
                    info, DataType::QAsymmU8, qAsymmU8Params.first, qAsymmU8Params.second);
                break;
            case DataType::QAsymmS8:
                TestQuantizationParamsImpl(
                    info, DataType::QAsymmS8, qAsymmS8Params.first, qAsymmS8Params.second);
                break;
            case DataType::QSymmS8:
                TestQuantizationParamsImpl(
                        info, DataType::QSymmS8, qSymmS8Params.first, qSymmS8Params.second);
                break;
            case DataType::QSymmS16:
                TestQuantizationParamsImpl(
                    info, DataType::QSymmS16, qSymmS16Params.first, qSymmS16Params.second);
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
                                        DataType dataType = DataType::QAsymmU8)
    {
        IgnoreUnused(dataType);
        TestQuantizationParamsImpl(info, dataType, params.first, params.second);
    }

    void TestBiasQuantizationParams(const TensorInfo& info,
                                    const OffsetScalePair& qAsymmU8Params,
                                    const OffsetScalePair& qAsymmS8Params,
                                    const OffsetScalePair& qSymmS8Params,
                                    const OffsetScalePair& qSymmS16Params,
                                    DataType dataType = DataType::QAsymmU8)
    {
        switch (m_QuantizerOptions.m_ActivationFormat)
        {
            case DataType::QAsymmU8:
                TestQuantizationParamsImpl(info, dataType, qAsymmU8Params.first, qAsymmU8Params.second);
                break;
            case DataType::QAsymmS8:
                TestQuantizationParamsImpl(info, dataType, qAsymmS8Params.first, qAsymmS8Params.second);
                break;
            case DataType::QSymmS8:
                TestQuantizationParamsImpl(info, dataType, qSymmS8Params.first, qSymmS8Params.second);
                break;
            case DataType::QSymmS16:
                TestQuantizationParamsImpl(info, dataType, qSymmS16Params.first, qSymmS16Params.second);
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
        float inputScaleQAsymmU8 = 30.0f / g_AsymmU8QuantizationBase;
        float inputScaleQAsymmS8 = 30.0f / g_AsymmS8QuantizationBase;
        float inputScaleQSymmS8  = 15.0f / g_SymmS8QuantizationBase;
        float inputScaleQSymmS16 = 15.0f / g_SymmS16QuantizationBase;
        float weightsScale       = 3.0f / g_AsymmU8QuantizationBase;

        // Based off default static range [-15.0f, 15.0f]
        TestQuantizationParams(info, {inputScaleQAsymmU8, 128},
                                     {inputScaleQAsymmS8, 0},
                                     {inputScaleQSymmS8, 0},
                                     {inputScaleQSymmS16, 0});

        TestConstantQuantizationParams(weights.GetInfo(), {weightsScale, 85});

        if (biases.has_value())
        {
            TestBiasQuantizationParams(biases.value().GetInfo(),
                                       {inputScaleQAsymmU8 * weightsScale, 0},
                                       {inputScaleQAsymmS8 * weightsScale, 0},
                                       {inputScaleQSymmS8  * weightsScale, 0},
                                       {inputScaleQSymmS16 * weightsScale, 0},
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
    auto network = PolymorphicDowncast<const Network*>(inputNetwork);
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
        IgnoreUnused(name);
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        // Based off default static range [-20.0f, 20.0f]
        TestQuantizationParams(info, {40.0f / g_AsymmU8QuantizationBase, 128},
                                     {40.0f / g_AsymmS8QuantizationBase, 0},
                                     {20.0f / g_SymmS8QuantizationBase,  0},
                                     {20.0f / g_SymmS16QuantizationBase, 0});
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

    const QuantizerOptions qAsymmU8Options(DataType::QAsymmU8);
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get(), qAsymmU8Options)->ExportNetwork();
    TestAdditionQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestAdditionQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestAdditionQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestAdditionQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
        IgnoreUnused(descriptor, name);

        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        // Based off default static range [0.0f, 15.0f]
        TestQuantizationParams(info, {15.0f / g_AsymmU8QuantizationBase, 0},
                                     {15.0f / g_AsymmS8QuantizationBase, -128},
                                     {15.0f / g_SymmS8QuantizationBase, 0},
                                     {15.0f / g_SymmS16QuantizationBase, 0});
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
        ARMNN_ASSERT_MSG(inputLayer->GetNumOutputSlots() == 1, "Input layer should have exactly 1 output slot");
        return inputLayer->GetOutputSlot(0).GetTensorInfo();
    }
    throw InvalidArgumentException("Network has no input layers");
}

BOOST_AUTO_TEST_CASE(InputOutputLayerDynamicQuant)
{
    INetworkPtr network = CreateNetworkWithInputOutputLayers();

    armnn::TensorInfo tensorInfo = GetInputTensorInfo(PolymorphicDowncast<const Network*>(network.get()));

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
    // according to QU8 Quantization Scheme
    std::unique_ptr<IQuantizationScheme> quantizationScheme = std::make_unique<QAsymmU8QuantizationScheme>();
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
            IgnoreUnused(id, name);
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

    const QuantizerOptions qAsymmU8Options(DataType::QAsymmU8);
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get(), qAsymmU8Options)->ExportNetwork();
    TestActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeLinearActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Linear;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::ReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeSoftReLuActivation)
{
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::SoftReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(descriptor, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [0.0f, 3.5f]
            TestQuantizationParams(info, {3.5f / g_AsymmU8QuantizationBase, 0},
                                         {3.5f / g_AsymmS8QuantizationBase, -128},
                                         {3.5f / g_SymmS8QuantizationBase,  0},
                                         {3.5f / g_SymmS16QuantizationBase, 0});
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::BoundedReLu;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBoundedReluActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestBoundedReluActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestBoundedReluActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestBoundedReluActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(descriptor, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-1.0f, 1.0f]
            TestQuantizationParams(
                info, {2.0f / g_AsymmU8QuantizationBase, 128},
                      {2.0f / g_AsymmS8QuantizationBase,   0},
                      {1.0f / g_SymmS8QuantizationBase ,   0},
                      {1.0f / g_SymmS16QuantizationBase,   0});
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::TanH;
    descriptor.m_A        = 3.5f;
    descriptor.m_B        = -10.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestTanHActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestTanHActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestTanHActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestTanHActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
        IgnoreUnused(descriptor, name);
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

        // Based off default static range [-5.0f, 15.0f]
        TestQuantizationParams(info, {20.0f / g_AsymmU8QuantizationBase, 64},
                                     {20.0f / g_AsymmS8QuantizationBase,-64},
                                     {15.0f / g_SymmS8QuantizationBase ,  0},
                                     {15.0f / g_SymmS16QuantizationBase,  0});
    }

protected:
    // Used by the descendant classes which test layers
    // that are forwarding their parent layer settings
    void CheckForwardedQuantizationSettings(const IConnectableLayer* layer)
    {
        TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();
        TestQuantizationParams(info, {20.0f / g_AsymmU8QuantizationBase, 64},
                                     {20.0f / g_AsymmS8QuantizationBase,-64},
                                     {15.0f / g_SymmS8QuantizationBase,   0},
                                     {15.0f / g_SymmS16QuantizationBase,  0});
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestLeakyReLuActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestLeakyReLuActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestLeakyReLuActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestLeakyReLuActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}


BOOST_AUTO_TEST_CASE(QuantizeELuActivation)
{
    class TestEluActivationQuantization : public TestQuantization
    {
    public:
        TestEluActivationQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestEluActivationQuantization(const QuantizerOptions& options,
                                       const TensorShape& inputShape,
                                       const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

        void VisitActivationLayer(const IConnectableLayer* layer,
                                  const ActivationDescriptor& descriptor,
                                  const char* name = nullptr) override
        {
            IgnoreUnused(descriptor, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-15.0f, 15.0f]
            TestQuantizationParams(
                info, {30.0f / g_AsymmU8QuantizationBase, 128},
                      {30.0f / g_AsymmS8QuantizationBase, 0},
                      {15.0f / g_SymmS8QuantizationBase,  0},
                      {15.0f / g_SymmS16QuantizationBase, 0});
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Elu;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestEluActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestEluActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestEluActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestEluActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}
BOOST_AUTO_TEST_CASE(QuantizeHardSwishActivation)
{
    class TestHardSwishActivationQuantization : public TestQuantization
    {
    public:
        TestHardSwishActivationQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
            : TestQuantization(inputShape, outputShape) {}

        TestHardSwishActivationQuantization(const QuantizerOptions& options,
                                      const TensorShape& inputShape,
                                      const TensorShape& outputShape)
            : TestQuantization(options, inputShape, outputShape) {}

        void VisitActivationLayer(const IConnectableLayer* layer,
                                  const ActivationDescriptor& descriptor,
                                  const char* name = nullptr) override
        {
            IgnoreUnused(descriptor, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-15.0f, 15.0f]
            TestQuantizationParams(
                info, {30.0f / g_AsymmU8QuantizationBase, 128},
                {30.0f / g_AsymmS8QuantizationBase, 0},
                {15.0f / g_SymmS8QuantizationBase,  0},
                {15.0f / g_SymmS16QuantizationBase, 0});
        }
    };

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::HardSwish;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithActivationLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestHardSwishActivationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestHardSwishActivationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestHardSwishActivationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestHardSwishActivationQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(desc, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [-15.0f, 15.0f]
            TestQuantizationParams(
                info, {30.0f / g_AsymmU8QuantizationBase, 128},
                      {30.0f / g_AsymmS8QuantizationBase,  0},
                      {15.0f / g_SymmS8QuantizationBase,  0},
                      {15.0f / g_SymmS16QuantizationBase, 0});

            // Test constants
            TestConstantQuantizationParams(mean.GetInfo(), {3.0f / g_AsymmU8QuantizationBase, 85});
            TestConstantQuantizationParams(variance.GetInfo(), {3.0f / g_AsymmU8QuantizationBase, 85});
            TestConstantQuantizationParams(beta.GetInfo(), {3.0f / g_AsymmU8QuantizationBase, 85});
            TestConstantQuantizationParams(gamma.GetInfo(), {3.0f / g_AsymmU8QuantizationBase, 85});
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBatchNormalizationQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestBatchNormalizationQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestBatchNormalizationQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions QQsymm16Options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), QQsymm16Options)->ExportNetwork();
    TestBatchNormalizationQuantization validatorQSymmS16(QQsymm16Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeDepthToSpace)
{
    class TestDepthToSpaceQuantization : public TestQuantization
    {
    public:
        TestDepthToSpaceQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
            : TestQuantization(inputShape, outputShape) {}

        TestDepthToSpaceQuantization(const QuantizerOptions& options,
                                     const TensorShape& inputShape,
                                     const TensorShape& outputShape)
            : TestQuantization(options, inputShape, outputShape) {}

        virtual void VisitDepthToSpaceLayer(const IConnectableLayer* layer,
                                            const DepthToSpaceDescriptor& desc,
                                            const char* name = nullptr)
        {
            IgnoreUnused(desc, name);
            const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();

            const OffsetScalePair qAsymmU8Params{ 30.0f / g_AsymmU8QuantizationBase, 128 };
            const OffsetScalePair qAsymmS8Params{ 30.0f / g_AsymmS8QuantizationBase, 0 };
            const OffsetScalePair qSymmS8Params { 15.0f / g_SymmS8QuantizationBase,  0 };
            const OffsetScalePair qSymmS16Params{ 15.0f / g_SymmS16QuantizationBase, 0 };

            TestQuantizationParams(info, qAsymmU8Params, qAsymmS8Params, qSymmS8Params, qSymmS16Params);
        }
    };

    const TensorShape inputShape { 1, 2, 2, 4 };
    const TensorShape outputShape{ 1, 4, 4, 1 };

    const TensorInfo inputInfo (inputShape,  DataType::Float32);
    const TensorInfo outputInfo(outputShape, DataType::Float32);

    INetworkPtr network = INetwork::Create();
    const DepthToSpaceDescriptor descriptor(2, armnn::DataLayout::NHWC);

    IConnectableLayer* inputLayer        = network->AddInputLayer(0);
    IConnectableLayer* depthToSpaceLayer = network->AddDepthToSpaceLayer(descriptor);
    IConnectableLayer* outputLayer       = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(depthToSpaceLayer->GetInputSlot(0));
    depthToSpaceLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    depthToSpaceLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // test QAsymmU8 quantization
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestDepthToSpaceQuantization validatorQAsymmU8(inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    // test QAsymmS8 quantization
    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestDepthToSpaceQuantization validatorQAsymmS8(qAsymmS8Options, inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    // test QSymmS8 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestDepthToSpaceQuantization validatorQSymmS8(qSymmS8Options, inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    // test QSymmS16 quantization
    const QuantizerOptions Qsymm16Options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), Qsymm16Options)->ExportNetwork();
    TestDepthToSpaceQuantization validatorQSymmS16(Qsymm16Options, inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(desc, name);
            TestQuantizationOnLayersWithBiases(layer, weights, biases);
        }
    };

    const TensorShape shape{3U};
    INetworkPtr network = CreateNetworkWithFullyConnectedLayer(biasEnabled, shape, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestFullyConnectedQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestFullyConnectedQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestFullyConnectedQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions Qsymm16Options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), Qsymm16Options)->ExportNetwork();
    TestFullyConnectedQuantization validatorQSymmS16(Qsymm16Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeFill)
{
    class TestFillQuantization : public TestQuantization
    {
    public:
        TestFillQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
        : TestQuantization(inputShape, outputShape) {}

        TestFillQuantization(const QuantizerOptions& options,
                             const TensorShape& inputShape,
                             const TensorShape& outputShape)
        : TestQuantization(options, inputShape, outputShape) {}

        virtual void VisitFillLayer(const IConnectableLayer* layer,
                                    const FillDescriptor& desc,
                                    const char* name = nullptr)
        {
            IgnoreUnused(desc, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            const OffsetScalePair qAsymmU8Params{ 30.0f / g_AsymmU8QuantizationBase, 128 };
            const OffsetScalePair qAsymmS8Params { 30.0f / g_AsymmS8QuantizationBase,  0};
            const OffsetScalePair qSymmS8Params { 15.0f / g_SymmS8QuantizationBase,  0};
            const OffsetScalePair qSymmS16Params{ 15.0f / g_SymmS16QuantizationBase, 0 };

            TestQuantizationParams(info, qAsymmU8Params, qAsymmS8Params, qSymmS8Params, qSymmS16Params);
        }
    };

    const TensorShape tensorShape{ 1U };
    const TensorInfo tensorInfo(tensorShape, DataType::Float32);

    INetworkPtr network = INetwork::Create();

    FillDescriptor descriptor;
    descriptor.m_Value = 1;

    IConnectableLayer* inputLayer = network->AddInputLayer(0);
    IConnectableLayer* fillLayer = network->AddFillLayer(descriptor);
    IConnectableLayer* outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fillLayer->GetInputSlot(0));
    fillLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    fillLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // test QAsymmU8 quantization
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestFillQuantization validatorQAsymmU8(tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    // test QAsymmS8 quantization
    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestFillQuantization validatorQAsymmS8(qAsymmS8Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    // test QSymmS8 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestFillQuantization validatorQSymmS8(qSymmS8Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    // test QuantisedSymmS16 quantization
    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestFillQuantization validatorQSymmS16(qSymmS16options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(convolution2dDescriptor, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestConv2dQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestConv2dQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestConv2dQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions Qsymm16Options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), Qsymm16Options)->ExportNetwork();
    TestConv2dQuantization validatorQSymmS16(Qsymm16Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(convolution2dDescriptor, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestDepthwiseConv2dQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestDepthwiseConv2dQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestDepthwiseConv2dQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions Qsymm16Options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), Qsymm16Options)->ExportNetwork();
    TestDepthwiseConv2dQuantization validatorQSymmS16(Qsymm16Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeDepthwiseConvolution2d)
{
    TestQuantizeDepthwiseConvolution2d(false);
}

BOOST_AUTO_TEST_CASE(QuantizeDepthwiseConvolution2dWithBiases)
{
    TestQuantizeDepthwiseConvolution2d(true);
}

BOOST_AUTO_TEST_CASE(QuantizeInstanceNormalization)
{
    class TestInstanceNormalizationQuantization : public TestQuantization
    {
    public:
        TestInstanceNormalizationQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
            : TestQuantization(inputShape, outputShape) {}

        TestInstanceNormalizationQuantization(const QuantizerOptions& options,
                                              const TensorShape& inputShape,
                                              const TensorShape& outputShape)
            : TestQuantization(options, inputShape, outputShape) {}

        virtual void VisitInstanceNormalizationLayer(const IConnectableLayer* layer,
                                                     const InstanceNormalizationDescriptor& descriptor,
                                                     const char* name = nullptr)
        {
            IgnoreUnused(descriptor, name);
            const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();

            const OffsetScalePair qAsymmU8Params{ 30.0f / g_AsymmU8QuantizationBase, 128 };
            const OffsetScalePair qAsymmS8Params { 30.0f / g_AsymmS8QuantizationBase,  0};
            const OffsetScalePair qSymmS8Params { 15.0f / g_SymmS8QuantizationBase,  0};
            const OffsetScalePair qSymmS16Params{ 15.0f / g_SymmS16QuantizationBase, 0 };

            TestQuantizationParams(info, qAsymmU8Params, qAsymmS8Params, qSymmS8Params, qSymmS16Params);
        }
    };

    const TensorShape tensorShape{ 1, 4, 4, 1 };
    const TensorInfo tensorInfo(tensorShape, DataType::Float32);

    INetworkPtr network = INetwork::Create();

    IConnectableLayer* inputLayer        = network->AddInputLayer(0);
    IConnectableLayer* instanceNormLayer = network->AddInstanceNormalizationLayer(InstanceNormalizationDescriptor());
    IConnectableLayer* outputLayer       = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(instanceNormLayer->GetInputSlot(0));
    instanceNormLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    instanceNormLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // test QAsymmU8 quantization
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestInstanceNormalizationQuantization validatorQAsymmU8(tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    //test QAsymmS8 quantization
    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestInstanceNormalizationQuantization validatorQAsymmS8(qAsymmS8Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    // test QSymmS8 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestInstanceNormalizationQuantization validatorQSymmS8(qSymmS8Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    // test QSymmS16 quantization
    const QuantizerOptions qSymmS16Options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16Options)->ExportNetwork();
    TestInstanceNormalizationQuantization validatorQSymmS16(qSymmS16Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeLogSoftmax)
{
    class TestLogSoftmaxQuantization : public TestQuantization
    {
    public:
        TestLogSoftmaxQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
            : TestQuantization(inputShape, outputShape) {}

        TestLogSoftmaxQuantization(const QuantizerOptions& options,
                                   const TensorShape& inputShape,
                                   const TensorShape& outputShape)
            : TestQuantization(options, inputShape, outputShape) {}

        void VisitLogSoftmaxLayer(const IConnectableLayer* layer,
                                  const SoftmaxDescriptor& descriptor,
                                  const char* name = nullptr) override
        {
            IgnoreUnused(descriptor, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            const OffsetScalePair qAsymmU8Params{ 30.0f / g_AsymmU8QuantizationBase, 128 };
            const OffsetScalePair qAsymmS8Params { 30.0f / g_AsymmS8QuantizationBase,  0};
            const OffsetScalePair qSymmS8Params { 15.0f / g_SymmS8QuantizationBase,  0};
            const OffsetScalePair qSymmS16Params{ 15.0f / g_SymmS16QuantizationBase, 0 };

            TestQuantizationParams(info, qAsymmU8Params, qAsymmS8Params, qSymmS8Params, qSymmS16Params);
        }
    };

    const TensorShape tensorShape{ 1U };
    const TensorInfo tensorInfo(tensorShape, DataType::Float32);

    INetworkPtr network = INetwork::Create();

    LogSoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;

    IConnectableLayer* inputLayer        = network->AddInputLayer(0);
    IConnectableLayer* logSoftmaxLayer   = network->AddLogSoftmaxLayer(descriptor);
    IConnectableLayer* outputLayer       = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(logSoftmaxLayer->GetInputSlot(0));
    logSoftmaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    logSoftmaxLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // test QAsymmU8 quantization
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestLogSoftmaxQuantization validatorQAsymmU8(tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    // test QAsymmS8 quantization
    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestLogSoftmaxQuantization validatorQAsymmS8(qAsymmS8Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    // test QSymmS8 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestLogSoftmaxQuantization validatorQSymmS8(qSymmS8Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    // test QuantisedSymmS16 quantization
    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestLogSoftmaxQuantization validatorQSymmS16(qSymmS16options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(descriptor, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off default static range [0.0f, 1.0f]
            TestQuantizationParams(info, {1.0f / g_AsymmU8QuantizationBase, 0},
                                         {1.0f / g_AsymmS8QuantizationBase, -128},
                                         {1.0f / g_SymmS8QuantizationBase,  0},
                                         {1.0f / g_SymmS16QuantizationBase, 0});
        }
    };

    SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;

    const TensorShape shape{1U};
    INetworkPtr network = CreateNetworkWithSoftmaxLayer(descriptor, shape);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSoftmaxQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestSoftmaxQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    // test QSymmS8 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestSoftmaxQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestSoftmaxQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeStandIn)
{
    const TensorShape tensorShape{ 1U };
    const TensorInfo tensorInfo(tensorShape, DataType::Float32);

    INetworkPtr network = INetwork::Create();

    StandInDescriptor descriptor;
    descriptor.m_NumInputs = 1;
    descriptor.m_NumOutputs = 1;

    IConnectableLayer* inputLayer     = network->AddInputLayer(0);
    IConnectableLayer* standInLayer   = network->AddStandInLayer(descriptor);
    IConnectableLayer* outputLayer    = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(standInLayer->GetInputSlot(0));
    standInLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    standInLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // test QAsymmU8 quantization
    BOOST_CHECK_THROW(INetworkQuantizer::Create(network.get())->ExportNetwork(),
                      armnn::UnimplementedException);

    // test QAsymmS8 quantization
    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    BOOST_CHECK_THROW(INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork(),
                      armnn::UnimplementedException);

    // test QuantisedSymmS16 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    BOOST_CHECK_THROW(INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork(),
                      armnn::UnimplementedException);

    // test QuantisedSymmS16 quantization
    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    BOOST_CHECK_THROW(INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork(),
                      armnn::UnimplementedException);
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
            IgnoreUnused(desc, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPermuteQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestPermuteQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestPermuteQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestPermuteQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(spaceToBatchNdDescriptor, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSpaceToBatchQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestSpaceToBatchQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestSpaceToBatchQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestSpaceToBatchQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
                                  { 30.0f / g_AsymmU8QuantizationBase, 128 },
                                  { 30.0f / g_AsymmS8QuantizationBase, 0   },
                                  { 15.0f / g_SymmS8QuantizationBase,  0   },
                                  { 15.0f / g_SymmS16QuantizationBase, 0   });
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape shape{ 1u };
    TensorInfo info(shape, DataType::Float32);

    IConnectableLayer* activation   = CreateStartOfLeakyReluNetwork(network.get(), info);
    IConnectableLayer* spaceToDepth = network->AddSpaceToDepthLayer(SpaceToDepthDescriptor());

    CompleteLeakyReluNetwork(network.get(), activation, spaceToDepth, info);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSpaceToDepthQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestSpaceToDepthQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestSpaceToDepthQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestSpaceToDepthQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(desc, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPooling2dQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestPooling2dQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestPooling2dQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestPooling2dQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(input, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            // Based off the range of values in the const tensor used for the test: [-2.0f, 6.0f]
            TestQuantizationParams(info, {8.0f / g_AsymmU8QuantizationBase, 64},
                                         {8.0f / g_AsymmS8QuantizationBase, -64},
                                         {6.0f / g_SymmS8QuantizationBase,  0},
                                         {6.0f / g_SymmS16QuantizationBase, 0});
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestConstantQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestConstantQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestConstantQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestConstantQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeArgMinMax)
{
    class TestArgMinMaxQuantization : public TestQuantization
    {
    public:
        TestArgMinMaxQuantization(const TensorShape& inputShape, const TensorShape& outputShape)  :
                TestQuantization(inputShape, outputShape) {}

        TestArgMinMaxQuantization(const QuantizerOptions& options,
                                  const TensorShape& inputShape,
                                  const TensorShape& outputShape) :
                TestQuantization(options, inputShape, outputShape)
        {}

        void VisitInputLayer(const IConnectableLayer* layer,
                             LayerBindingId id,
                             const char* name = nullptr) override
        {
            IgnoreUnused(layer, id, name);
        }

        void VisitOutputLayer(const IConnectableLayer* layer,
                              LayerBindingId id,
                              const char* name = nullptr) override
        {
            IgnoreUnused(layer, id, name);
        }
        void VisitArgMinMaxLayer(const IConnectableLayer* layer,
                                 const ArgMinMaxDescriptor& argMinMaxDescriptor,
                                 const char* name = nullptr) override
        {
                IgnoreUnused(argMinMaxDescriptor, name);
                TensorInfo outputInfo = layer->GetOutputSlot(0).GetTensorInfo();

                TestQuantizationParams(outputInfo,
                                       { 30.0f / g_AsymmU8QuantizationBase, 128 },
                                       { 30.0f / g_AsymmS8QuantizationBase,  0},
                                       { 15.0f / g_SymmS8QuantizationBase,  0},
                                       { 15.0f / g_SymmS16QuantizationBase, 0 });
        }
    };

    INetworkPtr network = INetwork::Create();

    const TensorShape inputShape{ 1, 1, 1, 5 };
    const TensorShape outputShape{ 1, 1, 1 };

    TensorInfo inputInfo(inputShape, DataType::Float32);
    TensorInfo outputInfo(outputShape, DataType::Float32);

    // Add the input layers
    IConnectableLayer* input = network->AddInputLayer(0);

    // Add the layer under test
    ArgMinMaxDescriptor argMinMaxDescriptor;
    argMinMaxDescriptor.m_Function = ArgMinMaxFunction::Max;
    IConnectableLayer* argMinMaxLayer = network->AddArgMinMaxLayer(argMinMaxDescriptor);

    // Add the output layers
    IConnectableLayer* output = network->AddOutputLayer(1);

    // Establish connections
    input->GetOutputSlot(0).Connect(argMinMaxLayer->GetInputSlot(0));
    argMinMaxLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Set tensor info
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    argMinMaxLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestArgMinMaxQuantization validatorQAsymmU8(inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestArgMinMaxQuantization validatorQAsymmS8(qAsymmS8Options, inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestArgMinMaxQuantization validatorQSymmS8(qSymmS8Options, inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestArgMinMaxQuantization validatorQSymmS16(qSymmS16options, inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeComparison)
{
    class TestComparisonQuantization : public TestQuantization
    {
    public:
        TestComparisonQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
            : TestQuantization(inputShape, outputShape) {}

        TestComparisonQuantization(const QuantizerOptions& options,
                                   const TensorShape& inputShape,
                                   const TensorShape& outputShape)
            : TestQuantization(options, inputShape, outputShape) {}

        void VisitComparisonLayer(const IConnectableLayer* layer,
                                  const ComparisonDescriptor& descriptor,
                                  const char* name = nullptr) override
        {
            IgnoreUnused(descriptor, name);
            TensorInfo info = layer->GetOutputSlot(0).GetTensorInfo();

            const OffsetScalePair qAsymmU8Params{ 30.0f / g_AsymmU8QuantizationBase, 128 };
            const OffsetScalePair qAsymmS8Params { 30.0f / g_AsymmS8QuantizationBase,  0};
            const OffsetScalePair qSymmS8Params { 15.0f / g_SymmS8QuantizationBase,  0};
            const OffsetScalePair qSymmS16Params{ 15.0f / g_SymmS16QuantizationBase, 0 };

            TestQuantizationParams(info, qAsymmU8Params, qAsymmS8Params, qSymmS8Params, qSymmS16Params);
        }
    };

    const TensorShape tensorShape{ 1u };
    const TensorInfo tensorInfo(tensorShape, DataType::Float32);

    INetworkPtr network = INetwork::Create();
    ComparisonDescriptor descriptor(ComparisonOperation::LessOrEqual);

    IConnectableLayer* inputLayer0     = network->AddInputLayer(0);
    IConnectableLayer* inputLayer1     = network->AddInputLayer(1);
    IConnectableLayer* comparisonLayer = network->AddComparisonLayer(descriptor);
    IConnectableLayer* outputLayer     = network->AddOutputLayer(0);

    inputLayer0->GetOutputSlot(0).Connect(comparisonLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(comparisonLayer->GetInputSlot(1));
    comparisonLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer0->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    inputLayer1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    comparisonLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // test QAsymmU8 quantization
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestComparisonQuantization validatorQAsymmU8(tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    // test QAsymmS8 quantization
    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestComparisonQuantization validatorQAsymmS8(qAsymmS8Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    // test QSymmS8 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestComparisonQuantization validatorQSymmS8(qSymmS8Options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    // test QuantisedSymmS16 quantization
    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestComparisonQuantization validatorQSymmS16(qSymmS16options, tensorShape, tensorShape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
        {
            IgnoreUnused(layer, id, name);
        }
        void VisitOutputLayer(const IConnectableLayer* layer,
                              LayerBindingId id,
                              const char* name = nullptr) override
        {
            IgnoreUnused(layer, id, name);
        }
        void VisitConcatLayer(const IConnectableLayer* layer,
                              const OriginsDescriptor& originsDescriptor,
                              const char* name = nullptr) override
        {
            IgnoreUnused(originsDescriptor, name);
            TensorInfo outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
            TestQuantizationParams(
                outputInfo, {60.8f / g_AsymmU8QuantizationBase, 65},
                            {60.8f / g_SymmS8QuantizationBase,  -63},
                            {45.3f / g_SymmS8QuantizationBase,  0},
                            {45.3f / g_SymmS16QuantizationBase, 0});

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

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkQuantizerPtr quantizerPtrQAsymmU8 =  INetworkQuantizer::Create(network.get());
    INetworkQuantizerPtr quantizerPtrQSymmS8  =  INetworkQuantizer::Create(network.get(), qSymmS8Options);
    INetworkQuantizerPtr quantizerPtrQSymmS16 =  INetworkQuantizer::Create(network.get(), qSymmS16options);
    // Override the input ranges
    float min = -15.5f;
    float max = 45.3f;

    quantizerPtrQAsymmU8->OverrideInputRange(0, (min + 2.1f), (max - 3.2f));
    quantizerPtrQAsymmU8->OverrideInputRange(1, (min + 6.7f), max);
    quantizerPtrQAsymmU8->OverrideInputRange(2, min, (max - 7.8f));

    quantizerPtrQSymmS8->OverrideInputRange(0, (min + 2.1f), (max - 3.2f));
    quantizerPtrQSymmS8->OverrideInputRange(1, (min + 6.7f), max);
    quantizerPtrQSymmS8->OverrideInputRange(2, min, (max - 7.8f));

    quantizerPtrQSymmS16->OverrideInputRange(0, (min + 2.1f), (max - 3.2f));
    quantizerPtrQSymmS16->OverrideInputRange(1, (min + 6.7f), max);
    quantizerPtrQSymmS16->OverrideInputRange(2, min, (max - 7.8f));

    INetworkPtr quantizedNetworkQAsymmU8 = quantizerPtrQAsymmU8->ExportNetwork();
    TestConcatQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    INetworkPtr quantizedNetworkQSymmS8 = quantizerPtrQSymmS8->ExportNetwork();
    TestConcatQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    INetworkPtr quantizedNetworkQSymmS16 = quantizerPtrQSymmS16->ExportNetwork();
    TestConcatQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(reshapeDescriptor, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestReshapeQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestReshapeQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestReshapeQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestReshapeQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(desc, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSplitterQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestSplitterQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestSplitterQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestSplitterQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(resizeDescriptor, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestResizeQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestResizeQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestResizeQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestResizeQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(desc, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestStridedSliceQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestStridedSliceQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestStridedSliceQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestStridedSliceQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(batchToSpaceNdDescriptor, name);
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestBatchToSpaceQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestBatchToSpaceQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestBatchToSpaceQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestBatchToSpaceQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(id, name);
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
                                   { 30.0f / g_AsymmU8QuantizationBase, 128 }, // QASymmU8
                                   { 30.0f / g_AsymmS8QuantizationBase,  0},   // QASymmS8
                                   { 15.0f / g_SymmS8QuantizationBase,  0},    // QSymmS8
                                   { 15.0f / g_SymmS16QuantizationBase, 0 });  // QSymmS16
        }

        void VisitOutputLayer(const IConnectableLayer* layer,
                              LayerBindingId id,
                              const char* name = nullptr) override
        {
            IgnoreUnused(id, name);
            const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
            BOOST_TEST(m_OutputShape == info.GetShape());
        }

        void VisitPreluLayer(const IConnectableLayer* layer,
                             const char* name = nullptr) override
        {
            IgnoreUnused(name);
            const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();
            TestQuantizationParams(info,
                                   { 30.0f / g_AsymmU8QuantizationBase, 128 }, // QASymmU8
                                   { 30.0f / g_AsymmS8QuantizationBase,  0},   // QAsymmS8
                                   { 15.0f / g_SymmS8QuantizationBase,  0},    // QSymmS8
                                   { 15.0f / g_SymmS16QuantizationBase, 0 });  // QSymmS16
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestPreluQuantization validatorQAsymmU8(inputShape, alphaShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestPreluQuantization validatorQAsymmS8(qAsymmS8Options, inputShape, alphaShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestPreluQuantization validatorQSymmS8(qSymmS8Options, inputShape, alphaShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestPreluQuantization validatorQSymmS16(qSymmS16options, inputShape, alphaShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
            IgnoreUnused(descriptor, name);
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

    // test QAsymmU8 quantization
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestTransposeConvolution2dQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    //test QAsymmS8 quantization
    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestTransposeConvolution2dQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    // test QSymmS8 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestTransposeConvolution2dQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    // test QSymmS16 quantization
    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestTransposeConvolution2dQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
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
        {
            IgnoreUnused(layer, id, name);
        }
        void VisitOutputLayer(const IConnectableLayer* layer,
                              LayerBindingId id,
                              const char* name = nullptr) override
        {
            IgnoreUnused(layer, id, name);
        }

        void VisitStackLayer(const IConnectableLayer* layer,
                             const StackDescriptor& descriptor,
                             const char* name = nullptr) override
        {
            IgnoreUnused(descriptor, name);
            TensorInfo outputInfo = layer->GetOutputSlot(0).GetTensorInfo();

            TestQuantizationParams(outputInfo,
                { 30.0f / g_AsymmU8QuantizationBase, 128 },
                { 30.0f / g_AsymmS8QuantizationBase, 0},
                { 15.0f / g_SymmS8QuantizationBase,  0},
                { 15.0f / g_SymmS16QuantizationBase, 0 });
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

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestStackQuantization validatorQAsymmU8(inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestStackQuantization validatorQAsymmS8(qAsymmS8Options, inputShape, inputShape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestStackQuantization validatorQSymmS8(qSymmS8Options, inputShape, inputShape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestStackQuantization validatorQSymmS16(qSymmS16options, inputShape, outputShape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

BOOST_AUTO_TEST_CASE(QuantizeSlice)
{
    class TestSliceQuantization : public TestQuantization
    {
    public:
        TestSliceQuantization(const TensorShape& inputShape, const TensorShape& outputShape)
            : TestQuantization(inputShape, outputShape)
        {}

        TestSliceQuantization(const QuantizerOptions& options,
                              const TensorShape& inputShape,
                              const TensorShape& outputShape)
            : TestQuantization(options, inputShape, outputShape)
        {}

        virtual void VisitSliceLayer(const IConnectableLayer* layer,
                                     const SliceDescriptor& desc,
                                     const char* name = nullptr)
        {
            IgnoreUnused(desc, name);
            const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();

            const OffsetScalePair qAsymmU8Params{ 30.0f / g_AsymmU8QuantizationBase, 128 };
            const OffsetScalePair qAsymmS8Params{ 30.0f / g_AsymmS8QuantizationBase, 0 };
            const OffsetScalePair qSymmS8Params { 15.0f / g_SymmS8QuantizationBase,  0 };
            const OffsetScalePair qSymmS16Params{ 15.0f / g_SymmS16QuantizationBase, 0 };

            TestQuantizationParams(info, qAsymmU8Params, qAsymmS8Params, qSymmS8Params, qSymmS16Params);
        }
    };

    TensorShape shape{ 3 };
    TensorInfo info(shape, DataType::Float32);

    INetworkPtr network = INetwork::Create();

    IConnectableLayer* inputLayer  = network->AddInputLayer(0);
    IConnectableLayer* sliceLayer  = network->AddSliceLayer(SliceDescriptor());
    IConnectableLayer* outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(sliceLayer->GetInputSlot(0));
    sliceLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(info);
    sliceLayer->GetOutputSlot(0).SetTensorInfo(info);

    // test QAsymmU8 quantization
    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get())->ExportNetwork();
    TestSliceQuantization validatorQAsymmU8(shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);

    // test QASymmS8 quantization
    const QuantizerOptions qAsymmS8Options(DataType::QAsymmS8);
    INetworkPtr quantizedNetworkQAsymmS8 = INetworkQuantizer::Create(network.get(), qAsymmS8Options)->ExportNetwork();
    TestSliceQuantization validatorQAsymmS8(qAsymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmS8.get(), validatorQAsymmS8);

    // test QSymmS8 quantization
    const QuantizerOptions qSymmS8Options(DataType::QSymmS8);
    INetworkPtr quantizedNetworkQSymmS8 = INetworkQuantizer::Create(network.get(), qSymmS8Options)->ExportNetwork();
    TestSliceQuantization validatorQSymmS8(qSymmS8Options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS8.get(), validatorQSymmS8);

    // test QSymmS16 quantization
    const QuantizerOptions qSymmS16options(DataType::QSymmS16);
    INetworkPtr quantizedNetworkQSymmS16 = INetworkQuantizer::Create(network.get(), qSymmS16options)->ExportNetwork();
    TestSliceQuantization validatorQSymmS16(qSymmS16options, shape, shape);
    VisitLayersTopologically(quantizedNetworkQSymmS16.get(), validatorQSymmS16);
}

std::vector<uint8_t> SetupQuantize(float value)
{
    armnn::TensorInfo inputInfo({ 1, 2, 2 }, armnn::DataType::Float32);
    inputInfo.SetQuantizationScale(1.0f);
    inputInfo.SetQuantizationOffset(1);
    std::vector<float> input({ value, 0.0f, 0.0f, 1.0f });
    const std::vector<float> &inputRef = input;

    auto output = armnnUtils::QuantizedVector<uint8_t>(inputRef,
                                                       inputInfo.GetQuantizationScale(),
                                                       inputInfo.GetQuantizationOffset());

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
        IgnoreUnused(id, name);
        const TensorInfo& info = layer->GetOutputSlot(0).GetTensorInfo();
        BOOST_TEST(GetDataTypeName(info.GetDataType()) == GetDataTypeName(m_DataType));
        BOOST_TEST(m_InputShape == info.GetShape());
    }

    void VisitOutputLayer(const IConnectableLayer* layer,
                          LayerBindingId id,
                          const char* name = nullptr) override
    {
        IgnoreUnused(id, name);
        const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
        BOOST_TEST(GetDataTypeName(info.GetDataType()) == GetDataTypeName(m_DataType));
        BOOST_TEST(m_OutputShape == info.GetShape());
    }

    void VisitQuantizeLayer(const IConnectableLayer* layer,
                            const char* name = nullptr) override
    {
        IgnoreUnused(layer, name);
        m_VisitedQuantizeLayer = true;
    }

    void VisitDequantizeLayer(const IConnectableLayer* layer,
                              const char* name = nullptr) override
    {
        IgnoreUnused(layer, name);
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

    QuantizerOptions options = dataType == DataType::Float32 ?
            QuantizerOptions(DataType::QAsymmU8, true) : QuantizerOptions(dataType, true);

    INetworkPtr quantizedNetworkQAsymmU8 = INetworkQuantizer::Create(network.get(), options)->ExportNetwork();
    TestPreserveType validatorQAsymmU8(options, dataType, shape, shape);
    VisitLayersTopologically(quantizedNetworkQAsymmU8.get(), validatorQAsymmU8);
    validatorQAsymmU8.CheckQuantizeDequantizeLayerVisited(
    dataType == DataType::Float32 || dataType == DataType::Float16);
}

BOOST_AUTO_TEST_CASE(PreserveTypeFloat32)
{
    PreserveTypeTestImpl(DataType::Float32);
}

BOOST_AUTO_TEST_CASE(PreserveTypeQAsymmU8)
{
    PreserveTypeTestImpl(DataType::QAsymmU8);
}

BOOST_AUTO_TEST_CASE(PreserveTypeQsymm8)
{
    PreserveTypeTestImpl(DataType::QSymmS8);
}

BOOST_AUTO_TEST_CASE(PreserveTypeQsymm16)
{
    PreserveTypeTestImpl(DataType::QSymmS16);
}

BOOST_AUTO_TEST_CASE(TestConnectionPreservationAfterDynamicQuant)
{
    class TestConnectionPreservation : public LayerVisitorBase<VisitorNoThrowPolicy>
    {
    public:
        TestConnectionPreservation(const Graph& graph)
            : LayerVisitorBase<VisitorNoThrowPolicy>()
            , m_Graph(graph)
        {}

        void VisitAdditionLayer(const IConnectableLayer* layer, const char*) override
        {
            CheckLayerName(layer->GetInputSlot(0).GetConnection()->GetOwningLayerGuid(), "reLU1");
            CheckLayerName(layer->GetInputSlot(1).GetConnection()->GetOwningLayerGuid(), "reLU2");
        }

        void CheckLayerName(LayerGuid guid, std::string expectedName)
        {
            bool guidFound = false;
            for (Layer* layer : m_Graph)
            {
                if (layer->GetGuid() == guid)
                {
                    BOOST_CHECK_EQUAL(layer->GetName(), expectedName.c_str());
                    guidFound = true;
                    break;
                }
            }
            if (!guidFound)
            {
                BOOST_FAIL("No layer matching the GUID was found");
            }
        }

    private:
        Graph m_Graph;
    };

    INetworkPtr network = INetwork::Create();

    IConnectableLayer* inputLayer =  network->AddInputLayer(0,"inputLayer1");
    armnn::ActivationDescriptor ReLUDesc;
    ReLUDesc.m_Function = ActivationFunction::ReLu;

    IConnectableLayer* reLULayer1 = network->AddActivationLayer(ReLUDesc, "reLU1");
    IConnectableLayer* reLULayer2 = network->AddActivationLayer(ReLUDesc, "reLU2");
    IConnectableLayer* addLayer1 = network->AddAdditionLayer("addLayer1");
    IConnectableLayer* outputLayer = network->AddOutputLayer(0,"outPutLayer1");

    inputLayer->GetOutputSlot(0).Connect(reLULayer1->GetInputSlot(0));
    reLULayer1->GetOutputSlot(0).Connect(reLULayer2->GetInputSlot(0));
    reLULayer1->GetOutputSlot(0).Connect(addLayer1->GetInputSlot(0));
    reLULayer2->GetOutputSlot(0).Connect(addLayer1->GetInputSlot(1));
    addLayer1->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(TensorInfo(TensorShape({1, 2, 2, 1}), DataType::Float32));
    reLULayer1->GetOutputSlot(0).SetTensorInfo(TensorInfo(TensorShape({1, 2, 2, 1}), DataType::Float32));
    reLULayer2->GetOutputSlot(0).SetTensorInfo(TensorInfo(TensorShape({1, 2, 2, 1}), DataType::Float32));
    addLayer1->GetOutputSlot(0).SetTensorInfo(TensorInfo(TensorShape({1, 2, 2, 1}), DataType::Float32));

    TestConnectionPreservation visitor1(PolymorphicDowncast<const Network*>(network.get())->GetGraph());
    VisitLayersTopologically(network.get(), visitor1);

    armnn::INetworkQuantizerPtr quantizer = armnn::INetworkQuantizer::Create(network.get());

    armnn::TensorInfo tensorInfo = GetInputTensorInfo(PolymorphicDowncast<const Network*>(network.get()));

    std::vector<float> inputData({0, 2, 0, 4});
    armnn::ConstTensor inputTensor(tensorInfo, inputData.data());

    InputTensors inputTensors;
    inputTensors.push_back(std::make_pair(0, inputTensor));
    quantizer->Refine(inputTensors);

    INetworkPtr quantNetwork = quantizer->ExportNetwork();

    TestConnectionPreservation visitor2(PolymorphicDowncast<const Network*>(quantNetwork.get())->GetGraph());
    VisitLayersTopologically(quantNetwork.get(), visitor2);
}

BOOST_AUTO_TEST_SUITE_END()
} // namespace armnn
