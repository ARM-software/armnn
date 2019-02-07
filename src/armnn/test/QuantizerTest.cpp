//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/INetworkQuantizer.hpp>
#include <armnn/Types.hpp>

#include "../LayerVisitorBase.hpp"
#include "../Network.hpp"
#include "../Graph.hpp"

#include <boost/test/unit_test.hpp>

namespace armnn
{
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
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f/255.0f, 0.000001f );
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
    for (auto layer : graph)
    {
        layer->Accept(visitor);
    }
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
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 40.0f/255.0f, 0.000001f );
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
        BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 15.0f/255.0f, 0.000001f );
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
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 3.5f/255.0f, 0.000001f );
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
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 2.0f/255.0f, 0.000001f );
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
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 20.0f/255.0f, 0.000001f );
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
            BOOST_CHECK_CLOSE(info.GetQuantizationScale(), 30.0f/255.0f, 0.000001f );

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

BOOST_AUTO_TEST_SUITE_END()
} //namespace armnn
