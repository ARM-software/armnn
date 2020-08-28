//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include <Graph.hpp>
#include <Network.hpp>

#include <reference/RefWorkloadFactory.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(OptimizedNetwork)

BOOST_AUTO_TEST_CASE(SerializeToDot)
{
    armnn::Network net;

    //Defines layers.
    auto input = net.AddInputLayer(0);
    auto add = net.AddAdditionLayer();
    auto output = net.AddOutputLayer(0);

    // Connects layers.
    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorShape shape({4});
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec());

    std::ostringstream ss;
    optimizedNet->SerializeToDot(ss);

    auto inputId = input->GetGuid();
    auto addId = add->GetGuid();
    auto outputId = output->GetGuid();

    std::stringstream expected;
    expected <<
        "digraph Optimized {\n"
        "    node [shape=\"record\"];\n"
        "    edge [fontsize=8 fontcolor=\"blue\" fontname=\"arial-bold\"];\n"
        "    " << inputId << " [label=\"{Input|Guid : " << inputId << "\\lLayerType : Input\\l"
                             "BackendID : CpuRef\\l}\"];\n"
        "    " << addId << " [label=\"{Addition|Guid : " << addId << "\\lLayerType : Addition\\l"
                           "BackendID : CpuRef\\l}\"];\n"
        "    " << outputId << " [label=\"{Output|Guid : " << outputId << "\\lLayerType : Output\\l"
                              "BackendID : CpuRef\\l}\"];\n"
        "    " << inputId << " -> " << addId << " [label=< [4] >];\n"
        "    " << inputId << " -> " << addId << " [label=< [4] >];\n"
        "    " << addId << " -> " << outputId << " [label=< [4] >];\n"
        "}\n";

    BOOST_TEST(ss.str() == expected.str());
}

BOOST_AUTO_TEST_CASE(OptimizeValidateDeviceNonSupportLayerNoFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc and isn't allowed to fall back, so Optimize will return null.
    armnn::NormalizationDescriptor descriptor;
    armnn::IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    std::vector<std::string> errMessages;

    try
    {
        Optimize(*net, backends, runtime->GetDeviceSpec(), armnn::OptimizerOptions(), errMessages);
        BOOST_FAIL("Should have thrown an exception.");
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        // Different exceptions are thrown on different backends
    }
    BOOST_CHECK(errMessages.size() > 0);
}

BOOST_AUTO_TEST_CASE(OptimizeValidateDeviceNonSupportLayerWithFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc but it allows to fallback to CpuRef.
    armnn::NormalizationDescriptor descriptor;
    armnn::IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc, armnn::Compute::CpuRef };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_REQUIRE(optNet);

    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        // If NEON is enabled, Input and Output layers are supported by CpuAcc,
        // the other layers are supported by CpuRef.
        // If NEON is not enabled, all layers are supported by CpuRef.
#if defined(ARMCOMPUTENEON_ENABLED)
        if (layer->GetType() == armnn::LayerType::Input || layer->GetType() == armnn::LayerType::Output)
        {
            BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuAcc);
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        }
#else
        BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
#endif
    }
}

BOOST_AUTO_TEST_CASE(OptimizeValidateWorkloadsUndefinedComputeDevice)
{
    const armnn::TensorInfo desc({3, 5}, armnn::DataType::Float32);

    armnn::Network  net;

    armnn::NormalizationDescriptor nmDesc;
    armnn::ActivationDescriptor acDesc;

    //    in
    //     |
    //    nm
    //   /  |
    //  ac  |
    //   \  |
    //    ml
    //     |
    //    sm
    //     |
    //    ot
    armnn::IConnectableLayer* layer = net.AddInputLayer(0, "in");
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    armnn::IConnectableLayer* const normLayer = net.AddNormalizationLayer(nmDesc, "nm");

    layer->GetOutputSlot(0).Connect(normLayer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).SetTensorInfo(desc);

    layer = net.AddActivationLayer(acDesc, "ac");

    normLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    armnn::IConnectableLayer* prevLayer = layer;
    layer = net.AddMultiplicationLayer("ml");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    prevLayer = layer;
    armnn::SoftmaxDescriptor softmaxDescriptor;
    layer = net.AddSoftmaxLayer(softmaxDescriptor, "sm");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    prevLayer = layer;
    layer = net.AddOutputLayer(0, "ot");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::Undefined };
    std::vector<std::string> errMessages;

    try
    {
        Optimize(net, backends, runtime->GetDeviceSpec(), armnn::OptimizerOptions(), errMessages);
        BOOST_FAIL("Should have thrown an exception.");
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        // Different exceptions are thrown on different backends
    }
    BOOST_CHECK(errMessages.size() > 0);
}

BOOST_AUTO_TEST_CASE(OptimizeValidateWorkloadsUndefinedComputeDeviceWithFallback)
{
    const armnn::TensorInfo desc({3, 5}, armnn::DataType::Float32);

    armnn::Network  net;

    armnn::NormalizationDescriptor nmDesc;
    armnn::ActivationDescriptor acDesc;

    //    in
    //     |
    //    nm
    //   /  |
    //  ac  |
    //   \  |
    //    ml
    //     |
    //    sm
    //     |
    //    ot
    armnn::IConnectableLayer* layer = net.AddInputLayer(0, "in");
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    armnn::IConnectableLayer* const normLayer = net.AddNormalizationLayer(nmDesc, "nm");

    layer->GetOutputSlot(0).Connect(normLayer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).SetTensorInfo(desc);

    layer = net.AddActivationLayer(acDesc, "ac");

    normLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    armnn::IConnectableLayer* prevLayer = layer;
    layer = net.AddMultiplicationLayer("ml");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    prevLayer = layer;
    armnn::SoftmaxDescriptor softmaxDescriptor;
    layer = net.AddSoftmaxLayer(softmaxDescriptor, "sm");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    prevLayer = layer;
    layer = net.AddOutputLayer(0, "ot");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::Undefined, armnn::Compute::CpuRef };

    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);

    // validate workloads
    armnn::RefWorkloadFactory fact;
    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        BOOST_CHECK_NO_THROW(
            layer->CreateWorkload(fact));
    }
}

BOOST_AUTO_TEST_CASE(OptimizeValidateWorkloadsDuplicateComputeDeviceWithFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc but it allows to fallback to CpuRef.
    armnn::NormalizationDescriptor descriptor;
    armnn::IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                             armnn::Compute::GpuAcc,
                                             armnn::Compute::CpuRef };

    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_REQUIRE(optNet);

    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        // If NEON is enabled, Input and Output layers are supported by CpuAcc,
        // the other layers are supported by CpuRef.
        // If only CL is enabled, Input and Output layers are supported by GpuAcc,
        // the other layers are supported by CpuRef.
        // If neither NEON, nor CL is enabled, all layers are supported by CpuRef.
#if defined(ARMCOMPUTENEON_ENABLED)
        if (layer->GetType() == armnn::LayerType::Input || layer->GetType() == armnn::LayerType::Output)
        {
            BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuAcc);
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        }
#elif defined(ARMCOMPUTECL_ENABLED)
        if (layer->GetType() == armnn::LayerType::Input || layer->GetType() == armnn::LayerType::Output)
        {
            BOOST_CHECK(layer->GetBackendId() == armnn::Compute::GpuAcc);
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        }
#else
        BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
#endif
    }
}

BOOST_AUTO_TEST_SUITE_END()
