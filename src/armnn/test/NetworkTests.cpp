//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>

#include "armnn/ArmNN.hpp"
#include "Network.hpp"
#include "Graph.hpp"
#include "backends/RefWorkloadFactory.hpp"
#include "backends/ClWorkloadFactory.hpp"
#include "backends/NeonWorkloadFactory.hpp"

#include "GraphUtils.hpp"

namespace
{

bool AreAllLayerInputSlotsConnected(const armnn::IConnectableLayer& layer)
{
    bool allConnected = true;
    for (unsigned int i = 0; i < layer.GetNumInputSlots(); ++i)
    {
        const bool inputConnected = layer.GetInputSlot(i).GetConnection() != nullptr;
        allConnected &= inputConnected;
    }
    return allConnected;
}

}

BOOST_AUTO_TEST_SUITE(Network)

BOOST_AUTO_TEST_CASE(LayerGuids)
{
    armnn::Network net;
    armnn::LayerGuid inputId = net.AddInputLayer(0)->GetGuid();
    armnn::LayerGuid addId = net.AddAdditionLayer()->GetGuid();
    armnn::LayerGuid outputId = net.AddOutputLayer(0)->GetGuid();

    BOOST_TEST(inputId != addId);
    BOOST_TEST(addId != outputId);
    BOOST_TEST(inputId != outputId);
}

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

    std::vector<armnn::Compute> backends = {armnn::Compute::CpuRef};
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
        "    " << inputId << " [label=\"{Input}\"];\n"
        "    " << addId << " [label=\"{Addition}\"];\n"
        "    " << outputId << " [label=\"{Output}\"];\n"
        "    " << inputId << " -> " << addId << " [label=< [4] >];\n"
        "    " << inputId << " -> " << addId << " [label=< [4] >];\n"
        "    " << addId << " -> " << outputId << " [label=< [4] >];\n"
        "}\n";

    BOOST_TEST(ss.str() == expected.str());
}

BOOST_AUTO_TEST_CASE(NetworkBasic)
{
    armnn::Network net;
    BOOST_TEST(net.PrintGraph() == armnn::Status::Success);
}

BOOST_AUTO_TEST_CASE(LayerNamesAreOptionalForINetwork)
{
    armnn::Network net;
    armnn::INetwork& inet = net;
    inet.AddInputLayer(0);
    inet.AddAdditionLayer();
    inet.AddActivationLayer(armnn::ActivationDescriptor());
    inet.AddOutputLayer(0);
}

BOOST_AUTO_TEST_CASE(LayerNamesAreOptionalForNetwork)
{
    armnn::Network net;
    net.AddInputLayer(0);
    net.AddAdditionLayer();
    net.AddActivationLayer(armnn::ActivationDescriptor());
    net.AddOutputLayer(0);
}

BOOST_AUTO_TEST_CASE(NetworkModification)
{
    armnn::Network net;

    armnn::IConnectableLayer* const inputLayer = net.AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    unsigned int dims[] = { 10,1,1,1 };
    std::vector<float> convWeightsData(10);
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::Float32), convWeightsData);

    armnn::Convolution2dDescriptor convDesc2d;
    armnn::IConnectableLayer* const convLayer = net.AddConvolution2dLayer(convDesc2d, weights, "conv layer");
    BOOST_TEST(convLayer);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));

    armnn::FullyConnectedDescriptor fullyConnectedDesc;
    armnn::IConnectableLayer* const fullyConnectedLayer = net.AddFullyConnectedLayer(fullyConnectedDesc,
                                                                                     weights,
                                                                                     "fully connected");
    BOOST_TEST(fullyConnectedLayer);

    convLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));

    armnn::Pooling2dDescriptor pooling2dDesc;
    armnn::IConnectableLayer* const poolingLayer = net.AddPooling2dLayer(pooling2dDesc, "pooling2d");
    BOOST_TEST(poolingLayer);

    fullyConnectedLayer->GetOutputSlot(0).Connect(poolingLayer->GetInputSlot(0));

    armnn::ActivationDescriptor activationDesc;
    armnn::IConnectableLayer* const activationLayer = net.AddActivationLayer(activationDesc, "activation");
    BOOST_TEST(activationLayer);

    poolingLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));

    armnn::NormalizationDescriptor normalizationDesc;
    armnn::IConnectableLayer* const normalizationLayer = net.AddNormalizationLayer(normalizationDesc, "normalization");
    BOOST_TEST(normalizationLayer);

    activationLayer->GetOutputSlot(0).Connect(normalizationLayer->GetInputSlot(0));

    armnn::SoftmaxDescriptor softmaxDesc;
    armnn::IConnectableLayer* const softmaxLayer = net.AddSoftmaxLayer(softmaxDesc, "softmax");
    BOOST_TEST(softmaxLayer);

    normalizationLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));

    armnn::BatchNormalizationDescriptor batchNormDesc;

    armnn::TensorInfo tensorInfo({ 1 }, armnn::DataType::Float32);
    std::vector<float> data(tensorInfo.GetNumBytes() / sizeof(float));
    armnn::ConstTensor invalidTensor(tensorInfo, data);

    armnn::IConnectableLayer* const batchNormalizationLayer = net.AddBatchNormalizationLayer(batchNormDesc,
        invalidTensor,
        invalidTensor,
        invalidTensor,
        invalidTensor,
        "batch norm");
    BOOST_TEST(batchNormalizationLayer);

    softmaxLayer->GetOutputSlot(0).Connect(batchNormalizationLayer->GetInputSlot(0));

    armnn::IConnectableLayer* const additionLayer = net.AddAdditionLayer("addition");
    BOOST_TEST(additionLayer);

    batchNormalizationLayer->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
    batchNormalizationLayer->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const multiplicationLayer = net.AddMultiplicationLayer("multiplication");
    BOOST_TEST(multiplicationLayer);

    additionLayer->GetOutputSlot(0).Connect(multiplicationLayer->GetInputSlot(0));
    additionLayer->GetOutputSlot(0).Connect(multiplicationLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = net.AddOutputLayer(0, "output layer");
    BOOST_TEST(outputLayer);

    multiplicationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    //Tests that all layers are present in the graph.
    BOOST_TEST(net.GetGraph().GetNumLayers() == 11);

    //Tests that the vertices exist and have correct names.
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "input layer"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "conv layer"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "fully connected"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "pooling2d"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "activation"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "normalization"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "softmax"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "batch norm"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "addition"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "multiplication"));
    BOOST_TEST(GraphHasNamedLayer(net.GetGraph(), "output layer"));

    auto checkOneOutputToOneInputConnection = []
        (const armnn::IConnectableLayer* const srcLayer,
         const armnn::IConnectableLayer* const tgtLayer,
         int expectedSrcNumInputs = 1,
         int expectedDstNumOutputs = 1)
        {
            BOOST_TEST(srcLayer->GetNumInputSlots() == expectedSrcNumInputs);
            BOOST_TEST(srcLayer->GetNumOutputSlots() == 1);
            BOOST_TEST(tgtLayer->GetNumInputSlots() == 1);
            BOOST_TEST(tgtLayer->GetNumOutputSlots() == expectedDstNumOutputs);

            BOOST_TEST(srcLayer->GetOutputSlot(0).GetNumConnections() == 1);
            BOOST_TEST(srcLayer->GetOutputSlot(0).GetConnection(0) == &tgtLayer->GetInputSlot(0));
            BOOST_TEST(&srcLayer->GetOutputSlot(0) == tgtLayer->GetInputSlot(0).GetConnection());
        };
    auto checkOneOutputToTwoInputsConnections = []
        (const armnn::IConnectableLayer* const srcLayer,
         const armnn::IConnectableLayer* const tgtLayer,
         int expectedSrcNumInputs,
         int expectedDstNumOutputs = 1)
        {
            BOOST_TEST(srcLayer->GetNumInputSlots() == expectedSrcNumInputs);
            BOOST_TEST(srcLayer->GetNumOutputSlots() == 1);
            BOOST_TEST(tgtLayer->GetNumInputSlots() == 2);
            BOOST_TEST(tgtLayer->GetNumOutputSlots() == expectedDstNumOutputs);

            BOOST_TEST(srcLayer->GetOutputSlot(0).GetNumConnections() == 2);
            for (unsigned int i = 0; i < srcLayer->GetOutputSlot(0).GetNumConnections(); ++i)
            {
                BOOST_TEST(srcLayer->GetOutputSlot(0).GetConnection(i) == &tgtLayer->GetInputSlot(i));
                BOOST_TEST(&srcLayer->GetOutputSlot(0) == tgtLayer->GetInputSlot(i).GetConnection());
            }
        };

    BOOST_TEST(AreAllLayerInputSlotsConnected(*convLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*fullyConnectedLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*poolingLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*activationLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*normalizationLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*softmaxLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*batchNormalizationLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*additionLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*multiplicationLayer));
    BOOST_TEST(AreAllLayerInputSlotsConnected(*outputLayer));

    // Checks connectivity.
    checkOneOutputToOneInputConnection(inputLayer, convLayer, 0);
    checkOneOutputToOneInputConnection(convLayer, fullyConnectedLayer);
    checkOneOutputToOneInputConnection(fullyConnectedLayer, poolingLayer);
    checkOneOutputToOneInputConnection(poolingLayer, activationLayer);
    checkOneOutputToOneInputConnection(activationLayer, normalizationLayer);
    checkOneOutputToOneInputConnection(normalizationLayer, softmaxLayer);
    checkOneOutputToOneInputConnection(softmaxLayer, batchNormalizationLayer);
    checkOneOutputToTwoInputsConnections(batchNormalizationLayer, additionLayer, 1);
    checkOneOutputToTwoInputsConnections(additionLayer, multiplicationLayer, 2);
    checkOneOutputToOneInputConnection(multiplicationLayer, outputLayer, 2, 0);
}

BOOST_AUTO_TEST_CASE(NetworkModification_SplitterMerger)
{
    armnn::Network net;

    // Adds an input layer and an input tensor descriptor.
    armnn::IConnectableLayer* inputLayer = net.AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    // Adds a splitter layer.
    armnn::ViewsDescriptor splitterDesc(2,4);

    armnn::IConnectableLayer* splitterLayer = net.AddSplitterLayer(splitterDesc, "splitter layer");
    BOOST_TEST(splitterLayer);

    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));

    // Adds a softmax layer 1.
    armnn::SoftmaxDescriptor softmaxDescriptor;
    armnn::IConnectableLayer* softmaxLayer1 = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_1");
    BOOST_TEST(softmaxLayer1);

    splitterLayer->GetOutputSlot(0).Connect(softmaxLayer1->GetInputSlot(0));

    // Adds a softmax layer 2.
    armnn::IConnectableLayer* softmaxLayer2 = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_2");
    BOOST_TEST(softmaxLayer2);

    splitterLayer->GetOutputSlot(1).Connect(softmaxLayer2->GetInputSlot(0));

    // Adds a merger layer.
    armnn::OriginsDescriptor mergerDesc(2, 4);

    armnn::IConnectableLayer* mergerLayer = net.AddMergerLayer(mergerDesc, "merger layer");
    BOOST_TEST(mergerLayer);

    softmaxLayer1->GetOutputSlot(0).Connect(mergerLayer->GetInputSlot(0));
    softmaxLayer2->GetOutputSlot(0).Connect(mergerLayer->GetInputSlot(1));

    // Adds an output layer.
    armnn::IConnectableLayer* outputLayer = net.AddOutputLayer(0, "output layer");
    BOOST_TEST(outputLayer);

    mergerLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    BOOST_TEST(splitterLayer->GetNumOutputSlots() == 2);
    BOOST_TEST(splitterLayer->GetOutputSlot(0).GetConnection(0) == &softmaxLayer1->GetInputSlot(0));
    BOOST_TEST(&splitterLayer->GetOutputSlot(0) == softmaxLayer1->GetInputSlot(0).GetConnection());
    BOOST_TEST(splitterLayer->GetOutputSlot(1).GetConnection(0) == &softmaxLayer2->GetInputSlot(0));
    BOOST_TEST(&splitterLayer->GetOutputSlot(1) == softmaxLayer2->GetInputSlot(0).GetConnection());

    BOOST_TEST(mergerLayer->GetNumInputSlots() == 2);
    BOOST_TEST(softmaxLayer1->GetOutputSlot(0).GetConnection(0) == &mergerLayer->GetInputSlot(0));
    BOOST_TEST(&softmaxLayer1->GetOutputSlot(0) == mergerLayer->GetInputSlot(0).GetConnection());
    BOOST_TEST(softmaxLayer2->GetOutputSlot(0).GetConnection(0) == &mergerLayer->GetInputSlot(1));
    BOOST_TEST(&softmaxLayer2->GetOutputSlot(0) == mergerLayer->GetInputSlot(1).GetConnection());
}

BOOST_AUTO_TEST_CASE(NetworkModification_SplitterAddition)
{
    armnn::Network net;

    // Adds an input layer and an input tensor descriptor.
    armnn::IConnectableLayer* layer = net.AddInputLayer(0, "input layer");
    BOOST_TEST(layer);

    // Adds a splitter layer.
    armnn::ViewsDescriptor splitterDesc(2,4);

    armnn::IConnectableLayer* const splitterLayer = net.AddSplitterLayer(splitterDesc, "splitter layer");
    BOOST_TEST(splitterLayer);

    layer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));

    // Adds a softmax layer 1.
    armnn::SoftmaxDescriptor softmaxDescriptor;
    armnn::IConnectableLayer* const softmax1Layer = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_1");
    BOOST_TEST(softmax1Layer);

    splitterLayer->GetOutputSlot(0).Connect(softmax1Layer->GetInputSlot(0));

    // Adds a softmax layer 2.
    armnn::IConnectableLayer* const softmax2Layer = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_2");
    BOOST_TEST(softmax2Layer);

    splitterLayer->GetOutputSlot(1).Connect(softmax2Layer->GetInputSlot(0));

    // Adds addition layer.
    layer = net.AddAdditionLayer("add layer");
    BOOST_TEST(layer);

    softmax1Layer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    softmax2Layer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));

    // Adds an output layer.
    armnn::IConnectableLayer* prevLayer = layer;
    layer = net.AddOutputLayer(0, "output layer");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

    BOOST_TEST(layer);
}

BOOST_AUTO_TEST_CASE(NetworkModification_SplitterMultiplication)
{
    armnn::Network net;

    // Adds an input layer and an input tensor descriptor.
    armnn::IConnectableLayer* layer = net.AddInputLayer(0, "input layer");
    BOOST_TEST(layer);

    // Adds a splitter layer.
    armnn::ViewsDescriptor splitterDesc(2,4);
    armnn::IConnectableLayer* const splitterLayer = net.AddSplitterLayer(splitterDesc, "splitter layer");
    BOOST_TEST(splitterLayer);

    layer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));

    // Adds a softmax layer 1.
    armnn::SoftmaxDescriptor softmaxDescriptor;
    armnn::IConnectableLayer* const softmax1Layer = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_1");
    BOOST_TEST(softmax1Layer);

    splitterLayer->GetOutputSlot(0).Connect(softmax1Layer->GetInputSlot(0));

    // Adds a softmax layer 2.
    armnn::IConnectableLayer* const softmax2Layer = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_2");
    BOOST_TEST(softmax2Layer);

    splitterLayer->GetOutputSlot(1).Connect(softmax2Layer->GetInputSlot(0));

    // Adds multiplication layer.
    layer = net.AddMultiplicationLayer("multiplication layer");
    BOOST_TEST(layer);

    softmax1Layer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    softmax2Layer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));

    // Adds an output layer.
    armnn::IConnectableLayer* prevLayer = layer;
    layer = net.AddOutputLayer(0, "output layer");
    BOOST_TEST(layer);

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(OptimizeValidateCpuRefWorkloads)
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

    std::vector<armnn::Compute> backends = { armnn::Compute::CpuRef };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec());
    static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph().AllocateDynamicBuffers();
    BOOST_CHECK(optNet);

    // Validates workloads.
    armnn::RefWorkloadFactory fact;
    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK_NO_THROW(
            layer->CreateWorkload(static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph(), fact));
    }
}

#if ARMCOMPUTENEON_ENABLED
BOOST_AUTO_TEST_CASE(OptimizeValidateCpuAccDeviceSupportLayerNoFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::Compute> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);
    // validate workloads
    armnn::NeonWorkloadFactory fact;
    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK_EQUAL(armnn::Compute::CpuAcc, layer->GetComputeDevice());
        BOOST_CHECK_NO_THROW(
            layer->CreateWorkload(static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph(), fact));
    }
}
#endif // ARMCOMPUTENEON_ENABLED

#if ARMCOMPUTECL_ENABLED
BOOST_AUTO_TEST_CASE(OptimizeValidateGpuDeviceSupportLayerNoFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::Compute> backends = { armnn::Compute::GpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);
    // validate workloads
    armnn::ClWorkloadFactory fact;
    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK_EQUAL(armnn::Compute::GpuAcc, layer->GetComputeDevice());
        BOOST_CHECK_NO_THROW(
            layer->CreateWorkload(static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph(), fact));
    }
}
#endif // ARMCOMPUTECL_ENABLED

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

    std::vector<armnn::Compute> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(!optNet);
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

    std::vector<armnn::Compute> backends = { armnn::Compute::CpuAcc, armnn::Compute::CpuRef };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_REQUIRE(optNet);

    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        // If NEON is enabled, Input and Output layers are supported by CpuAcc,
        // the other layers are supported by CpuRef.
        // If NEON is not enabled, all layers are supported by CpuRef.
#if ARMCOMPUTENEON_ENABLED
        if (layer->GetType() == armnn::LayerType::Input || layer->GetType() == armnn::LayerType::Output)
        {
            BOOST_CHECK_EQUAL(armnn::Compute::CpuAcc, layer->GetComputeDevice());
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            BOOST_CHECK_EQUAL(armnn::Compute::CpuRef, layer->GetComputeDevice());
        }
#else
        BOOST_CHECK_EQUAL(armnn::Compute::CpuRef, layer->GetComputeDevice());
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

    std::vector<armnn::Compute> backends = { armnn::Compute::Undefined };

    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(!optNet);

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

    std::vector<armnn::Compute> backends = { armnn::Compute::Undefined, armnn::Compute::CpuRef };

    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);

    // validate workloads
    armnn::RefWorkloadFactory fact;
    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK_EQUAL(armnn::Compute::CpuRef, layer->GetComputeDevice());
        BOOST_CHECK_NO_THROW(
            layer->CreateWorkload(static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph(), fact));
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

    std::vector<armnn::Compute> backends = { armnn::Compute::CpuAcc,
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
#if ARMCOMPUTENEON_ENABLED
        if (layer->GetType() == armnn::LayerType::Input || layer->GetType() == armnn::LayerType::Output)
        {
            BOOST_CHECK_EQUAL(armnn::Compute::CpuAcc, layer->GetComputeDevice());
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            BOOST_CHECK_EQUAL(armnn::Compute::CpuRef, layer->GetComputeDevice());
        }
#elif ARMCOMPUTECL_ENABLED
        if (layer->GetType() == armnn::LayerType::Input || layer->GetType() == armnn::LayerType::Output)
        {
            BOOST_CHECK_EQUAL(armnn::Compute::GpuAcc, layer->GetComputeDevice());
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            BOOST_CHECK_EQUAL(armnn::Compute::CpuRef, layer->GetComputeDevice());
        }
#else
        BOOST_CHECK_EQUAL(armnn::Compute::CpuRef, layer->GetComputeDevice());
#endif
    }
}

BOOST_AUTO_TEST_CASE(OptimizeValidateWorkloadsCpuRefPermuteLayer)
{
    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::Compute> backends = {armnn::Compute::CpuRef};

    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    armnn::PermuteDescriptor descriptor({0, 2, 3, 1});
    armnn::IConnectableLayer* permute = net->AddPermuteLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(permute->GetInputSlot(0));
    permute->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    permute->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 4, 1, 4 }, armnn::DataType::Float32));

    // optimize the network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());

    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK_EQUAL(armnn::Compute::CpuRef, layer->GetComputeDevice());
    }
}

BOOST_AUTO_TEST_CASE(FP16TurboModeTestOnCpuRef)
{
    // Test to check when FP16 Turbo mode set
    // it converts the FP32 network to FP16 Network
    // add FP32ToFP16 conversion layer after the InputLayer
    // add FP16ToFP32 conversion layer after the OutputLayer
    // checks the other layers if they are supported in FP16
    // if they are not put the conversion layers before and after
    // if they are not supported in FP16 use FP32 instead
    // if there are inverse conversion layers remove them with optimization
    // at the moment FloorLayer is not supported in FP16 so it rolls back to FP32
    // and inverse conversion layers are removed by the optimizer
    armnn::Network net;

    // Defines layers.
    auto input = net.AddInputLayer(0);
    auto floor = net.AddFloorLayer();
    auto output = net.AddOutputLayer(0);

    // Connects layers.
    input->GetOutputSlot(0).Connect(floor->GetInputSlot(0));
    floor->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorShape shape({4});
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(info);
    floor->GetOutputSlot(0).SetTensorInfo(info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::Compute> backends = {armnn::Compute::CpuRef};

    armnn::OptimizerOptions optimizerOptions;
    optimizerOptions.m_ReduceFp32ToFp16 = true;

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec(),
                                                               optimizerOptions);

    std::ostringstream ss;
    optimizedNet->SerializeToDot(ss);

    auto inputId = input->GetGuid();
    auto floorId = floor->GetGuid();
    auto outputId = output->GetGuid();

    std::stringstream expected;
    expected <<
             "digraph Optimized {\n"
             "    node [shape=\"record\"];\n"
             "    edge [fontsize=8 fontcolor=\"blue\" fontname=\"arial-bold\"];\n"
             "    " << inputId << " [label=\"{Input}\"];\n"
             "    " << floorId << " [label=\"{Floor}\"];\n"
             "    " << outputId << " [label=\"{Output}\"];\n"
             "    " << inputId << " -> " << floorId << " [label=< [4] >];\n"
             "    " << floorId << " -> " << outputId << " [label=< [4] >];\n"
             "}\n";

    BOOST_TEST(ss.str() == expected.str());
}

#if ARMCOMPUTECL_ENABLED
BOOST_AUTO_TEST_CASE(FP16TurboModeTestOnGpuAcc)
{
    // Test to check when Fp16 Turbo mode set
    // it converts the Fp32 network to Fp16 Network
    // add Fp32ToFp16 conversion layer after the InputLayer
    // add Fp16ToFp32 conversion layer after the OutputLayer
    // checks the other layers if they are supported in Fp16
    // if they are not put the conversion layers before and after
    // if they are not supported in Fp16 use Fp32 instead
    // if there are inverse conversion layers remove them with optimization
    // at the moment FloorLayer is not supported in Fp16 so it rolls back to Fp32
    // and inverse conversion layers are removed by the optimizer
    armnn::Network net;

    // Defines layers.
    auto input = net.AddInputLayer(0, "input layer");
    // ReLu1
    armnn::ActivationDescriptor activation1Descriptor;
    activation1Descriptor.m_Function = armnn::ActivationFunction::BoundedReLu;
    activation1Descriptor.m_A = 1.f;
    activation1Descriptor.m_B = -1.f;
    auto activation = net.AddActivationLayer(activation1Descriptor, "activation layer");
    auto output = net.AddOutputLayer(0, "output layer");

    // Connects layers.
    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorShape shape({4});
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::Compute> backends = {armnn::Compute::GpuAcc};

    armnn::OptimizerOptions optimizerOptions;
    optimizerOptions.m_ReduceFp32ToFp16 = true;

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec(),
                                                               optimizerOptions);

    const armnn::Graph& graph = static_cast<armnn::OptimizedNetwork*>(optimizedNet.get())->GetGraph();

    // Tests that all layers are present in the graph.
    BOOST_TEST(graph.GetNumLayers() == 5);

    // Tests that the vertices exist and have correct names.
    BOOST_TEST(GraphHasNamedLayer(graph, "input layer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "convert_fp32_to_fp16-0-input layer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "activation layer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "convert_fp16_to_fp32-0-output layer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "output layer"));
}
#endif

BOOST_AUTO_TEST_SUITE_END()
