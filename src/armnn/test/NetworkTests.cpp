//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <GraphUtils.hpp>


#include <Network.hpp>

#include <doctest/doctest.h>

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

TEST_SUITE("Network")
{
TEST_CASE("LayerGuids")
{
    armnn::NetworkImpl net;
    LayerGuid inputId = net.AddInputLayer(0)->GetGuid();
    LayerGuid addId = net.AddElementwiseBinaryLayer(armnn::BinaryOperation::Add)->GetGuid();
    LayerGuid outputId = net.AddOutputLayer(0)->GetGuid();

    CHECK(inputId != addId);
    CHECK(addId != outputId);
    CHECK(inputId != outputId);
}

TEST_CASE("NetworkBasic")
{
    armnn::NetworkImpl net;
    CHECK(net.PrintGraph() == armnn::Status::Success);
}

TEST_CASE("LayerNamesAreOptionalForINetwork")
{
    armnn::INetworkPtr inet(armnn::INetwork::Create());
    inet->AddInputLayer(0);
    inet->AddElementwiseBinaryLayer(armnn::BinaryOperation::Add);
    inet->AddActivationLayer(armnn::ActivationDescriptor());
    inet->AddOutputLayer(0);
}

TEST_CASE("LayerNamesAreOptionalForNetwork")
{
    armnn::NetworkImpl net;
    net.AddInputLayer(0);
    net.AddElementwiseBinaryLayer(armnn::BinaryOperation::Add);
    net.AddActivationLayer(armnn::ActivationDescriptor());
    net.AddOutputLayer(0);
}

TEST_CASE("NetworkModification")
{
    armnn::NetworkImpl net;

    armnn::IConnectableLayer* const inputLayer = net.AddInputLayer(0, "input layer");
    CHECK(inputLayer);

    unsigned int dims[] = { 10,1,1,1 };
    std::vector<float> convWeightsData(10);
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::Float32, 0.0f, 0, true), convWeightsData);

    armnn::Convolution2dDescriptor convDesc2d;
    armnn::IConnectableLayer* const weightsLayer = net.AddConstantLayer(weights, "conv const weights");
    armnn::IConnectableLayer* const convLayer = net.AddConvolution2dLayer(convDesc2d, "conv layer");
    CHECK(convLayer);
    CHECK(weightsLayer);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1));

    armnn::FullyConnectedDescriptor fullyConnectedDesc;

    // Constant layer that now holds weights data for FullyConnected
    armnn::IConnectableLayer* const constantWeightsLayer = net.AddConstantLayer(weights, "fc const weights");
    armnn::IConnectableLayer* const fullyConnectedLayer = net.AddFullyConnectedLayer(fullyConnectedDesc,
                                                                                     "fully connected");
    CHECK(constantWeightsLayer);
    CHECK(fullyConnectedLayer);

    constantWeightsLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    convLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));

    armnn::Pooling2dDescriptor pooling2dDesc;
    armnn::IConnectableLayer* const poolingLayer = net.AddPooling2dLayer(pooling2dDesc, "pooling2d");
    CHECK(poolingLayer);

    fullyConnectedLayer->GetOutputSlot(0).Connect(poolingLayer->GetInputSlot(0));

    armnn::ActivationDescriptor activationDesc;
    armnn::IConnectableLayer* const activationLayer = net.AddActivationLayer(activationDesc, "activation");
    CHECK(activationLayer);

    poolingLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));

    armnn::NormalizationDescriptor normalizationDesc;
    armnn::IConnectableLayer* const normalizationLayer = net.AddNormalizationLayer(normalizationDesc, "normalization");
    CHECK(normalizationLayer);

    activationLayer->GetOutputSlot(0).Connect(normalizationLayer->GetInputSlot(0));

    armnn::SoftmaxDescriptor softmaxDesc;
    armnn::IConnectableLayer* const softmaxLayer = net.AddSoftmaxLayer(softmaxDesc, "softmax");
    CHECK(softmaxLayer);

    normalizationLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));

    armnn::BatchNormalizationDescriptor batchNormDesc;

    armnn::TensorInfo tensorInfo({ 1 }, armnn::DataType::Float32, 0.0f, 0, true);
    std::vector<float> data(tensorInfo.GetNumBytes() / sizeof(float));
    armnn::ConstTensor invalidTensor(tensorInfo, data);

    armnn::IConnectableLayer* const batchNormalizationLayer = net.AddBatchNormalizationLayer(batchNormDesc,
        invalidTensor,
        invalidTensor,
        invalidTensor,
        invalidTensor,
        "batch norm");
    CHECK(batchNormalizationLayer);

    softmaxLayer->GetOutputSlot(0).Connect(batchNormalizationLayer->GetInputSlot(0));

    armnn::IConnectableLayer* const additionLayer = net.AddElementwiseBinaryLayer(armnn::BinaryOperation::Add,
                                                                                  "addition");
    CHECK(additionLayer);

    batchNormalizationLayer->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
    batchNormalizationLayer->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const multiplicationLayer = net.AddElementwiseBinaryLayer(armnn::BinaryOperation::Mul,
                                                                                        "multiplication");
    CHECK(multiplicationLayer);

    additionLayer->GetOutputSlot(0).Connect(multiplicationLayer->GetInputSlot(0));
    additionLayer->GetOutputSlot(0).Connect(multiplicationLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = net.AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

    multiplicationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    //Tests that all layers are present in the graph.
    CHECK(net.GetGraph().GetNumLayers() == 13);

    //Tests that the vertices exist and have correct names.
    CHECK(GraphHasNamedLayer(net.GetGraph(), "input layer"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "conv layer"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "conv const weights"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "fc const weights"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "fully connected"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "pooling2d"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "activation"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "normalization"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "softmax"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "batch norm"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "addition"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "multiplication"));
    CHECK(GraphHasNamedLayer(net.GetGraph(), "output layer"));

    auto checkOneOutputToOneInputConnection = []
        (const armnn::IConnectableLayer* const srcLayer,
         const armnn::IConnectableLayer* const tgtLayer,
         int expectedSrcNumInputs = 1,
         int expectedDstNumOutputs = 1)
        {
            CHECK(srcLayer->GetNumInputSlots() == expectedSrcNumInputs);
            CHECK(srcLayer->GetNumOutputSlots() == 1);
            CHECK(tgtLayer->GetNumInputSlots() == 1);
            CHECK(tgtLayer->GetNumOutputSlots() == expectedDstNumOutputs);

            CHECK(srcLayer->GetOutputSlot(0).GetNumConnections() == 1);
            CHECK(srcLayer->GetOutputSlot(0).GetConnection(0) == &tgtLayer->GetInputSlot(0));
            CHECK(&srcLayer->GetOutputSlot(0) == tgtLayer->GetInputSlot(0).GetConnection());
        };
    auto checkOneOutputToTwoInputsConnections = []
        (const armnn::IConnectableLayer* const srcLayer,
         const armnn::IConnectableLayer* const tgtLayer,
         int expectedSrcNumInputs,
         int expectedDstNumOutputs = 1)
        {
            CHECK(srcLayer->GetNumInputSlots() == expectedSrcNumInputs);
            CHECK(srcLayer->GetNumOutputSlots() == 1);
            CHECK(tgtLayer->GetNumInputSlots() == 2);
            CHECK(tgtLayer->GetNumOutputSlots() == expectedDstNumOutputs);

            CHECK(srcLayer->GetOutputSlot(0).GetNumConnections() == 2);
            for (unsigned int i = 0; i < srcLayer->GetOutputSlot(0).GetNumConnections(); ++i)
            {
                CHECK(srcLayer->GetOutputSlot(0).GetConnection(i) == &tgtLayer->GetInputSlot(i));
                CHECK(&srcLayer->GetOutputSlot(0) == tgtLayer->GetInputSlot(i).GetConnection());
            }
        };
    auto checkOneOutputToTwoInputConnectionForTwoDifferentLayers = []
        (const armnn::IConnectableLayer* const srcLayer1,
         const armnn::IConnectableLayer* const srcLayer2,
         const armnn::IConnectableLayer* const tgtLayer,
         int expectedSrcNumInputs1 = 1,
         int expectedSrcNumInputs2 = 1,
         int expectedDstNumOutputs = 1)
        {
            CHECK(srcLayer1->GetNumInputSlots() == expectedSrcNumInputs1);
            CHECK(srcLayer1->GetNumOutputSlots() == 1);
            CHECK(srcLayer2->GetNumInputSlots() == expectedSrcNumInputs2);
            CHECK(srcLayer2->GetNumOutputSlots() == 1);
            CHECK(tgtLayer->GetNumInputSlots() == 2);
            CHECK(tgtLayer->GetNumOutputSlots() == expectedDstNumOutputs);

            CHECK(srcLayer1->GetOutputSlot(0).GetNumConnections() == 1);
            CHECK(srcLayer2->GetOutputSlot(0).GetNumConnections() == 1);
            CHECK(srcLayer1->GetOutputSlot(0).GetConnection(0) == &tgtLayer->GetInputSlot(0));
            CHECK(srcLayer2->GetOutputSlot(0).GetConnection(0) == &tgtLayer->GetInputSlot(1));
            CHECK(&srcLayer1->GetOutputSlot(0) == tgtLayer->GetInputSlot(0).GetConnection());
            CHECK(&srcLayer2->GetOutputSlot(0) == tgtLayer->GetInputSlot(1).GetConnection());
        };

    CHECK(AreAllLayerInputSlotsConnected(*convLayer));
    CHECK(AreAllLayerInputSlotsConnected(*fullyConnectedLayer));
    CHECK(AreAllLayerInputSlotsConnected(*poolingLayer));
    CHECK(AreAllLayerInputSlotsConnected(*activationLayer));
    CHECK(AreAllLayerInputSlotsConnected(*normalizationLayer));
    CHECK(AreAllLayerInputSlotsConnected(*softmaxLayer));
    CHECK(AreAllLayerInputSlotsConnected(*batchNormalizationLayer));
    CHECK(AreAllLayerInputSlotsConnected(*additionLayer));
    CHECK(AreAllLayerInputSlotsConnected(*multiplicationLayer));
    CHECK(AreAllLayerInputSlotsConnected(*outputLayer));

    // Checks connectivity.
    checkOneOutputToTwoInputConnectionForTwoDifferentLayers(inputLayer, weightsLayer, convLayer, 0, 0);
    checkOneOutputToTwoInputConnectionForTwoDifferentLayers(convLayer, constantWeightsLayer, fullyConnectedLayer, 2, 0);
    checkOneOutputToOneInputConnection(fullyConnectedLayer, poolingLayer, 2, 1);
    checkOneOutputToOneInputConnection(poolingLayer, activationLayer);
    checkOneOutputToOneInputConnection(activationLayer, normalizationLayer);
    checkOneOutputToOneInputConnection(normalizationLayer, softmaxLayer);
    checkOneOutputToOneInputConnection(softmaxLayer, batchNormalizationLayer);
    checkOneOutputToTwoInputsConnections(batchNormalizationLayer, additionLayer, 1);
    checkOneOutputToTwoInputsConnections(additionLayer, multiplicationLayer, 2);
    checkOneOutputToOneInputConnection(multiplicationLayer, outputLayer, 2, 0);
}

TEST_CASE("NetworkModification_SplitterConcat")
{
    armnn::NetworkImpl net;

    // Adds an input layer and an input tensor descriptor.
    armnn::IConnectableLayer* inputLayer = net.AddInputLayer(0, "input layer");
    CHECK(inputLayer);

    // Adds a splitter layer.
    armnn::ViewsDescriptor splitterDesc(2,4);

    armnn::IConnectableLayer* splitterLayer = net.AddSplitterLayer(splitterDesc, "splitter layer");
    CHECK(splitterLayer);

    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));

    // Adds a softmax layer 1.
    armnn::SoftmaxDescriptor softmaxDescriptor;
    armnn::IConnectableLayer* softmaxLayer1 = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_1");
    CHECK(softmaxLayer1);

    splitterLayer->GetOutputSlot(0).Connect(softmaxLayer1->GetInputSlot(0));

    // Adds a softmax layer 2.
    armnn::IConnectableLayer* softmaxLayer2 = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_2");
    CHECK(softmaxLayer2);

    splitterLayer->GetOutputSlot(1).Connect(softmaxLayer2->GetInputSlot(0));

    // Adds a concat layer.
    armnn::OriginsDescriptor concatDesc(2, 4);

    armnn::IConnectableLayer* concatLayer = net.AddConcatLayer(concatDesc, "concat layer");
    CHECK(concatLayer);

    softmaxLayer1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    softmaxLayer2->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));

    // Adds an output layer.
    armnn::IConnectableLayer* outputLayer = net.AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    CHECK(splitterLayer->GetNumOutputSlots() == 2);
    CHECK(splitterLayer->GetOutputSlot(0).GetConnection(0) == &softmaxLayer1->GetInputSlot(0));
    CHECK(&splitterLayer->GetOutputSlot(0) == softmaxLayer1->GetInputSlot(0).GetConnection());
    CHECK(splitterLayer->GetOutputSlot(1).GetConnection(0) == &softmaxLayer2->GetInputSlot(0));
    CHECK(&splitterLayer->GetOutputSlot(1) == softmaxLayer2->GetInputSlot(0).GetConnection());

    CHECK(concatLayer->GetNumInputSlots() == 2);
    CHECK(softmaxLayer1->GetOutputSlot(0).GetConnection(0) == &concatLayer->GetInputSlot(0));
    CHECK(&softmaxLayer1->GetOutputSlot(0) == concatLayer->GetInputSlot(0).GetConnection());
    CHECK(softmaxLayer2->GetOutputSlot(0).GetConnection(0) == &concatLayer->GetInputSlot(1));
    CHECK(&softmaxLayer2->GetOutputSlot(0) == concatLayer->GetInputSlot(1).GetConnection());
}

TEST_CASE("NetworkModification_SplitterAddition")
{
    armnn::NetworkImpl net;

    // Adds an input layer and an input tensor descriptor.
    armnn::IConnectableLayer* layer = net.AddInputLayer(0, "input layer");
    CHECK(layer);

    // Adds a splitter layer.
    armnn::ViewsDescriptor splitterDesc(2,4);

    armnn::IConnectableLayer* const splitterLayer = net.AddSplitterLayer(splitterDesc, "splitter layer");
    CHECK(splitterLayer);

    layer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));

    // Adds a softmax layer 1.
    armnn::SoftmaxDescriptor softmaxDescriptor;
    armnn::IConnectableLayer* const softmax1Layer = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_1");
    CHECK(softmax1Layer);

    splitterLayer->GetOutputSlot(0).Connect(softmax1Layer->GetInputSlot(0));

    // Adds a softmax layer 2.
    armnn::IConnectableLayer* const softmax2Layer = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_2");
    CHECK(softmax2Layer);

    splitterLayer->GetOutputSlot(1).Connect(softmax2Layer->GetInputSlot(0));

    // Adds addition layer.
    layer = net.AddElementwiseBinaryLayer(armnn::BinaryOperation::Add, "add layer");
    CHECK(layer);

    softmax1Layer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    softmax2Layer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));

    // Adds an output layer.
    armnn::IConnectableLayer* prevLayer = layer;
    layer = net.AddOutputLayer(0, "output layer");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

    CHECK(layer);
}

TEST_CASE("NetworkModification_SplitterMultiplication")
{
    armnn::NetworkImpl net;

    // Adds an input layer and an input tensor descriptor.
    armnn::IConnectableLayer* layer = net.AddInputLayer(0, "input layer");
    CHECK(layer);

    // Adds a splitter layer.
    armnn::ViewsDescriptor splitterDesc(2,4);
    armnn::IConnectableLayer* const splitterLayer = net.AddSplitterLayer(splitterDesc, "splitter layer");
    CHECK(splitterLayer);

    layer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));

    // Adds a softmax layer 1.
    armnn::SoftmaxDescriptor softmaxDescriptor;
    armnn::IConnectableLayer* const softmax1Layer = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_1");
    CHECK(softmax1Layer);

    splitterLayer->GetOutputSlot(0).Connect(softmax1Layer->GetInputSlot(0));

    // Adds a softmax layer 2.
    armnn::IConnectableLayer* const softmax2Layer = net.AddSoftmaxLayer(softmaxDescriptor, "softmax_2");
    CHECK(softmax2Layer);

    splitterLayer->GetOutputSlot(1).Connect(softmax2Layer->GetInputSlot(0));

    // Adds multiplication layer.
    layer = net.AddElementwiseBinaryLayer(armnn::BinaryOperation::Mul, "multiplication layer");
    CHECK(layer);

    softmax1Layer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    softmax2Layer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));

    // Adds an output layer.
    armnn::IConnectableLayer* prevLayer = layer;
    layer = net.AddOutputLayer(0, "output layer");
    CHECK(layer);

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
}

TEST_CASE("Network_AddQuantize")
{
    struct Test : public armnn::IStrategy
    {
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
                case armnn::LayerType::Quantize:
                {
                    m_Visited = true;

                    CHECK(layer);

                    std::string expectedName = std::string("quantize");
                    CHECK(std::string(layer->GetName()) == expectedName);
                    CHECK(std::string(name) == expectedName);

                    CHECK(layer->GetNumInputSlots() == 1);
                    CHECK(layer->GetNumOutputSlots() == 1);

                    const armnn::TensorInfo& infoIn = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
                    CHECK((infoIn.GetDataType() == armnn::DataType::Float32));

                    const armnn::TensorInfo& infoOut = layer->GetOutputSlot(0).GetTensorInfo();
                    CHECK((infoOut.GetDataType() == armnn::DataType::QAsymmU8));
                    break;
                }
                default:
                {
                    // nothing
                }
            }
        }

        bool m_Visited = false;
    };


    auto graph = armnn::INetwork::Create();

    auto input = graph->AddInputLayer(0, "input");
    auto quantize = graph->AddQuantizeLayer("quantize");
    auto output = graph->AddOutputLayer(1, "output");

    input->GetOutputSlot(0).Connect(quantize->GetInputSlot(0));
    quantize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorInfo infoIn({3,1}, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(infoIn);

    armnn::TensorInfo infoOut({3,1}, armnn::DataType::QAsymmU8);
    quantize->GetOutputSlot(0).SetTensorInfo(infoOut);

    Test testQuantize;
    graph->ExecuteStrategy(testQuantize);

    CHECK(testQuantize.m_Visited == true);

}

TEST_CASE("Network_AddMerge")
{
    struct Test : public armnn::IStrategy
    {
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
                    m_Visited = true;

                    CHECK(layer);

                    std::string expectedName = std::string("merge");
                    CHECK(std::string(layer->GetName()) == expectedName);
                    CHECK(std::string(name) == expectedName);

                    CHECK(layer->GetNumInputSlots() == 2);
                    CHECK(layer->GetNumOutputSlots() == 1);

                    const armnn::TensorInfo& infoIn0 = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
                    CHECK((infoIn0.GetDataType() == armnn::DataType::Float32));

                    const armnn::TensorInfo& infoIn1 = layer->GetInputSlot(1).GetConnection()->GetTensorInfo();
                    CHECK((infoIn1.GetDataType() == armnn::DataType::Float32));

                    const armnn::TensorInfo& infoOut = layer->GetOutputSlot(0).GetTensorInfo();
                    CHECK((infoOut.GetDataType() == armnn::DataType::Float32));
                    break;
                }
                default:
                {
                    // nothing
                }
            }
        }

        bool m_Visited = false;
    };

    armnn::INetworkPtr network = armnn::INetwork::Create();

    armnn::IConnectableLayer* input0 = network->AddInputLayer(0);
    armnn::IConnectableLayer* input1 = network->AddInputLayer(1);
    armnn::IConnectableLayer* merge = network->AddMergeLayer("merge");
    armnn::IConnectableLayer* output = network->AddOutputLayer(0);

    input0->GetOutputSlot(0).Connect(merge->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(merge->GetInputSlot(1));
    merge->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    const armnn::TensorInfo info({3,1}, armnn::DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    merge->GetOutputSlot(0).SetTensorInfo(info);

    Test testMerge;
    network->ExecuteStrategy(testMerge);

    CHECK(testMerge.m_Visited == true);
}

TEST_CASE("StandInLayerNetworkTest")
{
    // Create a simple network with a StandIn some place in it.
    armnn::NetworkImpl net;
    auto input = net.AddInputLayer(0);

    // Add some valid layer.
    auto floor = net.AddFloorLayer("Floor");

    // Add a standin layer
    armnn::StandInDescriptor standInDescriptor;
    standInDescriptor.m_NumInputs  = 1;
    standInDescriptor.m_NumOutputs = 1;
    auto standIn = net.AddStandInLayer(standInDescriptor, "StandIn");

    // Finally the output.
    auto output = net.AddOutputLayer(0);

    // Connect up the layers
    input->GetOutputSlot(0).Connect(floor->GetInputSlot(0));

    floor->GetOutputSlot(0).Connect(standIn->GetInputSlot(0));

    standIn->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Check that the layer is there.
    CHECK(GraphHasNamedLayer(net.GetGraph(), "StandIn"));
    // Check that it is connected as expected.
    CHECK(input->GetOutputSlot(0).GetConnection(0) == &floor->GetInputSlot(0));
    CHECK(floor->GetOutputSlot(0).GetConnection(0) == &standIn->GetInputSlot(0));
    CHECK(standIn->GetOutputSlot(0).GetConnection(0) == &output->GetInputSlot(0));
}

TEST_CASE("StandInLayerSingleInputMultipleOutputsNetworkTest")
{
    // Another test with one input and two outputs on the StandIn layer.
    armnn::NetworkImpl net;

    // Create the input.
    auto input = net.AddInputLayer(0);

    // Add a standin layer
    armnn::StandInDescriptor standInDescriptor;
    standInDescriptor.m_NumInputs  = 1;
    standInDescriptor.m_NumOutputs = 2;
    auto standIn = net.AddStandInLayer(standInDescriptor, "StandIn");

    // Add two outputs.
    auto output0 = net.AddOutputLayer(0);
    auto output1 = net.AddOutputLayer(1);

    // Connect up the layers
    input->GetOutputSlot(0).Connect(standIn->GetInputSlot(0));

    // Connect the two outputs of the Standin to the two outputs.
    standIn->GetOutputSlot(0).Connect(output0->GetInputSlot(0));
    standIn->GetOutputSlot(1).Connect(output1->GetInputSlot(0));

    // Check that the layer is there.
    CHECK(GraphHasNamedLayer(net.GetGraph(), "StandIn"));
    // Check that it is connected as expected.
    CHECK(input->GetOutputSlot(0).GetConnection(0) == &standIn->GetInputSlot(0));
    CHECK(standIn->GetOutputSlot(0).GetConnection(0) == &output0->GetInputSlot(0));
    CHECK(standIn->GetOutputSlot(1).GetConnection(0) == &output1->GetInputSlot(0));
}

TEST_CASE("ObtainConv2DDescriptorFromIConnectableLayer")
{
    armnn::NetworkImpl net;

    armnn::Convolution2dDescriptor convDesc2d;
    convDesc2d.m_PadLeft = 2;
    convDesc2d.m_PadRight = 3;
    convDesc2d.m_PadTop = 4;
    convDesc2d.m_PadBottom = 5;
    convDesc2d.m_StrideX = 2;
    convDesc2d.m_StrideY = 1;
    convDesc2d.m_DilationX = 3;
    convDesc2d.m_DilationY = 3;
    convDesc2d.m_BiasEnabled = false;
    convDesc2d.m_DataLayout = armnn::DataLayout::NCHW;
    armnn::IConnectableLayer* const convLayer = net.AddConvolution2dLayer(convDesc2d, "conv layer");
    CHECK(convLayer);

    const armnn::BaseDescriptor& descriptor = convLayer->GetParameters();
    CHECK(descriptor.IsNull() == false);
    const armnn::Convolution2dDescriptor& originalDescriptor =
        static_cast<const armnn::Convolution2dDescriptor&>(descriptor);
    CHECK(originalDescriptor.m_PadLeft == 2);
    CHECK(originalDescriptor.m_PadRight == 3);
    CHECK(originalDescriptor.m_PadTop == 4);
    CHECK(originalDescriptor.m_PadBottom == 5);
    CHECK(originalDescriptor.m_StrideX == 2);
    CHECK(originalDescriptor.m_StrideY == 1);
    CHECK(originalDescriptor.m_DilationX == 3);
    CHECK(originalDescriptor.m_DilationY == 3);
    CHECK(originalDescriptor.m_BiasEnabled == false);
    CHECK(originalDescriptor.m_DataLayout == armnn::DataLayout::NCHW);
}

TEST_CASE("CheckNotNullDescriptor")
{
    armnn::NetworkImpl net;
    armnn::IConnectableLayer* const addLayer = net.AddElementwiseBinaryLayer(armnn::BinaryOperation::Add);

    CHECK(addLayer);

    const armnn::BaseDescriptor& descriptor = addLayer->GetParameters();
    // additional layer has no descriptor so a NullDescriptor will be returned
    CHECK(descriptor.IsNull() == false);
}

TEST_CASE("CheckNullDescriptor")
{
    armnn::NetworkImpl net;
    armnn::IConnectableLayer* const addLayer = net.AddPreluLayer();

    CHECK(addLayer);

    const armnn::BaseDescriptor& descriptor = addLayer->GetParameters();
    // Prelu has no descriptor so a NullDescriptor will be returned
    CHECK(descriptor.IsNull() == true);
}

}
