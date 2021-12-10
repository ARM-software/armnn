//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <CommonTestUtils.hpp>

#include <Graph.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <doctest/doctest.h>

#include <utility>

using namespace armnn;
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////
// The following test are created specifically to test ReleaseConstantData() method in the Layer
// They build very simple graphs including the layer will be checked.
// Checks weights and biases before the method called and after.
/////////////////////////////////////////////////////////////////////////////////////////////

TEST_SUITE("LayerReleaseConstantDataTest")
{
TEST_CASE("ReleaseBatchNormalizationLayerConstantDataTest")
{
    Graph graph;

    // create the layer we're testing
    BatchNormalizationDescriptor layerDesc;
    layerDesc.m_Eps = 0.05f;
    BatchNormalizationLayer* const layer = graph.AddLayer<BatchNormalizationLayer>(layerDesc, "layer");

    armnn::TensorInfo weightInfo({3}, armnn::DataType::Float32);
    layer->m_Mean     = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Variance = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Beta     = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Gamma    = std::make_unique<ScopedTensorHandle>(weightInfo);
    layer->m_Mean->Allocate();
    layer->m_Variance->Allocate();
    layer->m_Beta->Allocate();
    layer->m_Gamma->Allocate();

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    armnn::TensorInfo tensorInfo({2, 3, 1, 1}, armnn::DataType::Float32);
    Connect(input, layer, tensorInfo);
    Connect(layer, output, tensorInfo);

    // check the constants that they are not NULL
    CHECK(layer->m_Mean != nullptr);
    CHECK(layer->m_Variance != nullptr);
    CHECK(layer->m_Beta != nullptr);
    CHECK(layer->m_Gamma != nullptr);

    // free up the constants..
    layer->ReleaseConstantData();

    // check the constants that they are NULL now
    CHECK(layer->m_Mean == nullptr);
    CHECK(layer->m_Variance == nullptr);
    CHECK(layer->m_Beta == nullptr);
    CHECK(layer->m_Gamma == nullptr);

 }


 TEST_CASE("ReleaseConvolution2dLayerConstantDataTest")
 {
     Graph graph;

     // create the layer we're testing
     Convolution2dDescriptor layerDesc;
     layerDesc.m_PadLeft = 3;
     layerDesc.m_PadRight = 3;
     layerDesc.m_PadTop = 1;
     layerDesc.m_PadBottom = 1;
     layerDesc.m_StrideX = 2;
     layerDesc.m_StrideY = 4;
     layerDesc.m_BiasEnabled = true;

     Convolution2dLayer* const layer = graph.AddLayer<Convolution2dLayer>(layerDesc, "layer");

     layer->m_Weight = std::make_unique<ScopedTensorHandle>(TensorInfo({2, 3, 5, 3},
                                                                          armnn::DataType::Float32));
     layer->m_Bias   = std::make_unique<ScopedTensorHandle>
             (TensorInfo({2}, GetBiasDataType(armnn::DataType::Float32)));

     layer->m_Weight->Allocate();
     layer->m_Bias->Allocate();

     // create extra layers
     Layer* const input = graph.AddLayer<InputLayer>(0, "input");
     Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

     // connect up
     Connect(input, layer, TensorInfo({2, 3, 8, 16}, armnn::DataType::Float32));
     Connect(layer, output, TensorInfo({2, 2, 2, 10}, armnn::DataType::Float32));

     // check the constants that they are not NULL
     CHECK(layer->m_Weight != nullptr);
     CHECK(layer->m_Bias != nullptr);

     // free up the constants..
     layer->ReleaseConstantData();

     // check the constants that they are NULL now
     CHECK(layer->m_Weight == nullptr);
     CHECK(layer->m_Bias == nullptr);
}

TEST_CASE("ReleaseDepthwiseConvolution2dLayerConstantDataTest")
{
    Graph graph;

    // create the layer we're testing
    DepthwiseConvolution2dDescriptor layerDesc;
    layerDesc.m_PadLeft         = 3;
    layerDesc.m_PadRight        = 3;
    layerDesc.m_PadTop          = 1;
    layerDesc.m_PadBottom       = 1;
    layerDesc.m_StrideX         = 2;
    layerDesc.m_StrideY         = 4;
    layerDesc.m_BiasEnabled     = true;

    DepthwiseConvolution2dLayer* const layer = graph.AddLayer<DepthwiseConvolution2dLayer>(layerDesc, "layer");

    layer->m_Weight = std::make_unique<ScopedTensorHandle>(TensorInfo({3, 3, 5, 3}, DataType::Float32));
    layer->m_Bias   = std::make_unique<ScopedTensorHandle>(TensorInfo({9}, DataType::Float32));
    layer->m_Weight->Allocate();
    layer->m_Bias->Allocate();

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    Connect(input, layer, TensorInfo({2, 3, 8, 16}, armnn::DataType::Float32));
    Connect(layer, output, TensorInfo({2, 9, 2, 10}, armnn::DataType::Float32));

    // check the constants that they are not NULL
    CHECK(layer->m_Weight != nullptr);
    CHECK(layer->m_Bias != nullptr);

    // free up the constants..
    layer->ReleaseConstantData();

    // check the constants that they are NULL now
    CHECK(layer->m_Weight == nullptr);
    CHECK(layer->m_Bias == nullptr);
}

TEST_CASE("ReleaseFullyConnectedLayerConstantDataTest")
{
    Graph graph;

    // create the layer we're testing
    FullyConnectedDescriptor layerDesc;
    layerDesc.m_BiasEnabled = true;
    layerDesc.m_TransposeWeightMatrix = true;

    FullyConnectedLayer* const layer = graph.AddLayer<FullyConnectedLayer>(layerDesc, "layer");

    float inputsQScale = 1.0f;
    float outputQScale = 2.0f;

    layer->m_Weight = std::make_unique<ScopedTensorHandle>(TensorInfo({7, 20},
                                                          DataType::QAsymmU8, inputsQScale, 0));
    layer->m_Bias   = std::make_unique<ScopedTensorHandle>(TensorInfo({7},
                                                          GetBiasDataType(DataType::QAsymmU8), inputsQScale));
    layer->m_Weight->Allocate();
    layer->m_Bias->Allocate();

    // create extra layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // connect up
    Connect(input, layer, TensorInfo({3, 1, 4, 5}, DataType::QAsymmU8, inputsQScale));
    Connect(layer, output, TensorInfo({3, 7}, DataType::QAsymmU8, outputQScale));

    // check the constants that they are not NULL
    CHECK(layer->m_Weight != nullptr);
    CHECK(layer->m_Bias != nullptr);

    // free up the constants..
    layer->ReleaseConstantData();

    // check the constants that they are NULL now
    CHECK(layer->m_Weight == nullptr);
    CHECK(layer->m_Bias == nullptr);
}

}

