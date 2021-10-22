//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConstTensorLayerVisitor.hpp"
#include "Network.hpp"

#include <doctest/doctest.h>

namespace armnn
{

void TestConvolution2dLayerVisitor::CheckDescriptor(const Convolution2dDescriptor &convolution2dDescriptor)
{
    CHECK(m_Descriptor.m_PadLeft == convolution2dDescriptor.m_PadLeft);
    CHECK(m_Descriptor.m_PadRight == convolution2dDescriptor.m_PadRight);
    CHECK(m_Descriptor.m_PadTop == convolution2dDescriptor.m_PadTop);
    CHECK(m_Descriptor.m_PadBottom == convolution2dDescriptor.m_PadBottom);
    CHECK(m_Descriptor.m_StrideX == convolution2dDescriptor.m_StrideX);
    CHECK(m_Descriptor.m_StrideY == convolution2dDescriptor.m_StrideY);
    CHECK(m_Descriptor.m_BiasEnabled == convolution2dDescriptor.m_BiasEnabled);
    CHECK(m_Descriptor.m_DataLayout == convolution2dDescriptor.m_DataLayout);
}

void TestDepthwiseConvolution2dLayerVisitor::CheckDescriptor(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor)
{
    CHECK(m_Descriptor.m_PadLeft == convolution2dDescriptor.m_PadLeft);
    CHECK(m_Descriptor.m_PadRight == convolution2dDescriptor.m_PadRight);
    CHECK(m_Descriptor.m_PadTop == convolution2dDescriptor.m_PadTop);
    CHECK(m_Descriptor.m_PadBottom == convolution2dDescriptor.m_PadBottom);
    CHECK(m_Descriptor.m_StrideX == convolution2dDescriptor.m_StrideX);
    CHECK(m_Descriptor.m_StrideY == convolution2dDescriptor.m_StrideY);
    CHECK(m_Descriptor.m_BiasEnabled == convolution2dDescriptor.m_BiasEnabled);
    CHECK(m_Descriptor.m_DataLayout == convolution2dDescriptor.m_DataLayout);
}

void TestFullyConnectedLayerVistor::CheckDescriptor(const FullyConnectedDescriptor& descriptor)
{
    CHECK(m_Descriptor.m_BiasEnabled == descriptor.m_BiasEnabled);
    CHECK(m_Descriptor.m_TransposeWeightMatrix == descriptor.m_TransposeWeightMatrix);
}

void TestBatchNormalizationLayerVisitor::CheckDescriptor(const BatchNormalizationDescriptor& descriptor)
{
    CHECK(m_Descriptor.m_Eps == descriptor.m_Eps);
    CHECK(m_Descriptor.m_DataLayout == descriptor.m_DataLayout);
}

void TestLstmLayerVisitor::CheckDescriptor(const LstmDescriptor& descriptor)
{
    CHECK(m_Descriptor.m_ActivationFunc == descriptor.m_ActivationFunc);
    CHECK(m_Descriptor.m_ClippingThresCell == descriptor.m_ClippingThresCell);
    CHECK(m_Descriptor.m_ClippingThresProj == descriptor.m_ClippingThresProj);
    CHECK(m_Descriptor.m_CifgEnabled == descriptor.m_CifgEnabled);
    CHECK(m_Descriptor.m_PeepholeEnabled == descriptor.m_PeepholeEnabled);
    CHECK(m_Descriptor.m_ProjectionEnabled == descriptor.m_ProjectionEnabled);
}

void TestQLstmLayerVisitor::CheckDescriptor(const QLstmDescriptor& descriptor)
{
    CHECK(m_Descriptor.m_CellClip == descriptor.m_CellClip);
    CHECK(m_Descriptor.m_ProjectionClip == descriptor.m_ProjectionClip);
    CHECK(m_Descriptor.m_CifgEnabled == descriptor.m_CifgEnabled);
    CHECK(m_Descriptor.m_PeepholeEnabled == descriptor.m_PeepholeEnabled);
    CHECK(m_Descriptor.m_ProjectionEnabled == descriptor.m_ProjectionEnabled);
}

void TestQuantizedLstmLayerVisitor::CheckInputParameters(const QuantizedLstmInputParams& inputParams)
{
    CheckConstTensorPtrs("InputToInputWeights",
                         m_InputParams.m_InputToInputWeights,
                         inputParams.m_InputToInputWeights);

    CheckConstTensorPtrs("InputToForgetWeights",
                         m_InputParams.m_InputToForgetWeights,
                         inputParams.m_InputToForgetWeights);

    CheckConstTensorPtrs("InputToCellWeights",
                         m_InputParams.m_InputToCellWeights,
                         inputParams.m_InputToCellWeights);

    CheckConstTensorPtrs("InputToOutputWeights",
                         m_InputParams.m_InputToOutputWeights,
                         inputParams.m_InputToOutputWeights);

    CheckConstTensorPtrs("RecurrentToInputWeights",
                         m_InputParams.m_RecurrentToInputWeights,
                         inputParams.m_RecurrentToInputWeights);

    CheckConstTensorPtrs("RecurrentToForgetWeights",
                         m_InputParams.m_RecurrentToForgetWeights,
                         inputParams.m_RecurrentToForgetWeights);

    CheckConstTensorPtrs("RecurrentToCellWeights",
                         m_InputParams.m_RecurrentToCellWeights,
                         inputParams.m_RecurrentToCellWeights);

    CheckConstTensorPtrs("RecurrentToOutputWeights",
                         m_InputParams.m_RecurrentToOutputWeights,
                         inputParams.m_RecurrentToOutputWeights);

    CheckConstTensorPtrs("InputGateBias",  m_InputParams.m_InputGateBias,  inputParams.m_InputGateBias);
    CheckConstTensorPtrs("ForgetGateBias", m_InputParams.m_ForgetGateBias, inputParams.m_ForgetGateBias);
    CheckConstTensorPtrs("CellBias",       m_InputParams.m_CellBias,       inputParams.m_CellBias);
    CheckConstTensorPtrs("OutputGateBias", m_InputParams.m_OutputGateBias, inputParams.m_OutputGateBias);
}

TEST_SUITE("TestConstTensorLayerVisitor")
{
TEST_CASE("CheckConvolution2dLayer")
{
    Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 3;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadTop = 5;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    TestConvolution2dLayerVisitor visitor(descriptor, weights, EmptyOptional());

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddConvolution2dLayer(descriptor, weights, EmptyOptional());
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedConvolution2dLayer")
{
    const char* layerName = "Convolution2dLayer";
    Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 3;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadTop = 5;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    TestConvolution2dLayerVisitor visitor(descriptor, weights, EmptyOptional(), layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddConvolution2dLayer(descriptor, weights, EmptyOptional(), layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckConvolution2dLayerWithBiases")
{
    Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 3;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadTop = 5;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = DataLayout::NHWC;
    descriptor.m_BiasEnabled = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    ConstTensor biases(TensorInfo(4, biasDimensions.data(), DataType::Float32, 0.0f, 0, true), biasData);
    Optional<ConstTensor> optionalBiases(biases);

    TestConvolution2dLayerVisitor visitor(descriptor, weights, optionalBiases);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddConvolution2dLayer(descriptor, weights, optionalBiases);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedConvolution2dLayerWithBiases")
{
    const char* layerName = "Convolution2dLayer";
    Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 3;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadTop = 5;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = DataLayout::NHWC;
    descriptor.m_BiasEnabled = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    ConstTensor biases(TensorInfo(4, biasDimensions.data(), DataType::Float32, 0.0f, 0, true), biasData);
    Optional<ConstTensor> optionalBiases(biases);

    TestConvolution2dLayerVisitor visitor(descriptor, weights, optionalBiases, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddConvolution2dLayer(descriptor, weights, optionalBiases, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckDepthwiseConvolution2dLayer")
{
    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 3;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadTop = 5;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    TestDepthwiseConvolution2dLayerVisitor visitor(descriptor, weights, EmptyOptional());

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddDepthwiseConvolution2dLayer(descriptor, weights, EmptyOptional());
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedDepthwiseConvolution2dLayer")
{
    const char* layerName = "DepthwiseConvolution2dLayer";
    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 3;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadTop = 5;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    TestDepthwiseConvolution2dLayerVisitor visitor(descriptor, weights, EmptyOptional(), layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddDepthwiseConvolution2dLayer(descriptor,
                                                                        weights,
                                                                        EmptyOptional(),
                                                                        layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckDepthwiseConvolution2dLayerWithBiases")
{
    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 3;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadTop = 5;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = DataLayout::NHWC;
    descriptor.m_BiasEnabled = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    ConstTensor biases(TensorInfo(4, biasDimensions.data(), DataType::Float32, 0.0f, 0, true), biasData);
    Optional<ConstTensor> optionalBiases(biases);

    TestDepthwiseConvolution2dLayerVisitor visitor(descriptor, weights, optionalBiases);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddDepthwiseConvolution2dLayer(descriptor, weights, optionalBiases);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedDepthwiseConvolution2dLayerWithBiases")
{
    const char* layerName = "DepthwiseConvolution2dLayer";
    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft = 2;
    descriptor.m_PadRight = 3;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadTop = 5;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = DataLayout::NHWC;
    descriptor.m_BiasEnabled = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    ConstTensor biases(TensorInfo(4, biasDimensions.data(), DataType::Float32, 0.0f, 0, true), biasData);
    Optional<ConstTensor> optionalBiases(biases);

    TestDepthwiseConvolution2dLayerVisitor visitor(descriptor, weights, optionalBiases, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddDepthwiseConvolution2dLayer(descriptor, weights, optionalBiases, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckFullyConnectedLayer")
{
    FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;
    descriptor.m_ConstantWeights = true;
    descriptor.m_BiasEnabled = false;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    TestConstantLayerVisitor weightsVisitor(weights);
    TestFullyConnectedLayerVistor visitor(descriptor);

    NetworkImpl net;

    IConnectableLayer* const weightsLayer = net.AddConstantLayer(weights);
    IConnectableLayer* const layer = net.AddFullyConnectedLayer(descriptor);
    weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));

    weightsLayer->ExecuteStrategy(weightsVisitor);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedFullyConnectedLayer")
{
    const char* layerName = "FullyConnectedLayer";
    FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;
    descriptor.m_ConstantWeights = true;
    descriptor.m_BiasEnabled = false;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    TestConstantLayerVisitor weightsVisitor(weights);
    TestFullyConnectedLayerVistor visitor(descriptor, layerName);

    NetworkImpl net;

    IConnectableLayer* const weightsLayer = net.AddConstantLayer(weights);
    IConnectableLayer* const layer = net.AddFullyConnectedLayer(descriptor, layerName);
    weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));

    weightsLayer->ExecuteStrategy(weightsVisitor);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckFullyConnectedLayerWithBiases")
{
    FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;
    descriptor.m_ConstantWeights = true;
    descriptor.m_BiasEnabled = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    ConstTensor biases(TensorInfo(4, biasDimensions.data(), DataType::Float32, 0.0f, 0, true), biasData);

    TestConstantLayerVisitor weightsVisitor(weights);
    TestConstantLayerVisitor biasesVisitor(biases);
    TestFullyConnectedLayerVistor visitor(descriptor);

    NetworkImpl net;

    IConnectableLayer* const weightsLayer = net.AddConstantLayer(weights);
    IConnectableLayer* const biasesLayer = net.AddConstantLayer(biases);
    IConnectableLayer* const layer = net.AddFullyConnectedLayer(descriptor);
    weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));
    biasesLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2));

    weightsLayer->ExecuteStrategy(weightsVisitor);
    biasesLayer->ExecuteStrategy(biasesVisitor);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedFullyConnectedLayerWithBiases")
{
    const char* layerName = "FullyConnectedLayer";
    FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;
    descriptor.m_ConstantWeights = true;
    descriptor.m_BiasEnabled = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor weights(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    ConstTensor biases(TensorInfo(4, biasDimensions.data(), DataType::Float32, 0.0f, 0, true), biasData);

    TestConstantLayerVisitor weightsVisitor(weights);
    TestConstantLayerVisitor biasesVisitor(biases);
    TestFullyConnectedLayerVistor visitor(descriptor, layerName);

    NetworkImpl net;

    IConnectableLayer* const weightsLayer = net.AddConstantLayer(weights);
    IConnectableLayer* const biasesLayer = net.AddConstantLayer(biases);
    IConnectableLayer* const layer = net.AddFullyConnectedLayer(descriptor, layerName);
    weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));
    biasesLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2));

    weightsLayer->ExecuteStrategy(weightsVisitor);
    biasesLayer->ExecuteStrategy(biasesVisitor);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckBatchNormalizationLayer")
{
    BatchNormalizationDescriptor descriptor;
    descriptor.m_Eps = 0.0002f;
    descriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor mean(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    std::vector<float> varianceData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> varianceDimensions = {1, 1, 3, 3};
    ConstTensor variance(TensorInfo(4, varianceDimensions.data(), DataType::Float32, 0.0f, 0, true), varianceData);

    std::vector<float> betaData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> betaDimensions = {1, 1, 3, 3};
    ConstTensor beta(TensorInfo(4, betaDimensions.data(), DataType::Float32, 0.0f, 0, true), betaData);

    std::vector<float> gammaData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> gammaDimensions = {1, 1, 3, 3};
    ConstTensor gamma(TensorInfo(4, gammaDimensions.data(), DataType::Float32, 0.0f, 0, true), gammaData);

    TestBatchNormalizationLayerVisitor visitor(descriptor, mean, variance, beta, gamma);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddBatchNormalizationLayer(descriptor, mean, variance, beta, gamma);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedBatchNormalizationLayer")
{
    const char* layerName = "BatchNormalizationLayer";
    BatchNormalizationDescriptor descriptor;
    descriptor.m_Eps = 0.0002f;
    descriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor mean(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    std::vector<float> varianceData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> varianceDimensions = {1, 1, 3, 3};
    ConstTensor variance(TensorInfo(4, varianceDimensions.data(), DataType::Float32, 0.0f, 0, true), varianceData);

    std::vector<float> betaData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> betaDimensions = {1, 1, 3, 3};
    ConstTensor beta(TensorInfo(4, betaDimensions.data(), DataType::Float32, 0.0f, 0, true), betaData);

    std::vector<float> gammaData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> gammaDimensions = {1, 1, 3, 3};
    ConstTensor gamma(TensorInfo(4, gammaDimensions.data(), DataType::Float32, 0.0f, 0, true), gammaData);

    TestBatchNormalizationLayerVisitor visitor(descriptor, mean, variance, beta, gamma, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddBatchNormalizationLayer(
            descriptor, mean, variance, beta, gamma, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckConstLayer")
{
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor input(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    TestConstantLayerVisitor visitor(input);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddConstantLayer(input);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedConstLayer")
{
    const char* layerName = "ConstantLayer";
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    ConstTensor input(TensorInfo(4, dimensions.data(), DataType::Float32, 0.0f, 0, true), data);

    TestConstantLayerVisitor visitor(input, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddConstantLayer(input, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckLstmLayerBasic")
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    TestLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedLstmLayerBasic")
{
    const char* layerName = "LstmLayer";
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    TestLstmLayerVisitor visitor(descriptor, params, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckLstmLayerCifgDisabled")
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = false; // if this is true then we DON'T need to set the OptCifgParams

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    std::vector<float> inputToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToInputWeightsData);

    std::vector<float> recurrentToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToInputWeights(
            TensorInfo(4, recurrentToInputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToInputWeightsData);

    std::vector<float> inputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputGateBiasData);

    LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    params.m_InputToInputWeights = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_InputGateBias = &inputGateBias;

    TestLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedLstmLayerCifgDisabled")
{
    const char* layerName = "LstmLayer";
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = false; // if this is true then we DON'T need to set the OptCifgParams

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    std::vector<float> inputToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToInputWeightsData);

    std::vector<float> recurrentToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToInputWeights(
            TensorInfo(4, recurrentToInputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToInputWeightsData);

    std::vector<float> inputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputGateBiasData);

    LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    params.m_InputToInputWeights = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_InputGateBias = &inputGateBias;

    TestLstmLayerVisitor visitor(descriptor, params, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params, layerName);
    layer->ExecuteStrategy(visitor);
}

// TODO add one with peephole
TEST_CASE("CheckLstmLayerPeephole")
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_PeepholeEnabled = true;

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    std::vector<float> cellToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToForgetWeights(
            TensorInfo(4, cellToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToOutputWeights(
            TensorInfo(4, cellToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellToOutputWeightsData);

    LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    TestLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckLstmLayerPeepholeCifgDisabled")
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = false;
    descriptor.m_PeepholeEnabled = true;

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    std::vector<float> cellToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToInputWeights(
            TensorInfo(4, cellToInputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellToInputWeightsData);

    std::vector<float> cellToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToForgetWeights(
            TensorInfo(4, cellToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToOutputWeights(
            TensorInfo(4, cellToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellToOutputWeightsData);

    std::vector<float> inputToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToInputWeightsData);

    std::vector<float> recurrentToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToInputWeights(
            TensorInfo(4, recurrentToInputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToInputWeightsData);

    std::vector<float> inputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputGateBiasData);

    LstmInputParams params;
    // Basic params
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    // Peephole params
    params.m_CellToInputWeights  = &cellToInputWeights;
    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    // Cifg params
    params.m_InputToInputWeights = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_InputGateBias = &inputGateBias;

    TestLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedLstmLayerPeephole")
{
    const char* layerName = "LstmLayer";
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_PeepholeEnabled = true;

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    std::vector<float> cellToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToForgetWeights(
            TensorInfo(4, cellToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToOutputWeights(
            TensorInfo(4, cellToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellToOutputWeightsData);

    LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    TestLstmLayerVisitor visitor(descriptor, params, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params, layerName);
    layer->ExecuteStrategy(visitor);
}

// TODO add one with projection
TEST_CASE("CheckLstmLayerProjection")
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_ProjectionEnabled = true;

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    std::vector<float> projectionBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> projectionBiasDimensions = {1, 1, 3, 3};
    ConstTensor projectionBias(
            TensorInfo(4, projectionBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            projectionBiasData);

    std::vector<float> projectionWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> projectionWeightsDimensions = {1, 1, 3, 3};
    ConstTensor projectionWeights(
            TensorInfo(4, projectionWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            projectionWeightsData);

    LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    params.m_ProjectionWeights = &projectionWeights;
    params.m_ProjectionBias = &projectionBias;

    TestLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedLstmLayerProjection")
{
    const char* layerName = "LstmLayer";
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_ProjectionEnabled = true;

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            outputGateBiasData);

    std::vector<float> projectionBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> projectionBiasDimensions = {1, 1, 3, 3};
    ConstTensor projectionBias(
            TensorInfo(4, projectionBiasDimensions.data(), DataType::Float32, 0.0f, 0, true),
            projectionBiasData);

    std::vector<float> projectionWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> projectionWeightsDimensions = {1, 1, 3, 3};
    ConstTensor projectionWeights(
            TensorInfo(4, projectionWeightsDimensions.data(), DataType::Float32, 0.0f, 0, true),
            projectionWeightsData);

    LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    params.m_ProjectionWeights = &projectionWeights;
    params.m_ProjectionBias = &projectionBias;

    TestLstmLayerVisitor visitor(descriptor, params, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckQLstmLayerBasic")
{
    QLstmDescriptor descriptor;
    descriptor.m_ProjectionClip = 0.5f;
    descriptor.m_CellClip = 0.3f;
    descriptor.m_CifgEnabled = true;

    // Basic params ONLY
    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    LstmInputParams params;
    params.m_InputToForgetWeights     = &inputToForgetWeights;
    params.m_InputToCellWeights       = &inputToCellWeights;
    params.m_InputToOutputWeights     = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias           = &forgetGateBias;
    params.m_CellBias                 = &cellBias;
    params.m_OutputGateBias           = &outputGateBias;

    TestQLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedQLstmLayerBasic")
{
    const char* layerName = "QLstmLayer";
    QLstmDescriptor descriptor;
    descriptor.m_ProjectionClip = 0.5f;
    descriptor.m_CellClip = 0.3f;
    descriptor.m_CifgEnabled = true;

    // Basic params ONLY
    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    LstmInputParams params;
    params.m_InputToForgetWeights     = &inputToForgetWeights;
    params.m_InputToCellWeights       = &inputToCellWeights;
    params.m_InputToOutputWeights     = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias           = &forgetGateBias;
    params.m_CellBias                 = &cellBias;
    params.m_OutputGateBias           = &outputGateBias;

    TestQLstmLayerVisitor visitor(descriptor, params, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQLstmLayer(descriptor, params, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckQLstmLayerCifgDisabled")
{
    QLstmDescriptor descriptor;
    descriptor.m_ProjectionClip = 0.5f;
    descriptor.m_CellClip = 0.3f;
    descriptor.m_CifgEnabled = false;

    // Basic params
    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    // CIFG disabled params
    std::vector<uint8_t> inputToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToInputWeightsData);

    std::vector<uint8_t> recurrentToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToInputWeights(
            TensorInfo(4, recurrentToInputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToInputWeightsData);

    std::vector<int32_t> inputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            inputGateBiasData);

    LstmInputParams params;

    // Basic params
    params.m_InputToForgetWeights     = &inputToForgetWeights;
    params.m_InputToCellWeights       = &inputToCellWeights;
    params.m_InputToOutputWeights     = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias           = &forgetGateBias;
    params.m_CellBias                 = &cellBias;
    params.m_OutputGateBias           = &outputGateBias;

    // CIFG disabled params
    params.m_InputToInputWeights     = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_InputGateBias           = &inputGateBias;

    TestQLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckQLstmLayerCifgDisabledPeepholeEnabled")
{
    QLstmDescriptor descriptor;
    descriptor.m_ProjectionClip = 0.5f;
    descriptor.m_CellClip = 0.3f;
    descriptor.m_CifgEnabled = false;
    descriptor.m_PeepholeEnabled = true;

    // Basic params
    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    // CIFG disabled params
    std::vector<uint8_t> inputToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToInputWeightsData);

    std::vector<uint8_t> recurrentToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToInputWeights(
            TensorInfo(4, recurrentToInputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToInputWeightsData);

    std::vector<int32_t> inputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            inputGateBiasData);

    // Peephole enabled, CIFG disabled params
    std::vector<int16_t> cellToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToInputWeights(
            TensorInfo(4, cellToInputWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            cellToInputWeightsData);

    std::vector<int16_t> cellToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToForgetWeights(
            TensorInfo(4, cellToForgetWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            cellToForgetWeightsData);

    std::vector<int16_t> cellToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToOutputWeights(
            TensorInfo(4, cellToOutputWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            cellToOutputWeightsData);

    LstmInputParams params;

    // Basic params
    params.m_InputToForgetWeights     = &inputToForgetWeights;
    params.m_InputToCellWeights       = &inputToCellWeights;
    params.m_InputToOutputWeights     = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias           = &forgetGateBias;
    params.m_CellBias                 = &cellBias;
    params.m_OutputGateBias           = &outputGateBias;

    // CIFG disabled params
    params.m_InputToInputWeights     = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_InputGateBias           = &inputGateBias;

    // Peephole enabled, CIFG disabled params
    params.m_CellToInputWeights  = &cellToInputWeights;
    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    TestQLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckQLstmLayerCifgEnabledPeepholeEnabled")
{
    QLstmDescriptor descriptor;
    descriptor.m_ProjectionClip = 0.5f;
    descriptor.m_CellClip = 0.3f;
    descriptor.m_CifgEnabled = true;
    descriptor.m_PeepholeEnabled = true;

    // Basic params
    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    // Peephole enabled and CIFG enabled params
    std::vector<int16_t> cellToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToForgetWeights(
            TensorInfo(4, cellToForgetWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            cellToForgetWeightsData);

    std::vector<int16_t> cellToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellToOutputWeights(
            TensorInfo(4, cellToOutputWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            cellToOutputWeightsData);

    LstmInputParams params;

    // Basic params
    params.m_InputToForgetWeights     = &inputToForgetWeights;
    params.m_InputToCellWeights       = &inputToCellWeights;
    params.m_InputToOutputWeights     = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias           = &forgetGateBias;
    params.m_CellBias                 = &cellBias;
    params.m_OutputGateBias           = &outputGateBias;

    // Peephole enabled and CIFG enabled params
    params.m_CellToForgetWeights = &cellToForgetWeights;
    params.m_CellToOutputWeights = &cellToOutputWeights;

    TestQLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckQLstmLayerProjectionEnabled")
{
    QLstmDescriptor descriptor;
    descriptor.m_ProjectionClip = 0.5f;
    descriptor.m_CellClip = 0.3f;
    descriptor.m_CifgEnabled = true;
    descriptor.m_ProjectionEnabled = true;

    // Basic params ONLY
    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    // Projection enabled params
    std::vector<uint8_t> projectionWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> projectionWeightsDimensions = {1, 1, 3, 3};
    ConstTensor projectionWeights(
            TensorInfo(4, projectionWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            projectionWeightsData);

    std::vector<int32_t> projectionBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> projectionBiasDimensions = {1, 1, 3, 3};
    ConstTensor projectionBias(
            TensorInfo(4, projectionBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            projectionBiasData);

    LstmInputParams params;

    // Basic params
    params.m_InputToForgetWeights     = &inputToForgetWeights;
    params.m_InputToCellWeights       = &inputToCellWeights;
    params.m_InputToOutputWeights     = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias           = &forgetGateBias;
    params.m_CellBias                 = &cellBias;
    params.m_OutputGateBias           = &outputGateBias;

    // Projection enabled params
    params.m_ProjectionWeights = &projectionWeights;
    params.m_ProjectionBias    = &projectionBias;

    TestQLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckQLstmLayerCifgDisabledLayerNormEnabled")
{
    QLstmDescriptor descriptor;
    descriptor.m_ProjectionClip = 0.5f;
    descriptor.m_CellClip = 0.3f;
    descriptor.m_CifgEnabled = false;
    descriptor.m_LayerNormEnabled = true;

    // Basic params
    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToOutputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToOutputWeightsData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    // CIFG disabled params
    std::vector<uint8_t> inputToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToInputWeightsData);

    std::vector<uint8_t> recurrentToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToInputWeights(
            TensorInfo(4, recurrentToInputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToInputWeightsData);

    std::vector<int32_t> inputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            inputGateBiasData);

    // Layer Norm enabled, CIFG disabled params
    std::vector<int16_t> inputLayerNormWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputLayerNormWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputLayerNormWeights(
            TensorInfo(4, inputLayerNormWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            inputLayerNormWeightsData);

    std::vector<int16_t> forgetLayerNormWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetLayerNormWeightsDimensions = {1, 1, 3, 3};
    ConstTensor forgetLayerNormWeights(
            TensorInfo(4, forgetLayerNormWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            forgetLayerNormWeightsData);

    std::vector<int16_t> cellLayerNormWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellLayerNormWeightsDimensions = {1, 1, 3, 3};
    ConstTensor cellLayerNormWeights(
            TensorInfo(4, cellLayerNormWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            cellLayerNormWeightsData);

    std::vector<int16_t> outputLayerNormWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputLayerNormWeightsDimensions = {1, 1, 3, 3};
    ConstTensor outputLayerNormWeights(
            TensorInfo(4, outputLayerNormWeightsDimensions.data(), DataType::QSymmS16, 0.0f, 0, true),
            outputLayerNormWeightsData);

    LstmInputParams params;

    // Basic params
    params.m_InputToForgetWeights     = &inputToForgetWeights;
    params.m_InputToCellWeights       = &inputToCellWeights;
    params.m_InputToOutputWeights     = &inputToOutputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
    params.m_ForgetGateBias           = &forgetGateBias;
    params.m_CellBias                 = &cellBias;
    params.m_OutputGateBias           = &outputGateBias;

    // CIFG disabled params
    params.m_InputToInputWeights     = &inputToInputWeights;
    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_InputGateBias           = &inputGateBias;

    // Layer Norm enabled, CIFG disabled params
    params.m_InputLayerNormWeights  = &inputLayerNormWeights;
    params.m_ForgetLayerNormWeights = &forgetLayerNormWeights;
    params.m_CellLayerNormWeights   = &cellLayerNormWeights;
    params.m_OutputLayerNormWeights = &outputLayerNormWeights;

    TestQLstmLayerVisitor visitor(descriptor, params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQLstmLayer(descriptor, params);
    layer->ExecuteStrategy(visitor);
}


TEST_CASE("CheckQuantizedLstmLayer")
{
    std::vector<uint8_t> inputToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToInputWeightsData);

    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            inputToOutputWeightsData);


    std::vector<uint8_t> recurrentToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToInputWeights(
            TensorInfo(4, recurrentToInputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToInputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QSymmS8, 0.0f, 0, true),
            recurrentToOutputWeightsData);


    std::vector<int32_t> inputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            inputGateBiasData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    QuantizedLstmInputParams params;

    params.m_InputToInputWeights = &inputToInputWeights;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;

    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;

    params.m_InputGateBias = &inputGateBias;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    TestQuantizedLstmLayerVisitor visitor(params);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQuantizedLstmLayer(params);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckNamedQuantizedLstmLayer")
{
    const char* layerName = "LstmLayer";
    std::vector<uint8_t> inputToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), DataType::QAsymmU8, 0.0f, 0, true),
            inputToInputWeightsData);

    std::vector<uint8_t> inputToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), DataType::QAsymmU8, 0.0f, 0, true),
            inputToForgetWeightsData);

    std::vector<uint8_t> inputToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), DataType::QAsymmU8, 0.0f, 0, true),
            inputToCellWeightsData);

    std::vector<uint8_t> inputToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), DataType::QAsymmU8, 0.0f, 0, true),
            inputToOutputWeightsData);


    std::vector<uint8_t> recurrentToInputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToInputWeights(
            TensorInfo(4, recurrentToInputWeightsDimensions.data(), DataType::QAsymmU8, 0.0f, 0, true),
            recurrentToInputWeightsData);

    std::vector<uint8_t> recurrentToForgetWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToForgetWeights(
            TensorInfo(4, recurrentToForgetWeightsDimensions.data(), DataType::QAsymmU8, 0.0f, 0, true),
            recurrentToForgetWeightsData);

    std::vector<uint8_t> recurrentToCellWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToCellWeights(
            TensorInfo(4, recurrentToCellWeightsDimensions.data(), DataType::QAsymmU8, 0.0f, 0, true),
            recurrentToCellWeightsData);

    std::vector<uint8_t> recurrentToOutputWeightsData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    ConstTensor recurrentToOutputWeights(
            TensorInfo(4, recurrentToOutputWeightsDimensions.data(), DataType::QAsymmU8, 0.0f, 0, true),
            recurrentToOutputWeightsData);


    std::vector<int32_t> inputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            inputGateBiasData);

    std::vector<int32_t> forgetGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor forgetGateBias(
            TensorInfo(4, forgetGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            forgetGateBiasData);

    std::vector<int32_t> cellBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    ConstTensor cellBias(
            TensorInfo(4, cellBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            cellBiasData);

    std::vector<int32_t> outputGateBiasData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    ConstTensor outputGateBias(
            TensorInfo(4, outputGateBiasDimensions.data(), DataType::Signed32, 0.0f, 0, true),
            outputGateBiasData);

    QuantizedLstmInputParams params;

    params.m_InputToInputWeights = &inputToInputWeights;
    params.m_InputToForgetWeights = &inputToForgetWeights;
    params.m_InputToCellWeights = &inputToCellWeights;
    params.m_InputToOutputWeights = &inputToOutputWeights;

    params.m_RecurrentToInputWeights = &recurrentToInputWeights;
    params.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
    params.m_RecurrentToCellWeights = &recurrentToCellWeights;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeights;

    params.m_InputGateBias = &inputGateBias;
    params.m_ForgetGateBias = &forgetGateBias;
    params.m_CellBias = &cellBias;
    params.m_OutputGateBias = &outputGateBias;

    TestQuantizedLstmLayerVisitor visitor(params, layerName);

    NetworkImpl net;

    IConnectableLayer* const layer = net.AddQuantizedLstmLayer(params, layerName);
    layer->ExecuteStrategy(visitor);
}

}

} // namespace armnn
