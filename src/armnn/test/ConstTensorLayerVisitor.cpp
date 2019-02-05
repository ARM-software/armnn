//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConstTensorLayerVisitor.hpp"
#include "Network.hpp"

#include <boost/test/unit_test.hpp>

namespace armnn
{

void TestConvolution2dLayerVisitor::CheckDescriptor(const Convolution2dDescriptor &convolution2dDescriptor)
{
    BOOST_CHECK(m_Descriptor.m_PadLeft == convolution2dDescriptor.m_PadLeft);
    BOOST_CHECK(m_Descriptor.m_PadRight == convolution2dDescriptor.m_PadRight);
    BOOST_CHECK(m_Descriptor.m_PadTop == convolution2dDescriptor.m_PadTop);
    BOOST_CHECK(m_Descriptor.m_PadBottom == convolution2dDescriptor.m_PadBottom);
    BOOST_CHECK(m_Descriptor.m_StrideX == convolution2dDescriptor.m_StrideX);
    BOOST_CHECK(m_Descriptor.m_StrideY == convolution2dDescriptor.m_StrideY);
    BOOST_CHECK(m_Descriptor.m_BiasEnabled == convolution2dDescriptor.m_BiasEnabled);
    BOOST_CHECK(m_Descriptor.m_DataLayout == convolution2dDescriptor.m_DataLayout);
}

void TestDepthwiseConvolution2dLayerVisitor::CheckDescriptor(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor)
{
    BOOST_CHECK(m_Descriptor.m_PadLeft == convolution2dDescriptor.m_PadLeft);
    BOOST_CHECK(m_Descriptor.m_PadRight == convolution2dDescriptor.m_PadRight);
    BOOST_CHECK(m_Descriptor.m_PadTop == convolution2dDescriptor.m_PadTop);
    BOOST_CHECK(m_Descriptor.m_PadBottom == convolution2dDescriptor.m_PadBottom);
    BOOST_CHECK(m_Descriptor.m_StrideX == convolution2dDescriptor.m_StrideX);
    BOOST_CHECK(m_Descriptor.m_StrideY == convolution2dDescriptor.m_StrideY);
    BOOST_CHECK(m_Descriptor.m_BiasEnabled == convolution2dDescriptor.m_BiasEnabled);
    BOOST_CHECK(m_Descriptor.m_DataLayout == convolution2dDescriptor.m_DataLayout);
}

void TestFullyConnectedLayerVistor::CheckDescriptor(const FullyConnectedDescriptor& descriptor)
{
    BOOST_CHECK(m_Descriptor.m_BiasEnabled == descriptor.m_BiasEnabled);
    BOOST_CHECK(m_Descriptor.m_TransposeWeightMatrix == descriptor.m_TransposeWeightMatrix);
}

void TestBatchNormalizationLayerVisitor::CheckDescriptor(const BatchNormalizationDescriptor& descriptor)
{
    BOOST_CHECK(m_Descriptor.m_Eps == descriptor.m_Eps);
    BOOST_CHECK(m_Descriptor.m_DataLayout == descriptor.m_DataLayout);
}

void TestLstmLayerVisitor::CheckDescriptor(const LstmDescriptor& descriptor)
{
    BOOST_CHECK(m_Descriptor.m_ActivationFunc == descriptor.m_ActivationFunc);
    BOOST_CHECK(m_Descriptor.m_ClippingThresCell == descriptor.m_ClippingThresCell);
    BOOST_CHECK(m_Descriptor.m_ClippingThresProj == descriptor.m_ClippingThresProj);
    BOOST_CHECK(m_Descriptor.m_CifgEnabled == descriptor.m_CifgEnabled);
    BOOST_CHECK(m_Descriptor.m_PeepholeEnabled == descriptor.m_PeepholeEnabled);
    BOOST_CHECK(m_Descriptor.m_ProjectionEnabled == descriptor.m_ProjectionEnabled);
}

void TestLstmLayerVisitor::CheckConstTensorPtrs(const std::string& name,
                                                const ConstTensor* expected,
                                                const ConstTensor* actual)
{
    if (expected == nullptr)
    {
        BOOST_TEST(actual == nullptr, name + " actual should have been a nullptr");
    }
    else
    {
        BOOST_TEST(actual != nullptr, name + " actual should have been set");
        if (actual != nullptr)
        {
            CheckConstTensors(*expected, *actual);
        }
    }
}

void TestLstmLayerVisitor::CheckInputParameters(const LstmInputParams& inputParams)
{
    CheckConstTensorPtrs("ProjectionBias", m_InputParams.m_ProjectionBias, inputParams.m_ProjectionBias);
    CheckConstTensorPtrs("ProjectionWeights", m_InputParams.m_ProjectionWeights, inputParams.m_ProjectionWeights);
    CheckConstTensorPtrs("OutputGateBias", m_InputParams.m_OutputGateBias, inputParams.m_OutputGateBias);
    CheckConstTensorPtrs("InputToInputWeights",
        m_InputParams.m_InputToInputWeights, inputParams.m_InputToInputWeights);
    CheckConstTensorPtrs("InputToForgetWeights",
        m_InputParams.m_InputToForgetWeights, inputParams.m_InputToForgetWeights);
    CheckConstTensorPtrs("InputToCellWeights", m_InputParams.m_InputToCellWeights, inputParams.m_InputToCellWeights);
    CheckConstTensorPtrs(
        "InputToOutputWeights", m_InputParams.m_InputToOutputWeights, inputParams.m_InputToOutputWeights);
    CheckConstTensorPtrs(
        "RecurrentToInputWeights", m_InputParams.m_RecurrentToInputWeights, inputParams.m_RecurrentToInputWeights);
    CheckConstTensorPtrs(
        "RecurrentToForgetWeights", m_InputParams.m_RecurrentToForgetWeights, inputParams.m_RecurrentToForgetWeights);
    CheckConstTensorPtrs(
        "RecurrentToCellWeights", m_InputParams.m_RecurrentToCellWeights, inputParams.m_RecurrentToCellWeights);
    CheckConstTensorPtrs(
        "RecurrentToOutputWeights", m_InputParams.m_RecurrentToOutputWeights, inputParams.m_RecurrentToOutputWeights);
    CheckConstTensorPtrs(
        "CellToInputWeights", m_InputParams.m_CellToInputWeights, inputParams.m_CellToInputWeights);
    CheckConstTensorPtrs(
        "CellToForgetWeights", m_InputParams.m_CellToForgetWeights, inputParams.m_CellToForgetWeights);
    CheckConstTensorPtrs(
        "CellToOutputWeights", m_InputParams.m_CellToOutputWeights, inputParams.m_CellToOutputWeights);
    CheckConstTensorPtrs("InputGateBias", m_InputParams.m_InputGateBias, inputParams.m_InputGateBias);
    CheckConstTensorPtrs("ForgetGateBias", m_InputParams.m_ForgetGateBias, inputParams.m_ForgetGateBias);
    CheckConstTensorPtrs("CellBias", m_InputParams.m_CellBias, inputParams.m_CellBias);
}

BOOST_AUTO_TEST_SUITE(TestConstTensorLayerVisitor)

BOOST_AUTO_TEST_CASE(CheckConvolution2dLayer)
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
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    TestConvolution2dLayerVisitor visitor(descriptor, weights);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddConvolution2dLayer(descriptor, weights);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedConvolution2dLayer)
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
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    TestConvolution2dLayerVisitor visitor(descriptor, weights, layerName);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddConvolution2dLayer(descriptor, weights, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckConvolution2dLayerWithBiases)
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
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor biases(TensorInfo(4, biasDimensions.data(), armnn::DataType::Float32), biasData);


    TestConvolution2dWithBiasLayerVisitor visitor(descriptor, weights, biases);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddConvolution2dLayer(descriptor, weights, biases);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedConvolution2dLayerWithBiases)
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
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor biases(TensorInfo(4, biasDimensions.data(), armnn::DataType::Float32), biasData);

    TestConvolution2dWithBiasLayerVisitor visitor(descriptor, weights, biases, layerName);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddConvolution2dLayer(descriptor, weights, biases, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckDepthwiseConvolution2dLayer)
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
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    TestDepthwiseConvolution2dLayerVisitor visitor(descriptor, weights);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddDepthwiseConvolution2dLayer(descriptor, weights);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedDepthwiseConvolution2dLayer)
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
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    TestDepthwiseConvolution2dLayerVisitor visitor(descriptor, weights, layerName);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddDepthwiseConvolution2dLayer(descriptor, weights, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckDepthwiseConvolution2dLayerWithBiases)
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
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor biases(TensorInfo(4, biasDimensions.data(), armnn::DataType::Float32), biasData);

    TestDepthwiseConvolution2dWithBiasLayerVisitor visitor(descriptor, weights, biases);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddDepthwiseConvolution2dLayer(descriptor, weights, biases);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedDepthwiseConvolution2dLayerWithBiases)
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
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor biases(TensorInfo(4, biasDimensions.data(), armnn::DataType::Float32), biasData);

    TestDepthwiseConvolution2dWithBiasLayerVisitor visitor(descriptor, weights, biases, layerName);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddDepthwiseConvolution2dLayer(descriptor, weights, biases, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckFullyConnectedLayer)
{
    FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    TestFullyConnectedLayerVistor visitor(descriptor, weights);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddFullyConnectedLayer(descriptor, weights);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedFullyConnectedLayer)
{
    const char* layerName = "FullyConnectedLayer";
    FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    TestFullyConnectedLayerVistor visitor(descriptor, weights, layerName);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddFullyConnectedLayer(descriptor, weights, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckFullyConnectedLayerWithBiases)
{
    FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;
    descriptor.m_BiasEnabled = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor biases(TensorInfo(4, biasDimensions.data(), armnn::DataType::Float32), biasData);

    TestFullyConnectedLayerWithBiasesVisitor visitor(descriptor, weights, biases);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddFullyConnectedLayer(descriptor, weights, biases);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedFullyConnectedLayerWithBiases)
{
    const char* layerName = "FullyConnectedLayer";
    FullyConnectedDescriptor descriptor;
    descriptor.m_TransposeWeightMatrix = true;
    descriptor.m_BiasEnabled = true;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    armnn::ConstTensor weights(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    std::vector<float> biasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> biasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor biases(TensorInfo(4, biasDimensions.data(), armnn::DataType::Float32), biasData);

    TestFullyConnectedLayerWithBiasesVisitor visitor(descriptor, weights, biases, layerName);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddFullyConnectedLayer(descriptor, weights, biases, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckBatchNormalizationLayer)
{
    BatchNormalizationDescriptor descriptor;
    descriptor.m_Eps = 0.0002f;
    descriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    armnn::ConstTensor mean(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    std::vector<float> varianceData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> varianceDimensions = {1, 1, 3, 3};
    armnn::ConstTensor variance(TensorInfo(4, varianceDimensions.data(), armnn::DataType::Float32), varianceData);

    std::vector<float> betaData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> betaDimensions = {1, 1, 3, 3};
    armnn::ConstTensor beta(TensorInfo(4, betaDimensions.data(), armnn::DataType::Float32), betaData);

    std::vector<float> gammaData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> gammaDimensions = {1, 1, 3, 3};
    armnn::ConstTensor gamma(TensorInfo(4, gammaDimensions.data(), armnn::DataType::Float32), gammaData);

    TestBatchNormalizationLayerVisitor visitor(descriptor, mean, variance, beta, gamma);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddBatchNormalizationLayer(descriptor, mean, variance, beta, gamma);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedBatchNormalizationLayer)
{
    const char* layerName = "BatchNormalizationLayer";
    BatchNormalizationDescriptor descriptor;
    descriptor.m_Eps = 0.0002f;
    descriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    armnn::ConstTensor mean(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    std::vector<float> varianceData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> varianceDimensions = {1, 1, 3, 3};
    armnn::ConstTensor variance(TensorInfo(4, varianceDimensions.data(), armnn::DataType::Float32), varianceData);

    std::vector<float> betaData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> betaDimensions = {1, 1, 3, 3};
    armnn::ConstTensor beta(TensorInfo(4, betaDimensions.data(), armnn::DataType::Float32), betaData);

    std::vector<float> gammaData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> gammaDimensions = {1, 1, 3, 3};
    armnn::ConstTensor gamma(TensorInfo(4, gammaDimensions.data(), armnn::DataType::Float32), gammaData);

    TestBatchNormalizationLayerVisitor visitor(descriptor, mean, variance, beta, gamma, layerName);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddBatchNormalizationLayer(
            descriptor, mean, variance, beta, gamma, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckConstLayer)
{
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    armnn::ConstTensor input(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    TestConstantLayerVisitor visitor(input);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddConstantLayer(input);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedConstLayer)
{
    const char* layerName = "ConstantLayer";
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> dimensions = {1, 1, 3, 3};
    armnn::ConstTensor input(TensorInfo(4, dimensions.data(), armnn::DataType::Float32), data);

    TestConstantLayerVisitor visitor(input, layerName);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddConstantLayer(input, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckLstmLayerBasic)
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), armnn::DataType::Float32), inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), armnn::DataType::Float32), inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), armnn::DataType::Float32), inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToForgetWeights(TensorInfo(
            4, recurrentToForgetWeightsDimensions.data(), armnn::DataType::Float32), recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToCellWeights(TensorInfo(
            4, recurrentToCellWeightsDimensions.data(), armnn::DataType::Float32), recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToOutputWeights(TensorInfo(
            4, recurrentToOutputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor forgetGateBias(TensorInfo(
            4, forgetGateBiasDimensions.data(), armnn::DataType::Float32), forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellBias(TensorInfo(
            4, cellBiasDimensions.data(), armnn::DataType::Float32), cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor outputGateBias(TensorInfo(
            4, outputGateBiasDimensions.data(), armnn::DataType::Float32), outputGateBiasData);

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

    armnn::Network net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedLstmLayerBasic)
{
    const char* layerName = "LstmLayer";
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), armnn::DataType::Float32), inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), armnn::DataType::Float32), inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), armnn::DataType::Float32), inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToForgetWeights(TensorInfo(
            4, recurrentToForgetWeightsDimensions.data(), armnn::DataType::Float32), recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToCellWeights(TensorInfo(
            4, recurrentToCellWeightsDimensions.data(), armnn::DataType::Float32), recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToOutputWeights(TensorInfo(
            4, recurrentToOutputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor forgetGateBias(TensorInfo(
            4, forgetGateBiasDimensions.data(), armnn::DataType::Float32), forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellBias(TensorInfo(
            4, cellBiasDimensions.data(), armnn::DataType::Float32), cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor outputGateBias(TensorInfo(
            4, outputGateBiasDimensions.data(), armnn::DataType::Float32), outputGateBiasData);

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

    armnn::Network net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckLstmLayerCifgDisabled)
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = false; // if this is true then we DON'T need to set the OptCifgParams

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), armnn::DataType::Float32), inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), armnn::DataType::Float32), inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), armnn::DataType::Float32), inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToForgetWeights(TensorInfo(
            4, recurrentToForgetWeightsDimensions.data(), armnn::DataType::Float32), recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToCellWeights(TensorInfo(
            4, recurrentToCellWeightsDimensions.data(), armnn::DataType::Float32), recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToOutputWeights(TensorInfo(
            4, recurrentToOutputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor forgetGateBias(TensorInfo(
            4, forgetGateBiasDimensions.data(), armnn::DataType::Float32), forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellBias(TensorInfo(
            4, cellBiasDimensions.data(), armnn::DataType::Float32), cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor outputGateBias(TensorInfo(
            4, outputGateBiasDimensions.data(), armnn::DataType::Float32), outputGateBiasData);

    std::vector<float> inputToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), armnn::DataType::Float32), inputToInputWeightsData);

    std::vector<float> recurrentToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToInputWeights(TensorInfo(
            4, recurrentToInputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToInputWeightsData);

    std::vector<float> cellToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToInputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellToInputWeights(
            TensorInfo(4, cellToInputWeightsDimensions.data(), armnn::DataType::Float32), cellToInputWeightsData);

    std::vector<float> inputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), armnn::DataType::Float32), inputGateBiasData);

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
    params.m_CellToInputWeights = &cellToInputWeights;
    params.m_InputGateBias = &inputGateBias;

    TestLstmLayerVisitor visitor(descriptor, params);

    armnn::Network net;

    IConnectableLayer* const layer = net.AddLstmLayer(descriptor, params);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedLstmLayerCifgDisabled)
{
    const char* layerName = "LstmLayer";
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = false; // if this is true then we DON'T need to set the OptCifgParams

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), armnn::DataType::Float32), inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), armnn::DataType::Float32), inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), armnn::DataType::Float32), inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToForgetWeights(TensorInfo(
            4, recurrentToForgetWeightsDimensions.data(), armnn::DataType::Float32), recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToCellWeights(TensorInfo(
            4, recurrentToCellWeightsDimensions.data(), armnn::DataType::Float32), recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToOutputWeights(TensorInfo(
            4, recurrentToOutputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor forgetGateBias(TensorInfo(
            4, forgetGateBiasDimensions.data(), armnn::DataType::Float32), forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellBias(TensorInfo(
            4, cellBiasDimensions.data(), armnn::DataType::Float32), cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor outputGateBias(TensorInfo(
            4, outputGateBiasDimensions.data(), armnn::DataType::Float32), outputGateBiasData);

    std::vector<float> inputToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToInputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToInputWeights(
            TensorInfo(4, inputToInputWeightsDimensions.data(), armnn::DataType::Float32), inputToInputWeightsData);

    std::vector<float> recurrentToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToInputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToInputWeights(TensorInfo(
            4, recurrentToInputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToInputWeightsData);

    std::vector<float> cellToInputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToInputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellToInputWeights(
            TensorInfo(4, cellToInputWeightsDimensions.data(), armnn::DataType::Float32), cellToInputWeightsData);

    std::vector<float> inputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputGateBias(
            TensorInfo(4, inputGateBiasDimensions.data(), armnn::DataType::Float32), inputGateBiasData);

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
    params.m_CellToInputWeights = &cellToInputWeights;
    params.m_InputGateBias = &inputGateBias;

    TestLstmLayerVisitor visitor(descriptor, params, layerName);

    armnn::Network net;

    IConnectableLayer *const layer = net.AddLstmLayer(descriptor, params, layerName);
    layer->Accept(visitor);
}

// TODO add one with peephole
BOOST_AUTO_TEST_CASE(CheckLstmLayerPeephole)
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_PeepholeEnabled = true;

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), armnn::DataType::Float32), inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), armnn::DataType::Float32), inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), armnn::DataType::Float32), inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToForgetWeights(TensorInfo(
            4, recurrentToForgetWeightsDimensions.data(), armnn::DataType::Float32), recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToCellWeights(TensorInfo(
            4, recurrentToCellWeightsDimensions.data(), armnn::DataType::Float32), recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToOutputWeights(TensorInfo(
            4, recurrentToOutputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor forgetGateBias(TensorInfo(
            4, forgetGateBiasDimensions.data(), armnn::DataType::Float32), forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellBias(TensorInfo(
            4, cellBiasDimensions.data(), armnn::DataType::Float32), cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor outputGateBias(TensorInfo(
            4, outputGateBiasDimensions.data(), armnn::DataType::Float32), outputGateBiasData);

    std::vector<float> cellToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellToForgetWeights(
            TensorInfo(4, cellToForgetWeightsDimensions.data(), armnn::DataType::Float32), cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellToOutputWeights(
            TensorInfo(4, cellToOutputWeightsDimensions.data(), armnn::DataType::Float32), cellToOutputWeightsData);

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

    armnn::Network net;

    IConnectableLayer *const layer = net.AddLstmLayer(descriptor, params);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedLstmLayerPeephole)
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
    armnn::ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), armnn::DataType::Float32), inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), armnn::DataType::Float32), inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), armnn::DataType::Float32), inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToForgetWeights(TensorInfo(
            4, recurrentToForgetWeightsDimensions.data(), armnn::DataType::Float32), recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToCellWeights(TensorInfo(
            4, recurrentToCellWeightsDimensions.data(), armnn::DataType::Float32), recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToOutputWeights(TensorInfo(
            4, recurrentToOutputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor forgetGateBias(TensorInfo(
            4, forgetGateBiasDimensions.data(), armnn::DataType::Float32), forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellBias(TensorInfo(
            4, cellBiasDimensions.data(), armnn::DataType::Float32), cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor outputGateBias(TensorInfo(
            4, outputGateBiasDimensions.data(), armnn::DataType::Float32), outputGateBiasData);

    std::vector<float> cellToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellToForgetWeights(
            TensorInfo(4, cellToForgetWeightsDimensions.data(), armnn::DataType::Float32), cellToForgetWeightsData);

    std::vector<float> cellToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellToOutputWeights(
            TensorInfo(4, cellToOutputWeightsDimensions.data(), armnn::DataType::Float32), cellToOutputWeightsData);

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

    armnn::Network net;

    IConnectableLayer *const layer = net.AddLstmLayer(descriptor, params, layerName);
    layer->Accept(visitor);
}

// TODO add one with projection
BOOST_AUTO_TEST_CASE(CheckLstmLayerProjection)
{
    LstmDescriptor descriptor;
    descriptor.m_ActivationFunc = 3;
    descriptor.m_ClippingThresProj = 0.5f;
    descriptor.m_ClippingThresCell = 0.3f;
    descriptor.m_CifgEnabled = true; // if this is true then we DON'T need to set the OptCifgParams
    descriptor.m_ProjectionEnabled = true;

    std::vector<float> inputToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), armnn::DataType::Float32), inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), armnn::DataType::Float32), inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), armnn::DataType::Float32), inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToForgetWeights(TensorInfo(
            4, recurrentToForgetWeightsDimensions.data(), armnn::DataType::Float32), recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToCellWeights(TensorInfo(
            4, recurrentToCellWeightsDimensions.data(), armnn::DataType::Float32), recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToOutputWeights(TensorInfo(
            4, recurrentToOutputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor forgetGateBias(TensorInfo(
            4, forgetGateBiasDimensions.data(), armnn::DataType::Float32), forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellBias(TensorInfo(
            4, cellBiasDimensions.data(), armnn::DataType::Float32), cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor outputGateBias(TensorInfo(
            4, outputGateBiasDimensions.data(), armnn::DataType::Float32), outputGateBiasData);

    std::vector<float> projectionBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> projectionBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor projectionBias(
            TensorInfo(4, projectionBiasDimensions.data(), armnn::DataType::Float32), projectionBiasData);

    std::vector<float> projectionWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> projectionWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor projectionWeights(
            TensorInfo(4, projectionWeightsDimensions.data(), armnn::DataType::Float32), projectionWeightsData);

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

    armnn::Network net;

    IConnectableLayer *const layer = net.AddLstmLayer(descriptor, params);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckNamedLstmLayerProjection)
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
    armnn::ConstTensor inputToForgetWeights(
            TensorInfo(4, inputToForgetWeightsDimensions.data(), armnn::DataType::Float32), inputToForgetWeightsData);

    std::vector<float> inputToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToCellWeights(
            TensorInfo(4, inputToCellWeightsDimensions.data(), armnn::DataType::Float32), inputToCellWeightsData);

    std::vector<float> inputToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> inputToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor inputToOutputWeights(
            TensorInfo(4, inputToOutputWeightsDimensions.data(), armnn::DataType::Float32), inputToOutputWeightsData);

    std::vector<float> recurrentToForgetWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToForgetWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToForgetWeights(TensorInfo(
            4, recurrentToForgetWeightsDimensions.data(), armnn::DataType::Float32), recurrentToForgetWeightsData);

    std::vector<float> recurrentToCellWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToCellWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToCellWeights(TensorInfo(
            4, recurrentToCellWeightsDimensions.data(), armnn::DataType::Float32), recurrentToCellWeightsData);

    std::vector<float> recurrentToOutputWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> recurrentToOutputWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor recurrentToOutputWeights(TensorInfo(
            4, recurrentToOutputWeightsDimensions.data(), armnn::DataType::Float32), recurrentToOutputWeightsData);

    std::vector<float> forgetGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> forgetGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor forgetGateBias(TensorInfo(
            4, forgetGateBiasDimensions.data(), armnn::DataType::Float32), forgetGateBiasData);

    std::vector<float> cellBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> cellBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor cellBias(TensorInfo(
            4, cellBiasDimensions.data(), armnn::DataType::Float32), cellBiasData);

    std::vector<float> outputGateBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> outputGateBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor outputGateBias(TensorInfo(
            4, outputGateBiasDimensions.data(), armnn::DataType::Float32), outputGateBiasData);

    std::vector<float> projectionBiasData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> projectionBiasDimensions = {1, 1, 3, 3};
    armnn::ConstTensor projectionBias(
            TensorInfo(4, projectionBiasDimensions.data(), armnn::DataType::Float32), projectionBiasData);

    std::vector<float> projectionWeightsData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned int> projectionWeightsDimensions = {1, 1, 3, 3};
    armnn::ConstTensor projectionWeights(
            TensorInfo(4, projectionWeightsDimensions.data(), armnn::DataType::Float32), projectionWeightsData);

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

    armnn::Network net;

    IConnectableLayer *const layer = net.AddLstmLayer(descriptor, params, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace armnn
