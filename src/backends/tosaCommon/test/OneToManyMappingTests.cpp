//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AvgPool2DIgnoreValueChecker.hpp"
#include "FullyConnectedChecker.hpp"
#include "QuantizeChecker.hpp"
#include "SplitChecker.hpp"
#include "CommonTestUtils.hpp"

#include <armnn/IRuntime.hpp>

using namespace armnn;
using namespace tosa;

TEST_SUITE("TosaOperatorMappingOneToManyTests")
{
TEST_CASE("GetTosaMapping_AvgPool2DIgnoreValueLayer")
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape         = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> intermediateShape  = {{ 1, 1, 6, 6 }};
    std::vector<std::vector<int32_t>> outputShape        = {{ 1, 1, 3, 3 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(nullptr, LayerType::Pooling2d, {&inputTensorInfo}, {&outputTensorInfo}, descriptor);
    VerifyAvgPool2DIgnoreValue(basicBlock,
                               inputShape,
                               outputShape,
                               intermediateShape,
                               descriptor);
}

TEST_CASE("GetTosaMappingFromLayer_AvgPool2DIgnoreValueLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    IConnectableLayer* input0  = net->AddInputLayer(0, "input0");
    IConnectableLayer* pool    = net->AddPooling2dLayer(descriptor, "pool");
    IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(pool->GetInputSlot(0));
    pool->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape         = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> intermediateShape  = {{ 1, 1, 6, 6 }};
    std::vector<std::vector<int32_t>> outputShape        = {{ 1, 1, 3, 3 }};

    input0->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    pool->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    TosaSerializationBasicBlock* basicBlock = GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(pool));
    VerifyAvgPool2DIgnoreValue(basicBlock,
                              inputShape,
                              outputShape,
                              intermediateShape,
                              descriptor);
}

TEST_CASE("GetTosaMapping_FullyConnectedLayer")
{
    FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = true;

    constexpr static unsigned int inputWidth = 3u;
    constexpr static unsigned int inputHeight = 2u;
    constexpr static unsigned int inputChannels = 1u;
    constexpr static unsigned int inputSize = inputWidth * inputHeight * inputChannels;
    constexpr static unsigned int outputChannels = 2u;

    const armnn::TensorInfo inputInfo({ 1, inputChannels, inputHeight, inputWidth }, DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, outputChannels }, DataType::Float32);
    const armnn::TensorInfo weightsInfo({ outputChannels, inputSize }, DataType::Float32, 1.0, 0, true);
    const armnn::TensorInfo biasesInfo({ outputChannels }, DataType::Float32, 1.0, 0, true);

    std::vector<std::vector<int32_t>> inputShapes  = {{ 1, inputChannels, inputHeight, inputWidth },
                                                      { outputChannels, inputSize },
                                                      { outputChannels }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, outputChannels }};


    TosaSerializationBasicBlock* basicBlock = GetTosaMapping(nullptr,
                                                             LayerType::FullyConnected,
                                                             {&inputInfo, &weightsInfo, &biasesInfo},
                                                             {&outputInfo},
                                                             descriptor);

    VerifyFullyConnected(basicBlock,
                         inputShapes,
                         outputShape,
                         descriptor);
}

TEST_CASE("GetTosaMappingFromLayer_FullyConnectedLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = true;

    constexpr static unsigned int inputWidth = 3u;
    constexpr static unsigned int inputHeight = 2u;
    constexpr static unsigned int inputChannels = 1u;
    constexpr static unsigned int inputSize = inputWidth * inputHeight * inputChannels;
    constexpr static unsigned int outputChannels = 2u;

    const armnn::TensorInfo inputInfo({ 1, inputChannels, inputHeight, inputWidth }, DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, outputChannels }, DataType::Float32);
    const armnn::TensorInfo weightsInfo({ outputChannels, inputSize }, DataType::Float32, 1.0, 0, true);
    const armnn::TensorInfo biasesInfo({ outputChannels }, DataType::Float32, 1.0, 0, true);

    std::vector<std::vector<int32_t>> inputShapes = {{ 1, inputChannels, inputHeight, inputWidth }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, outputChannels }};

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    ConstTensor weights(weightsInfo, weightsData);

    std::vector<float> biasesData = GenerateRandomData<float>(biasesInfo.GetNumElements());
    ConstTensor biases(biasesInfo, biasesData);

    IConnectableLayer* const inputLayer  = net->AddInputLayer(0, "input");
    IConnectableLayer* const weightsLayer = net->AddConstantLayer(weights, "weights");
    IConnectableLayer* const biasesLayer = net->AddConstantLayer(biases, "biases");
    IConnectableLayer* const fullyConnectedLayer = net->AddFullyConnectedLayer(descriptor, "fully_connected");
    IConnectableLayer* const outputLayer = net->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    biasesLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(2));
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);
    biasesLayer->GetOutputSlot(0).SetTensorInfo(biasesInfo);
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    TosaSerializationBasicBlock* basicBlock = GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(fullyConnectedLayer));

    VerifyFullyConnected(basicBlock,
                         inputShapes,
                         outputShape,
                         descriptor);
}

TEST_CASE("GetTosaMapping_QuantizeLayer")
{
    NullDescriptor descriptor;
    DataType outputDataType = DataType::Signed32;

    TensorInfo inputTensorInfo({ 1, 3, 3, 1 }, DataType::Float32);
    TensorInfo outputTensorInfo({ 1, 3, 3, 1 }, outputDataType);
    std::vector<int32_t> shape = { 1, 3, 3, 1 };

    TosaSerializationBasicBlock* basicBlock =
            GetTosaMapping(nullptr, LayerType::Quantize, {&inputTensorInfo}, {&outputTensorInfo}, descriptor);
    VerifyQuantize(basicBlock, shape, ArmNNToDType(DataType::Float32), ArmNNToDType(outputDataType));
}

TEST_CASE("GetTosaMappingFromLayer_QuantizeLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());
    NullDescriptor descriptor;
    DataType outputDataType = DataType::Signed32;

    IConnectableLayer* input0   = net->AddInputLayer(0, "input0");
    IConnectableLayer* quantize = net->AddQuantizeLayer("quantize");
    IConnectableLayer* output   = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(quantize->GetInputSlot(0));
    quantize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorInfo inputTensorInfo({ 1, 3, 3, 1 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 3, 3, 1 }, outputDataType);
    std::vector<int32_t> shape = { 1, 3, 3, 1 };

    input0->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    quantize->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    TosaSerializationBasicBlock* basicBlock = GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(quantize));
    VerifyQuantize(basicBlock, shape, ArmNNToDType(DataType::Float32), ArmNNToDType(outputDataType));
}

TEST_CASE("GetTosaMapping_SplitLayer")
{
    const unsigned int numViews = 3;
    const unsigned int numDimensions = 4;
    armnn::ViewsDescriptor descriptor(numViews, numDimensions);
    descriptor.SetAxis(static_cast<int32_t>(1));

    std::vector<std::vector<int32_t>> inShape  = {{ 1, 18, 4, 4 }};
    std::vector<std::vector<int32_t>> outShape = {{ 1, 6, 4, 4 },{ 1, 6, 4, 4 },{ 1, 6, 4, 4 }};

    armnn::TensorInfo inputTensorInfo({1, 18, 4, 4}, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({1, 6, 4, 4}, DataType::Float32);

    TosaSerializationBasicBlock* basicBlock = GetTosaMapping(nullptr,
                                                             LayerType::Splitter,
                                                             {&inputTensorInfo},
                                                             {&outputTensorInfo, &outputTensorInfo, &outputTensorInfo},
                                                             descriptor);

    VerifySplit(basicBlock,
                inShape,
                outShape,
                descriptor);
}

TEST_CASE("GetTosaMappingFromLayer_SplitLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    const unsigned int numViews = 3;
    const unsigned int numDimensions = 4;
    armnn::ViewsDescriptor descriptor(numViews, numDimensions);
    descriptor.SetAxis(static_cast<int32_t>(1));

    std::vector<std::vector<int32_t>> inShape  = {{ 1, 18, 4, 4 }};
    std::vector<std::vector<int32_t>> outShape = {{ 1, 6, 4, 4 },{ 1, 6, 4, 4 },{ 1, 6, 4, 4 }};

    IConnectableLayer* input0   = net->AddInputLayer(0, "input0");
    IConnectableLayer* split    = net->AddSplitterLayer(descriptor, "split");
    IConnectableLayer* output0  = net->AddOutputLayer(0, "output0");
    IConnectableLayer* output1  = net->AddOutputLayer(1, "output1");
    IConnectableLayer* output2  = net->AddOutputLayer(2, "output2");

    input0->GetOutputSlot(0).Connect(split->GetInputSlot(0));
    split->GetOutputSlot(0).Connect(output0->GetInputSlot(0));
    split->GetOutputSlot(1).Connect(output1->GetInputSlot(0));
    split->GetOutputSlot(2).Connect(output2->GetInputSlot(0));

    armnn::TensorInfo inputTensorInfo({1, 18, 4, 4}, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({1, 6, 4, 4}, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    split->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    split->GetOutputSlot(1).SetTensorInfo(outputTensorInfo);
    split->GetOutputSlot(2).SetTensorInfo(outputTensorInfo);

    TosaSerializationBasicBlock* basicBlock = GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(split));

    VerifySplit(basicBlock,
                inShape,
                outShape,
                descriptor);
}

}
