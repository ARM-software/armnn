//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>
#include <ResolveType.hpp>
#include <armnnUtils/QuantizeHelper.hpp>

#include <armnn/INetwork.hpp>

#include <doctest/doctest.h>

using namespace armnn;

namespace
{
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void MulMaxMinTest(Compute backendId, size_t numLayers)
{
    const TensorInfo input0TensorInfo({ 1, 2, 2, 3 },
                                      ArmnnType,
                                      IsQuantizedType<T>() ? 0.25f : 1,
                                      IsQuantizedType<T>() ? 10 : 0,
                                      true);
    const TensorInfo input1TensorInfo({ 1, 1, 1, 1 },
                                      ArmnnType,
                                      IsQuantizedType<T>() ? 0.25f : 1,
                                      IsQuantizedType<T>() ? 11 : 0,
                                      true);
    const TensorInfo maxInput1TensorInfo({ 1, 1, 1, 1 },
                                         ArmnnType,
                                         IsQuantizedType<T>() ? 0.25f : 1,
                                         IsQuantizedType<T>() ? 12 : 0,
                                         true);
    const TensorInfo minInput1TensorInfo({ 1, 1, 1, 1 },
                                         ArmnnType,
                                         IsQuantizedType<T>() ? 0.25f : 1,
                                         IsQuantizedType<T>() ? 13 : 0,
                                         true);
    const TensorInfo outputTensorInfo({ 1, 2, 2, 3 },
                                      ArmnnType,
                                      IsQuantizedType<T>() ? 0.5f : 1,
                                      IsQuantizedType<T>() ? 14 : 0);

    std::vector<float> input0Data
    {
         0.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -2.0f, -2.0f, -2.0f
    };
    std::vector<float> input1Data
    {
         1.0f
    };
    std::vector<float> maxInput1Data
    {
         -100.0f
    };
    std::vector<float> minInput1Data
    {
         100.0f
    };
    std::vector<float> outputExpectedData =
    {
         0.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -2.0f, -2.0f, -2.0f
    };

    std::vector<T> input0 = armnnUtils::QuantizedVector<T>(input0Data,
                                                           input0TensorInfo.GetQuantizationScale(),
                                                           input0TensorInfo.GetQuantizationOffset());
    std::vector<T> input1 = armnnUtils::QuantizedVector<T>(input1Data,
                                                           input1TensorInfo.GetQuantizationScale(),
                                                           input1TensorInfo.GetQuantizationOffset());
    std::vector<T> maxInput1 = armnnUtils::QuantizedVector<T>(maxInput1Data,
                                                              maxInput1TensorInfo.GetQuantizationScale(),
                                                              maxInput1TensorInfo.GetQuantizationOffset());
    std::vector<T> minInput1 = armnnUtils::QuantizedVector<T>(minInput1Data,
                                                              minInput1TensorInfo.GetQuantizationScale(),
                                                              minInput1TensorInfo.GetQuantizationOffset());
    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>(outputExpectedData,
                                                                   outputTensorInfo.GetQuantizationScale(),
                                                                   outputTensorInfo.GetQuantizationOffset());
    std::vector<T> outputActual(outputTensorInfo.GetNumElements());

    // Create a network
    INetworkPtr network = INetwork::Create();

    // add layers to network
    IConnectableLayer* const input0Layer = network->AddInputLayer(0);
    IConnectableLayer* const input1Layer = network->AddInputLayer(1);
    IConnectableLayer* const mulLayer = network->AddElementwiseBinaryLayer(BinaryOperation::Mul, "mul");

    auto constMaxInput1Tensor = ConstTensor(maxInput1TensorInfo, maxInput1);
    IConnectableLayer* const maxInput1Layer = network->AddConstantLayer(constMaxInput1Tensor, "maxInput1");
    IConnectableLayer* const maxLayer = network->AddElementwiseBinaryLayer(BinaryOperation::Maximum, "max");

    auto constMinInput1Tensor = ConstTensor(minInput1TensorInfo, minInput1);
    IConnectableLayer* const minInput1Layer = network->AddConstantLayer(constMinInput1Tensor, "minInput1");
    IConnectableLayer* const minLayer = network->AddElementwiseBinaryLayer(BinaryOperation::Minimum, "min");

    IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    // set tensor info to output slots
    input0Layer->GetOutputSlot(0).SetTensorInfo(input0TensorInfo);
    input1Layer->GetOutputSlot(0).SetTensorInfo(input1TensorInfo);
    mulLayer->GetOutputSlot(0).SetTensorInfo(input0TensorInfo);
    maxInput1Layer->GetOutputSlot(0).SetTensorInfo(maxInput1TensorInfo);
    maxLayer->GetOutputSlot(0).SetTensorInfo(input0TensorInfo);
    minInput1Layer->GetOutputSlot(0).SetTensorInfo(minInput1TensorInfo);
    minLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // connect layers.
    // In0  In1
    //  \   /
    //   Mul  maxIn1
    //    |   /
    //   Max    minIn1
    //    |     /
    //     Min
    //     |
    //    Out
    input0Layer   ->GetOutputSlot(0).Connect(mulLayer->GetInputSlot(0));
    input1Layer   ->GetOutputSlot(0).Connect(mulLayer->GetInputSlot(1));
    mulLayer      ->GetOutputSlot(0).Connect(maxLayer->GetInputSlot(0));
    maxInput1Layer->GetOutputSlot(0).Connect(maxLayer->GetInputSlot(1));
    maxLayer      ->GetOutputSlot(0).Connect(minLayer->GetInputSlot(0));
    minInput1Layer->GetOutputSlot(0).Connect(minLayer->GetInputSlot(1));
    minLayer      ->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create ArmNN runtime
    IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions());

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNet = Optimize(*network, {backendId}, run->GetDeviceSpec());

    Graph& graph = GetGraphForTesting(optNet.get());

    auto checkMul = [ ](const armnn::Layer* const layer) -> bool
    {
        auto* mulLayer = PolymorphicDowncast<const ElementwiseBinaryLayer*>(layer);

        return IsLayerOfType<ElementwiseBinaryLayer>(layer) &&
               (mulLayer->GetParameters().m_Operation == BinaryOperation::Mul);
    };

    auto checkBoundedRelu = [ ](const armnn::Layer* const layer) -> bool
    {
        auto* activationLayer = PolymorphicDowncast<const ActivationLayer*>(layer);

        return IsLayerOfType<ActivationLayer>(layer) &&
               (activationLayer->GetParameters().m_Function == ActivationFunction::BoundedReLu);
    };

    // 2 inputs, mul, activation(in CpuRef and CpuAcc), output
    CHECK((graph.GetNumLayers() == numLayers));
    if (numLayers == 4)
    {
        CHECK(CheckSequence(graph.cbegin(),
                            graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<InputLayer>,
                            checkMul,
                            &IsLayerOfType<OutputLayer>));
    }
    else if (numLayers == 5)
    {
        CHECK(CheckSequence(graph.cbegin(),
                            graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<InputLayer>,
                            checkMul,
                            checkBoundedRelu,
                            &IsLayerOfType<OutputLayer>));
    }

    // Load network into runtime
    NetworkId networkIdentifier;
    run->LoadNetwork(networkIdentifier, std::move(optNet));

    // Create input and output tensors
    InputTensors inputTensors
    {
        {0, ConstTensor(input0TensorInfo, input0.data())},
        {1, ConstTensor(input1TensorInfo, input1.data())}
    };
    OutputTensors outputTensors
    {
        {0, Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputActual.data())}
    };

    // Run inference
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // Checks the results
    CHECK(outputActual == outputExpected);
}
}

TEST_SUITE("Optimizer")
{
#if defined(ARMNNREF_ENABLED)
TEST_CASE("FuseMulMaxMinTest_Float_CpuRef")
{
    MulMaxMinTest<DataType::Float32>(Compute::CpuRef, 5);
}
#endif
#if defined(ARMCOMPUTENEON_ENABLED)
TEST_CASE("FuseMulMaxMinTest_Float_CpuAcc")
{
    MulMaxMinTest<DataType::Float32>(Compute::CpuAcc, 5);
}
#endif
#if defined(ARMCOMPUTECL_ENABLED)
TEST_CASE("FuseMulMaxMinTest_Float_GpuAcc")
{
    MulMaxMinTest<DataType::Float32>(Compute::GpuAcc, 4);
}
#endif
}