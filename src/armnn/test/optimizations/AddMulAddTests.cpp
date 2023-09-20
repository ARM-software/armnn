//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <GraphUtils.hpp>
#include <TestUtils.hpp>
#include <ResolveType.hpp>
#include <armnnUtils/QuantizeHelper.hpp>

#include <armnn/INetwork.hpp>

#include <doctest/doctest.h>

using namespace armnn;

namespace
{
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void AddMulAddTest(Compute backendId, bool addOutput, bool addRelu)
{
    const TensorInfo input0TensorInfo({ 1, 2, 2, 3 },
                                      ArmnnType,
                                      IsQuantizedType<T>() ? 0.25f : 1,
                                      IsQuantizedType<T>() ? 10 : 0,
                                      true);
    const TensorInfo input1TensorInfo({ 1, 2, 2, 3 },
                                      ArmnnType,
                                      IsQuantizedType<T>() ? 0.25f : 1,
                                      IsQuantizedType<T>() ? 11 : 0,
                                      true);
    const TensorInfo mulInput1TensorInfo({ 3 },
                                         ArmnnType,
                                         IsQuantizedType<T>() ? 0.25f : 1,
                                         IsQuantizedType<T>() ? 12 : 0,
                                         true);
    const TensorInfo addInput1TensorInfo({ 3 },
                                         ArmnnType,
                                         IsQuantizedType<T>() ? 0.25f : 1,
                                         IsQuantizedType<T>() ? 13 : 0,
                                         true);
    const TensorInfo output0TensorInfo({ 1, 2, 2, 3 },
                                       ArmnnType,
                                       IsQuantizedType<T>() ? 0.5f : 1,
                                       IsQuantizedType<T>() ? 14 : 0);
    const TensorInfo output1TensorInfo({ 1, 2, 2, 3 },
                                       ArmnnType,
                                       IsQuantizedType<T>() ? 0.5f : 1,
                                       IsQuantizedType<T>() ? 15 : 0);

    std::vector<float> input0Data
    {
         0.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -2.0f, -2.0f, -2.0f
    };
    std::vector<float> input1Data
    {
         0.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -2.0f, -2.0f, -2.0f
    };
    std::vector<float> mulInput1Data
    {
         2.0f, 1.0f, 1.0f
    };
    std::vector<float> addInput1Data
    {
         3.0f, 0.0f, 0.0f
    };
    std::vector<float> output0ExpectedData =
    {
         0.0f,  0.0f,  0.0f,
         2.0f,  2.0f,  2.0f,
        -2.0f, -2.0f, -2.0f,
        -4.0f, -4.0f, -4.0f
    };
    std::vector<float> output1ExpectedData =
    {
         3.0f,  0.0f,  0.0f,
         7.0f,  2.0f,  2.0f,
        -1.0f, -2.0f, -2.0f,
        -5.0f, -4.0f, -4.0f
    };
    std::vector<float> output1ReluExpectedData =
    {
         3.0f,  0.0f,  0.0f,
         7.0f,  2.0f,  2.0f,
         0.0f,  0.0f,  0.0f,
         0.0f,  0.0f,  0.0f
    };

    std::vector<T> input0 = armnnUtils::QuantizedVector<T>(input0Data,
                                                           input0TensorInfo.GetQuantizationScale(),
                                                           input0TensorInfo.GetQuantizationOffset());
    std::vector<T> input1 = armnnUtils::QuantizedVector<T>(input1Data,
                                                           input1TensorInfo.GetQuantizationScale(),
                                                           input1TensorInfo.GetQuantizationOffset());
    std::vector<T> mulInput1 = armnnUtils::QuantizedVector<T>(mulInput1Data,
                                                              mulInput1TensorInfo.GetQuantizationScale(),
                                                              mulInput1TensorInfo.GetQuantizationOffset());
    std::vector<T> addInput1 = armnnUtils::QuantizedVector<T>(addInput1Data,
                                                              addInput1TensorInfo.GetQuantizationScale(),
                                                              addInput1TensorInfo.GetQuantizationOffset());
    std::vector<T> output0Expected = armnnUtils::QuantizedVector<T>(output0ExpectedData,
                                                                    output0TensorInfo.GetQuantizationScale(),
                                                                    output0TensorInfo.GetQuantizationOffset());
    std::vector<T> output1Expected = armnnUtils::QuantizedVector<T>(output1ExpectedData,
                                                                    output1TensorInfo.GetQuantizationScale(),
                                                                    output1TensorInfo.GetQuantizationOffset());
    std::vector<T> output1ReluExpected = armnnUtils::QuantizedVector<T>(output1ReluExpectedData,
                                                                    output1TensorInfo.GetQuantizationScale(),
                                                                    output1TensorInfo.GetQuantizationOffset());

    std::vector<T> output0Actual(output0TensorInfo.GetNumElements());
    std::vector<T> output1Actual(output1TensorInfo.GetNumElements());

    // Create a network
    INetworkPtr network = INetwork::Create();

    IConnectableLayer* const input0Layer = network->AddInputLayer(0);
    IConnectableLayer* const input1Layer = network->AddInputLayer(1);
    IConnectableLayer* const add0Layer = network->AddElementwiseBinaryLayer(BinaryOperation::Add, "add0");
    IConnectableLayer* outputAddLayer = nullptr;
    if (addOutput)
    {
        outputAddLayer = network->AddOutputLayer(0);
    }

    auto constMulInput1Tensor = armnn::ConstTensor(mulInput1TensorInfo, mulInput1);
    IConnectableLayer* const mulInput1Layer = network->AddConstantLayer(constMulInput1Tensor, "mulInput1");
    IConnectableLayer* const mulLayer = network->AddElementwiseBinaryLayer(BinaryOperation::Mul, "mul");

    auto constAddInput1Tensor = armnn::ConstTensor(addInput1TensorInfo, addInput1);
    IConnectableLayer* const addInput1Layer = network->AddConstantLayer(constAddInput1Tensor, "addInput1");
    IConnectableLayer* const add1Layer = network->AddElementwiseBinaryLayer(BinaryOperation::Add, "add1");
    IConnectableLayer* const outputLayer = network->AddOutputLayer(1);

    IConnectableLayer* relu = nullptr;
    if (addRelu)
    {
        relu = network->AddActivationLayer(ActivationFunction::ReLu, "relu");
    }

    input0Layer->GetOutputSlot(0).SetTensorInfo(input0TensorInfo);
    input1Layer->GetOutputSlot(0).SetTensorInfo(input1TensorInfo);
    add0Layer->GetOutputSlot(0).SetTensorInfo(output0TensorInfo);
    mulInput1Layer->GetOutputSlot(0).SetTensorInfo(mulInput1TensorInfo);
    mulLayer->GetOutputSlot(0).SetTensorInfo(input0TensorInfo);
    addInput1Layer->GetOutputSlot(0).SetTensorInfo(addInput1TensorInfo);
    add1Layer->GetOutputSlot(0).SetTensorInfo(output1TensorInfo);
    if (addRelu)
    {
        relu->GetOutputSlot(0).SetTensorInfo(output1TensorInfo);
    }

    input0Layer->GetOutputSlot(0).Connect(add0Layer->GetInputSlot(0));
    input1Layer->GetOutputSlot(0).Connect(add0Layer->GetInputSlot(1));
    if (addOutput)
    {
        add0Layer->GetOutputSlot(0).Connect(outputAddLayer->GetInputSlot(0));
    }

    add0Layer->GetOutputSlot(0).Connect(mulLayer->GetInputSlot(0));
    mulInput1Layer->GetOutputSlot(0).Connect(mulLayer->GetInputSlot(1));
    mulLayer->GetOutputSlot(0).Connect(add1Layer->GetInputSlot(0));
    addInput1Layer->GetOutputSlot(0).Connect(add1Layer->GetInputSlot(1));

    if (addRelu)
    {
        add1Layer->GetOutputSlot(0).Connect(relu->GetInputSlot(0));
        relu->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    }
    else
    {
        add1Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    }

    // Create ArmNN runtime
    IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions());

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNet = Optimize(*network, {backendId}, run->GetDeviceSpec());

    Graph& graph = GetGraphForTesting(optNet.get());

    // There are 9 add layers above, so we should be left with 7 layers in the graph after the optimization.
    // Numbers reduced by 1 in addOutput == false case
    unsigned int expectedLayerNum = (addOutput) ? 7 : 6;
    CHECK((graph.GetNumLayers() == expectedLayerNum));

    if (addOutput)
    {
        CHECK(CheckSequence(graph.cbegin(),
                            graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<ConstantLayer>,
                            &IsLayerOfType<ConstantLayer>,
                            &IsLayerOfType<FusedLayer>,
                            &IsLayerOfType<OutputLayer>,
                            &IsLayerOfType<OutputLayer>));
    }
    else
    {
        CHECK(CheckSequence(graph.cbegin(),
                            graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<ConstantLayer>,
                            &IsLayerOfType<ConstantLayer>,
                            &IsLayerOfType<FusedLayer>,
                            &IsLayerOfType<OutputLayer>));
    }

    // Load network into runtime
    NetworkId networkIdentifier;
    run->LoadNetwork(networkIdentifier, std::move(optNet));

    // Create input and output tensors
    InputTensors inputTensors
    {
        {0, armnn::ConstTensor(input0TensorInfo, input0.data())},
        {1, armnn::ConstTensor(input1TensorInfo, input1.data())}
    };
    OutputTensors outputTensors;
    if (addOutput)
    {
        outputTensors.push_back(
                {0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), output0Actual.data())});
        outputTensors.push_back(
                {1, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 1), output1Actual.data())});
    }
    else
    {
        outputTensors.push_back(
                {1, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 1), output1Actual.data())});
    }

    // Run inference
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // Checks the results
    if (addOutput)
    {
        CHECK(output0Actual == output0Expected);
    }

    if (addRelu)
    {
        CHECK(output1Actual == output1ReluExpected);
    }
    else
    {
        CHECK(output1Actual == output1Expected);
    }

}
}

#if defined(ARMCOMPUTENEON_ENABLED)
TEST_SUITE("Optimizer_AddMulAdd")
{

TEST_CASE("AddMulAdd2OutputsFloat32Test")
{
    AddMulAddTest<DataType::Float32>(Compute::CpuAcc, true, false);
}

TEST_CASE("AddMulAdd2OutputsInt8Test")
{
    AddMulAddTest<DataType::QAsymmS8>(Compute::CpuAcc, true, false);
}

TEST_CASE("AddMulAdd2OutputsUint8Test")
{
    AddMulAddTest<DataType::QAsymmU8>(Compute::CpuAcc, true, false);
}

TEST_CASE("AddMulAdd1OutputFloat32Test")
{
    AddMulAddTest<DataType::Float32>(Compute::CpuAcc, false, false);
}

TEST_CASE("AddMulAdd1OutputInt8Test")
{
    AddMulAddTest<DataType::QAsymmS8>(Compute::CpuAcc, false, false);
}

TEST_CASE("AddMulAdd1OutputUint8Test")
{
    AddMulAddTest<DataType::QAsymmU8>(Compute::CpuAcc, false, false);
}

//
// Relu tests
//
TEST_CASE("AddMulAddRelu2OutputsFloat32Test")
{
    AddMulAddTest<DataType::Float32>(Compute::CpuAcc, true, true);
}

TEST_CASE("AddMulAddRelu1OutputFloat32Test")
{
    AddMulAddTest<DataType::Float32>(Compute::CpuAcc, false, true);
}

}
#endif