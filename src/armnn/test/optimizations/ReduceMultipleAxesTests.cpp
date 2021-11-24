//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <GraphUtils.hpp>
#include <TestUtils.hpp>

#include <armnn/INetwork.hpp>

#include <doctest/doctest.h>

using namespace armnn;

namespace
{
#if defined(ARMCOMPUTENEON_ENABLED)||defined(ARMCOMPUTECL_ENABLED)
INetworkPtr CreateSimpleReduceNetwork(ReduceDescriptor reduceDescriptor,
                                      TensorShape& inputShape,
                                      TensorShape& outputShape)
{
    // Create a network
    INetworkPtr network = INetwork::Create();

    const std::string layerName("reduce_layer");
    const TensorInfo inputInfo(inputShape, DataType::Float32);
    const TensorInfo outputInfo(outputShape, DataType::Float32);

    IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    IConnectableLayer* const reduceLayer = network->AddReduceLayer(reduceDescriptor, layerName.c_str());
    IConnectableLayer* const outputLayer1 = network->AddOutputLayer(0);
    IConnectableLayer* const outputLayer2 = network->AddOutputLayer(1);

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    reduceLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    inputLayer->GetOutputSlot(0).Connect(reduceLayer->GetInputSlot(0));
    reduceLayer->GetOutputSlot(0).Connect(outputLayer1->GetInputSlot(0));
    reduceLayer->GetOutputSlot(0).Connect(outputLayer2->GetInputSlot(0));

    return network;
}

void ReduceWithMultipleAxesTest(INetworkPtr& network,
                                const TensorShape& outputShape,
                                const std::vector<float>& inputData,
                                const std::vector<float>& expectedOutput,
                                const size_t numOfAxes,
                                Compute backendId)
{
    // Create ArmNN runtime
    IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions());

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNet = Optimize(*network, {backendId}, run->GetDeviceSpec());

    Graph& graph = GetGraphForTesting(optNet.get());
    if (numOfAxes == 2)
    {
        CHECK(graph.GetNumLayers() == 5);
        CHECK(CheckSequence(graph.cbegin(),
                            graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<ReduceLayer>,
                            &IsLayerOfType<ReduceLayer>,
                            &IsLayerOfType<OutputLayer>,
                            &IsLayerOfType<OutputLayer>));
    } else
    {
        CHECK(graph.GetNumLayers() == 6);
        CHECK(CheckSequence(graph.cbegin(),
                            graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<ReduceLayer>,
                            &IsLayerOfType<ReduceLayer>,
                            &IsLayerOfType<ReduceLayer>,
                            &IsLayerOfType<OutputLayer>,
                            &IsLayerOfType<OutputLayer>));
    }

    // Get last layer in new chain, layers name follow 0, 1, 2 pattern
    std::string layerName = "reduce_layer_" + std::to_string(numOfAxes - 1);
    Layer* const reduceLayer = GetFirstLayerWithName(graph, layerName);
    CHECK(reduceLayer);
    auto reduceTensorInfo = reduceLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    CHECK((reduceTensorInfo.GetShape() == outputShape));
    CHECK((reduceTensorInfo.GetDataType() == DataType::Float32));

    // Load network into runtime
    NetworkId networkIdentifier;
    run->LoadNetwork(networkIdentifier, std::move(optNet));

    // Create input and output tensors
    std::vector<float> outputData(expectedOutput.size());
    armnn::TensorInfo inputTensorInfo = run->GetInputTensorInfo(networkIdentifier, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
        {
            {0, armnn::ConstTensor(inputTensorInfo, inputData.data())}
        };
    OutputTensors outputTensors
        {
            {0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputData.data())},
            {1, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 1), outputData.data())}
        };

    // Run inference
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // Checks the results
    CHECK(outputData == expectedOutput);
}

void ReduceSumWithTwoAxesKeepDimsTest(Compute backendId)
{
    armnn::ReduceDescriptor reduceDescriptor;
    reduceDescriptor.m_vAxis = {1, 2};
    reduceDescriptor.m_KeepDims = true;
    reduceDescriptor.m_ReduceOperation = armnn::ReduceOperation::Sum;

    TensorShape inputShape = {1, 3, 2, 4};
    TensorShape outputShape = {1, 1, 1, 4};

    // Construct ArmNN network
    INetworkPtr network = CreateSimpleReduceNetwork(reduceDescriptor, inputShape, outputShape);

    // Creates structures for input & output.
    const std::vector<float> inputData({1.0f, 2.0f, 3.0f, 4.0f,
                                        5.0f, 6.0f, 7.0f, 8.0f,

                                        10.0f, 20.0f, 30.0f, 40.0f,
                                        50.0f, 60.0f, 70.0f, 80.0f,

                                        100.0f, 200.0f, 300.0f, 400.0f,
                                        500.0f, 600.0f, 700.0f, 800.0f});
    const std::vector<float> expectedOutput({666.0f, 888.0f, 1110.0f, 1332.0f});

    ReduceWithMultipleAxesTest(network,
                               outputShape,
                               inputData,
                               expectedOutput,
                               reduceDescriptor.m_vAxis.size(),
                               backendId);
}

void ReduceSumWithTwoAxesTest(Compute backendId)
{
    armnn::ReduceDescriptor reduceDescriptor;
    reduceDescriptor.m_vAxis = {1, 2};
    reduceDescriptor.m_KeepDims = false;
    reduceDescriptor.m_ReduceOperation = armnn::ReduceOperation::Sum;

    TensorShape inputShape = {1, 3, 2, 4};
    TensorShape outputShape = {1, 4};

    // Construct ArmNN network
    INetworkPtr network = CreateSimpleReduceNetwork(reduceDescriptor, inputShape, outputShape);

    // Creates structures for input & output.
    const std::vector<float> inputData({1.0f, 2.0f, 3.0f, 4.0f,
                                        5.0f, 6.0f, 7.0f, 8.0f,

                                        10.0f, 20.0f, 30.0f, 40.0f,
                                        50.0f, 60.0f, 70.0f, 80.0f,

                                        100.0f, 200.0f, 300.0f, 400.0f,
                                        500.0f, 600.0f, 700.0f, 800.0f});
    const std::vector<float> expectedOutput({666.0f, 888.0f, 1110.0f, 1332.0f});

    ReduceWithMultipleAxesTest(network,
                               outputShape,
                               inputData,
                               expectedOutput,
                               reduceDescriptor.m_vAxis.size(),
                               backendId);
}

void ReduceSumWithThreeAxesKeepDimsTest(Compute backendId)
{
    armnn::ReduceDescriptor reduceDescriptor;
    reduceDescriptor.m_vAxis = {0, 2, 3};
    reduceDescriptor.m_KeepDims = true;
    reduceDescriptor.m_ReduceOperation = armnn::ReduceOperation::Sum;

    TensorShape inputShape = {2, 2, 2, 2};
    TensorShape outputShape = {1, 2, 1, 1};

    // Construct ArmNN network
    INetworkPtr network = CreateSimpleReduceNetwork(reduceDescriptor, inputShape, outputShape);

    // Creates structures for input & output.
    const std::vector<float> inputData({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        10.0f, 20.0f,
                                        30.0f, 40.0f,

                                        50.0f, 60.0f,
                                        70.0f, 80.0f});
    const std::vector<float> expectedOutput({110.0f, 286.0f});

    ReduceWithMultipleAxesTest(network,
                               outputShape,
                               inputData,
                               expectedOutput,
                               reduceDescriptor.m_vAxis.size(),
                               backendId);
}

void ReduceSumWithThreeAxesTest(Compute backendId)
{
    armnn::ReduceDescriptor reduceDescriptor;
    reduceDescriptor.m_vAxis = {0, 2, 3};
    reduceDescriptor.m_KeepDims = false;
    reduceDescriptor.m_ReduceOperation = armnn::ReduceOperation::Sum;

    TensorShape inputShape = {2, 2, 2, 2};
    TensorShape outputShape = {2};

    // Construct ArmNN network
    INetworkPtr network = CreateSimpleReduceNetwork(reduceDescriptor, inputShape, outputShape);

    // Creates structures for input & output.
    const std::vector<float> inputData({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        10.0f, 20.0f,
                                        30.0f, 40.0f,

                                        50.0f, 60.0f,
                                        70.0f, 80.0f});
    const std::vector<float> expectedOutput({110.0f, 286.0f});

    ReduceWithMultipleAxesTest(network,
                               outputShape,
                               inputData,
                               expectedOutput,
                               reduceDescriptor.m_vAxis.size(),
                               backendId);
}
#endif
}

#if defined(ARMCOMPUTENEON_ENABLED)
TEST_SUITE("Optimizer_ReduceMultipleAxesCpu")
{
TEST_CASE("ReduceSumWithTwoAxesKeepDimsCpuAccTest")
{
    ReduceSumWithTwoAxesKeepDimsTest(Compute::CpuAcc);
}

TEST_CASE("ReduceSumWithTwoAxesCpuAccTest")
{
    ReduceSumWithTwoAxesTest(Compute::CpuAcc);
}

TEST_CASE("ReduceSumWithThreeAxesKeepDimsCpuAccTest")
{
    ReduceSumWithThreeAxesKeepDimsTest(Compute::CpuAcc);
}

TEST_CASE("ReduceSumWithThreeAxesCpuAccTest")
{
    ReduceSumWithThreeAxesTest(Compute::CpuAcc);
}
}
#endif

#if defined(ARMCOMPUTECL_ENABLED)
TEST_SUITE("Optimizer_ReduceMultipleAxesGpu")
{
TEST_CASE("ReduceSumWithTwoAxesKeepDimsGpuAccTest")
{
    ReduceSumWithTwoAxesKeepDimsTest(Compute::GpuAcc);
}

TEST_CASE("ReduceSumWithTwoAxesGpuAccTest")
{
    ReduceSumWithTwoAxesTest(Compute::GpuAcc);
}

TEST_CASE("ReduceSumWithThreeAxesKeepDimsGpuAccTest")
{
    ReduceSumWithThreeAxesKeepDimsTest(Compute::GpuAcc);
}

TEST_CASE("ReduceSumWithThreeAxesGpuAccTest")
{
    ReduceSumWithThreeAxesTest(Compute::GpuAcc);
}
}
#endif
