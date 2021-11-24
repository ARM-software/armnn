//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Network.hpp>
#include <Optimizer.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

namespace
{

/// Shared function for the below tests, so that we test the same network in both cases.
std::unique_ptr<NetworkImpl> CreateTestNetworkImpl()
{
    std::unique_ptr<NetworkImpl> network(new NetworkImpl());

    auto input = network->AddInputLayer(0, "input");
    const TensorInfo inputInfo({ 1, 2, 3, 4 }, DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);

    // Insert Permute which swaps batches and channels dimensions
    auto permute = network->AddPermuteLayer(PermuteDescriptor(PermutationVector{ 3, 1, 2, 0 }), "permute");
    const TensorInfo permuteInfo({ 4, 2, 3, 1 }, DataType::Float32);
    permute->GetOutputSlot(0).SetTensorInfo(permuteInfo);
    input->GetOutputSlot(0).Connect(permute->GetInputSlot(0));

    // Insert BatchToSpace
    BatchToSpaceNdDescriptor batchToSpaceDesc;
    batchToSpaceDesc.m_BlockShape = { 2, 2 };
    batchToSpaceDesc.m_DataLayout = DataLayout::NHWC;
    auto batchToSpace             = network->AddBatchToSpaceNdLayer(batchToSpaceDesc, "batchToSpace");
    const TensorInfo batchToSpaceInfo({ 1, 4, 6, 1 }, DataType::Float32);
    batchToSpace->GetOutputSlot(0).SetTensorInfo(batchToSpaceInfo);
    permute->GetOutputSlot(0).Connect(batchToSpace->GetInputSlot(0));

    auto output = network->AddOutputLayer(0, "output");
    batchToSpace->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    return network;
}

/// Shared function for the below tests, so that we test the same network in both cases.
std::unique_ptr<NetworkImpl> CreateTransposeTestNetworkImpl()
{
    // Create a network
    std::unique_ptr<NetworkImpl> network(new NetworkImpl());

    auto input = network->AddInputLayer(0, "input");
    const TensorInfo inputInfo({ 1, 2, 3, 4 }, DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);

    // Insert Permute which swaps batches and channels dimensions
    auto permute = network->AddTransposeLayer(TransposeDescriptor(PermutationVector{ 3, 1, 2, 0 }), "permute");
    const TensorInfo permuteInfo({ 4, 2, 3, 1 }, DataType::Float32);
    permute->GetOutputSlot(0).SetTensorInfo(permuteInfo);
    input->GetOutputSlot(0).Connect(permute->GetInputSlot(0));

    // Insert BatchToSpace
    BatchToSpaceNdDescriptor batchToSpaceDesc;
    batchToSpaceDesc.m_BlockShape = { 2, 2 };
    batchToSpaceDesc.m_DataLayout = DataLayout::NHWC;
    auto batchToSpace             = network->AddBatchToSpaceNdLayer(batchToSpaceDesc, "batchToSpace");
    const TensorInfo batchToSpaceInfo({ 1, 4, 6, 1 }, DataType::Float32);
    batchToSpace->GetOutputSlot(0).SetTensorInfo(batchToSpaceInfo);
    permute->GetOutputSlot(0).Connect(batchToSpace->GetInputSlot(0));

    auto output = network->AddOutputLayer(0, "output");
    batchToSpace->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    return network;
}

}    // namespace

/// Tests that the optimization performed by PermuteAndBatchToSpaceAsDepthToSpace is as expected.
/// Note this does not ensure the correctness of the optimization - that is done in the below test.
TEST_CASE("PermuteAndBatchToSpaceAsDepthToSpaceOptimizerTest")
{
    std::unique_ptr<NetworkImpl> network = CreateTestNetworkImpl();
    Graph graph         = network.get()->GetGraph();

    // Confirm initial graph is as we expect
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<InputLayer>, &IsLayerOfType<PermuteLayer>,
                             &IsLayerOfType<BatchToSpaceNdLayer>, &IsLayerOfType<OutputLayer>));

    // Perform the optimization which should merge the two layers into a DepthToSpace
    armnn::Optimizer::Pass(graph, MakeOptimizations(PermuteAndBatchToSpaceAsDepthToSpace()));

    // Check that the replacement has been made as expected
    auto checkDepthToSpace = [](const Layer* const layer) -> bool {
        return IsLayerOfType<DepthToSpaceLayer>(layer) &&
               static_cast<const DepthToSpaceLayer*>(layer)->GetParameters().m_BlockSize == 2 &&
               static_cast<const DepthToSpaceLayer*>(layer)->GetParameters().m_DataLayout == DataLayout::NHWC &&
               layer->GetOutputHandler().GetTensorInfo() == TensorInfo({ 1, 4, 6, 1 }, DataType::Float32);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<InputLayer>, checkDepthToSpace,
                             &IsLayerOfType<OutputLayer>));

    // Check the new layer has the two merged layers listed as related layers
    std::list<std::string> testRelatedLayers = { "batchToSpace", "permute" };
    CHECK(CheckRelatedLayers<DepthToSpaceLayer>(graph, testRelatedLayers));
}

/// Tests that the optimization performed by PermuteAndBatchToSpaceAsDepthToSpace is as expected.
/// Note this does not ensure the correctness of the optimization - that is done in the below test.
TEST_CASE("TransposeAndBatchToSpaceAsDepthToSpaceOptimizerTest")
{
    std::unique_ptr<NetworkImpl> network = CreateTransposeTestNetworkImpl();
    Graph graph         = network.get()->GetGraph();

    // Confirm initial graph is as we expect
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<InputLayer>, &IsLayerOfType<TransposeLayer>,
                             &IsLayerOfType<BatchToSpaceNdLayer>, &IsLayerOfType<OutputLayer>));

    // Perform the optimization which should merge the two layers into a DepthToSpace
    armnn::Optimizer::Pass(graph, MakeOptimizations(TransposeAndBatchToSpaceAsDepthToSpace()));

    // Check that the replacement has been made as expected
    auto checkDepthToSpace = [](const Layer* const layer) -> bool {
        return IsLayerOfType<DepthToSpaceLayer>(layer) &&
               static_cast<const DepthToSpaceLayer*>(layer)->GetParameters().m_BlockSize == 2 &&
               static_cast<const DepthToSpaceLayer*>(layer)->GetParameters().m_DataLayout == DataLayout::NHWC &&
               layer->GetOutputHandler().GetTensorInfo() == TensorInfo({ 1, 4, 6, 1 }, DataType::Float32);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<InputLayer>, checkDepthToSpace,
                             &IsLayerOfType<OutputLayer>));

    // Check the new layer has the two merged layers listed as related layers
    std::list<std::string> testRelatedLayers = { "batchToSpace", "permute" };
    CHECK(CheckRelatedLayers<DepthToSpaceLayer>(graph, testRelatedLayers));
}

// This unit test needs the reference backend, it's not available if the reference backend is not built
#if defined(ARMNNREF_ENABLED)

/// Shared function for the below tests, so that we test the same network in both cases.
INetworkPtr CreateTestNetwork()
{
    // Create a network
    INetworkPtr network = INetwork::Create();

    auto input = network->AddInputLayer(0, "input");
    const TensorInfo inputInfo({ 1, 2, 3, 4 }, DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);

    // Insert Permute which swaps batches and channels dimensions
    auto permute = network->AddPermuteLayer(PermuteDescriptor(PermutationVector{ 3, 1, 2, 0 }), "permute");
    const TensorInfo permuteInfo({ 4, 2, 3, 1 }, DataType::Float32);
    permute->GetOutputSlot(0).SetTensorInfo(permuteInfo);
    input->GetOutputSlot(0).Connect(permute->GetInputSlot(0));

    // Insert BatchToSpace
    BatchToSpaceNdDescriptor batchToSpaceDesc;
    batchToSpaceDesc.m_BlockShape = { 2, 2 };
    batchToSpaceDesc.m_DataLayout = DataLayout::NHWC;
    auto batchToSpace             = network->AddBatchToSpaceNdLayer(batchToSpaceDesc, "batchToSpace");
    const TensorInfo batchToSpaceInfo({ 1, 4, 6, 1 }, DataType::Float32);
    batchToSpace->GetOutputSlot(0).SetTensorInfo(batchToSpaceInfo);
    permute->GetOutputSlot(0).Connect(batchToSpace->GetInputSlot(0));

    auto output = network->AddOutputLayer(0, "output");
    batchToSpace->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    return network;
}

/// Shared function for the below tests, so that we test the same network in both cases.
INetworkPtr CreateTransposeTestNetwork()
{
    // Create a network
    INetworkPtr network = INetwork::Create();

    auto input = network->AddInputLayer(0, "input");
    const TensorInfo inputInfo({ 1, 2, 3, 4 }, DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);

    // Insert Permute which swaps batches and channels dimensions
    auto permute = network->AddTransposeLayer(TransposeDescriptor(PermutationVector{ 3, 1, 2, 0 }), "permute");
    const TensorInfo permuteInfo({ 4, 2, 3, 1 }, DataType::Float32);
    permute->GetOutputSlot(0).SetTensorInfo(permuteInfo);
    input->GetOutputSlot(0).Connect(permute->GetInputSlot(0));

    // Insert BatchToSpace
    BatchToSpaceNdDescriptor batchToSpaceDesc;
    batchToSpaceDesc.m_BlockShape = { 2, 2 };
    batchToSpaceDesc.m_DataLayout = DataLayout::NHWC;
    auto batchToSpace             = network->AddBatchToSpaceNdLayer(batchToSpaceDesc, "batchToSpace");
    const TensorInfo batchToSpaceInfo({ 1, 4, 6, 1 }, DataType::Float32);
    batchToSpace->GetOutputSlot(0).SetTensorInfo(batchToSpaceInfo);
    permute->GetOutputSlot(0).Connect(batchToSpace->GetInputSlot(0));

    auto output = network->AddOutputLayer(0, "output");
    batchToSpace->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    return network;
}

/// Tests that a optimization performed by PermuteAndBatchToSpaceAsDepthToSpace does not change the behaviour
/// of the network (i.e. it still produces the correct output).
TEST_CASE("PermuteAndBatchToSpaceAsDepthToSpaceCorrectnessTest")
{
    INetworkPtr network = CreateTestNetwork();

    IRuntimePtr runtime = IRuntime::Create(IRuntime::CreationOptions());
    IOptimizedNetworkPtr optimizedNetwork = Optimize(*network, { Compute::CpuRef }, runtime->GetDeviceSpec());

    // Confirm that the optimization has actually taken place
    const Graph& optGraph = GetGraphForTesting(optimizedNetwork.get());
    CHECK(CheckSequence(optGraph.cbegin(), optGraph.cend(), &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<DepthToSpaceLayer>, &IsLayerOfType<OutputLayer>));

    // Load the graph into a runtime so we can check it produces the correct output
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optimizedNetwork));

    std::vector<float> inputData{
        // Each row here is a row of pixels where each pixel has 4 channels
        // clang-format off
        1.0f,  2.0f,  3.0f,  4.0f,      10.0f,  20.0f,  30.0f,  40.0f,      100.0f,  200.0f,  300.0f,  400.0f,
        -1.0f, -2.0f, -3.0f, -4.0f,    -10.0f, -20.0f, -30.0f, -40.0f,     -100.0f, -200.0f, -300.0f, -400.0f,
        // clang-format on
    };
    ConstTensor input(TensorInfo({ 1, 2, 3, 4 }, DataType::Float32, 0.0f, 0, true), inputData);
    InputTensors inputs = { { 0, input } };
    std::vector<float> outputData(4 * 6);
    Tensor output(TensorInfo({ 1, 4, 6, 1 }, DataType::Float32), outputData.data());
    OutputTensors outputs = { { 0, output } };
    runtime->EnqueueWorkload(netId, inputs, outputs);

    // Check the output is as expected.
    // Note this output has been generated by running the network *without* the optimization.
    std::vector<float> expectedOutput = {
        // Rows and columns here match exactly with the tensor, as there is only 1 channel.
        // clang-format off
        1.0f,  2.0f,     10.0f,  20.0f,     100.0f,  200.0f,
        3.0f,  4.0f,     30.0f,  40.0f,     300.0f,  400.0f,

        -1.0f, -2.0f,   -10.0f, -20.0f,    -100.0f, -200.0f,
        -3.0f, -4.0f,   -30.0f, -40.0f,    -300.0f, -400.0f,
        // clang-format on
    };
    CHECK(outputData == expectedOutput);
}

/// Tests that a optimization performed by PermuteAndBatchToSpaceAsDepthToSpace does not change the behaviour
/// of the network (i.e. it still produces the correct output).
TEST_CASE("TransposeAndBatchToSpaceAsDepthToSpaceCorrectnessTest")
{
    INetworkPtr network = CreateTransposeTestNetwork();

    IRuntimePtr runtime = IRuntime::Create(IRuntime::CreationOptions());
    IOptimizedNetworkPtr optimizedNetwork = Optimize(*network, { Compute::CpuRef }, runtime->GetDeviceSpec());

    // Confirm that the optimization has actually taken place
    const Graph& optGraph = GetGraphForTesting(optimizedNetwork.get());
    CHECK(CheckSequence(optGraph.cbegin(), optGraph.cend(), &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<DepthToSpaceLayer>, &IsLayerOfType<OutputLayer>));

    // Load the graph into a runtime so we can check it produces the correct output
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optimizedNetwork));

    std::vector<float> inputData{
            // Each row here is a row of pixels where each pixel has 4 channels
            // clang-format off
            1.0f,  2.0f,  3.0f,  4.0f,      10.0f,  20.0f,  30.0f,  40.0f,      100.0f,  200.0f,  300.0f,  400.0f,
            -1.0f, -2.0f, -3.0f, -4.0f,    -10.0f, -20.0f, -30.0f, -40.0f,     -100.0f, -200.0f, -300.0f, -400.0f,
            // clang-format on
    };
    ConstTensor input(TensorInfo({ 1, 2, 3, 4 }, DataType::Float32, 0.0f, 0, true), inputData);
    InputTensors inputs = { { 0, input } };
    std::vector<float> outputData(4 * 6);
    Tensor output(TensorInfo({ 1, 4, 6, 1 }, DataType::Float32), outputData.data());
    OutputTensors outputs = { { 0, output } };
    runtime->EnqueueWorkload(netId, inputs, outputs);

    // Check the output is as expected.
    // Note this output has been generated by running the network *without* the optimization.
    std::vector<float> expectedOutput = {
            // Rows and columns here match exactly with the tensor, as there is only 1 channel.
            // clang-format off
            1.0f,  2.0f,     10.0f,  20.0f,     100.0f,  200.0f,
            3.0f,  4.0f,     30.0f,  40.0f,     300.0f,  400.0f,

            -1.0f, -2.0f,   -10.0f, -20.0f,    -100.0f, -200.0f,
            -3.0f, -4.0f,   -30.0f, -40.0f,    -300.0f, -400.0f,
            // clang-format on
    };
    CHECK(outputData == expectedOutput);
}
#endif

}