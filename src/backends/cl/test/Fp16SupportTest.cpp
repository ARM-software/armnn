//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>
#include <Half.hpp>

#include <Graph.hpp>
#include <Optimizer.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>

#include <set>

using namespace armnn;

TEST_SUITE("Fp16Support")
{
TEST_CASE("Fp16DataTypeSupport")
{
    Graph graph;

    Layer* const inputLayer1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const inputLayer2 = graph.AddLayer<InputLayer>(2, "input2");

    Layer* const additionLayer = graph.AddLayer<AdditionLayer>("addition");
    Layer* const outputLayer = graph.AddLayer<armnn::OutputLayer>(0, "output");

    TensorInfo fp16TensorInfo({1, 2, 3, 5}, armnn::DataType::Float16);
    inputLayer1->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
    inputLayer2->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(1));
    additionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer1->GetOutputSlot().SetTensorInfo(fp16TensorInfo);
    inputLayer2->GetOutputSlot().SetTensorInfo(fp16TensorInfo);
    additionLayer->GetOutputSlot().SetTensorInfo(fp16TensorInfo);

    CHECK(inputLayer1->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);
    CHECK(inputLayer2->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);
    CHECK(additionLayer->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);
}

TEST_CASE("Fp16AdditionTest")
{
   using namespace half_float::literal;
   // Create runtime in which test will run
   IRuntime::CreationOptions options;
   IRuntimePtr runtime(IRuntime::Create(options));

   // Builds up the structure of the network.
   INetworkPtr net(INetwork::Create());

   IConnectableLayer* inputLayer1 = net->AddInputLayer(0);
   IConnectableLayer* inputLayer2 = net->AddInputLayer(1);
   IConnectableLayer* additionLayer = net->AddAdditionLayer();
   IConnectableLayer* outputLayer = net->AddOutputLayer(0);

   inputLayer1->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
   inputLayer2->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(1));
   additionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

   //change to float16
   TensorInfo fp16TensorInfo(TensorShape({4}), DataType::Float16);
   inputLayer1->GetOutputSlot(0).SetTensorInfo(fp16TensorInfo);
   inputLayer2->GetOutputSlot(0).SetTensorInfo(fp16TensorInfo);
   additionLayer->GetOutputSlot(0).SetTensorInfo(fp16TensorInfo);

   // optimize the network
   std::vector<BackendId> backends = {Compute::GpuAcc};
   IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

   // Loads it into the runtime.
   NetworkId netId;
   runtime->LoadNetwork(netId, std::move(optNet));

   std::vector<Half> input1Data
   {
       1.0_h, 2.0_h, 3.0_h, 4.0_h
   };

   std::vector<Half> input2Data
   {
       100.0_h, 200.0_h, 300.0_h, 400.0_h
   };

   TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
   inputTensorInfo.SetConstant(true);
   InputTensors inputTensors
   {
       {0,ConstTensor(inputTensorInfo, input1Data.data())},
       {1,ConstTensor(inputTensorInfo, input2Data.data())}
   };

   std::vector<Half> outputData(input1Data.size());
   OutputTensors outputTensors
   {
       {0,Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
   };

   // Does the inference.
   runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

   // Checks the results.
   CHECK(outputData == std::vector<Half>({ 101.0_h, 202.0_h, 303.0_h, 404.0_h})); // Add
}

}
