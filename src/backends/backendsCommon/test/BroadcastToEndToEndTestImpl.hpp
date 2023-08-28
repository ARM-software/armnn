//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "armnn/INetwork.hpp"
#include "armnnUtils/QuantizeHelper.hpp"
#include "ElementwiseBinaryEndToEndTestImpl.hpp"
#include "Optimizer.hpp"
#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>
#include <doctest/doctest.h>

namespace
{
    using namespace armnn;
    armnn::INetworkPtr CreateBroadcastToNetwork(BroadcastToDescriptor& descriptor,
                                                const armnn::TensorInfo& inputInfo,
                                                const armnn::TensorInfo& outputInfo)
    {
        INetworkPtr network(INetwork::Create());
        IConnectableLayer* inputLayer  = network->AddInputLayer(0, "input");
        IConnectableLayer* broadcastLayer   = network->AddBroadcastToLayer(descriptor, "broadcast_to");
        IConnectableLayer* outputLayer = network->AddOutputLayer(0, "output");
        Connect(inputLayer,   broadcastLayer, inputInfo,  0, 0);
        Connect(broadcastLayer, outputLayer,  outputInfo, 0, 0);
        return network;
    }

    armnn::INetworkPtr CreateBroadcastToNetworkWithElementWiseBinary(BroadcastToDescriptor& descriptor,
                                                                     const ElementwiseBinaryDescriptor&
                                                                     elementWiseDescriptor,
                                                                     const armnn::TensorInfo& inputInfo,
                                                                     const armnn::TensorInfo& inputInfoElementWise,
                                                                     const armnn::TensorInfo& outputInfo)
    {
        INetworkPtr network(INetwork::Create());
        IConnectableLayer* inputLayer  = network->AddInputLayer(0, "input");
        IConnectableLayer* inputLayerElementWise  = network->AddInputLayer(1, "inputElementWiseBinary");
        IConnectableLayer* broadcastLayer   = network->AddBroadcastToLayer(descriptor, "broadcast_to");
        IConnectableLayer* multiplicationLayer =
                network->AddElementwiseBinaryLayer(elementWiseDescriptor,
                                                   "multiplication");
        IConnectableLayer* outputLayer = network->AddOutputLayer(0, "output");
        Connect(inputLayer,   broadcastLayer, inputInfo,  0, 0);
        Connect(inputLayerElementWise,   multiplicationLayer,
                inputInfoElementWise,  0, 1);
        Connect(broadcastLayer,   multiplicationLayer, inputInfo,  0, 0);
        Connect(multiplicationLayer, outputLayer,  outputInfo, 0, 0);
        return network;
    }

    template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
    void BroadcastToEndToEnd(const std::vector<BackendId>& backends)
    {
        float   qScale  = 1.0f;
        int32_t qOffset = 0;
        bool    qConst  = true;

        const TensorShape inputTensorShape =  { {1, 4} };
        const TensorShape outputTensorShape = { {4, 4} };

        TensorInfo inputInfo  (inputTensorShape, ArmnnType, qScale,
                               qOffset, qConst);
        TensorInfo outputInfo (outputTensorShape, ArmnnType,qScale,
                               qOffset);

        std::vector<T> inputData = armnnUtils::QuantizedVector<T>({
                                                                          65, 144, 91, 161
                                                                  }, qScale, qOffset);

        std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>({
                                                                                   65, 144, 91, 161,
                                                                                   65, 144, 91, 161,
                                                                                   65, 144, 91, 161,
                                                                                   65, 144, 91, 161
                                                                           }, qScale, qOffset);

        auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape({ 4, 4 }));
        CHECK(descriptor.m_BroadcastToShape == outputTensorShape);
        INetworkPtr network = CreateBroadcastToNetwork(descriptor, inputInfo, outputInfo);

        std::map<int, std::vector<T>> inputTensor          = { { 0, inputData  } };
        std::map<int, std::vector<T>> expectedOutputTensor = { { 0, expectedOutputData } };
        EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),inputTensor,
                                                    expectedOutputTensor, backends);
    }

    template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
    void BroadcastToEndToEndElementWiseBinary(const std::vector<BackendId>& backends)
    {
        float   qScale  = 1.0f;
        int32_t qOffset = 0;
        bool    qConst  = true;

        const TensorShape inputTensorShape =  { {1, 4} };
        const TensorShape outputTensorShape = { {4, 4} };

        const TensorInfo inputInfo  (inputTensorShape, ArmnnType, qScale,
                                    qOffset, qConst);
        const TensorInfo inputInfoElementWise  (outputTensorShape, ArmnnType, qScale,
                                                qOffset, qConst);
        const TensorInfo outputInfo (outputTensorShape, ArmnnType,qScale,
                                     qOffset);

        std::vector<T> inputData = armnnUtils::QuantizedVector<T>({
                                                                          65, 144, 91, 161
                                                                  }, qScale, qOffset);

        std::vector<T> inputDataElementWise = armnnUtils::QuantizedVector<T>({
                                                                                   1, 1, 1, 1,
                                                                                   1, 1, 1, 1,
                                                                                   1, 1, 1, 1,
                                                                                   1, 1, 1, 1
                                                                           }, qScale, qOffset);

        std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>({
                                                                                   65, 144, 91, 161,
                                                                                   65, 144, 91, 161,
                                                                                   65, 144, 91, 161,
                                                                                   65, 144, 91, 161
                                                                           }, qScale, qOffset);

        auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape({ 4, 4 }));
        CHECK(descriptor.m_BroadcastToShape == outputTensorShape);
        INetworkPtr network = CreateBroadcastToNetworkWithElementWiseBinary(descriptor,
                                                                            BinaryOperation::Mul,
                                                                            inputInfo,
                                                                            inputInfoElementWise,
                                                                            outputInfo);
        // Create ArmNN runtime
        IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions());

        // Optimise ArmNN network
        IOptimizedNetworkPtr optNet = Optimize(*network, {Compute::CpuRef},
                                               run->GetDeviceSpec());

        Graph& graph = GetGraphForTesting(optNet.get());

        Optimizer::Pass(graph,
                        armnn::MakeOptimizations(armnn::optimizations::BroadcastToOptimizationLayer()));

        std::map<int, std::vector<T>> inputTensor          = { { 0, inputData  }, {1, inputDataElementWise} };
        std::map<int, std::vector<T>> expectedOutputTensor = { { 0, expectedOutputData } };
        EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),inputTensor,
                                                    expectedOutputTensor, backends);
    }

} // anonymous namespace