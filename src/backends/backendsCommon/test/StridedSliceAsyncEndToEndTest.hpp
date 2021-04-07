//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ResolveType.hpp>

#include <armnn/IWorkingMemHandle.hpp>
#include <armnn/INetwork.hpp>

#include <backendsCommon/test/CommonTestUtils.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

namespace armnn
{

namespace experimental
{

template<DataType ArmnnIType, DataType ArmnnOType,
        typename TInput = ResolveType <ArmnnIType>, typename TOutput = ResolveType <ArmnnOType>>
void AsyncEndToEndTestImpl(INetworkPtr network,
                           const std::map<int, std::vector<TInput>>& inputTensorData,
                           const std::map<int, std::vector<TOutput>>& expectedOutputData,
                           std::vector<BackendId> backends,
                           float tolerance = 0.000001f)
{
    // Create Runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Optimize the Network
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec());

    // Creates AsyncNetwork
    NetworkId networkId = 0;
    std::string errorMessage;
    const INetworkProperties networkProperties(false, false, true);
    runtime->LoadNetwork(networkId, std::move(optNet), errorMessage, networkProperties);

    InputTensors inputTensors;
    inputTensors.reserve(inputTensorData.size());
    for (auto&& it : inputTensorData)
    {
        inputTensors.push_back({it.first,
                                ConstTensor(runtime->GetInputTensorInfo(networkId, it.first), it.second.data())});
    }

    OutputTensors outputTensors;
    outputTensors.reserve(expectedOutputData.size());
    std::map<int, std::vector<TOutput>> outputStorage;
    for (auto&& it : expectedOutputData)
    {
        std::vector<TOutput> out(it.second.size());
        outputStorage.emplace(it.first, out);
        outputTensors.push_back({it.first,
                                 Tensor(runtime->GetOutputTensorInfo(networkId, it.first),
                                        outputStorage.at(it.first).data())});
    }

    // Create WorkingMemHandle for this async network
    std::unique_ptr<IWorkingMemHandle> workingMemHandle = runtime->CreateWorkingMemHandle(networkId);
    IWorkingMemHandle& workingMemHandleRef = *workingMemHandle.get();

    // Run the async network
    runtime->Execute(workingMemHandleRef, inputTensors, outputTensors);

    // Checks the results.
    for (auto&& it : expectedOutputData)
    {
        std::vector<TOutput> out = outputStorage.at(it.first);
        for (unsigned int i = 0; i < out.size(); ++i)
        {
            BOOST_CHECK(Compare<ArmnnOType>(it.second[i], out[i], tolerance) == true);
        }
    }
}

template<typename armnn::DataType DataType>
INetworkPtr CreateStridedSliceNetwork(const TensorShape& inputShape,
                                      const TensorShape& outputShape,
                                      const std::vector<int>& beginData,
                                      const std::vector<int>& endData,
                                      const std::vector<int>& stridesData,
                                      int beginMask = 0,
                                      int endMask = 0,
                                      int shrinkAxisMask = 0,
                                      int ellipsisMask = 0,
                                      int newAxisMask = 0,
                                      const float qScale = 1.0f,
                                      const int32_t qOffset = 0)
{
    using namespace armnn;
    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset);
    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);

    armnn::StridedSliceDescriptor stridedSliceDescriptor;
    stridedSliceDescriptor.m_Begin = beginData;
    stridedSliceDescriptor.m_End = endData;
    stridedSliceDescriptor.m_Stride = stridesData;
    stridedSliceDescriptor.m_BeginMask = beginMask;
    stridedSliceDescriptor.m_EndMask = endMask;
    stridedSliceDescriptor.m_ShrinkAxisMask = shrinkAxisMask;
    stridedSliceDescriptor.m_EllipsisMask = ellipsisMask;
    stridedSliceDescriptor.m_NewAxisMask = newAxisMask;

    IConnectableLayer* input = net->AddInputLayer(0, "Input_Layer");
    IConnectableLayer* stridedSlice = net->AddStridedSliceLayer(stridedSliceDescriptor, "splitter");
    IConnectableLayer* output = net->AddOutputLayer(0);

    Connect(input, stridedSlice, inputTensorInfo, 0, 0);
    Connect(stridedSlice, output, outputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType>
void StridedSlicedEndToEndTest(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    const TensorShape& inputShape = {3, 2, 3, 1};
    const TensorShape& outputShape = {1, 2, 3, 1};
    const std::vector<int>& beginData = {1, 0, 0, 0};
    const std::vector<int>& endData = {2, 2, 3, 1};
    const std::vector<int>& stridesData = {1, 1, 1, 1};
    int beginMask = 0;
    int endMask = 0;
    int shrinkAxisMask = 0;
    int ellipsisMask = 0;
    int newAxisMask = 0;

    // Builds up the structure of the network
    INetworkPtr net = CreateStridedSliceNetwork<ArmnnType>(inputShape,
                                                           outputShape,
                                                           beginData,
                                                           endData,
                                                           stridesData,
                                                           beginMask,
                                                           endMask,
                                                           shrinkAxisMask,
                                                           ellipsisMask,
                                                           newAxisMask);

    BOOST_TEST_CHECKPOINT("create a network");

    // Creates structures for input & output.
    std::vector<T> inputData{
            1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

            3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

            5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
    };

    std::vector<T> outputExpected{
            3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f
    };

    std::map<int, std::vector<T>> inputTensorData = {{0, inputData}};
    std::map<int, std::vector<T>> expectedOutputData = {{0, outputExpected}};

    AsyncEndToEndTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

} // experimental namespace

} // armnn namespace

