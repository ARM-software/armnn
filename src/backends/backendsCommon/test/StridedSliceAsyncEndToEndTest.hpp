//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ResolveType.hpp>

#include <armnn/IWorkingMemHandle.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/Threadpool.hpp>
#include <armnn/IAsyncExecutionCallback.hpp>

#include <AsyncExecutionCallback.hpp>
#include <CommonTestUtils.hpp>

#include <doctest/doctest.h>

#include <vector>

namespace armnn
{

namespace experimental
{

template<DataType ArmnnIType, DataType ArmnnOType,
        typename TInput = ResolveType <ArmnnIType>, typename TOutput = ResolveType <ArmnnOType>>
void AsyncThreadedEndToEndTestImpl(INetworkPtr network,
                                   const std::vector<std::map<int, std::vector<TInput>>>& inputTensorData,
                                   const std::vector<std::map<int, std::vector<TOutput>>>& expectedOutputData,
                                   std::vector<BackendId> backends,
                                   const size_t numberOfInferences,
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
    const INetworkProperties networkProperties(true, MemorySource::Undefined, MemorySource::Undefined);
    runtime->LoadNetwork(networkId, std::move(optNet), errorMessage, networkProperties);

    std::vector<InputTensors> inputTensorsVec;
    std::vector<OutputTensors> outputTensorsVec;
    std::vector<std::map<int, std::vector<TOutput>>> outputStorageVec;
    std::vector<std::unique_ptr<IWorkingMemHandle>> workingMemHandles;

    for (unsigned int i = 0; i < numberOfInferences; ++i)
    {
        InputTensors inputTensors;
        OutputTensors outputTensors;
        outputStorageVec.emplace_back(std::map<int, std::vector<TOutput>>());

        inputTensors.reserve(inputTensorData.size());
        for (auto&& it : inputTensorData[i])
        {
            TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(networkId, it.first);
            inputTensorInfo.SetConstant(true);
            inputTensors.push_back({it.first,
                                    ConstTensor(inputTensorInfo, it.second.data())});
        }

        outputTensors.reserve(expectedOutputData.size());
        for (auto&& it : expectedOutputData[i])
        {
            std::vector<TOutput> out(it.second.size());
            outputStorageVec[i].emplace(it.first, out);
            outputTensors.push_back({it.first,
                                     Tensor(runtime->GetOutputTensorInfo(networkId, it.first),
                                            outputStorageVec[i].at(it.first).data())});
        }

        inputTensorsVec.push_back(inputTensors);
        outputTensorsVec.push_back(outputTensors);

        workingMemHandles.push_back(runtime->CreateWorkingMemHandle(networkId));
    }

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < numberOfInferences; ++i)
    {
        // Access the vectors before we do anything multi-threaded
        InputTensors& inputTensors = inputTensorsVec[i];
        OutputTensors& outputTensors = outputTensorsVec[i];
        IWorkingMemHandle& workingMemHandle = *workingMemHandles[i].get();

        threads.emplace_back([&]()
        {
            // Run the async network
            runtime->Execute(workingMemHandle, inputTensors, outputTensors);
        });
    }

    for (unsigned int i = 0; i < numberOfInferences; ++i)
    {
        threads[i].join();
    }

    // Checks the results.
    for (unsigned int i = 0; i < numberOfInferences; ++i)
    {
        for (auto &&it : expectedOutputData[i])
        {
            std::vector<TOutput> out = outputStorageVec[i].at(it.first);
            for (unsigned int j = 0; j < out.size(); ++j)
            {
                CHECK(Compare<ArmnnOType>(it.second[j], out[j], tolerance) == true);
            }
        }
    }

}

template<DataType ArmnnIType, DataType ArmnnOType,
    typename TInput = ResolveType<ArmnnIType>, typename TOutput = ResolveType<ArmnnOType>>
void AsyncEndToEndTestImpl(INetworkPtr network,
                           const std::map<int, std::vector<TInput>>& inputTensorData,
                           const std::map<int, std::vector<TOutput>>& expectedOutputData,
                           std::vector<BackendId> backends,
                           float tolerance = 0.000001f,
                           size_t numThreads = 1)
{
    ARMNN_ASSERT(numThreads >= 1);
    const unsigned int numberOfInferences = numThreads == 1 ? 1 : 1000;

    // Create Runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr               runtime(IRuntime::Create(options));

    // Optimize the Network
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec());

    // Creates AsyncNetwork
    NetworkId networkId = 0;

    std::string errorMessage;

    const INetworkProperties networkProperties(true, MemorySource::Undefined, MemorySource::Undefined);

    runtime->LoadNetwork(networkId, std::move(optNet), errorMessage, networkProperties);

    InputTensors inputTensors;
    inputTensors.reserve(inputTensorData.size());
    for (auto&& it : inputTensorData)
    {
        TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(networkId, it.first);
        inputTensorInfo.SetConstant(true);
        inputTensors.push_back({it.first,
                                ConstTensor(inputTensorInfo, it.second.data())});
    }

    std::vector<OutputTensors> outputTensorsVec;
    std::vector<std::map<int, std::vector<TOutput>>> outputStorageVec;

    outputTensorsVec.reserve(numberOfInferences);
    outputStorageVec.reserve(numberOfInferences);
    for (unsigned int i = 0; i < numberOfInferences; ++i)
    {
        OutputTensors outputTensors;
        outputStorageVec.emplace_back(std::map<int, std::vector<TOutput>>());

        outputTensors.reserve(expectedOutputData.size());
        for (auto&& it : expectedOutputData)
        {
            std::vector<TOutput> out(it.second.size());
            outputStorageVec[i].emplace(it.first, out);
            outputTensors.push_back({it.first,
                                     Tensor(runtime->GetOutputTensorInfo(networkId, it.first),
                                            outputStorageVec[i].at(it.first).data())});
        }

        outputTensorsVec.push_back(outputTensors);
    }

    if (numThreads == 1)
    {
        // Create WorkingMemHandle for this async network
        std::unique_ptr<IWorkingMemHandle> workingMemHandle = runtime->CreateWorkingMemHandle(networkId);
        IWorkingMemHandle& workingMemHandleRef = *workingMemHandle.get();

        // Run the async network
        runtime->Execute(workingMemHandleRef, inputTensors, outputTensorsVec[0]);
    }
    else
    {
        std::vector<std::shared_ptr<IWorkingMemHandle>> memHandles;

        for (size_t i = 0; i < numThreads; ++i)
        {
            memHandles.emplace_back(runtime->CreateWorkingMemHandle(networkId));
        }

        Threadpool threadpool(numThreads, runtime.get(), memHandles);
        AsyncCallbackManager callbackManager;

        // For the asyncronous execution, we are adding a pool of working memory handles (1 per thread) in the
        // LoadedNetwork with each scheduled inference having a random priority
        for (size_t i = 0; i < numberOfInferences; ++i)
        {
            threadpool.Schedule(networkId,
                                inputTensors,
                                outputTensorsVec[i],
                                static_cast<QosExecPriority>(rand()%3),
                                callbackManager.GetNewCallback());
        }

        // Wait until the execution signals a notify
        for (size_t i = 0; i < numberOfInferences; ++i)
        {
            auto cb = callbackManager.GetNotifiedCallback();

            // Checks the results.
            CHECK(cb->GetStatus() == Status::Success);
        }
    }

    for (auto&& outputStorage : outputStorageVec)
    {
        for (auto&& it : expectedOutputData)
        {
            std::vector<TOutput> out = outputStorage.at(it.first);

            for (unsigned int i = 0; i < out.size(); ++i)
            {
                //CHECK(Compare<ArmnnOType>(it.second[i], out[i], tolerance) == true);
                CHECK(it.second[i] == doctest::Approx(out[i]).epsilon(tolerance));
            }
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
void StridedSlicedEndToEndTest(const std::vector<BackendId>& backends, size_t numThreads)
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

    CHECK(net);
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

    AsyncEndToEndTestImpl<ArmnnType, ArmnnType>(move(net),
                                                inputTensorData,
                                                expectedOutputData,
                                                backends,
                                                0.000001f,
                                                numThreads);
}

template<armnn::DataType ArmnnType>
void StridedSlicedMultiThreadedEndToEndTest(const std::vector<BackendId>& backends)
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

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData1{
            1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

            3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,

            5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
    };

    std::vector<T> outputExpected1{ 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f };

    // Creates structures for input & output.
    std::vector<T> inputData2{
            1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,

            8.0f, 8.0f, 8.0f, 7.0f, 7.0f, 7.0f,

            5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
    };

    std::vector<T> outputExpected2{ 8.0f, 8.0f, 8.0f, 7.0f, 7.0f, 7.0f };

    std::vector<std::map<int, std::vector<T>>> inputTensors;
    std::vector<std::map<int, std::vector<T>>> outputTensors;

    inputTensors.push_back(std::map<int, std::vector<T>> {{0, inputData1}});
    inputTensors.push_back(std::map<int, std::vector<T>> {{0, inputData2}});
    outputTensors.push_back(std::map<int, std::vector<T>> {{0, outputExpected1}});
    outputTensors.push_back(std::map<int, std::vector<T>> {{0, outputExpected2}});

    AsyncThreadedEndToEndTestImpl<ArmnnType, ArmnnType>(move(net), inputTensors, outputTensors, backends, 2);
}

} // experimental namespace

} // armnn namespace

