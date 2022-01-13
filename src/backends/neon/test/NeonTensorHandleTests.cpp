//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <Graph.hpp>
#include <Network.hpp>

#include <neon/NeonTensorHandle.hpp>
#include <neon/NeonTensorHandleFactory.hpp>

#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <GraphUtils.hpp>
#include <arm_compute/runtime/Allocator.h>
#include <CommonTestUtils.hpp>

#include <doctest/doctest.h>
#include <armnn/utility/Assert.hpp>

TEST_SUITE("NeonTensorHandleTests")
{
using namespace armnn;

TEST_CASE("NeonTensorHandleGetCapabilitiesNoPadding")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>();
    NeonTensorHandleFactory handleFactory(memoryManager);

    INetworkPtr network(INetwork::Create());

    // Add the layers
    IConnectableLayer* input = network->AddInputLayer(0);
    SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;
    IConnectableLayer* softmax = network->AddSoftmaxLayer(descriptor);
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // No padding required for input
    std::vector<Capability> capabilities = handleFactory.GetCapabilities(input,
                                                                         softmax,
                                                                         CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());

    // No padding required for Softmax
    capabilities = handleFactory.GetCapabilities(softmax, output, CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());

    // No padding required for output
    capabilities = handleFactory.GetCapabilities(output, nullptr, CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());
}

TEST_CASE("NeonTensorHandleGetCapabilitiesPadding")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>();
    NeonTensorHandleFactory handleFactory(memoryManager);

    INetworkPtr network(INetwork::Create());

    // Add the layers
    IConnectableLayer* input = network->AddInputLayer(0);
    Pooling2dDescriptor descriptor;
    IConnectableLayer* pooling = network->AddPooling2dLayer(descriptor);
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // No padding required for input
    std::vector<Capability> capabilities = handleFactory.GetCapabilities(input,
                                                                         pooling,
                                                                         CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());

    // No padding required for output
    capabilities = handleFactory.GetCapabilities(output, nullptr, CapabilityClass::PaddingRequired);
    CHECK(capabilities.empty());

    // Padding required for Pooling2d
    capabilities = handleFactory.GetCapabilities(pooling, output, CapabilityClass::PaddingRequired);
    CHECK(capabilities.size() == 1);
    CHECK((capabilities[0].m_CapabilityClass == CapabilityClass::PaddingRequired));
    CHECK(capabilities[0].m_Value);
}

TEST_CASE("ConcatOnXorYSubTensorsNoPaddingRequiredTest")
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    // Set up tensor infos
    const armnn::TensorInfo inputInfo = armnn::TensorInfo({2, 3, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo intermediateInfo = armnn::TensorInfo({2, 3, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({2, 3, 4, 2}, armnn::DataType::Float32);

    armnn::ElementwiseUnaryDescriptor descriptor(armnn::UnaryOperation::Abs);

    // Create the network
    armnn::IConnectableLayer* const input0Layer = net->AddInputLayer(0, "input_0");
    input0Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    armnn::IConnectableLayer* elementwiseUnaryLayer0 = net->AddElementwiseUnaryLayer(descriptor, "elementwiseUnary_0");
    elementwiseUnaryLayer0->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input0Layer->GetOutputSlot(0).Connect(elementwiseUnaryLayer0->GetInputSlot(0));

    armnn::IConnectableLayer* const input1Layer = net->AddInputLayer(1, "input_1");
    input1Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    armnn::IConnectableLayer* elementwiseUnaryLayer1 = net->AddElementwiseUnaryLayer(descriptor, "elementwiseUnary_1");
    elementwiseUnaryLayer1->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input1Layer->GetOutputSlot(0).Connect(elementwiseUnaryLayer1->GetInputSlot(0));

    std::array<armnn::TensorShape, 2> concatInputShapes = { intermediateInfo.GetShape(), intermediateInfo.GetShape() };
    armnn::IConnectableLayer* const concatLayer = net->AddConcatLayer(armnn::CreateDescriptorForConcatenation(
        concatInputShapes.begin(), concatInputShapes.end(), 2), "concatenation");
    concatLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    elementwiseUnaryLayer0->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    elementwiseUnaryLayer1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());

    const armnn::Graph& theGraph = GetGraphForTesting(optimizedNet.get());

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));

    // now check the concat how many sub-tensors it is using..
    auto TraceSubTensorHandleAncestry = [](armnn::ITensorHandle* const subTensorHandle)
    {
        if (subTensorHandle && subTensorHandle->GetParent())
        {
            return true;
        }
        return false;
    };

    for (auto&& layer : theGraph)
    {
        if(layer->GetType() == armnn::LayerType::Concat)
        {
            unsigned int numberOfSubTensors = 0;
            for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
            {
                const armnn::OutputSlot* slot = layer->GetInputSlot(i).GetConnectedOutputSlot();
                if (TraceSubTensorHandleAncestry(slot->GetOutputHandler().GetData()))
                {
                    ++numberOfSubTensors;
                }
            }
            // sub-tensors should be supported in this configuration
            ARMNN_ASSERT(numberOfSubTensors > 0);
        }
    }
}

TEST_CASE("ConcatonXorYPaddingRequiredTest")
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    // Set up tensor infos
    const armnn::TensorInfo inputInfo = armnn::TensorInfo({2, 3, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo intermediateInfo = armnn::TensorInfo({2, 3, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({2, 3, 4, 2}, armnn::DataType::Float32);

    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    // Create the network
    armnn::IConnectableLayer* const input0Layer = net->AddInputLayer(0, "input_0");
    input0Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    armnn::IConnectableLayer* pooling2dLayer0 = net->AddPooling2dLayer(descriptor, "pooling2d_0");
    pooling2dLayer0->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input0Layer->GetOutputSlot(0).Connect(pooling2dLayer0->GetInputSlot(0));

    armnn::IConnectableLayer* const input1Layer = net->AddInputLayer(1, "input_1");
    input1Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    armnn::IConnectableLayer* pooling2dLayer1 = net->AddPooling2dLayer(descriptor, "pooling2d_1");
    pooling2dLayer1->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input1Layer->GetOutputSlot(0).Connect(pooling2dLayer1->GetInputSlot(0));

    std::array<armnn::TensorShape, 2> concatInputShapes = { intermediateInfo.GetShape(), intermediateInfo.GetShape() };
    armnn::IConnectableLayer* const concatLayer = net->AddConcatLayer(armnn::CreateDescriptorForConcatenation(
        concatInputShapes.begin(), concatInputShapes.end(), 2), "concatenation");
    concatLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    pooling2dLayer0->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    pooling2dLayer1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());

    const armnn::Graph& theGraph = GetGraphForTesting(optimizedNet.get());

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));

    // now check the concat how many sub-tensors it is using..
    auto TraceSubTensorHandleAncestry = [](armnn::ITensorHandle* const subTensorHandle)
    {
        if (subTensorHandle && subTensorHandle->GetParent())
        {
            return true;
        }
        return false;
    };

    unsigned int numberOfSubTensors = 0;
    for (auto&& layer : theGraph)
    {
        if(layer->GetType() == armnn::LayerType::Concat)
        {
            for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
            {
                const armnn::OutputSlot* slot = layer->GetInputSlot(i).GetConnectedOutputSlot();
                if (TraceSubTensorHandleAncestry(slot->GetOutputHandler().GetData()))
                {
                    ++numberOfSubTensors;
                }
            }
        }
    }
    // sub-tensors should not be supported in this configuration
    ARMNN_ASSERT(numberOfSubTensors == 0);
}

TEST_CASE("SplitteronXorYNoPaddingRequiredTest")
{
    using namespace armnn;

    unsigned int splitAxis = 2;
    unsigned int numSplit = 2;

    const TensorShape& inputShape = { 2, 3, 4, 2 };
    const armnn::TensorInfo intermediateInfo = armnn::TensorInfo({ 2, 3, 2, 2 }, armnn::DataType::Float32);
    const std::vector<TensorShape> outputShapes{{ 2, 3, 2, 2 },
                                                { 2, 3, 2, 2 }};
    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    // Creates structures for input & output.
    std::vector<float> inputData{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12,
            13, 14,
            15, 16,
            17, 18,
            19, 20,
            21, 22,
            23, 24,
            25, 26,
            27, 28,
            29, 30,
            31, 32,
            33, 34,
            35, 36,
            37, 38,
            39, 40,
            41, 42,
            43, 44,
            45, 46,
            47, 48
    };

    std::vector<float> expectedOutput0{
            1, 2,
            3, 4,
            9, 10,
            11, 12,
            17, 18,
            19, 20,
            25, 26,
            27, 28,
            33, 34,
            35, 36,
            41, 42,
            43, 44
    };

    std::vector<float> expectedOutput1{
            5, 6,
            7, 8,
            13, 14,
            15, 16,
            21, 22,
            23, 24,
            29, 30,
            31, 32,
            37, 38,
            39, 40,
            45, 46,
            47, 48
    };

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32, qScale, qOffset);

    armnn::ElementwiseUnaryDescriptor descriptor(armnn::UnaryOperation::Abs);

    // Splitter
    std::vector<unsigned int> splitterDimSizes(inputShape.GetNumDimensions());

    // Add current input shape to splitterDimSizes
    for (unsigned int i = 0; i < inputShape.GetNumDimensions(); ++i)
    {
        splitterDimSizes[i] = inputTensorInfo.GetShape()[i];
    }

    if (splitterDimSizes[splitAxis] % numSplit != 0)
    {
        throw ParseException("Number of splits must evenly divide the dimension");
    }

    splitterDimSizes[splitAxis] /= numSplit;

    SplitterDescriptor splitDesc(numSplit, inputShape.GetNumDimensions());

    for (unsigned int g = 0; g < numSplit; ++g)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < splitterDimSizes.size(); ++dimIdx)
        {
            splitDesc.SetViewSize(g, dimIdx, splitterDimSizes[dimIdx]);
        }
        splitDesc.SetViewOriginCoord(g, splitAxis, splitterDimSizes[splitAxis] * g);
    }
    IConnectableLayer* input = net->AddInputLayer(0, "input");
    IConnectableLayer* elementWiseUnary0 = net->AddElementwiseUnaryLayer(descriptor, "elementwiseunary_0");
    IConnectableLayer* elementWiseUnary1 = net->AddElementwiseUnaryLayer(descriptor, "elementwiseunary_0");
    IConnectableLayer* splitter = net->AddSplitterLayer(splitDesc, "splitter");

    // Connections
    Connect(input, splitter, inputTensorInfo, 0, 0);
    Connect(splitter, elementWiseUnary0, intermediateInfo, 0, 0);
    Connect(splitter, elementWiseUnary1, intermediateInfo, 1, 0);

    std::vector<IConnectableLayer*> pooling2dLayers{elementWiseUnary0, elementWiseUnary1};

    for (unsigned int i = 0; i < outputShapes.size(); ++i)
    {
        TensorInfo outputTensorInfo(outputShapes[i], armnn::DataType::Float32, qScale, qOffset);
        IConnectableLayer* output = net->AddOutputLayer(armnn::numeric_cast<LayerBindingId>(i));
        Connect(pooling2dLayers[i], output, outputTensorInfo, 0, 0);
    }

    std::map<int, std::vector<float>> inputTensorData = {{ 0,inputData }};
    std::map<int, std::vector<float>> expectedOutputData = {{ 0, expectedOutput0 }, { 1, expectedOutput1 }};

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());

    const armnn::Graph& theGraph = GetGraphForTesting(optimizedNet.get());

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));

    // now check the concat how many sub-tensors it is using..
    auto TraceSubTensorHandleAncestry = [](armnn::ITensorHandle* const subTensorHandle)
    {
        if (subTensorHandle && subTensorHandle->GetParent())
        {
            return true;
        }
        return false;
    };

    for (auto&& layer : theGraph)
    {
        if(layer->GetType() == armnn::LayerType::ElementwiseUnary)
        {
            unsigned int numberOfSubTensors = 0;
            for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
            {
                const armnn::OutputSlot* slot = layer->GetInputSlot(i).GetConnectedOutputSlot();
                if (TraceSubTensorHandleAncestry(slot->GetOutputHandler().GetData()))
                {
                    ++numberOfSubTensors;
                }
            }
            // sub-tensors should be supported in this configuration
            ARMNN_ASSERT(numberOfSubTensors > 0);
        }
    }

    InputTensors inputTensors;
    inputTensors.reserve(inputTensorData.size());
    for (auto&& it : inputTensorData)
    {
        TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(networkIdentifier, it.first);
        inputTensorInfo.SetConstant(true);
        inputTensors.push_back({it.first,
                                ConstTensor(inputTensorInfo, it.second.data())});
    }
    OutputTensors outputTensors;
    outputTensors.reserve(expectedOutputData.size());
    std::map<int, std::vector<float>> outputStorage;
    for (auto&& it : expectedOutputData)
    {
        std::vector<float> out(it.second.size());
        outputStorage.emplace(it.first, out);
        outputTensors.push_back({it.first,
                                 Tensor(runtime->GetOutputTensorInfo(networkIdentifier, it.first),
                                               outputStorage.at(it.first).data())});
    }

    // Does the inference.
    runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // Checks the results.
    float tolerance = 0.000001f;
    for (auto&& it : expectedOutputData)
    {
        std::vector<float> out = outputStorage.at(it.first);
        for (unsigned int i = 0; i < out.size(); ++i)
        {
            CHECK_MESSAGE(Compare<armnn::DataType::Float32>(it.second[i], out[i], tolerance) == true,
                    "Actual output: " << out[i] << ". Expected output:" << it.second[i]);

        }
    }
}

TEST_CASE("SplitteronXorYPaddingRequiredTest")
{
    using namespace armnn;

    unsigned int splitAxis = 2;
    unsigned int numSplit = 2;

    const TensorShape& inputShape = { 1, 1, 4, 4 };
    const armnn::TensorInfo intermediateInfo = armnn::TensorInfo({ 1, 1, 2, 4 }, armnn::DataType::Float32);
    const std::vector<TensorShape> outputShapes{{ 1, 1, 2, 4 },
                                                { 1, 1, 2, 4 }};

    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    // Creates structures for input & output.
    std::vector<float> inputData{
        9.0f,   27.0f,  18.0f,  36.0f,
        18.0f,   9.0f,  18.0f,   9.0f,
        27.0f,  18.0f,   9.0f,  27.0f,
        9.0f,   27.0f,   9.0f,  18.0f,
    };

    std::vector<float> expectedOutput0{
         7.0f,  11.0f,  13.0f, 9.0f,
         7.0f,  11.0f,  13.0f, 9.0f
    };

    std::vector<float> expectedOutput1{
        9.0f,  11.0f,  12.0f, 7.0f,
        9.0f,  11.0f,  12.0f, 7.0f
    };

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32, qScale, qOffset);

    // Pooling
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 3;
    descriptor.m_StrideX = descriptor.m_StrideY = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    // Splitter
    std::vector<unsigned int> splitterDimSizes(inputShape.GetNumDimensions());

    // Add current input shape to splitterDimSizes
    for (unsigned int i = 0; i < inputShape.GetNumDimensions(); ++i)
    {
        splitterDimSizes[i] = inputTensorInfo.GetShape()[i];
    }

    if (splitterDimSizes[splitAxis] % numSplit != 0)
    {
        throw ParseException("Number of splits must evenly divide the dimension");
    }

    splitterDimSizes[splitAxis] /= numSplit;

    SplitterDescriptor splitDesc(numSplit, inputShape.GetNumDimensions());

    for (unsigned int g = 0; g < numSplit; ++g)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < splitterDimSizes.size(); ++dimIdx)
        {
            splitDesc.SetViewSize(g, dimIdx, splitterDimSizes[dimIdx]);
        }
        splitDesc.SetViewOriginCoord(g, splitAxis, splitterDimSizes[splitAxis] * g);
    }

    IConnectableLayer* input = net->AddInputLayer(0, "input");
    IConnectableLayer* pooling2d0 = net->AddPooling2dLayer(descriptor, "pooling2d_0");
    IConnectableLayer* pooling2d1 = net->AddPooling2dLayer(descriptor, "pooling2d_1");
    IConnectableLayer* splitter = net->AddSplitterLayer(splitDesc, "splitter");

    // Connections
    Connect(input, splitter, inputTensorInfo, 0, 0);
    Connect(splitter, pooling2d0, intermediateInfo, 0, 0);
    Connect(splitter, pooling2d1, intermediateInfo, 1, 0);

    std::vector<IConnectableLayer*> pooling2dLayers{pooling2d0, pooling2d1};

    for (unsigned int i = 0; i < outputShapes.size(); ++i)
    {
        TensorInfo outputTensorInfo(outputShapes[i], armnn::DataType::Float32, qScale, qOffset);
        IConnectableLayer* output = net->AddOutputLayer(armnn::numeric_cast<LayerBindingId>(i));
        Connect(pooling2dLayers[i], output, outputTensorInfo, 0, 0);
    }

    std::map<int, std::vector<float>> inputTensorData = {{ 0,inputData }};
    std::map<int, std::vector<float>> expectedOutputData = {{ 0, expectedOutput0 }, { 1, expectedOutput1 }};

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());

    const armnn::Graph& theGraph = GetGraphForTesting(optimizedNet.get());

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));

    // now check the concat how many sub-tensors it is using..
    auto TraceSubTensorHandleAncestry = [](armnn::ITensorHandle* const subTensorHandle)
    {
        if (subTensorHandle && subTensorHandle->GetParent())
        {
            return true;
        }
        return false;
    };

    for (auto&& layer : theGraph)
    {
        if(layer->GetType() == armnn::LayerType::Pooling2d)
        {
            unsigned int numberOfSubTensors = 0;
            for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
            {
                const armnn::OutputSlot* slot = layer->GetInputSlot(i).GetConnectedOutputSlot();
                if (TraceSubTensorHandleAncestry(slot->GetOutputHandler().GetData()))
                {
                    ++numberOfSubTensors;
                }
            }
            // sub-tensors should be supported in this configuration
            ARMNN_ASSERT(numberOfSubTensors == 0);
        }
    }

    InputTensors inputTensors;
    inputTensors.reserve(inputTensorData.size());
    for (auto&& it : inputTensorData)
    {
        TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(networkIdentifier, it.first);
        inputTensorInfo.SetConstant(true);
        inputTensors.push_back({it.first,
                                ConstTensor(inputTensorInfo, it.second.data())});
    }
    OutputTensors outputTensors;
    outputTensors.reserve(expectedOutputData.size());
    std::map<int, std::vector<float>> outputStorage;
    for (auto&& it : expectedOutputData)
    {
        std::vector<float> out(it.second.size());
        outputStorage.emplace(it.first, out);
        outputTensors.push_back({it.first,
                                 Tensor(runtime->GetOutputTensorInfo(networkIdentifier, it.first),
                                               outputStorage.at(it.first).data())});
    }

    // Does the inference.
    runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // Checks the results.
    float tolerance = 0.000001f;
    for (auto&& it : expectedOutputData)
    {
        std::vector<float> out = outputStorage.at(it.first);
        for (unsigned int i = 0; i < out.size(); ++i)
        {
            CHECK_MESSAGE(Compare<armnn::DataType::Float32>(it.second[i], out[i], tolerance) == true,
                    "Actual output: " << out[i] << ". Expected output:" << it.second[i]);

        }
    }
}

TEST_CASE("NeonTensorHandleFactoryMemoryManaged")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>(
        std::make_unique<arm_compute::Allocator>(),
        BaseMemoryManager::MemoryAffinity::Offset);
    NeonTensorHandleFactory handleFactory(memoryManager);
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);

    // create TensorHandle with memory managed
    auto handle = handleFactory.CreateTensorHandle(info, true);
    handle->Manage();
    handle->Allocate();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle->Map());
        CHECK(buffer != nullptr); // Yields a valid pointer
        buffer[0] = 1.5f;
        buffer[1] = 2.5f;
        CHECK(buffer[0] == 1.5f); // Memory is writable and readable
        CHECK(buffer[1] == 2.5f); // Memory is writable and readable
    }
    memoryManager->Release();

    memoryManager->Acquire();
    {
        float* buffer = reinterpret_cast<float*>(handle->Map());
        CHECK(buffer != nullptr); // Yields a valid pointer
        buffer[0] = 3.5f;
        buffer[1] = 4.5f;
        CHECK(buffer[0] == 3.5f); // Memory is writable and readable
        CHECK(buffer[1] == 4.5f); // Memory is writable and readable
    }
    memoryManager->Release();

    float testPtr[2] = { 2.5f, 5.5f };
    // Cannot import as import is disabled
    CHECK_THROWS_AS(handle->Import(static_cast<void*>(testPtr), MemorySource::Malloc), MemoryImportException);
}

TEST_CASE("NeonTensorHandleFactoryImport")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>(
        std::make_unique<arm_compute::Allocator>(),
        BaseMemoryManager::MemoryAffinity::Offset);
    NeonTensorHandleFactory handleFactory(memoryManager);
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);

    // create TensorHandle without memory managed
    auto handle = handleFactory.CreateTensorHandle(info, false);
    handle->Manage();
    handle->Allocate();
    memoryManager->Acquire();

    // No buffer allocated when import is enabled
    CHECK((PolymorphicDowncast<NeonTensorHandle*>(handle.get()))->GetTensor().buffer() == nullptr);

    float testPtr[2] = { 2.5f, 5.5f };
    // Correctly import
    CHECK(handle->Import(static_cast<void*>(testPtr), MemorySource::Malloc));
    float* buffer = reinterpret_cast<float*>(handle->Map());
    CHECK(buffer != nullptr); // Yields a valid pointer after import
    CHECK(buffer == testPtr); // buffer is pointing to testPtr
    // Memory is writable and readable with correct value
    CHECK(buffer[0] == 2.5f);
    CHECK(buffer[1] == 5.5f);
    buffer[0] = 3.5f;
    buffer[1] = 10.0f;
    CHECK(buffer[0] == 3.5f);
    CHECK(buffer[1] == 10.0f);
    memoryManager->Release();
}

TEST_CASE("NeonTensorHandleCanBeImported")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>(
        std::make_unique<arm_compute::Allocator>(),
        BaseMemoryManager::MemoryAffinity::Offset);
    NeonTensorHandleFactory handleFactory(memoryManager);
    TensorInfo info({ 1, 1, 2, 1 }, DataType::Float32);

    // create TensorHandle (Memory Managed status is irrelevant)
    auto handle = handleFactory.CreateTensorHandle(info, false);

    // Create an aligned buffer
    float alignedBuffer[2] = { 2.5f, 5.5f };
    // Check aligned buffers return true
    CHECK(handle->CanBeImported(&alignedBuffer, MemorySource::Malloc) == true);

    // Create a misaligned buffer from the aligned one
    float* misalignedBuffer = reinterpret_cast<float*>(reinterpret_cast<char*>(alignedBuffer) + 1);
    // Check misaligned buffers return false
    CHECK(handle->CanBeImported(static_cast<void*>(misalignedBuffer), MemorySource::Malloc) == false);
}

TEST_CASE("NeonTensorHandleSupportsInPlaceComputation")
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>();
    NeonTensorHandleFactory handleFactory(memoryManager);

    // NeonTensorHandleFactory supports InPlaceComputation
    ARMNN_ASSERT(handleFactory.SupportsInPlaceComputation());
}

}
