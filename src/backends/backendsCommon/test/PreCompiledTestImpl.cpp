//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PreCompiledTestImpl.hpp"

#include "TensorCopyUtils.hpp"

#include <Graph.hpp>
#include <Network.hpp>
#include <Runtime.hpp>

#include <armnn/Exceptions.hpp>
#include <armnn/INetwork.hpp>

#include <test/TensorHelpers.hpp>

#include <backendsCommon/WorkloadFactory.hpp>

#include <boost/polymorphic_pointer_cast.hpp>

using namespace armnn;

namespace
{

template<typename ConvolutionDescriptor>
struct PreCompiledConvolutionHelper
{
};

template<>
struct PreCompiledConvolutionHelper<Convolution2dDescriptor>
{
    static IConnectableLayer* AddConvolutionLayerToNetwork(
        Network& network,
        Convolution2dDescriptor descriptor,
        const ConstTensor& weights,
        const ConstTensor& biases)
    {
        return network.AddConvolution2dLayer(descriptor, weights, biases, "convolution");
    }
};

template<>
struct PreCompiledConvolutionHelper<DepthwiseConvolution2dDescriptor>
{
    static IConnectableLayer* AddConvolutionLayerToNetwork(
        Network& network,
        DepthwiseConvolution2dDescriptor descriptor,
        const ConstTensor& weights,
        const ConstTensor& biases)
    {
        return network.AddDepthwiseConvolution2dLayer(descriptor, weights, biases, "depthwiseConvolution");
    }
};

template<typename ConvolutionDescriptor>
ConvolutionDescriptor CreateConvolutionDescriptor(unsigned int stride, unsigned int padding)
{
    ConvolutionDescriptor descriptor;

    descriptor.m_StrideX     = stride;
    descriptor.m_StrideY     = stride;
    descriptor.m_PadLeft     = padding;
    descriptor.m_PadRight    = padding;
    descriptor.m_PadTop      = padding;
    descriptor.m_PadBottom   = padding;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = DataLayout::NHWC;

    return descriptor;
}

static std::vector<uint8_t> CreateIdentityConvolutionKernel(
    unsigned int kernelSize, unsigned int channels)
{
    BOOST_ASSERT(kernelSize % 2 == 1); // kernelSize need to be an odd number

    const unsigned int numElements = channels * (kernelSize * kernelSize);
    std::vector<uint8_t> kernel(numElements, 0u);

    unsigned int centerIndex = kernelSize / 2;
    for(unsigned int y = 0u; y < kernelSize; y++)
    {
        for(unsigned int x = 0u; x < kernelSize; x++)
        {
            for(unsigned int channel = 0u; channel < channels; channel++)
            {
                if (x == centerIndex && y == centerIndex)
                {
                    const unsigned int flatIndex =
                        (y * kernelSize * channels) + (x * channels) + channel;

                    kernel[flatIndex] = 1u;
                }
            }
        }
    }

    return kernel;
}

template<typename ConvolutionDescriptor>
std::vector<uint8_t> GetIdentityConvolutionExpectedOutputData(
    const TensorInfo& inputInfo,
    const TensorInfo& outputInfo,
    const ConvolutionDescriptor& descriptor,
    const std::vector<uint8_t>& inputData)
{
    const unsigned int outputDataSize = outputInfo.GetNumElements();
    std::vector<uint8_t> expectedOutputData(outputDataSize);

    const unsigned int channels = outputInfo.GetShape()[3];
    BOOST_ASSERT(channels == inputInfo.GetShape()[3]);

    const unsigned int inputW  = inputInfo.GetShape()[2];

    const unsigned int outputH = outputInfo.GetShape()[1];
    const unsigned int outputW = outputInfo.GetShape()[2];

    // Pick values from the input buffer, but after each iteration skip a number of
    // rows and columns equal to the stride in the respective dimension
    for (unsigned int inputY = 0, outputY = 0; outputY < outputH; inputY += descriptor.m_StrideY, outputY++)
    {
        for (unsigned int inputX = 0, outputX = 0; outputX < outputW; inputX += descriptor.m_StrideX, outputX++)
        {
            for (unsigned int channel = 0u; channel < channels; channel++)
            {
                const unsigned int inputIndex  =
                    (inputY * inputW * channels) + (inputX * channels) + channel;
                const unsigned int outputIndex =
                    (outputY * outputW * channels) + (outputX * channels) + channel;

                expectedOutputData[outputIndex] = inputData[inputIndex];
            }
        }
    }

    return expectedOutputData;
}

armnn::PreCompiledLayer* FindPreCompiledLayer(armnn::Graph& optimisedGraph)
{
    for (auto& layer : optimisedGraph)
    {
        if (layer->GetType() == armnn::LayerType::PreCompiled)
        {
            return boost::polymorphic_pointer_downcast<armnn::PreCompiledLayer>(layer);
        }
    }

    // No pre-compiled layer found
    return nullptr;
}

// NOTE: This only supports a single input and a single output
LayerTestResult<uint8_t, 4> OptimiseAndRunNetwork(armnn::IWorkloadFactory& workloadFactory,
                                                  Network& net,
                                                  TensorInfo inputInfo,
                                                  std::vector<uint8_t> inputData,
                                                  TensorInfo outputInfo,
                                                  std::vector<uint8_t> expectedOutputData)
{
    // Optimize the network for the backend supported by the factory
    std::vector<BackendId> backends = {workloadFactory.GetBackendId()};
    IRuntimePtr runtime(IRuntime::Create(IRuntime::CreationOptions()));
    IOptimizedNetworkPtr optimizedNet = Optimize(net, backends, runtime->GetDeviceSpec(), OptimizerOptions());
    if (!optimizedNet)
    {
        throw RuntimeException(std::string("Failed to optimize network for ") + std::string(backends[0]),
                               CHECK_LOCATION());
    }

    // Find the pre-compiled layer in the optimised graph
    Graph& optimisedGraph = static_cast<OptimizedNetwork*>(optimizedNet.get())->GetGraph();
    PreCompiledLayer* preCompiledLayer = FindPreCompiledLayer(optimisedGraph);
    if (!preCompiledLayer)
    {
        throw RuntimeException("Could not find pre-compiled layer in optimised graph", CHECK_LOCATION());
    }

    // Create the tensor handles
    for (auto&& layer : optimisedGraph.TopologicalSort())
    {
        layer->CreateTensorHandles(optimisedGraph, workloadFactory);
    }

    // Create the pre-compiled workload
    auto workload = preCompiledLayer->CreateWorkload(optimisedGraph, workloadFactory);

    // Set the input data
    boost::multi_array<uint8_t, 4> input = MakeTensor<uint8_t, 4>(inputInfo, inputData);
    const QueueDescriptor& workloadData =
        static_cast<BaseWorkload<PreCompiledQueueDescriptor>*>(workload.get())->GetData();
    CopyDataToITensorHandle(workloadData.m_Inputs[0], &input[0][0][0][0]);

    // Execute the workload
    workload->Execute();

    // Set the expected and actual outputs
    LayerTestResult<uint8_t, 4> result(outputInfo);
    result.outputExpected = MakeTensor<uint8_t, 4>(outputInfo, expectedOutputData);
    CopyDataFromITensorHandle(&result.output[0][0][0][0], workloadData.m_Outputs[0]);
    return result;
}

} // anonymous namespace

template<typename ConvolutionDescriptor>
LayerTestResult<uint8_t, 4> PreCompiledConvolution2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    unsigned int inputSize,
    unsigned int outputSize,
    unsigned int channels,
    unsigned int kernelSize,
    const ConvolutionDescriptor& descriptor,
    bool isDepthwiseConvolution = false)
{
    BOOST_ASSERT(descriptor.m_BiasEnabled == true);
    BOOST_ASSERT(descriptor.m_DataLayout  == DataLayout::NHWC);

    // Set up tensor shapes and infos
    const TensorShape inputShape ({1, inputSize,  inputSize,  channels});
    const TensorShape outputShape({1, outputSize, outputSize, channels});
    const TensorShape kernelShape = isDepthwiseConvolution
                                    // The format for the depthwise convolution is MIHW
                                    ? TensorShape({1, channels, kernelSize, kernelSize})
                                    // The format for the regular convolution depends on the layout of the inputs,
                                    // in this case is NHWC
                                    : TensorShape({1, kernelSize, kernelSize, channels});
    const TensorShape biasesShape({1, 1, 1, channels});

    // NOTE: inputScale * weightsScale / outputScale must be >= 0.0 and < 1.0
    TensorInfo inputInfo(inputShape, DataType::QuantisedAsymm8, 1.0f, 0);
    TensorInfo outputInfo(outputShape, DataType::QuantisedAsymm8, 2.0f, 0);
    TensorInfo weightsInfo(kernelShape, DataType::QuantisedAsymm8, 1.0f, 0);
    TensorInfo biasesInfo(biasesShape, DataType::Signed32, 1.0f, 0);

    // Populate weight and bias data
    std::vector<uint8_t> weightsData = CreateIdentityConvolutionKernel(kernelSize, channels);

    // NOTE: We need to multiply the elements of the identity kernel by 2
    // to compensate for the scaling factor
    std::transform(weightsData.begin(), weightsData.end(), weightsData.begin(),
        [](uint8_t w) -> uint8_t { return static_cast<uint8_t>(w * 2); });

    const unsigned int biasDataSize = biasesInfo.GetNumElements();
    std::vector<int32_t> biasesData(biasDataSize, 0);

    // Construct network
    Network network;
    ConstTensor weights(weightsInfo, weightsData);
    ConstTensor biases(biasesInfo, biasesData);

    IConnectableLayer* const inputLayer = network.AddInputLayer(0, "input");

    IConnectableLayer* const convolutionLayer =
        PreCompiledConvolutionHelper<ConvolutionDescriptor>
            ::AddConvolutionLayerToNetwork(network, descriptor, weights, biases);

    IConnectableLayer* const outputLayer = network.AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convolutionLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    convolutionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convolutionLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Generate input data: sequence [0, 1 .. 255]
    const unsigned int inputDataSize = inputInfo.GetNumElements();
    std::vector<uint8_t> inputData(inputDataSize);
    std::iota(inputData.begin(), inputData.end(), 0);

    // Set expected output
    std::vector<uint8_t> expectedOutputData =
        GetIdentityConvolutionExpectedOutputData(inputInfo,
                                                 outputInfo,
                                                 descriptor,
                                                 inputData);

    return OptimiseAndRunNetwork(workloadFactory,
                                 network,
                                 inputInfo,
                                 inputData,
                                 outputInfo,
                                 expectedOutputData);
}

LayerTestResult<uint8_t, 4> PreCompiledConvolution2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 1;
    const unsigned int padding    = 1;

    Convolution2dDescriptor descriptor =
        CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory,
                                            memoryManager,
                                            inputSize,
                                            outputSize,
                                            channels,
                                            kernelSize,
                                            descriptor);
}

LayerTestResult<uint8_t, 4> PreCompiledConvolution2dStride2x2TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 8;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 2;
    const unsigned int padding    = 1;

    Convolution2dDescriptor descriptor =
        CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory,
                                            memoryManager,
                                            inputSize,
                                            outputSize,
                                            channels,
                                            kernelSize,
                                            descriptor);
}

LayerTestResult<uint8_t, 4> PreCompiledDepthwiseConvolution2dTestImpl(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 3;
    const unsigned int kernelSize = 1;
    const unsigned int stride     = 1;
    const unsigned int padding    = 0;

    DepthwiseConvolution2dDescriptor descriptor =
        CreateConvolutionDescriptor<DepthwiseConvolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory,
                                            memoryManager,
                                            inputSize,
                                            outputSize,
                                            channels,
                                            kernelSize,
                                            descriptor,
                                            true);
}

LayerTestResult<uint8_t, 4> PreCompiledDepthwiseConvolution2dStride2x2TestImpl(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 8;
    const unsigned int channels   = 3;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 2;
    const unsigned int padding    = 1;

    DepthwiseConvolution2dDescriptor descriptor =
        CreateConvolutionDescriptor<DepthwiseConvolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory,
                                            memoryManager,
                                            inputSize,
                                            outputSize,
                                            channels,
                                            kernelSize,
                                            descriptor);
}

LayerTestResult<uint8_t, 4> PreCompiledMaxPooling2dTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    // Pooling cannot be run in isolation, it must be fused with the previous layer, e.g. Convolution2d.

    // Set up the Convolution descriptor
    Convolution2dDescriptor convDescriptor;
    convDescriptor.m_StrideX = 1;
    convDescriptor.m_StrideY = 1;
    convDescriptor.m_BiasEnabled = true;
    convDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Set up the Convolution weights
    TensorInfo weightsInfo(TensorShape({16, 1, 1, 16}), DataType::QuantisedAsymm8, 2.0f, 0);
    const unsigned int weightsDataSize = weightsInfo.GetNumElements();
    std::vector<uint8_t> weightsData(weightsDataSize);
    for (unsigned int i = 0; i < 16; ++i)
    {
        for (unsigned int j = 0; j < 16; ++j)
        {
            weightsData[(i * 16) + j] = i == j ? 1.0f : 0.0f;
        }
    }
    ConstTensor weights(weightsInfo, weightsData);

    // Set up the Convolution biases
    TensorInfo biasInfo(TensorShape({1, 1, 1, 16}), DataType::Signed32, 1.0f * 2.0f, 0);
    const unsigned int biasDataSize = biasInfo.GetNumElements();
    std::vector<int32_t> biasData(biasDataSize, 0);
    ConstTensor biases(biasInfo, biasData);

    // Set up the Convolution input
    TensorInfo inputInfo(TensorShape({1, 16, 16, 16 }), DataType::QuantisedAsymm8, 1.0f, 0);
    const unsigned int inputDataSize = inputInfo.GetNumElements();
    std::vector<uint8_t> inputData(inputDataSize);
    for (unsigned int i = 0; i < inputDataSize; ++i)
    {
        inputData[i] = boost::numeric_cast<uint8_t>((i * 4) % 250);
    }

    // Set up the Convolution output / Pooling input info
    TensorInfo convOutputInfo(TensorShape({1, 16, 16, 16 }), DataType::QuantisedAsymm8, 4.0f, 0);

    // Set up the Pooling descriptor
    Pooling2dDescriptor poolDescriptor;
    poolDescriptor.m_PoolType = PoolingAlgorithm::Max;
    poolDescriptor.m_PoolWidth = 2;
    poolDescriptor.m_PoolHeight = 2;
    poolDescriptor.m_StrideX = 2;
    poolDescriptor.m_StrideY = 2;
    poolDescriptor.m_PaddingMethod = PaddingMethod::Exclude;
    poolDescriptor.m_DataLayout = DataLayout::NHWC;

    // Set the expected output from the Pooling layer
    TensorInfo outputInfo(TensorShape({1, 8, 8, 16 }), DataType::QuantisedAsymm8, 4.0f, 0);
    const unsigned int outputDataSize = outputInfo.GetNumElements();
    std::vector<uint8_t> expectedOutputData(outputDataSize);
    // The Maxpooling inputs are the Convolution outputs, i.e. (Convolution inputs / 2) after scale adjustments
    // Maxpooling selects the max value in each pool from its inputs and our pool size is 2x2
    for (unsigned int channel = 0; channel < 16; ++channel)
    {
        for (unsigned int row = 0; row < 8; ++row)
        {
            for (unsigned int column = 0; column < 8; ++column)
            {
                // The input and output data indexes are calculated for NHWC data layout.
                // Output index: (row * columns * channels) + (column * channels) + channel
                auto outIndex = (row * 8 * 16) + (column * 16) + channel;
                // Input index: (row * strideY * columns * channels) + (column * strideX * channels) + channel
                //      and we take 4 entries for the 2x2 pool
                auto in0Index = ((row * 2) * 16 * 16) + ((column * 2) * 16) + channel;
                auto in1Index = ((row * 2) * 16 * 16) + (((column * 2) + 1) * 16) + channel;
                auto in2Index = (((row * 2) + 1) * 16 * 16) + ((column * 2) * 16) + channel;
                auto in3Index = (((row * 2) + 1) * 16 * 16) + (((column * 2) + 1) * 16) + channel;
                // output value is the maximum of the input pool values, adjusted for the quantization scale change
                auto maxIn = std::max<uint8_t>({inputData[in0Index],
                                                inputData[in1Index],
                                                inputData[in2Index],
                                                inputData[in3Index]});
                expectedOutputData[outIndex] = maxIn / 2;
            }
        }
    }

    // Construct the network
    Network net;
    IConnectableLayer* const inputLayer   = net.AddInputLayer(0, "input");
    IConnectableLayer* const convLayer    = net.AddConvolution2dLayer(convDescriptor, weights, biases, "conv");
    IConnectableLayer* const poolingLayer = net.AddPooling2dLayer(poolDescriptor, "pooling2d");
    IConnectableLayer* const outputLayer  = net.AddOutputLayer(0, "output");

    // Connect the layers
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayer->GetOutputSlot(0).Connect(poolingLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(convOutputInfo);
    poolingLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    poolingLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return OptimiseAndRunNetwork(workloadFactory,
                                 net,
                                 inputInfo,
                                 inputData,
                                 outputInfo,
                                 expectedOutputData);
}
