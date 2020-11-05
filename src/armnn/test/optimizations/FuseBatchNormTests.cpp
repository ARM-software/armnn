//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayersFwd.hpp"

#include <Network.hpp>
#include <ResolveType.hpp>
#include <armnn/INetwork.hpp>
#include <test/TestUtils.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(Optimizer)

namespace
{

class Conv2dTest
{
public:
    using ConvDescriptorType            = armnn::Convolution2dDescriptor;
    using ConvLayerType                 = armnn::Convolution2dLayer;

    static IConnectableLayer *AddConvolution(INetwork *network,
                                             const Convolution2dDescriptor &descriptor,
                                             const ConstTensor &weights,
                                             const Optional<ConstTensor> &biases,
                                             const char *name)
    {
        return network->AddConvolution2dLayer(descriptor, weights, biases, name);
    }
};

class DepthwiseConv2dTest
{
public:
    using ConvDescriptorType            = armnn::DepthwiseConvolution2dDescriptor;
    using ConvLayerType                 = armnn::DepthwiseConvolution2dLayer;

    static IConnectableLayer *AddConvolution(INetwork *network,
                                             const DepthwiseConvolution2dDescriptor &descriptor,
                                             const ConstTensor &weights,
                                             const Optional<ConstTensor> &biases,
                                             const char *name)
    {
        return network->AddDepthwiseConvolution2dLayer(descriptor, weights, biases, name);
    }
};

template<typename T>
std::vector<T> GetVector(unsigned int size, float initial, float increment)
{
    std::vector<float> typeVector(size, initial);
    std::vector<T> vector(size);

    if (size > 1)
    {
        for (unsigned int i = 0; i < size; ++i)
        {
            vector[i] = T(initial + (increment * static_cast<float>(i)));
        }
    }
    return vector;
}

} // namespace

template <typename Conv2dTest,
          armnn::DataType ArmnnType,
          typename ConvDescriptorType = typename Conv2dTest::ConvDescriptorType,
          typename T = armnn::ResolveType<ArmnnType>>
INetworkPtr CreatNetwork(bool depthwise, bool preventFusing)
{
    // Define layers information
    ConvDescriptorType convolution2dDescriptor;
    convolution2dDescriptor.m_BiasEnabled = false;
    convolution2dDescriptor.m_DataLayout = DataLayout::NHWC;
    convolution2dDescriptor.m_StrideX = 1;
    convolution2dDescriptor.m_StrideY = 1;
    BatchNormalizationDescriptor batchNormDescriptor;
    batchNormDescriptor.m_DataLayout = DataLayout::NHWC;

    const unsigned int inputDimensionSizes[] = {1, 4, 4, 3};  // NHWCin
    unsigned int weightsDimensionSizes[]     = {4, 2, 2, 3};  // CoutHWCin
    unsigned int outputDimensionSizes[]      = {1, 3, 3, 4};  // NHWCout

    if (depthwise)
    {
        //M Cin H W
        weightsDimensionSizes[0] = 4;
        weightsDimensionSizes[1] = 3;
        weightsDimensionSizes[2] = 2;
        weightsDimensionSizes[3] = 2;
        outputDimensionSizes[3]  = weightsDimensionSizes[0] * weightsDimensionSizes[1];
    }
    const unsigned int outputChannelSize[]   = {outputDimensionSizes[3]};  // Cout

    TensorInfo inputInfo(4, inputDimensionSizes, ArmnnType);
    TensorInfo outputInfo(4, outputDimensionSizes, ArmnnType);

    std::vector<int> weightsIntVector = { 1,  2,  3,  4,   5,  6,  7,  8,   9, 10, 11, 12,
                                         11, 12, 13, 14,  15, 16, 17, 18,  19, 20, 21, 22,
                                         21, 22, 23, 24,  25, 26, 27, 28,  29, 30, 31, 32,
                                         31, 32, 33, 34,  35, 36, 37, 38,  39, 40, 41, 42};
    std::vector<T> weightsVector(begin(weightsIntVector), end(weightsIntVector));
    TensorInfo weightsInfo(4, weightsDimensionSizes, ArmnnType);
    ConstTensor weights(weightsInfo, weightsVector);

    std::vector<T> biasVector = GetVector<T>(outputDimensionSizes[3], 3.3f, 0.1f);
    TensorInfo biasInfo(1, outputChannelSize, ArmnnType);
    ConstTensor bias(biasInfo, biasVector);
    Optional<ConstTensor> optionalBias = Optional<ConstTensor>(bias);

    std::vector<T> betaVector     = GetVector<T>(outputDimensionSizes[3], 0.0f, 0.2f);
    std::vector<T> gammaVector    = GetVector<T>(outputDimensionSizes[3], 0.5f, 0.1f);
    std::vector<T> meanVector     = GetVector<T>(outputDimensionSizes[3], 0.1f, 0.1f);
    std::vector<T> varianceVector = GetVector<T>(outputDimensionSizes[3], 1.0f, 0.1f);

    ConstTensor beta    (TensorInfo(1, outputChannelSize, ArmnnType), betaVector);
    ConstTensor gamma   (TensorInfo(1, outputChannelSize, ArmnnType), gammaVector);
    ConstTensor mean    (TensorInfo(1, outputChannelSize, ArmnnType), meanVector);
    ConstTensor variance(TensorInfo(1, outputChannelSize, ArmnnType), varianceVector);

    // Create a network
    INetworkPtr network = INetwork::Create();

    IConnectableLayer* inputLayer     = network->AddInputLayer(0);

    IConnectableLayer* convLayer      = Conv2dTest::AddConvolution(network.get(),
                                                                   convolution2dDescriptor,
                                                                   weights,
                                                                   optionalBias,
                                                                   "convolution");

    IConnectableLayer* batchNormLayer = network->AddBatchNormalizationLayer(batchNormDescriptor,
                                                                            mean,
                                                                            variance,
                                                                            beta,
                                                                            gamma,
                                                                            "batchNorm");

    IConnectableLayer* outputLayer    = network->AddOutputLayer(0);
    IConnectableLayer* output2Layer   = nullptr;

    if (preventFusing)
    {
        output2Layer                  = network->AddOutputLayer(1);
    }

    // Set layer information
    inputLayer    ->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayer     ->GetOutputSlot(0).SetTensorInfo(outputInfo);
    batchNormLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Connect layers
    inputLayer    ->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer     ->GetOutputSlot(0).Connect(batchNormLayer->GetInputSlot(0));
    batchNormLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    if (preventFusing)
    {
        convLayer ->GetOutputSlot(0).Connect(output2Layer->GetInputSlot(0));
    }

    return network;
}

template <typename Conv2dTest,
          armnn::DataType ArmnnType,
          typename ConvDescriptorType = typename Conv2dTest::ConvDescriptorType,
          typename ConvLayerType = typename Conv2dTest::ConvLayerType,
          typename T = armnn::ResolveType<ArmnnType>>
void FuseBatchNormIntoConvTest(bool depthwise, float tolerance, armnn::Compute backendId)
{
    // FIRST NETWORK: Fused
    // Construct ArmNN network
    INetworkPtr networkFused = CreatNetwork<Conv2dTest, ArmnnType>(depthwise, false);

    // Create ArmNN runtime
    IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions()); // default options

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNetFused = Optimize(*networkFused, {backendId}, run->GetDeviceSpec());

    Graph graphFused = PolymorphicDowncast<OptimizedNetwork*>(optNetFused.get())->GetGraph();

    auto checkFusedConv2d = [ ](const armnn::Layer* const layer) -> bool
    {
        return IsLayerOfType<ConvLayerType>(layer) &&
               (layer->GetNameStr() == "fused-batchNorm-into-convolution");
    };

    BOOST_CHECK(3 == graphFused.GetNumLayers());
    BOOST_TEST(CheckSequence(graphFused.cbegin(),
                             graphFused.cend(),
                             &IsLayerOfType<InputLayer>,
                             checkFusedConv2d,
                             &IsLayerOfType<OutputLayer>));

    // Load network into runtime
    NetworkId networkIdentifier;
    BOOST_TEST(run->LoadNetwork(networkIdentifier, std::move(optNetFused)) == Status::Success);

    //Creates structures for inputs and outputs.
    std::vector<T> inputDataFused = GetVector<T>(48, 1.0f, 0.1f);

    std::vector<T> outputDataFused(36);

    if (depthwise)
    {
        outputDataFused.resize(108);
    }

    InputTensors  inputTensorsFused {
            {0, ConstTensor(run->GetInputTensorInfo (networkIdentifier, 0), inputDataFused.data())}};
    OutputTensors outputTensorsFused{
            {0,      Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputDataFused.data())}};

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensorsFused, outputTensorsFused);

    // SECOND NETWORK: NotFused
    // Construct ArmNN network
    INetworkPtr networkNotFused = CreatNetwork<Conv2dTest, ArmnnType>(depthwise, true);

    // Create ArmNN runtime
    IRuntimePtr runNotFused = IRuntime::Create(IRuntime::CreationOptions()); // default options

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNetNotFused = Optimize(*networkNotFused, {backendId}, runNotFused->GetDeviceSpec());

    Graph graphNotFused = PolymorphicDowncast<OptimizedNetwork*>(optNetNotFused.get())->GetGraph();

    BOOST_CHECK(5 == graphNotFused.GetNumLayers());
    BOOST_TEST(CheckSequence(graphNotFused.cbegin(),
                             graphNotFused.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<ConvLayerType>,
                             &IsLayerOfType<armnn::BatchNormalizationLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    // Load network into runtime
    NetworkId networkIdentifierNotFused;
    BOOST_TEST(runNotFused->LoadNetwork(networkIdentifierNotFused, std::move(optNetNotFused)) == Status::Success);

    //Creates structures for inputs and outputs.
    std::vector<T> inputDataNotFused = GetVector<T>(48, 1.0f, 0.1f);

    std::vector<T> outputDataNotFused(36);
    std::vector<T> outputData2NotFused(36);

    if (depthwise)
    {
        outputDataNotFused.resize(108);
        outputData2NotFused.resize(108);
    }
    InputTensors inputTensorsNotFused{
            {0, ConstTensor(runNotFused->GetInputTensorInfo(networkIdentifierNotFused, 0), inputDataNotFused.data())}};
    OutputTensors outputTensorsNotFused{
            {0, Tensor(runNotFused->GetOutputTensorInfo(networkIdentifierNotFused, 0), outputDataNotFused.data())},
            {1, Tensor(runNotFused->GetOutputTensorInfo(networkIdentifierNotFused, 1), outputData2NotFused.data())}};

    // Execute network
    runNotFused->EnqueueWorkload(networkIdentifierNotFused, inputTensorsNotFused, outputTensorsNotFused);

    // Check the output of the fused-convolution matches with the output of the batchNormm in the "NotFused" network
    for (unsigned int n = 0; n < outputDataFused.size(); ++n)
    {
        BOOST_CHECK_CLOSE(outputDataFused[n], outputDataNotFused[n], T(tolerance));
    }
}

// This unit test needs the reference backend, it's not available if the reference backend is not built
#if defined(ARMNNREF_ENABLED)
BOOST_AUTO_TEST_CASE(FuseBatchNormIntoConv2DFloat32Test)
{
    FuseBatchNormIntoConvTest<Conv2dTest, DataType::Float32>(false, 0.0001f, armnn::Compute::CpuRef);
}

BOOST_AUTO_TEST_CASE(FuseBatchNormIntoConv2DFloat16Test)
{
    FuseBatchNormIntoConvTest<Conv2dTest, DataType::Float16>(false, 0.1f, armnn::Compute::CpuRef);
}

BOOST_AUTO_TEST_CASE(FuseBatchNormIntoDepthwiseConv2DFloat32Test)
{
    FuseBatchNormIntoConvTest<DepthwiseConv2dTest, DataType::Float32>(true, 0.0001f,armnn::Compute::CpuRef);
}

BOOST_AUTO_TEST_CASE(FuseBatchNormIntoDepthwiseConv2DFloat16Test)
{
    FuseBatchNormIntoConvTest<DepthwiseConv2dTest, DataType::Float16>(true, 0.1f,armnn::Compute::CpuRef);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
