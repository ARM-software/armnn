//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayersFwd.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Optimizer)
using namespace armnn;

// This unit test needs the reference backend, it's not available if the reference backend is not built
#if defined(ARMNNREF_ENABLED)
BOOST_AUTO_TEST_CASE(Fuse_batchNorm_into_Conv2D_Float32_Test)
{
    // Define layers information
    Convolution2dDescriptor convolution2dDescriptor;
    convolution2dDescriptor.m_BiasEnabled = false;
    convolution2dDescriptor.m_DataLayout = DataLayout::NHWC;
    convolution2dDescriptor.m_StrideX = 1;
    convolution2dDescriptor.m_StrideY = 1;
    BatchNormalizationDescriptor batchNormDescriptor;
    batchNormDescriptor.m_DataLayout = DataLayout::NHWC;

    const unsigned int inputDimensionSizes[]   = {1, 4, 4, 3};  // NHWCin
    const unsigned int weightsDimensionSizes[] = {4, 2, 2, 3};  // CoutHWCin
    const unsigned int outputDimensionSizes[]  = {1, 3, 3, 4};  // NHWCout
    const unsigned int outputChannelSize[]     = {outputDimensionSizes[3]};  // Cout

    TensorInfo inputInfo (4, inputDimensionSizes, DataType::Float32);
    TensorInfo outputInfo(4, outputDimensionSizes, DataType::Float32);

    std::vector<float> weightsVector = { 1,  2,  3,  4,    5,  6,  7, 8,    9,  10,  11,  12,
                                         11, 12, 13, 14,   15, 16, 17, 18,  19, 110, 111, 112,
                                         21, 22, 23, 24,   25, 26, 27, 28,  29, 210, 211, 212,
                                         31, 32, 33, 34,   35, 36, 37, 38,  39, 310, 311, 312};
    TensorInfo weightsInfo(4, weightsDimensionSizes, DataType::Float32);
    ConstTensor weights (weightsInfo, weightsVector);
    std::vector<float> biasVector     = {3.3f, 3.2f, 3.1f, 3.0f};
    TensorInfo biasInfo(1, outputChannelSize, DataType::Float32);
    ConstTensor bias (biasInfo, biasVector);
    Optional<ConstTensor> optionalBias = Optional<ConstTensor>(bias);

    std::vector<float> betaVector     = {0.0f, 0.2f, 0.3f, 0.4f};
    std::vector<float> gammaVector    = {0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> meanVector     = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> varianceVector = {1.0f, 1.1f, 1.2f, 1.3f};
    ConstTensor beta    (TensorInfo(1, outputChannelSize, DataType::Float32), betaVector);
    ConstTensor gamma   (TensorInfo(1, outputChannelSize, DataType::Float32), gammaVector);
    ConstTensor mean    (TensorInfo(1, outputChannelSize, DataType::Float32), meanVector);
    ConstTensor variance(TensorInfo(1, outputChannelSize, DataType::Float32), varianceVector);

    auto inputSize = inputDimensionSizes[0]*inputDimensionSizes[1]*inputDimensionSizes[2]*inputDimensionSizes[3];
    auto outputSize = outputDimensionSizes[0]*outputDimensionSizes[1]*outputDimensionSizes[2]*outputDimensionSizes[3];

    // FIRST NETWORK: Fused

    // Construct ArmNN network
    NetworkId networkIdentifier;
    INetworkPtr network = INetwork::Create();
    IConnectableLayer *inputLayer     = network->AddInputLayer(0);
    IConnectableLayer *convLayer      = network->AddConvolution2dLayer(convolution2dDescriptor,
                                                                       weights,
                                                                       optionalBias,
                                                                       "convolution");
    IConnectableLayer *batchNormLayer = network->AddBatchNormalizationLayer(batchNormDescriptor,
                                                                            mean,
                                                                            variance,
                                                                            beta,
                                                                            gamma,
                                                                            "batchNorm");
    IConnectableLayer *outputLayer    = network->AddOutputLayer(0);

    inputLayer     ->GetOutputSlot(0).Connect(convLayer     ->GetInputSlot(0));
    convLayer      ->GetOutputSlot(0).Connect(batchNormLayer->GetInputSlot(0));
    batchNormLayer ->GetOutputSlot(0).Connect(outputLayer   ->GetInputSlot(0));

    //Set the tensors in the network.
    inputLayer     ->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayer      ->GetOutputSlot(0).SetTensorInfo(outputInfo);
    batchNormLayer ->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    IRuntimePtr run = IRuntime::Create(options);

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNet = Optimize(*network, {Compute::CpuRef}, run->GetDeviceSpec());

    // Load graph into runtime
    BOOST_TEST(run->LoadNetwork(networkIdentifier, std::move(optNet)) == Status::Success);

    //Creates structures for inputs and outputs.
    std::vector<float> inputData(inputSize, 128);
    std::vector<float> outputData(outputSize);

    InputTensors inputTensors  {{0, ConstTensor(run->GetInputTensorInfo (networkIdentifier, 0), inputData.data())}};
    OutputTensors outputTensors{{0,      Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputData.data())}};

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // SECOND NETWORK: NotFused

    // Construct ArmNN network
    NetworkId networkIdentifierNotFused;
    INetworkPtr networkNotFused = INetwork::Create();
    IConnectableLayer *inputLayerNotFused     = networkNotFused->AddInputLayer(0);
    IConnectableLayer *convLayerNotFused      = networkNotFused->AddConvolution2dLayer(convolution2dDescriptor,
                                                                                       weights,
                                                                                       optionalBias,
                                                                                       "convolution");
    IConnectableLayer *batchNormLayerNotFused = networkNotFused->AddBatchNormalizationLayer(batchNormDescriptor,
                                                                                            mean,
                                                                                            variance,
                                                                                            beta,
                                                                                            gamma,
                                                                                            "batchNorm");
    IConnectableLayer *outputLayerNotFused    = networkNotFused->AddOutputLayer(0);
    IConnectableLayer *output2LayerNotFused   = networkNotFused->AddOutputLayer(1);

    inputLayerNotFused     ->GetOutputSlot(0).Connect(convLayerNotFused     ->GetInputSlot(0));
    convLayerNotFused      ->GetOutputSlot(0).Connect(batchNormLayerNotFused->GetInputSlot(0));
    batchNormLayerNotFused ->GetOutputSlot(0).Connect(outputLayerNotFused   ->GetInputSlot(0));
    convLayerNotFused      ->GetOutputSlot(0).Connect(output2LayerNotFused  ->GetInputSlot(0));

    //Set the tensors in the network.
    inputLayerNotFused     ->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayerNotFused      ->GetOutputSlot(0).SetTensorInfo(outputInfo);
    batchNormLayerNotFused ->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Create ArmNN runtime
    IRuntimePtr runNotFused = IRuntime::Create(options);

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNetNotFused = Optimize(*networkNotFused, {Compute::CpuRef}, runNotFused->GetDeviceSpec());

    // Load graph into runtime
    BOOST_TEST(runNotFused->LoadNetwork(networkIdentifierNotFused, std::move(optNetNotFused)) == Status::Success);

    //Creates structures for inputs and outputs.
    std::vector<float> inputDataNotFused(inputSize, 128);
    std::vector<float> outputDataNotFused(outputSize);
    std::vector<float> outputData2NotFused(outputSize);

    InputTensors inputTensorsNotFused{
            {0, ConstTensor(runNotFused->GetInputTensorInfo(networkIdentifierNotFused, 0), inputDataNotFused.data())}};
    OutputTensors outputTensorsNotFused{
            {0, Tensor(runNotFused->GetOutputTensorInfo(networkIdentifierNotFused, 0), outputDataNotFused.data())},
            {1, Tensor(runNotFused->GetOutputTensorInfo(networkIdentifierNotFused, 1), outputData2NotFused.data())}};

    // Execute network
    runNotFused->EnqueueWorkload(networkIdentifierNotFused, inputTensorsNotFused, outputTensorsNotFused);

    // Check the output of the fused-convolution matches with the output of the batchNormm in the "NotFused" network
    for (unsigned int n = 0; n < outputData.size(); ++n)
    {
        BOOST_CHECK_CLOSE(outputData[n], outputDataNotFused[n], 0.001);
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
