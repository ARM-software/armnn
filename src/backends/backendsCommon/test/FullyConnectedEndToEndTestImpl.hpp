//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <CommonTestUtils.hpp>

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <doctest/doctest.h>

#include <vector>

namespace
{

armnn::INetworkPtr CreateFullyConnectedNetworkNonConstWeights(const armnn::TensorInfo& inputTensorInfo,
                                                              const armnn::TensorInfo& outputTensorInfo,
                                                              const armnn::TensorInfo& weightsTensorInfo,
                                                              armnn::FullyConnectedDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* weightsInputLayer   = network->AddInputLayer(1, "Weights_Input");
    armnn::IConnectableLayer* fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor, "Fully_Connected");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fullyConnectedLayer, inputTensorInfo, 0, 0);
    Connect(weightsInputLayer, fullyConnectedLayer, weightsTensorInfo, 0, 1);
    Connect(fullyConnectedLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

armnn::INetworkPtr CreateFullyConnectedNetworkNonConstWeightsConstBias(const armnn::TensorInfo& inputTensorInfo,
                                                                       const armnn::TensorInfo& outputTensorInfo,
                                                                       const armnn::TensorInfo& weightsTensorInfo,
                                                                       const armnn::TensorInfo& biasTensorInfo,
                                                                       const armnn::ConstTensor& biasConstantTensor,
                                                                       armnn::FullyConnectedDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* weightsInputLayer   = network->AddInputLayer(1, "Weights_Input");
    armnn::IConnectableLayer* biasLayer  = network->AddConstantLayer(biasConstantTensor, "Weights");
    armnn::IConnectableLayer* fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor, "Fully_Connected");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fullyConnectedLayer, inputTensorInfo, 0, 0);
    Connect(weightsInputLayer, fullyConnectedLayer, weightsTensorInfo, 0, 1);
    Connect(biasLayer, fullyConnectedLayer, biasTensorInfo, 0, 2);
    Connect(fullyConnectedLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

armnn::INetworkPtr CreateFullyConnectedNetworkConstWeightsNonConstBias(const armnn::TensorInfo& inputTensorInfo,
                                                                       const armnn::TensorInfo& outputTensorInfo,
                                                                       const armnn::TensorInfo& weightsTensorInfo,
                                                                       const armnn::TensorInfo& biasTensorInfo,
                                                                       const armnn::ConstTensor& weightsConstantTensor,
                                                                       armnn::FullyConnectedDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* weightsLayer  = network->AddConstantLayer(weightsConstantTensor, "Weights");
    armnn::IConnectableLayer* biasLayer   = network->AddInputLayer(2, "Bias_Input");
    armnn::IConnectableLayer* fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor, "Fully_Connected");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fullyConnectedLayer, inputTensorInfo, 0, 0);
    Connect(weightsLayer, fullyConnectedLayer, weightsTensorInfo, 0, 1);
    Connect(biasLayer, fullyConnectedLayer, biasTensorInfo, 0, 2);
    Connect(fullyConnectedLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

armnn::INetworkPtr CreateFullyConnectedNetworkNoTensorInfoConstWeights(const armnn::TensorInfo& inputTensorInfo,
                                                                       const armnn::TensorInfo& outputTensorInfo,
                                                                       const armnn::ConstTensor& weightsConstantTensor,
                                                                       armnn::FullyConnectedDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* weightsLayer  = network->AddConstantLayer(weightsConstantTensor, "Weights");
    armnn::IConnectableLayer* fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor, "Fully_Connected");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fullyConnectedLayer, inputTensorInfo, 0, 0);
    weightsLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    Connect(fullyConnectedLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

armnn::INetworkPtr CreateFullyConnectedNetworkNoConnectedWeightsExplicit(const armnn::TensorInfo& inputTensorInfo,
                                                                         const armnn::TensorInfo& outputTensorInfo,
                                                                         const armnn::TensorInfo& biasTensorInfo,
                                                                         armnn::FullyConnectedDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* biasLayer   = network->AddInputLayer(2, "Bias_Input");
    armnn::IConnectableLayer* fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor, "Fully_Connected");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fullyConnectedLayer, inputTensorInfo, 0, 0);
    Connect(biasLayer, fullyConnectedLayer, biasTensorInfo, 0, 2);
    Connect(fullyConnectedLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

armnn::INetworkPtr CreateFullyConnectedNetworkNoConnectedWeightsAndBias(const armnn::TensorInfo& inputTensorInfo,
                                                                        const armnn::TensorInfo& outputTensorInfo,
                                                                        armnn::FullyConnectedDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor, "Fully_Connected");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fullyConnectedLayer, inputTensorInfo, 0, 0);
    Connect(fullyConnectedLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

armnn::INetworkPtr CreateFullyConnectedNetworkNoConnectedBiasExplicit(const armnn::TensorInfo& inputTensorInfo,
                                                                      const armnn::TensorInfo& outputTensorInfo,
                                                                      const armnn::TensorInfo& weightsTensorInfo,
                                                                      const armnn::ConstTensor& weightsConstantTensor,
                                                                      armnn::FullyConnectedDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* weightsLayer  = network->AddConstantLayer(weightsConstantTensor, "Weights");
    armnn::IConnectableLayer* fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor, "Fully_Connected");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fullyConnectedLayer, inputTensorInfo, 0, 0);
    Connect(weightsLayer, fullyConnectedLayer, weightsTensorInfo, 0, 1);
    Connect(fullyConnectedLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void FullyConnectedWithDynamicWeightsEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 2, 3 }, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(63);
    inputTensorInfo.SetConstant(true);

    armnn::TensorInfo outputTensorInfo({ 1, 2 }, ArmnnType);
    outputTensorInfo.SetQuantizationScale(5.f);
    outputTensorInfo.SetQuantizationOffset(10);

    armnn::TensorInfo weightsTensorInfo({ 2, 6 }, ArmnnType);
    weightsTensorInfo.SetQuantizationScale(0.2f);
    weightsTensorInfo.SetQuantizationOffset(93);
    weightsTensorInfo.SetConstant(true);

    FullyConnectedDescriptor descriptor;
    descriptor.m_ConstantWeights = false;
    descriptor.m_BiasEnabled     = false;
    descriptor.m_TransposeWeightMatrix = true;

    std::vector<T> inputData {
        -1.2f, 6.1f, -3.5f,
        18.8f, -5.5f, 2.9f
    };

    std::vector<T> weightsData {
        -8.4f, 20.0f, -10.4f, -8, 16.4f, -11.8f,
        23.4f, 10.4f, -14.0f, -3.8f, -11.8f, 11.4f
    };

    std::vector<T> floatExpectedOutputData {
        -107.04f, 110.f
    };
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatExpectedOutputData);

    armnn::INetworkPtr network = CreateFullyConnectedNetworkNonConstWeights(inputTensorInfo,
                                                                            outputTensorInfo,
                                                                            weightsTensorInfo,
                                                                            descriptor);

    CHECK(network);

    std::map<int, std::vector<T>> inputTensorData    = {{ 0, inputData }, {1, weightsData}};
    std::map<int, std::vector<T>> expectedOutputTensorData = {{ 0, expectedOutputData }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(network),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends,
                                                1.0f);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void FullyConnectedWithDynamicOrConstantInputsEndToEnd(const std::vector<armnn::BackendId>& backends,
                                                       const bool transposeWeights,
                                                       const bool constantWeightsOrBias)
{
    unsigned int inputWidth = 1;
    unsigned int inputHeight = 1;
    unsigned int inputChannels = 5;
    unsigned int inputNum = 2;

    unsigned int outputChannels = 3;
    unsigned int outputNum = 2;

    unsigned int inputShape[]   = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[]  = { outputNum, outputChannels };
    unsigned int weightsShape[] = { inputChannels, outputChannels };

    if (transposeWeights)
    {
        std::swap(weightsShape[0], weightsShape[1]);
    }

    unsigned int biasShape[] = { outputChannels };

    armnn::TensorInfo inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32, 0.0f, 0, true);
    armnn::TensorInfo outputTensorInfo = armnn::TensorInfo(2, outputShape, armnn::DataType::Float32);
    armnn::TensorInfo weightsDesc = armnn::TensorInfo(2, weightsShape, armnn::DataType::Float32, 0.0f, 0, true);
    armnn::TensorInfo biasesDesc = armnn::TensorInfo(1, biasShape, armnn::DataType::Float32, 0.0f, 0, true);

    std::vector<float> input =
    {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        5.0f, 4.0f, 3.0f, 2.0f, 1.0f
    };

    std::vector<float> weights =
    {
        .5f, 2.f, .5f,
        .5f, 2.f, 1.f,
        .5f, 2.f, 2.f,
        .5f, 2.f, 3.f,
        .5f, 2.f, 4.f
    };

    if (transposeWeights)
    {
        weights =
        {
            .5f, .5f, .5f, .5f, .5f,
            2.f, 2.f, 2.f, 2.f, 2.f,
            .5f, 1.f, 2.f, 3.f, 4.f
        };
    }

    std::vector<float> biasValues = std::vector<float>({10.f, 20.f, 30.f});

    std::vector<float> expectedOutput =
    {
        0.5f + 1.0f + 1.5f + 2.0f + 2.5f + biasValues[0],
        2.0f + 4.0f + 6.0f + 8.0f + 10.f + biasValues[1],
        0.5f + 2.0f + 6.0f + 12.f + 20.f + biasValues[2],

        2.5f + 2.0f + 1.5f + 1.0f + 0.5f + biasValues[0],
        10.0f + 8.0f + 6.0f + 4.0f + 2.f + biasValues[1],
        2.5f + 4.0f + 6.0f + 6.f + 4.f   + biasValues[2]
    };

    FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = true;
    descriptor.m_TransposeWeightMatrix = transposeWeights;
    descriptor.m_ConstantWeights = constantWeightsOrBias;

    if (!constantWeightsOrBias)
    {
        // Tests non constant weights and constant bias.
        ConstTensor biasConstantTensor(biasesDesc, biasValues.data());

        armnn::INetworkPtr network = CreateFullyConnectedNetworkNonConstWeightsConstBias(inputTensorInfo,
                                                                                         outputTensorInfo,
                                                                                         weightsDesc,
                                                                                         biasesDesc,
                                                                                         biasConstantTensor,
                                                                                         descriptor);
        CHECK(network);

        std::map<int, std::vector<T>> inputTensorData    = {{ 0, input }, {1, weights}};
        std::map<int, std::vector<T>> expectedOutputTensorData = {{ 0, expectedOutput }};

        EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(network),
                                                    inputTensorData,
                                                    expectedOutputTensorData,
                                                    backends,
                                                    1.0f);
    }
    else
    {
        // Tests constant weights and non constant bias.
        ConstTensor weightsConstantTensor(weightsDesc, weights.data());

        armnn::INetworkPtr network = CreateFullyConnectedNetworkConstWeightsNonConstBias(inputTensorInfo,
                                                                                         outputTensorInfo,
                                                                                         weightsDesc,
                                                                                         biasesDesc,
                                                                                         weightsConstantTensor,
                                                                                         descriptor);
        CHECK(network);

        std::map<int, std::vector<T>> inputTensorData    = {{ 0, input }, {2, biasValues}};
        std::map<int, std::vector<T>> expectedOutputTensorData = {{ 0, expectedOutput }};

        EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(network),
                                                    inputTensorData,
                                                    expectedOutputTensorData,
                                                    backends,
                                                    1.0f);
    }
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void FullyConnectedErrorChecking(const std::vector<armnn::BackendId>& backends,
                                 const bool explicitCheck,
                                 const bool biasEnabled,
                                 const bool connectedWeights,
                                 const bool connectedBias,
                                 const bool tensorInfoSet)
{
    unsigned int inputWidth = 1;
    unsigned int inputHeight = 1;
    unsigned int inputChannels = 5;
    unsigned int inputNum = 2;

    unsigned int outputChannels = 3;
    unsigned int outputNum = 2;

    unsigned int inputShape[]   = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[]  = { outputNum, outputChannels };
    unsigned int weightsShape[] = { inputChannels, outputChannels };

    unsigned int biasShape[] = { outputChannels };

    armnn::TensorInfo inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32, 0.0f, 0, true);
    armnn::TensorInfo outputTensorInfo = armnn::TensorInfo(2, outputShape, armnn::DataType::Float32);
    armnn::TensorInfo weightsDesc = armnn::TensorInfo(2, weightsShape, armnn::DataType::Float32, 0.0f, 0, true);
    armnn::TensorInfo biasesDesc = armnn::TensorInfo(1, biasShape, armnn::DataType::Float32, 0.0f, 0, true);

    std::vector<float> weights =
    {
        .5f, 2.f, .5f,
        .5f, 2.f, 1.f,
        .5f, 2.f, 2.f,
        .5f, 2.f, 3.f,
        .5f, 2.f, 4.f
    };

    FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled = biasEnabled;

    if(explicitCheck)
    {
        if(!biasEnabled)
        {
            try
            {
                CreateFullyConnectedNetworkNoConnectedWeightsExplicit(inputTensorInfo,
                                                                      outputTensorInfo,
                                                                      biasesDesc,
                                                                      descriptor);
                FAIL("LayerValidationException should have been thrown");
            }
            catch (const LayerValidationException& exc)
            {
                CHECK(strcmp(exc.what(), "Tried to connect bias to FullyConnected layer when bias is not enabled: "
                                         "Failed to connect to input slot 2 on FullyConnected layer "
                                         "\"Fully_Connected\" as the slot does not exist or is unavailable") == 0);
            }
        }
        else if (!connectedWeights)
        {
            armnn::INetworkPtr network = CreateFullyConnectedNetworkNoConnectedWeightsExplicit(inputTensorInfo,
                                                                                               outputTensorInfo,
                                                                                               biasesDesc,
                                                                                               descriptor);
            CHECK(network);

            // Create runtime in which test will run
            IRuntime::CreationOptions options;
            IRuntimePtr               runtime(IRuntime::Create(options));

            try
            {
                Optimize(*network, backends, runtime->GetDeviceSpec());
                FAIL("LayerValidationException should have been thrown");
            }
            catch (const LayerValidationException& exc)
            {
                CHECK(strcmp(exc.what(), "Fully_Connected layer weights not set: Input slot(s) 1 not connected "
                                         "to an output slot on FullyConnected layer \"Fully_Connected\"") == 0);
            }
        }
        else if (!connectedBias)
        {
            // Tests with constant weights.
            ConstTensor weightsConstantTensor(weightsDesc, weights.data());

            armnn::INetworkPtr network = CreateFullyConnectedNetworkNoConnectedBiasExplicit(inputTensorInfo,
                                                                                            outputTensorInfo,
                                                                                            weightsDesc,
                                                                                            weightsConstantTensor,
                                                                                            descriptor);
            CHECK(network);

            // Create runtime in which test will run
            IRuntime::CreationOptions options;
            IRuntimePtr               runtime(IRuntime::Create(options));

            try
            {
                Optimize(*network, backends, runtime->GetDeviceSpec());
                FAIL("LayerValidationException should have been thrown");
            }
            catch (const LayerValidationException& exc)
            {
                CHECK(strcmp(exc.what(), "Fully_Connected layer bias not set: Input slot(s) 2 not connected "
                                         "to an output slot on FullyConnected layer \"Fully_Connected\"") == 0);
            }
        }
    }
    else if(!connectedWeights && !connectedBias)
    {
        armnn::INetworkPtr network = CreateFullyConnectedNetworkNoConnectedWeightsAndBias(inputTensorInfo,
                                                                                          outputTensorInfo,
                                                                                          descriptor);
        CHECK(network);

        // Create runtime in which test will run
        IRuntime::CreationOptions options;
        IRuntimePtr               runtime(IRuntime::Create(options));

        try
        {
            Optimize(*network, backends, runtime->GetDeviceSpec());
            FAIL("LayerValidationException should have been thrown");
        }
        catch (const LayerValidationException& exc)
        {
            CHECK(strcmp(exc.what(), "Fully_Connected layer weights and bias not set: Input slot(s) 1 & 2 not "
                                     "connected to an output slot on FullyConnected layer \"Fully_Connected\"") == 0);
        }

    }
    else if(!tensorInfoSet)
    {
        // Tests with constant weights.
        ConstTensor weightsConstantTensor(weightsDesc, weights.data());

        armnn::INetworkPtr network = CreateFullyConnectedNetworkNoTensorInfoConstWeights(inputTensorInfo,
                                                                                         outputTensorInfo,
                                                                                         weightsConstantTensor,
                                                                                         descriptor);
        CHECK(network);

        // Create runtime in which test will run
        IRuntime::CreationOptions options;
        IRuntimePtr runtime(IRuntime::Create(options));

        try
        {
            Optimize(*network, backends, runtime->GetDeviceSpec());
            FAIL("LayerValidationException should have been thrown");
        }
        catch (const LayerValidationException& exc)
        {
            CHECK(strcmp(exc.what(), "Output slot TensorInfo not set on Constant layer \"Weights\"") == 0);
        }
    }
}

} // anonymous namespace
