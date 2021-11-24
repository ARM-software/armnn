//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QLstmEndToEndTestImpl.hpp"

#include <CommonTestUtils.hpp>
#include "EndToEndTestImpl.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/LstmParams.hpp>

#include <doctest/doctest.h>

namespace
{

// Checks if two values of an arithmetic type are close enough to each other
// with regard to a given tolerance value.
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, bool>::type
IsCloseEnough(T value1, T value2, T tolerance)
{
    if (tolerance < 0)
    {
        throw armnn::InvalidArgumentException("Tolerance cannot be < 0");
    }

    T diff = value1 >= value2 ? static_cast<T>(value1 - value2) : static_cast<T>(value2 - value1);
    return diff <= tolerance;
}

} // anonymous namespace

void QLstmEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    const unsigned int numBatches = 2;
    const unsigned int inputSize  = 5;
    const unsigned int outputSize = 4;
    const unsigned int numUnits   = 4;

    bool cifgEnabled       = true;
    bool peepholeEnabled   = false;
    bool projectionEnabled = false;
    bool layerNormEnabled  = true;

    // Scale/Offset quantization info
    const float inputScale    = 0.0078125f;
    const int32_t inputOffset = 0;

    const int32_t hiddenStateZeroPoint = 0;
    const float hiddenStateScale       = 0.007f;

    // if (!projectionEnabled) outputScale == hiddenStateScale
    const float outputScale    = hiddenStateScale;
    const int32_t outputOffset = hiddenStateZeroPoint;

    const float cellStateScale    = 3.05176e-05f;
    const int32_t cellStateOffset = 0;

    const float weightsScale    = 0.00784314f;
    const int32_t weightsOffset = 0;

    const float layerNormScale    = 3.05182e-05f;
    const int32_t layerNormOffset = 0;

    const float biasScale    = layerNormScale / 1024;
    const int32_t biasOffset = 0;

    const float inputIntermediateScale  = 0.007059f;
    const float forgetIntermediateScale = 0.007812f;
    const float cellIntermediateScale   = inputIntermediateScale;
    const float outputIntermediateScale = forgetIntermediateScale;

    const float cellClip       = 0.0f;
    const float projectionClip = 0.0f;

    // Weights and bias tensor info
    const armnn::TensorInfo inputWeightsInfo({outputSize, inputSize},
                                             armnn::DataType::QSymmS8,
                                             weightsScale,
                                             weightsOffset,
                                             true);

    const armnn::TensorInfo recurrentWeightsInfo({outputSize, outputSize},
                                                 armnn::DataType::QSymmS8,
                                                 weightsScale,
                                                 weightsOffset,
                                                 true);

    const armnn::TensorInfo biasInfo({outputSize},
                                     armnn::DataType::Signed32,
                                     biasScale,
                                     biasOffset,
                                     true);

    const armnn::TensorInfo layerNormWeightsInfo({numUnits},
                                                 armnn::DataType::QSymmS16,
                                                 layerNormScale,
                                                 layerNormOffset,
                                                 true);

    // Mandatory params
    const std::vector<int8_t> inputToForgetWeightsVector =
            {-77, -13, 38, 25, 115, -64, -25, -51, 38, -102, -51, 38, -64, -51, -77, 38, -51, -77, -64, -64};
    const std::vector<int8_t> inputToCellWeightsTensorVector =
            {-51, -38, -25, -13, -64, 64, -25, -38, -25, -77, 77, -13, -51, -38, -89, 89, -115, -64, 102, 77};
    const std::vector<int8_t> inputToOutputWeightsTensorVector =
            {-102, -51, -25, -115, -13, -89, 38, -38, -102, -25, 77, -25, 51, -89, -38, -64, 13, 64, -77, -51};

    armnn::ConstTensor inputToForgetWeightsTensor(inputWeightsInfo, inputToForgetWeightsVector.data());
    armnn::ConstTensor inputToCellWeightsTensor(inputWeightsInfo, inputToCellWeightsTensorVector.data());
    armnn::ConstTensor inputToOutputWeightsTensor(inputWeightsInfo, inputToOutputWeightsTensorVector.data());

    const std::vector<int8_t> recurrentToForgetWeightsTensorVector =
            {-64, -38, -64, -25, 77, 51, 115, 38, -13, 25, 64, 25, 25, 38, -13, 51};
    const std::vector<int8_t> recurrentToCellWeightsTensorVector =
            {-38, 25, 13, -38, 102, -10, -25, 38, 102, -77, -13, 25, 38, -13, 25, 64};
    const std::vector<int8_t> recurrentToOutputWeightsTensorVector =
            {38, -13, 13, -25, -64, -89, -25, -77, -13, -51, -89, -25, 13, 64, 25, -38};

    armnn::ConstTensor recurrentToForgetWeightsTensor(recurrentWeightsInfo,
                                                      recurrentToForgetWeightsTensorVector.data());
    armnn::ConstTensor recurrentToCellWeightsTensor(recurrentWeightsInfo,
                                                    recurrentToCellWeightsTensorVector.data());
    armnn::ConstTensor recurrentToOutputWeightsTensor(recurrentWeightsInfo,
                                                      recurrentToOutputWeightsTensorVector.data());

    const std::vector<int32_t> forgetGateBiasTensorVector = {2147484, -6442451, -4294968, 2147484};
    const std::vector<int32_t> cellBiasTensorVector       = {-1073742, 15461883, 5368709, 1717987};
    const std::vector<int32_t> outputGateBiasTensorVector = {1073742, -214748, 4294968, 2147484};

    armnn::ConstTensor forgetGateBiasTensor(biasInfo, forgetGateBiasTensorVector.data());
    armnn::ConstTensor cellBiasTensor(biasInfo, cellBiasTensorVector.data());
    armnn::ConstTensor outputGateBiasTensor(biasInfo, outputGateBiasTensorVector.data());

    // Layer Norm
    const std::vector<int16_t> forgetLayerNormWeightsVector = {6553, 6553, 13107, 9830};
    const std::vector<int16_t> cellLayerNormWeightsVector   = {22937, 6553, 9830, 26214};
    const std::vector<int16_t> outputLayerNormWeightsVector = {19660, 6553, 6553, 16384};

    armnn::ConstTensor forgetLayerNormWeights(layerNormWeightsInfo, forgetLayerNormWeightsVector.data());
    armnn::ConstTensor cellLayerNormWeights(layerNormWeightsInfo, cellLayerNormWeightsVector.data());
    armnn::ConstTensor outputLayerNormWeights(layerNormWeightsInfo, outputLayerNormWeightsVector.data());

    // Set up params
    armnn::LstmInputParams params;
    params.m_InputToForgetWeights = &inputToForgetWeightsTensor;
    params.m_InputToCellWeights   = &inputToCellWeightsTensor;
    params.m_InputToOutputWeights = &inputToOutputWeightsTensor;

    params.m_RecurrentToForgetWeights = &recurrentToForgetWeightsTensor;
    params.m_RecurrentToCellWeights   = &recurrentToCellWeightsTensor;
    params.m_RecurrentToOutputWeights = &recurrentToOutputWeightsTensor;

    params.m_ForgetGateBias = &forgetGateBiasTensor;
    params.m_CellBias       = &cellBiasTensor;
    params.m_OutputGateBias = &outputGateBiasTensor;

    params.m_ForgetLayerNormWeights = &forgetLayerNormWeights;
    params.m_CellLayerNormWeights   = &cellLayerNormWeights;
    params.m_OutputLayerNormWeights = &outputLayerNormWeights;

    QLstmDescriptor descriptor;
    descriptor.m_CifgEnabled       = cifgEnabled;
    descriptor.m_PeepholeEnabled   = peepholeEnabled;
    descriptor.m_ProjectionEnabled = projectionEnabled;
    descriptor.m_LayerNormEnabled  = layerNormEnabled;

    descriptor.m_CellClip       = cellClip;
    descriptor.m_ProjectionClip = projectionClip;

    descriptor.m_HiddenStateZeroPoint = hiddenStateZeroPoint;
    descriptor.m_HiddenStateScale     = hiddenStateScale;

    descriptor.m_InputIntermediateScale  = inputIntermediateScale;
    descriptor.m_ForgetIntermediateScale = forgetIntermediateScale;
    descriptor.m_CellIntermediateScale   = cellIntermediateScale;
    descriptor.m_OutputIntermediateScale = outputIntermediateScale;

    // Input/Output tensor info
    const armnn::TensorInfo inputInfo({numBatches , inputSize},
                                      armnn::DataType::QAsymmS8,
                                      inputScale,
                                      inputOffset,
                                      true);

    const armnn::TensorInfo cellStateInfo({numBatches , numUnits},
                                          armnn::DataType::QSymmS16,
                                          cellStateScale,
                                          cellStateOffset,
                                          true);

    const armnn::TensorInfo outputStateInfo({numBatches , outputSize},
                                            armnn::DataType::QAsymmS8,
                                            outputScale,
                                            outputOffset,
                                            true);

    // Input tensor data
    const std::vector<int8_t> inputVector         = {90, 102, 13, 26, 38, 102, 13, 26, 51, 64};
    const std::vector<int8_t> outputStateInVector = {0, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<int16_t> cellStateInVector  = {0, 0, 0, 0, 0, 0, 0, 0};

    // Expected output tensor data
    const std::vector<int8_t> outputStateOutVector = {-15, 21, 14, 20, -15, 15, 5, 27};
    const std::vector<int16_t> cellStateOutVector  = {-11692, 9960, 5491, 8861, -9422, 7726, 2056, 13149};
    const std::vector<int8_t> outputVector         = {-15, 21, 14, 20, -15, 15, 5, 27};

    // Build network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* const input         = net->AddInputLayer(0);
    armnn::IConnectableLayer* const outputStateIn = net->AddInputLayer(1);
    armnn::IConnectableLayer* const cellStateIn   = net->AddInputLayer(2);

    armnn::IConnectableLayer* const qLstmLayer = net->AddQLstmLayer(descriptor, params, "qLstm");

    armnn::IConnectableLayer* const outputStateOut = net->AddOutputLayer(0);
    armnn::IConnectableLayer* const cellStateOut   = net->AddOutputLayer(1);
    armnn::IConnectableLayer* const output         = net->AddOutputLayer(2);

    // Connect input/output slots
    Connect(input, qLstmLayer, inputInfo, 0, 0);
    Connect(outputStateIn, qLstmLayer, outputStateInfo, 0, 1);
    Connect(cellStateIn, qLstmLayer, cellStateInfo, 0, 2);

    Connect(qLstmLayer, outputStateOut, outputStateInfo, 0, 0);
    Connect(qLstmLayer, cellStateOut, cellStateInfo, 1, 0);
    Connect(qLstmLayer, output, outputStateInfo, 2, 0);

    // Create runtime
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Loads network into runtime
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    // Push back input tensors
    InputTensors inputTensors;
    inputTensors.reserve(3);

    inputTensors.push_back({0, ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputVector.data())});
    inputTensors.push_back({1, ConstTensor(runtime->GetInputTensorInfo(netId, 1), outputStateInVector.data())});
    inputTensors.push_back({2, ConstTensor(runtime->GetInputTensorInfo(netId, 2), cellStateInVector.data())});

    // Push back output tensors
    OutputTensors outputTensors;
    outputTensors.reserve(3);

    std::vector<int8_t> outputStateOutResult(outputStateOutVector.size());
    std::vector<int16_t> cellStateOutResult(cellStateOutVector.size());
    std::vector<int8_t> outputResult(outputStateOutVector.size());

    outputTensors.push_back({0, Tensor(runtime->GetOutputTensorInfo(netId, 0), outputStateOutResult.data())});
    outputTensors.push_back({1, Tensor(runtime->GetOutputTensorInfo(netId, 1), cellStateOutResult.data())});
    outputTensors.push_back({2, Tensor(runtime->GetOutputTensorInfo(netId, 2), outputResult.data())});

    // Execute inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    constexpr int8_t toleranceInt8 = 1;
    for (unsigned int i = 0u; i < outputStateOutResult.size(); ++i)
    {
        CHECK(IsCloseEnough(outputStateOutVector[i], outputStateOutResult[i], toleranceInt8));
    }

    for (unsigned int i = 0u; i < outputResult.size(); ++i)
    {
        CHECK(IsCloseEnough(outputVector[i], outputResult[i], toleranceInt8));
    }
}