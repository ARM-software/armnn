//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizedLstmEndToEndTestImpl.hpp"

#include <CommonTestUtils.hpp>
#include "EndToEndTestImpl.hpp"

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>
#include <armnn/QuantizedLstmParams.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <doctest/doctest.h>

#include <type_traits>

namespace
{

armnn::INetworkPtr CreateQuantizedLstmNetwork(armnn::TensorShape& inputShape,
                                              armnn::TensorShape& outputExpectedShape)
{
    auto batchSize = armnn::numeric_cast<unsigned int>(inputShape[0]);
    auto inputSize = armnn::numeric_cast<unsigned int>(inputShape[1]);
    auto outputSize = armnn::numeric_cast<unsigned int>(outputExpectedShape[1]);

    float inputOutputScale = 0.0078125f;
    int32_t inputOutputOffset = 128;

    float weightsScale = 0.00408021f;
    int32_t weightsOffset = 100;

    float biasScale = 3.1876640625e-05f;
    int32_t biasOffset = 0;

    float cellStateScale = 0.00048828125f;
    int32_t cellStateOffset = 0;

    armnn::TensorInfo inputWeightsInfo({outputSize, inputSize},
                                       armnn::DataType::QAsymmU8,
                                       weightsScale,
                                       weightsOffset,
                                       true);

    armnn::TensorInfo recurrentWeightsInfo({outputSize, outputSize},
                                           armnn::DataType::QAsymmU8,
                                           weightsScale,
                                           weightsOffset,
                                           true);

    armnn::TensorInfo biasInfo({outputSize}, armnn::DataType::Signed32, biasScale, biasOffset, true);

    armnn::QuantizedLstmInputParams data;

    const std::vector<uint8_t> inputToInputWeightsVector = {146, 250, 235, 171, 10, 218, 171, 108};
    armnn::ConstTensor inputToInputWeightsTensor(inputWeightsInfo, inputToInputWeightsVector.data());

    const std::vector<uint8_t> inputToForgetWeightsVector = {24, 50, 132, 179, 158, 110, 3, 169};
    armnn::ConstTensor inputToForgetWeightsTensor(inputWeightsInfo, inputToForgetWeightsVector.data());

    const std::vector<uint8_t> inputToCellWeightsTensorVector = {133, 34, 29, 49, 206, 109, 54, 183};
    armnn::ConstTensor inputToCellWeightsTensor(inputWeightsInfo, inputToCellWeightsTensorVector.data());

    const std::vector<uint8_t> inputToOutputWeightsTensorVector = {195, 187, 11, 99, 109, 10, 218, 48};
    armnn::ConstTensor inputToOutputWeightsTensor(inputWeightsInfo, inputToOutputWeightsTensorVector.data());

    const std::vector<uint8_t> recurrentToInputWeightsTensorVector =
            {254, 206, 77, 168, 71, 20, 215, 6, 223, 7, 118, 225, 59, 130, 174, 26};
    armnn::ConstTensor recurrentToInputWeightsTensor(recurrentWeightsInfo, recurrentToInputWeightsTensorVector.data());

    const std::vector<uint8_t> recurrentToForgetWeightsTensorVector =
            {137, 240, 103, 52, 68, 51, 237, 112, 0, 220, 89, 23, 69, 4, 207, 253};
    armnn::ConstTensor recurrentToForgetWeightsTensor(recurrentWeightsInfo,
                                                      recurrentToForgetWeightsTensorVector.data());

    const std::vector<uint8_t> recurrentToCellWeightsTensorVector =
            {172, 60, 205, 65, 14, 0, 140, 168, 240, 223, 133, 56, 142, 64, 246, 216};
    armnn::ConstTensor recurrentToCellWeightsTensor(recurrentWeightsInfo, recurrentToCellWeightsTensorVector.data());

    const std::vector<uint8_t> recurrentToOutputWeightsTensorVector =
            {106, 214, 67, 23, 59, 158, 45, 3, 119, 132, 49, 205, 129, 218, 11, 98};
    armnn::ConstTensor recurrentToOutputWeightsTensor(recurrentWeightsInfo,
                                                      recurrentToOutputWeightsTensorVector.data());

    const std::vector<int32_t> inputGateBiasTensorVector = {-7876, 13488, -726, 32839};
    armnn::ConstTensor inputGateBiasTensor(biasInfo, inputGateBiasTensorVector.data());

    const std::vector<int32_t> forgetGateBiasTensorVector = {9206, -46884, -11693, -38724};
    armnn::ConstTensor forgetGateBiasTensor(biasInfo, forgetGateBiasTensorVector.data());

    const std::vector<int32_t> cellBiasTensorVector = {39481, 48624, 48976, -21419};
    armnn::ConstTensor cellBiasTensor(biasInfo, cellBiasTensorVector.data());

    const std::vector<int32_t> outputGateBiasTensorVector = {-58999, -17050, -41852, -40538};
    armnn::ConstTensor outputGateBiasTensor(biasInfo, outputGateBiasTensorVector.data());

    data.m_InputToInputWeights = &inputToInputWeightsTensor;
    data.m_InputToForgetWeights = &inputToForgetWeightsTensor;
    data.m_InputToCellWeights = &inputToCellWeightsTensor;
    data.m_InputToOutputWeights = &inputToOutputWeightsTensor;
    data.m_RecurrentToInputWeights = &recurrentToInputWeightsTensor;
    data.m_RecurrentToForgetWeights = &recurrentToForgetWeightsTensor;
    data.m_RecurrentToCellWeights = &recurrentToCellWeightsTensor;
    data.m_RecurrentToOutputWeights = &recurrentToOutputWeightsTensor;
    data.m_InputGateBias = &inputGateBiasTensor;
    data.m_ForgetGateBias = &forgetGateBiasTensor;
    data.m_CellBias = &cellBiasTensor;
    data.m_OutputGateBias = &outputGateBiasTensor;

    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* const inputLayer   = net->AddInputLayer(0);
    armnn::IConnectableLayer* const cellStateIn = net->AddInputLayer(1);
    armnn::IConnectableLayer* const outputStateIn = net->AddInputLayer(2);
    armnn::IConnectableLayer* const quantizedLstmLayer = net->AddQuantizedLstmLayer(data, "quantizedLstm");
    armnn::IConnectableLayer* const cellStateOut  = net->AddOutputLayer(0);
    armnn::IConnectableLayer* const outputStateOut  = net->AddOutputLayer(1);

    armnn::TensorInfo inputTensorInfo({batchSize , inputSize},
                                      armnn::DataType::QAsymmU8,
                                      inputOutputScale,
                                      inputOutputOffset);

    armnn::TensorInfo cellStateInTensorInfo({batchSize , outputSize},
                                            armnn::DataType::QSymmS16,
                                            cellStateScale,
                                            cellStateOffset);

    armnn::TensorInfo outputStateInTensorInfo({batchSize , outputSize},
                                              armnn::DataType::QAsymmU8,
                                              inputOutputScale,
                                              inputOutputOffset);

    armnn::TensorInfo cellStateOutTensorInfo({batchSize, outputSize},
                                             armnn::DataType::QSymmS16,
                                             cellStateScale,
                                             cellStateOffset);

    armnn::TensorInfo outputTensorInfo({batchSize, outputSize},
                                       armnn::DataType::QAsymmU8,
                                       inputOutputScale,
                                       inputOutputOffset);

    // connect up
    // inputs
    Connect(inputLayer, quantizedLstmLayer, inputTensorInfo, 0, 0);
    Connect(cellStateIn, quantizedLstmLayer, cellStateInTensorInfo, 0, 1);
    Connect(outputStateIn, quantizedLstmLayer, outputStateInTensorInfo, 0, 2);

    // outputs
    Connect(quantizedLstmLayer, cellStateOut, cellStateOutTensorInfo, 0, 0);
    Connect(quantizedLstmLayer, outputStateOut, outputTensorInfo, 1, 0);

    return net;
}

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

void QuantizedLstmEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    std::vector<uint8_t> inputVector = {166, 179, 50, 150};
    armnn::TensorInfo inputDesc({2, 2}, armnn::DataType::QAsymmU8);

    std::vector<int16_t> cellStateInVector = {876, 1034, 955, -909, 761, 1029, 796, -1036};
    armnn::TensorInfo cellStateInDesc({2, 4}, armnn::DataType::QSymmS16);

    std::vector<uint8_t> outputStateInVector = {136, 150, 140, 115, 135, 152, 138, 112};
    armnn::TensorInfo outputStateInDesc({2, 4}, armnn::DataType::QAsymmU8);

    std::vector<int16_t> cellStateOutVector = {1485, 1177, 1373, -1023, 1019, 1355, 1097, -1235};
    armnn::TensorInfo cellStateOutVectorDesc({2, 4}, armnn::DataType::QSymmS16);

    std::vector<uint8_t> outputStateOutVector = {140, 151, 146, 112, 136, 156, 142, 112};
    armnn::TensorInfo outputDesc({2, 4}, armnn::DataType::QAsymmU8);

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateQuantizedLstmNetwork(inputDesc.GetShape(), outputDesc.GetShape());

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    InputTensors inputTensors;
    inputTensors.reserve(3);

    // input
    TensorInfo inputTensorInfo0 = runtime->GetInputTensorInfo(netId, 0);
    TensorInfo inputTensorInfo1 = runtime->GetInputTensorInfo(netId, 1);
    TensorInfo inputTensorInfo2 = runtime->GetInputTensorInfo(netId, 2);
    inputTensorInfo0.SetConstant(true);
    inputTensorInfo1.SetConstant(true);
    inputTensorInfo2.SetConstant(true);

    inputTensors.push_back({0, ConstTensor(inputTensorInfo0, inputVector.data())});
    inputTensors.push_back({1, ConstTensor(inputTensorInfo1, cellStateInVector.data())});
    inputTensors.push_back({2, ConstTensor(inputTensorInfo2, outputStateInVector.data())});

    OutputTensors outputTensors;
    outputTensors.reserve(2);

    //output
    std::vector<int16_t> cellStateOutResult(cellStateOutVector.size());
    std::vector<uint8_t> outputStateOutResult(outputStateOutVector.size());
    outputTensors.push_back({0, Tensor(runtime->GetOutputTensorInfo(netId, 0), cellStateOutResult.data())});
    outputTensors.push_back({1, Tensor(runtime->GetOutputTensorInfo(netId, 1), outputStateOutResult.data())});

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results
    constexpr int16_t toleranceInt16 = 2;
    for (unsigned int i = 0u; i < cellStateOutResult.size(); ++i)
    {
        CHECK(IsCloseEnough(cellStateOutVector[i], cellStateOutResult[i], toleranceInt16));
    }

    constexpr uint8_t toleranceUint8 = 1;
    for (unsigned int i = 0u; i < outputStateOutResult.size(); ++i)
    {
        CHECK(IsCloseEnough(outputStateOutVector[i], outputStateOutResult[i], toleranceUint8));
    }
}
