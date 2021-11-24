//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "EndToEndTestImpl.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/TypesUtils.hpp>

#include <CommonTestUtils.hpp>

#include <ResolveType.hpp>

namespace
{

/** Defines the acceptable tolerance of ActivationFunction-DataType combinations.
 *
 * @param activationFunction The activation function used
 * @param dataType  Data type used
 *
 * @return Tolerance depending on the activation function and data type
 */
float GetActivationTolerance(const armnn::ActivationFunction& activationFunction, DataType dataType)
{
    constexpr float defaultTolerance = 1e-6f;

    switch (activationFunction)
    {
        // The following values are taken from ArmComputeLibrary/tests/validation/CL/ActivationLayer.cpp
        case ActivationFunction::Elu:
            return (dataType == DataType::Float16 ? 0.01f : 0.00001f);
        case ActivationFunction::HardSwish:
            return (dataType == DataType::Float16 ? 0.01f : defaultTolerance);
        default:
            return defaultTolerance;
    }
}

/** Creates a network with one layer of the activation function specified in the activation descriptor.
 *
 * @param inputInfo  Tensor info of inputs
 * @param outputInfo  Tensor info of outputs
 * @param descriptor  Activation descriptor
 *
 * @return INetworkPtr  A pointer to the created network
 */
armnn::INetworkPtr CreateActivationNetwork(const armnn::TensorInfo& inputInfo,
                                           const armnn::TensorInfo& outputInfo,
                                           const armnn::ActivationDescriptor& descriptor)
{
    using namespace armnn;

    char const* ActivationName = GetActivationFunctionAsCString(descriptor.m_Function);

    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0, "input");
    IConnectableLayer* prelu = net->AddActivationLayer(descriptor, ActivationName);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    Connect(input, prelu, inputInfo, 0, 0);
    Connect(prelu, output, outputInfo, 0, 0);

    return net;
}

/** Specifies the implementation of end to end tests for activation functions.
 *
 *  - Converts input data and expected-output data to the data type that is desired for the test (ArmnnType)
 *  - Creates a network with one layer of the activation function specified in the activation descriptor.
 *  - Executes the network on specified backends and compares results to expected output values
 *
 * @tparam ArmnnType  The armnn data type for the input and expected-output data
 * @param backends  Backends to run test on
 * @param floatInputData  Input data given as vector of float
 * @param floatExpectedOutputData  Expected output data given as vector of float
 * @param inputInfo  Tensor info of inputs
 * @param outputInfo  Tensor info of outputs
 * @param descriptor  Activation descriptor
 */
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ActivationEndToEndImpl(const std::vector<armnn::BackendId>& backends,
                     const std::vector<float>& floatInputData,
                     const std::vector<float>& floatExpectedOutputData,
                     const armnn::TensorInfo&  inputInfo,
                     const armnn::TensorInfo&  outputInfo,
                     const armnn::ActivationDescriptor& descriptor)
{
    using namespace armnn;

    // Selectively quantizes/transforms float values to the needed data type
    std::vector<T> inputData          = armnnUtils::QuantizedVector<T>( floatInputData,
                                                                        inputInfo.GetQuantizationScale(),
                                                                        inputInfo.GetQuantizationOffset());
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>( floatExpectedOutputData,
                                                                        outputInfo.GetQuantizationScale(),
                                                                        outputInfo.GetQuantizationOffset());

    INetworkPtr net = CreateActivationNetwork(inputInfo, outputInfo, descriptor);

    std::map<int, std::vector<T>> inputTensorData          = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputTensorData = { { 0, expectedOutputData } };

    float tolerance = GetActivationTolerance(descriptor.m_Function, ArmnnType);

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends,
                                                tolerance);
}

/** Executes an end to end test for Elu activation with specific input and expected-output data
 *
 * @tparam ArmnnType  The armnn data type for the input and expected-output data
 * @param backends  The backends on which to run the test
 */
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void EluEndToEndTest(const std::vector<BackendId>& backends)
{
    std::vector<float> floatInputData{ -2.0f, -1.0f, -0.0f, 0.0f,
                                        1.0f,  2.0f,  3.0f, 4.0f };

    std::vector<float> floatExpectedOutputData{ -0.86466471676f,  -0.63212055882f,  -0.0f, 0.0f,
                                                 1.0f          ,   2.0f          ,   3.0f, 4.0f };

    float qScale = 1.0f;
    int32_t qOffset = 0;
    armnn::TensorInfo inputInfo({ 2, 2, 2, 1 }, ArmnnType, qScale, qOffset, true);
    armnn::TensorInfo outputInfo({ 2, 2, 2, 1 }, ArmnnType, qScale, qOffset);

    armnn::ActivationDescriptor descriptor(ActivationFunction::Elu, 1.0);

    ActivationEndToEndImpl<ArmnnType>(backends,
                                      floatInputData,
                                      floatExpectedOutputData,
                                      inputInfo,
                                      outputInfo,
                                      descriptor);
}

/** Executes an end to end test for HardSwish activation with specific input and expected-output data
 *
 * @tparam ArmnnType  The armnn data type for the input and expected-output data
 * @param backends  The backends on which to run the test
 */
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void HardSwishEndToEndTest(const std::vector<BackendId>& backends)
{
    std::vector<float> floatInputData{ -2.0f, -1.0f, -0.5f, 0.0f,
                                       1.0f,  2.0f,  3.0f, 4.0f };

    std::vector<float> floatExpectedOutputData{ -0.33333333333f,  -0.33333333333f, -0.208333f, 0.0f,
                                                 0.66666666667f,   1.66666666667f,  3.0f     , 4.0f };

    float qScale = 1.0f;
    int32_t qOffset = 0;
    armnn::TensorInfo inputInfo({ 2, 2, 2, 1 }, ArmnnType, qScale, qOffset, true);
    armnn::TensorInfo outputInfo({ 2, 2, 2, 1 }, ArmnnType, qScale, qOffset);

    armnn::ActivationDescriptor descriptor(ActivationFunction::HardSwish, 1.0);

    ActivationEndToEndImpl<ArmnnType>(backends,
                                      floatInputData,
                                      floatExpectedOutputData,
                                      inputInfo,
                                      outputInfo,
                                      descriptor);
}

} // anonymous namespace