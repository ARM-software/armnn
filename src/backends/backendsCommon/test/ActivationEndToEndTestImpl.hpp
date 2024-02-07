//
// Copyright © 2020-2021,2023-2024 Arm Ltd and Contributors. All rights reserved.
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

    IConnectableLayer* inputLayer = net->AddInputLayer(0, "input");
    IConnectableLayer* activationLayer = net->AddActivationLayer(descriptor, ActivationName);
    IConnectableLayer* outputLayer = net->AddOutputLayer(0, "output");

    Connect(inputLayer, activationLayer, inputInfo, 0, 0);
    Connect(activationLayer, outputLayer, outputInfo, 0, 0);

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

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends,
                                                tolerance);
}

std::vector<float> Activation(const std::vector<float>& input,
                              const ActivationDescriptor& descriptor)
{
    float a = descriptor.m_A;
    float b = descriptor.m_B;

    std::vector<float> output;
    output.reserve(input.size());

    // Compute the result of the activation function.
    switch (descriptor.m_Function)
    {
        case ActivationFunction::Linear:
        {
            for (auto in :input)
            {
                auto out = a * in + b;
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::Sigmoid:
        {
            for (auto in :input)
            {
                auto out = 1.f / (1.f + expf(-in));
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::ReLu:
        {
            for (auto in :input)
            {
                auto out = std::max(0.f, in);
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::BoundedReLu:
        {
            for (auto in :input)
            {
                auto out = std::min(a, std::max(b, in));
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::SoftReLu:
        {
            for (auto in :input)
            {
                auto out = logf(1.0f + expf(in));
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::LeakyReLu:
        {
            for (auto in :input)
            {
                auto out = in > 0.0f ? in : (in * a);
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::Abs:
        {
            for (auto in :input)
            {
                auto out = in < 0 ? -in : in;
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::Sqrt:
        {
            for (auto in :input)
            {
                auto out = sqrtf(in);
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::Square:
        {
            for (auto in :input)
            {
                auto out = in * in;
                output.push_back(out);
            }
            break;
       }
        case ActivationFunction::TanH:
        {
            for (auto in :input)
            {
                auto out = a * tanhf(b * in);
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::Elu:
        {
            for (auto in: input) {
                auto out = (in >= 0) ? in : a * (expf(in) - 1);
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::HardSwish:
        {
            for (auto in :input)
            {
                // hard_swish(x) = x * relu6(x+3) / 6
                // relu6(x) = min(max(x,0),6)
                auto out = in * (std::min(std::max((in + 3), 0.0f), 6.0f)) / 6;
                output.push_back(out);
            }
            break;
        }
        case ActivationFunction::Gelu:
        {
            for (auto in :input)
            {
                // gelu(x) = x * 1/2 * (1 + erf(x / sqrt(2))),
                // where erf is Gaussian error function
                auto out = in * (0.5f * (1.0f + erff(static_cast<float>(in / std::sqrt(2)))));
                output.push_back(out);
            }
            break;
        }
        default:
        {
            throw InvalidArgumentException("Unsupported activation function");
        }
    }
    return output;
}

/** Executes an end to end test for activation layers with specific input and expected-output data
 *
 * @tparam ArmnnType  The armnn data type for the input and expected-output data
 * @param backends  The backends on which to run the test
 */
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ActivationEndToEndTest(const std::vector<BackendId>& backends,
                            const ActivationFunction activationFunction,
                            const float qScale=1.0f,
                            const int32_t qOffset=0,
                            const float a = 1,
                            const float b = 0)
{
    std::vector<float> floatInputData{ -2.0f, -1.0f, -0.0f, 0.0f,
                                       1.0f,  2.0f,  3.0f, 4.0f };

    ActivationDescriptor descriptor(activationFunction, a, b);

    std::vector<float> floatExpectedOutputData = Activation(floatInputData, descriptor);

    armnn::TensorInfo inputInfo({ 2, 2, 2, 1 }, ArmnnType, qScale, qOffset, true);
    armnn::TensorInfo outputInfo({ 2, 2, 2, 1 }, ArmnnType, qScale, qOffset);

    ActivationEndToEndImpl<ArmnnType>(backends,
                                      floatInputData,
                                      floatExpectedOutputData,
                                      inputInfo,
                                      outputInfo,
                                      descriptor);
}

} // anonymous namespace
