//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/IRuntime.hpp>
#include <armnnTestUtils/TensorHelpers.hpp>

#include <Network.hpp>
#include <VerificationHelpers.hpp>

#include <doctest/doctest.h>
#include <fmt/format.h>

#include <iomanip>
#include <string>

namespace armnnUtils
{

template<typename TParser>
struct ParserPrototxtFixture
{
    ParserPrototxtFixture()
        : m_Parser(TParser::Create())
        , m_Runtime(armnn::IRuntime::Create(armnn::IRuntime::CreationOptions()))
        , m_NetworkIdentifier(-1)
    {
    }

    /// Parses and loads the network defined by the m_Prototext string.
    /// @{
    void SetupSingleInputSingleOutput(const std::string& inputName, const std::string& outputName);
    void SetupSingleInputSingleOutput(const armnn::TensorShape& inputTensorShape,
        const std::string& inputName,
        const std::string& outputName);
    void SetupSingleInputSingleOutput(const armnn::TensorShape& inputTensorShape,
                                      const armnn::TensorShape& outputTensorShape,
                                      const std::string& inputName,
                                      const std::string& outputName);
    void Setup(const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);
    void Setup(const std::map<std::string, armnn::TensorShape>& inputShapes);
    void Setup();
    armnn::IOptimizedNetworkPtr SetupOptimizedNetwork(
        const std::map<std::string,armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);
    /// @}

    /// Executes the network with the given input tensor and checks the result against the given output tensor.
    /// This overload assumes that the network has a single input and a single output.
    template <std::size_t NumOutputDimensions>
    void RunTest(const std::vector<float>& inputData, const std::vector<float>& expectedOutputData);

    /// Executes the network with the given input tensor and checks the result against the given output tensor.
    /// Calls RunTest with output type of uint8_t for checking comparison operators.
    template <std::size_t NumOutputDimensions>
    void RunComparisonTest(const std::map<std::string, std::vector<float>>& inputData,
                           const std::map<std::string, std::vector<uint8_t>>& expectedOutputData);

    /// Executes the network with the given input tensors and checks the results against the given output tensors.
    /// This overload supports multiple inputs and multiple outputs, identified by name.
    template <std::size_t NumOutputDimensions, typename T = float>
    void RunTest(const std::map<std::string, std::vector<float>>& inputData,
                 const std::map<std::string, std::vector<T>>& expectedOutputData);

    std::string                                         m_Prototext;
    std::unique_ptr<TParser, void(*)(TParser* parser)>  m_Parser;
    armnn::IRuntimePtr                                  m_Runtime;
    armnn::NetworkId                                    m_NetworkIdentifier;

    /// If the single-input-single-output overload of Setup() is called, these will store the input and output name
    /// so they don't need to be passed to the single-input-single-output overload of RunTest().
    /// @{
    std::string m_SingleInputName;
    std::string m_SingleOutputName;
    /// @}

    /// This will store the output shape so it don't need to be passed to the single-input-single-output overload
    /// of RunTest().
    armnn::TensorShape m_SingleOutputShape;
};

template<typename TParser>
void ParserPrototxtFixture<TParser>::SetupSingleInputSingleOutput(const std::string& inputName,
    const std::string& outputName)
{
    // Stores the input and output name so they don't need to be passed to the single-input-single-output RunTest().
    m_SingleInputName = inputName;
    m_SingleOutputName = outputName;
    Setup({ }, { outputName });
}

template<typename TParser>
void ParserPrototxtFixture<TParser>::SetupSingleInputSingleOutput(const armnn::TensorShape& inputTensorShape,
    const std::string& inputName,
    const std::string& outputName)
{
    // Stores the input and output name so they don't need to be passed to the single-input-single-output RunTest().
    m_SingleInputName = inputName;
    m_SingleOutputName = outputName;
    Setup({ { inputName, inputTensorShape } }, { outputName });
}

template<typename TParser>
void ParserPrototxtFixture<TParser>::SetupSingleInputSingleOutput(const armnn::TensorShape& inputTensorShape,
                                                                  const armnn::TensorShape& outputTensorShape,
                                                                  const std::string& inputName,
                                                                  const std::string& outputName)
{
    // Stores the input name, the output name and the output tensor shape
    // so they don't need to be passed to the single-input-single-output RunTest().
    m_SingleInputName = inputName;
    m_SingleOutputName = outputName;
    m_SingleOutputShape = outputTensorShape;
    Setup({ { inputName, inputTensorShape } }, { outputName });
}

template<typename TParser>
void ParserPrototxtFixture<TParser>::Setup(const std::map<std::string, armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    std::string errorMessage;

    armnn::INetworkPtr network =
        m_Parser->CreateNetworkFromString(m_Prototext.c_str(), inputShapes, requestedOutputs);
    auto optimized = Optimize(*network, { armnn::Compute::CpuRef }, m_Runtime->GetDeviceSpec());
    armnn::Status ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, move(optimized), errorMessage);
    if (ret != armnn::Status::Success)
    {
        throw armnn::Exception(fmt::format("LoadNetwork failed with error: '{0}' {1}",
                                           errorMessage,
                                           CHECK_LOCATION().AsString()));
    }
}

template<typename TParser>
void ParserPrototxtFixture<TParser>::Setup(const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    std::string errorMessage;

    armnn::INetworkPtr network =
        m_Parser->CreateNetworkFromString(m_Prototext.c_str(), inputShapes);
    auto optimized = Optimize(*network, { armnn::Compute::CpuRef }, m_Runtime->GetDeviceSpec());
    armnn::Status ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, move(optimized), errorMessage);
    if (ret != armnn::Status::Success)
    {
        throw armnn::Exception(fmt::format("LoadNetwork failed with error: '{0}' {1}",
                                           errorMessage,
                                           CHECK_LOCATION().AsString()));
    }
}

template<typename TParser>
void ParserPrototxtFixture<TParser>::Setup()
{
    std::string errorMessage;

    armnn::INetworkPtr network =
        m_Parser->CreateNetworkFromString(m_Prototext.c_str());
    auto optimized = Optimize(*network, { armnn::Compute::CpuRef }, m_Runtime->GetDeviceSpec());
    armnn::Status ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, move(optimized), errorMessage);
    if (ret != armnn::Status::Success)
    {
        throw armnn::Exception(fmt::format("LoadNetwork failed with error: '{0}' {1}",
                                           errorMessage,
                                           CHECK_LOCATION().AsString()));
    }
}

template<typename TParser>
armnn::IOptimizedNetworkPtr ParserPrototxtFixture<TParser>::SetupOptimizedNetwork(
    const std::map<std::string,armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    armnn::INetworkPtr network =
        m_Parser->CreateNetworkFromString(m_Prototext.c_str(), inputShapes, requestedOutputs);
    auto optimized = Optimize(*network, { armnn::Compute::CpuRef }, m_Runtime->GetDeviceSpec());
    return optimized;
}

template<typename TParser>
template <std::size_t NumOutputDimensions>
void ParserPrototxtFixture<TParser>::RunTest(const std::vector<float>& inputData,
                                             const std::vector<float>& expectedOutputData)
{
    RunTest<NumOutputDimensions>({ { m_SingleInputName, inputData } }, { { m_SingleOutputName, expectedOutputData } });
}

template<typename TParser>
template <std::size_t NumOutputDimensions>
void ParserPrototxtFixture<TParser>::RunComparisonTest(const std::map<std::string, std::vector<float>>& inputData,
                                                       const std::map<std::string, std::vector<uint8_t>>&
                                                       expectedOutputData)
{
    RunTest<NumOutputDimensions, uint8_t>(inputData, expectedOutputData);
}

template<typename TParser>
template <std::size_t NumOutputDimensions, typename T>
void ParserPrototxtFixture<TParser>::RunTest(const std::map<std::string, std::vector<float>>& inputData,
    const std::map<std::string, std::vector<T>>& expectedOutputData)
{
    // Sets up the armnn input tensors from the given vectors.
    armnn::InputTensors inputTensors;
    for (auto&& it : inputData)
    {
        armnn::BindingPointInfo bindingInfo = m_Parser->GetNetworkInputBindingInfo(it.first);
        bindingInfo.second.SetConstant(true);
        inputTensors.push_back({ bindingInfo.first, armnn::ConstTensor(bindingInfo.second, it.second.data()) });
        if (bindingInfo.second.GetNumElements() != it.second.size())
        {
            throw armnn::Exception(fmt::format("Input tensor {0} is expected to have {1} elements. "
                                               "{2} elements supplied. {3}",
                                               it.first,
                                               bindingInfo.second.GetNumElements(),
                                               it.second.size(),
                                               CHECK_LOCATION().AsString()));
        }
    }

    // Allocates storage for the output tensors to be written to and sets up the armnn output tensors.
    std::map<std::string, std::vector<T>> outputStorage;
    armnn::OutputTensors outputTensors;
    for (auto&& it : expectedOutputData)
    {
        armnn::BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(it.first);
        outputStorage.emplace(it.first, std::vector<T>(bindingInfo.second.GetNumElements()));
        outputTensors.push_back(
            { bindingInfo.first, armnn::Tensor(bindingInfo.second, outputStorage.at(it.first).data()) });
    }

    m_Runtime->EnqueueWorkload(m_NetworkIdentifier, inputTensors, outputTensors);

    // Compares each output tensor to the expected values.
    for (auto&& it : expectedOutputData)
    {
        armnn::BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(it.first);
        if (bindingInfo.second.GetNumElements() != it.second.size())
        {
            throw armnn::Exception(fmt::format("Output tensor {0} is expected to have {1} elements. "
                                               "{2} elements supplied. {3}",
                                               it.first,
                                               bindingInfo.second.GetNumElements(),
                                               it.second.size(),
                                               CHECK_LOCATION().AsString()));
        }

        // If the expected output shape is set, the output tensor checks will be carried out.
        if (m_SingleOutputShape.GetNumDimensions() != 0)
        {

            if (bindingInfo.second.GetShape().GetNumDimensions() == NumOutputDimensions &&
                bindingInfo.second.GetShape().GetNumDimensions() == m_SingleOutputShape.GetNumDimensions())
            {
                for (unsigned int i = 0; i < m_SingleOutputShape.GetNumDimensions(); ++i)
                {
                    if (m_SingleOutputShape[i] != bindingInfo.second.GetShape()[i])
                    {
                        // This exception message could not be created by fmt:format because of an oddity in
                        // the operator << of TensorShape.
                        std::stringstream message;
                        message << "Output tensor " << it.first << " is expected to have "
                                << bindingInfo.second.GetShape() << "shape. "
                                << m_SingleOutputShape << " shape supplied. "
                                << CHECK_LOCATION().AsString();
                        throw armnn::Exception(message.str());
                    }
                }
            }
            else
            {
                throw armnn::Exception(fmt::format("Output tensor {0} is expected to have {1} dimensions. "
                                                   "{2} dimensions supplied. {3}",
                                                   it.first,
                                                   bindingInfo.second.GetShape().GetNumDimensions(),
                                                   NumOutputDimensions,
                                                   CHECK_LOCATION().AsString()));
            }
        }

        auto outputExpected = it.second;
        auto shape = bindingInfo.second.GetShape();
        if (std::is_same<T, uint8_t>::value)
        {
            auto result = CompareTensors(outputExpected, outputStorage[it.first], shape, shape, true);
            CHECK_MESSAGE(result.m_Result, result.m_Message.str());
        }
        else
        {
            auto result = CompareTensors(outputExpected, outputStorage[it.first], shape, shape);
            CHECK_MESSAGE(result.m_Result, result.m_Message.str());
        }
    }
}

} // namespace armnnUtils
