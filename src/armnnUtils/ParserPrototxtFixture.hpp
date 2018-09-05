//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/IRuntime.hpp"
#include "armnnOnnxParser/IOnnxParser.hpp"
#include "test/TensorHelpers.hpp"
#include "VerificationHelpers.hpp"

#include <boost/format.hpp>
#include <string>

namespace armnnUtils
{

template<typename TParser>
struct ParserPrototxtFixture
{
    ParserPrototxtFixture()
        : m_Parser(TParser::Create())
        , m_NetworkIdentifier(-1)
    {
        armnn::IRuntime::CreationOptions options;
        m_Runtimes.push_back(std::make_pair(armnn::IRuntime::Create(options), armnn::Compute::CpuRef));

#if ARMCOMPUTENEON_ENABLED
        m_Runtimes.push_back(std::make_pair(armnn::IRuntime::Create(options), armnn::Compute::CpuAcc));
#endif

#if ARMCOMPUTECL_ENABLED
        m_Runtimes.push_back(std::make_pair(armnn::IRuntime::Create(options), armnn::Compute::GpuAcc));
#endif
    }

    /// Parses and loads the network defined by the m_Prototext string.
    /// @{
    void SetupSingleInputSingleOutput(const std::string& inputName, const std::string& outputName);
    void SetupSingleInputSingleOutput(const armnn::TensorShape& inputTensorShape,
        const std::string& inputName,
        const std::string& outputName);
    void Setup(const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);
    void Setup();
    /// @}

    /// Executes the network with the given input tensor and checks the result against the given output tensor.
    /// This overload assumes that the network has a single input and a single output.
    template <std::size_t NumOutputDimensions>
    void RunTest(const std::vector<float>& inputData, const std::vector<float>& expectedOutputData);

    /// Executes the network with the given input tensors and checks the results against the given output tensors.
    /// This overload supports multiple inputs and multiple outputs, identified by name.
    template <std::size_t NumOutputDimensions>
    void RunTest(const std::map<std::string, std::vector<float>>& inputData,
        const std::map<std::string, std::vector<float>>& expectedOutputData);

    std::string                                         m_Prototext;
    std::unique_ptr<TParser, void(*)(TParser* parser)>  m_Parser;
    std::vector<std::pair<armnn::IRuntimePtr, armnn::Compute>> m_Runtimes;
    armnn::NetworkId                                    m_NetworkIdentifier;

    /// If the single-input-single-output overload of Setup() is called, these will store the input and output name
    /// so they don't need to be passed to the single-input-single-output overload of RunTest().
    /// @{
    std::string m_SingleInputName;
    std::string m_SingleOutputName;
    /// @}
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
void ParserPrototxtFixture<TParser>::Setup(const std::map<std::string, armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    for (auto&& runtime : m_Runtimes)
    {
        std::string errorMessage;

        armnn::INetworkPtr network =
            m_Parser->CreateNetworkFromString(m_Prototext.c_str(), inputShapes, requestedOutputs);
        auto optimized = Optimize(*network, { runtime.second, armnn::Compute::CpuRef }, runtime.first->GetDeviceSpec());
        armnn::Status ret = runtime.first->LoadNetwork(m_NetworkIdentifier, move(optimized), errorMessage);
        if (ret != armnn::Status::Success)
        {
            throw armnn::Exception(boost::str(
                boost::format("LoadNetwork failed with error: '%1%' %2%")
                              % errorMessage
                              % CHECK_LOCATION().AsString()));
        }
    }
}

template<typename TParser>
void ParserPrototxtFixture<TParser>::Setup()
{
    for (auto&& runtime : m_Runtimes)
    {
        std::string errorMessage;

        armnn::INetworkPtr network =
            m_Parser->CreateNetworkFromString(m_Prototext.c_str());
        auto optimized = Optimize(*network, { runtime.second, armnn::Compute::CpuRef }, runtime.first->GetDeviceSpec());
        armnn::Status ret = runtime.first->LoadNetwork(m_NetworkIdentifier, move(optimized), errorMessage);
        if (ret != armnn::Status::Success)
        {
            throw armnn::Exception(boost::str(
                boost::format("LoadNetwork failed with error: '%1%' %2%")
                              % errorMessage
                              % CHECK_LOCATION().AsString()));
        }
    }
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
void ParserPrototxtFixture<TParser>::RunTest(const std::map<std::string, std::vector<float>>& inputData,
    const std::map<std::string, std::vector<float>>& expectedOutputData)
{
    for (auto&& runtime : m_Runtimes)
    {
        using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;

        // Sets up the armnn input tensors from the given vectors.
        armnn::InputTensors inputTensors;
        for (auto&& it : inputData)
        {
            BindingPointInfo bindingInfo = m_Parser->GetNetworkInputBindingInfo(it.first);
            inputTensors.push_back({ bindingInfo.first, armnn::ConstTensor(bindingInfo.second, it.second.data()) });
        }

        // Allocates storage for the output tensors to be written to and sets up the armnn output tensors.
        std::map<std::string, boost::multi_array<float, NumOutputDimensions>> outputStorage;
        armnn::OutputTensors outputTensors;
        for (auto&& it : expectedOutputData)
        {
            BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(it.first);
            outputStorage.emplace(it.first, MakeTensor<float, NumOutputDimensions>(bindingInfo.second));
            outputTensors.push_back(
                { bindingInfo.first, armnn::Tensor(bindingInfo.second, outputStorage.at(it.first).data()) });
        }

        runtime.first->EnqueueWorkload(m_NetworkIdentifier, inputTensors, outputTensors);

        // Compares each output tensor to the expected values.
        for (auto&& it : expectedOutputData)
        {
            BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(it.first);
            if (bindingInfo.second.GetNumElements() != it.second.size())
            {
                throw armnn::Exception(
                    boost::str(
                        boost::format("Output tensor %1% is expected to have %2% elements. "
                                      "%3% elements supplied. %4%") %
                                      it.first %
                                      bindingInfo.second.GetNumElements() %
                                      it.second.size() %
                                      CHECK_LOCATION().AsString()));
            }
            auto outputExpected = MakeTensor<float, NumOutputDimensions>(bindingInfo.second, it.second);
            BOOST_TEST(CompareTensors(outputExpected, outputStorage[it.first]));
        }
    }
}

} // namespace armnnUtils
