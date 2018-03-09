//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "armnn/IRuntime.hpp"
#include "test/TensorHelpers.hpp"
#include <string>

template<typename TParser>
struct ParserPrototxtFixture
{
    ParserPrototxtFixture()
        : m_Parser(TParser::Create())
        , m_Runtime(armnn::IRuntime::Create(armnn::Compute::CpuRef))
        , m_NetworkIdentifier(-1)
    {}

    /// Parses and loads the network defined by the m_Prototext string.
    /// @{
    void SetupSingleInputSingleOutput(const std::string& inputName, const std::string& outputName);
    void SetupSingleInputSingleOutput(const armnn::TensorShape& inputTensorShape,
        const std::string& inputName,
        const std::string& outputName);
    void Setup(const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);
    /// @}

    /// Executes the network with the given input tensor and checks the result against the given output tensor.
    /// This overload assumes the network has a single input and a single output.
    template <std::size_t NumOutputDimensions>
    void RunTest(const std::vector<float>& inputData, const std::vector<float>& expectedOutputData);

    /// Executes the network with the given input tensors and checks the results against the given output tensors.
    /// This overload supports multiple inputs and multiple outputs, identified by name.
    template <std::size_t NumOutputDimensions>
    void RunTest(const std::map<std::string, std::vector<float>>& inputData,
        const std::map<std::string, std::vector<float>>& expectedOutputData);

    std::string                 m_Prototext;
    std::unique_ptr<TParser, void(*)(TParser* parser)> m_Parser;
    armnn::IRuntimePtr          m_Runtime;
    armnn::NetworkId            m_NetworkIdentifier;

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
    // Store the input and output name so they don't need to be passed to the single-input-single-output RunTest().
    m_SingleInputName = inputName;
    m_SingleOutputName = outputName;
    Setup({ }, { outputName });
}

template<typename TParser>
void ParserPrototxtFixture<TParser>::SetupSingleInputSingleOutput(const armnn::TensorShape& inputTensorShape,
    const std::string& inputName,
    const std::string& outputName)
{
    // Store the input and output name so they don't need to be passed to the single-input-single-output RunTest().
    m_SingleInputName = inputName;
    m_SingleOutputName = outputName;
    Setup({ { inputName, inputTensorShape } }, { outputName });
}

template<typename TParser>
void ParserPrototxtFixture<TParser>::Setup(const std::map<std::string, armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    armnn::INetworkPtr network =
        m_Parser->CreateNetworkFromString(m_Prototext.c_str(), inputShapes, requestedOutputs);

    auto optimized = Optimize(*network, m_Runtime->GetDeviceSpec());
    armnn::Status ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, move(optimized));
    if (ret != armnn::Status::Success)
    {
        throw armnn::Exception("LoadNetwork failed");
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
    using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;

    // Setup the armnn input tensors from the given vectors.
    armnn::InputTensors inputTensors;
    for (auto&& it : inputData)
    {
        BindingPointInfo bindingInfo = m_Parser->GetNetworkInputBindingInfo(it.first);
        inputTensors.push_back({ bindingInfo.first, armnn::ConstTensor(bindingInfo.second, it.second.data()) });
    }

    // Allocate storage for the output tensors to be written to and setup the armnn output tensors.
    std::map<std::string, boost::multi_array<float, NumOutputDimensions>> outputStorage;
    armnn::OutputTensors outputTensors;
    for (auto&& it : expectedOutputData)
    {
        BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(it.first);
        outputStorage.emplace(it.first, MakeTensor<float, NumOutputDimensions>(bindingInfo.second));
        outputTensors.push_back(
            { bindingInfo.first, armnn::Tensor(bindingInfo.second, outputStorage.at(it.first).data()) });
    }

    m_Runtime->EnqueueWorkload(m_NetworkIdentifier, inputTensors, outputTensors);

    // Compare each output tensor to the expected values
    for (auto&& it : expectedOutputData)
    {
        BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(it.first);
        auto outputExpected = MakeTensor<float, NumOutputDimensions>(bindingInfo.second, it.second);
        BOOST_TEST(CompareTensors(outputExpected, outputStorage[it.first]));
    }
}
