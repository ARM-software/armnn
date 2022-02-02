//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "SchemaSerialize.hpp"
#include <armnnTestUtils/TensorHelpers.hpp>

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include <ArmnnSchema_generated.h>
#include <armnn/IRuntime.hpp>
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <ResolveType.hpp>

#include <fmt/format.h>
#include <doctest/doctest.h>

#include <vector>

using armnnDeserializer::IDeserializer;
using TensorRawPtr = armnnSerializer::TensorInfo*;

struct ParserFlatbuffersSerializeFixture
{
    ParserFlatbuffersSerializeFixture() :
        m_Parser(IDeserializer::Create()),
        m_Runtime(armnn::IRuntime::Create(armnn::IRuntime::CreationOptions())),
        m_NetworkIdentifier(-1)
    {
    }

    std::vector<uint8_t> m_GraphBinary;
    std::string m_JsonString;
    std::unique_ptr<IDeserializer, void (*)(IDeserializer* parser)> m_Parser;
    armnn::IRuntimePtr m_Runtime;
    armnn::NetworkId m_NetworkIdentifier;

    /// If the single-input-single-output overload of Setup() is called, these will store the input and output name
    /// so they don't need to be passed to the single-input-single-output overload of RunTest().
    std::string m_SingleInputName;
    std::string m_SingleOutputName;

    void Setup()
    {
        bool ok = ReadStringToBinary();
        if (!ok)
        {
            throw armnn::Exception("LoadNetwork failed while reading binary input");
        }

        armnn::INetworkPtr network =
                m_Parser->CreateNetworkFromBinary(m_GraphBinary);

        if (!network)
        {
            throw armnn::Exception("The parser failed to create an ArmNN network");
        }

        auto optimized = Optimize(*network, {armnn::Compute::CpuRef},
                                  m_Runtime->GetDeviceSpec());

        std::string errorMessage;
        armnn::Status ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, move(optimized), errorMessage);

        if (ret != armnn::Status::Success)
        {
            throw armnn::Exception(fmt::format("The runtime failed to load the network. "
                                               "Error was: {0}. in {1} [{2}:{3}]",
                                               errorMessage,
                                               __func__,
                                               __FILE__,
                                               __LINE__));
        }

    }

    void SetupSingleInputSingleOutput(const std::string& inputName, const std::string& outputName)
    {
        // Store the input and output name so they don't need to be passed to the single-input-single-output RunTest().
        m_SingleInputName = inputName;
        m_SingleOutputName = outputName;
        Setup();
    }

    bool ReadStringToBinary()
    {
        std::string schemafile(&deserialize_schema_start, &deserialize_schema_end);

        // parse schema first, so we can use it to parse the data after
        flatbuffers::Parser parser;

        bool ok = parser.Parse(schemafile.c_str());
        CHECK_MESSAGE(ok, std::string("Failed to parse schema file. Error was: " + parser.error_).c_str());

        ok &= parser.Parse(m_JsonString.c_str());
        CHECK_MESSAGE(ok, std::string("Failed to parse json input. Error was: " + parser.error_).c_str());

        if (!ok)
        {
            return false;
        }

        {
            const uint8_t* bufferPtr = parser.builder_.GetBufferPointer();
            size_t size = static_cast<size_t>(parser.builder_.GetSize());
            m_GraphBinary.assign(bufferPtr, bufferPtr+size);
        }
        return ok;
    }

    /// Executes the network with the given input tensor and checks the result against the given output tensor.
    /// This overload assumes the network has a single input and a single output.
    template<std::size_t NumOutputDimensions,
             armnn::DataType ArmnnType,
             typename DataType = armnn::ResolveType<ArmnnType>>
    void RunTest(unsigned int layersId,
                 const std::vector<DataType>& inputData,
                 const std::vector<DataType>& expectedOutputData);

    template<std::size_t NumOutputDimensions,
             armnn::DataType ArmnnInputType,
             armnn::DataType ArmnnOutputType,
             typename InputDataType = armnn::ResolveType<ArmnnInputType>,
             typename OutputDataType = armnn::ResolveType<ArmnnOutputType>>
    void RunTest(unsigned int layersId,
                 const std::vector<InputDataType>& inputData,
                 const std::vector<OutputDataType>& expectedOutputData);

    /// Executes the network with the given input tensors and checks the results against the given output tensors.
    /// This overload supports multiple inputs and multiple outputs, identified by name.
    template<std::size_t NumOutputDimensions,
             armnn::DataType ArmnnType,
             typename DataType = armnn::ResolveType<ArmnnType>>
    void RunTest(unsigned int layersId,
                 const std::map<std::string, std::vector<DataType>>& inputData,
                 const std::map<std::string, std::vector<DataType>>& expectedOutputData);

    template<std::size_t NumOutputDimensions,
             armnn::DataType ArmnnInputType,
             armnn::DataType ArmnnOutputType,
             typename InputDataType = armnn::ResolveType<ArmnnInputType>,
             typename OutputDataType = armnn::ResolveType<ArmnnOutputType>>
    void RunTest(unsigned int layersId,
                 const std::map<std::string, std::vector<InputDataType>>& inputData,
                 const std::map<std::string, std::vector<OutputDataType>>& expectedOutputData);

    void CheckTensors(const TensorRawPtr& tensors, size_t shapeSize, const std::vector<int32_t>& shape,
                      armnnSerializer::TensorInfo tensorType, const std::string& name,
                      const float scale, const int64_t zeroPoint)
    {
        armnn::IgnoreUnused(name);
        CHECK_EQ(shapeSize, tensors->dimensions()->size());
        CHECK(std::equal(shape.begin(), shape.end(),
                                      tensors->dimensions()->begin(), tensors->dimensions()->end()));
        CHECK_EQ(tensorType.dataType(), tensors->dataType());
        CHECK_EQ(scale, tensors->quantizationScale());
        CHECK_EQ(zeroPoint, tensors->quantizationOffset());
    }
};

template<std::size_t NumOutputDimensions, armnn::DataType ArmnnType, typename DataType>
void ParserFlatbuffersSerializeFixture::RunTest(unsigned int layersId,
                                                const std::vector<DataType>& inputData,
                                                const std::vector<DataType>& expectedOutputData)
{
    RunTest<NumOutputDimensions, ArmnnType, ArmnnType, DataType, DataType>(layersId, inputData, expectedOutputData);
}

template<std::size_t NumOutputDimensions,
         armnn::DataType ArmnnInputType,
         armnn::DataType ArmnnOutputType,
         typename InputDataType,
         typename OutputDataType>
void ParserFlatbuffersSerializeFixture::RunTest(unsigned int layersId,
                                                const std::vector<InputDataType>& inputData,
                                                const std::vector<OutputDataType>& expectedOutputData)
{
    RunTest<NumOutputDimensions, ArmnnInputType, ArmnnOutputType>(layersId,
                                                                  { { m_SingleInputName, inputData } },
                                                                  { { m_SingleOutputName, expectedOutputData } });
}

template<std::size_t NumOutputDimensions, armnn::DataType ArmnnType, typename DataType>
void ParserFlatbuffersSerializeFixture::RunTest(unsigned int layersId,
                                                const std::map<std::string, std::vector<DataType>>& inputData,
                                                const std::map<std::string, std::vector<DataType>>& expectedOutputData)
{
    RunTest<NumOutputDimensions, ArmnnType, ArmnnType, DataType, DataType>(layersId, inputData, expectedOutputData);
}

template<std::size_t NumOutputDimensions,
         armnn::DataType ArmnnInputType,
         armnn::DataType ArmnnOutputType,
         typename InputDataType,
         typename OutputDataType>
void ParserFlatbuffersSerializeFixture::RunTest(
    unsigned int layersId,
    const std::map<std::string, std::vector<InputDataType>>& inputData,
    const std::map<std::string, std::vector<OutputDataType>>& expectedOutputData)
{
    auto ConvertBindingInfo = [](const armnnDeserializer::BindingPointInfo& bindingInfo)
        {
            return std::make_pair(bindingInfo.m_BindingId, bindingInfo.m_TensorInfo);
        };

    // Setup the armnn input tensors from the given vectors.
    armnn::InputTensors inputTensors;
    for (auto&& it : inputData)
    {
        armnn::BindingPointInfo bindingInfo = ConvertBindingInfo(
            m_Parser->GetNetworkInputBindingInfo(layersId, it.first));
        bindingInfo.second.SetConstant(true);
        armnn::VerifyTensorInfoDataType(bindingInfo.second, ArmnnInputType);
        inputTensors.push_back({ bindingInfo.first, armnn::ConstTensor(bindingInfo.second, it.second.data()) });
    }

    // Allocate storage for the output tensors to be written to and setup the armnn output tensors.
    std::map<std::string, std::vector<OutputDataType>> outputStorage;
    armnn::OutputTensors outputTensors;
    for (auto&& it : expectedOutputData)
    {
        armnn::BindingPointInfo bindingInfo = ConvertBindingInfo(
            m_Parser->GetNetworkOutputBindingInfo(layersId, it.first));
        armnn::VerifyTensorInfoDataType(bindingInfo.second, ArmnnOutputType);
        outputStorage.emplace(it.first, std::vector<OutputDataType>(bindingInfo.second.GetNumElements()));
        outputTensors.push_back(
                { bindingInfo.first, armnn::Tensor(bindingInfo.second, outputStorage.at(it.first).data()) });
    }

    m_Runtime->EnqueueWorkload(m_NetworkIdentifier, inputTensors, outputTensors);

    // Compare each output tensor to the expected values
    for (auto&& it : expectedOutputData)
    {
        armnn::BindingPointInfo bindingInfo = ConvertBindingInfo(
            m_Parser->GetNetworkOutputBindingInfo(layersId, it.first));
        auto outputExpected = it.second;
        auto result = CompareTensors(outputExpected, outputStorage[it.first],
                                     bindingInfo.second.GetShape(), bindingInfo.second.GetShape());
        CHECK_MESSAGE(result.m_Result, result.m_Message.str());
    }
}
