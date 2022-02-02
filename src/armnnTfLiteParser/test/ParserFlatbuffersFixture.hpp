//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Schema.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/BackendRegistry.hpp>

#include "../TfLiteParser.hpp"

#include <ResolveType.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <fmt/format.h>
#include <doctest/doctest.h>

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "flatbuffers/flexbuffers.h"

#include <schema_generated.h>


using armnnTfLiteParser::ITfLiteParser;
using armnnTfLiteParser::ITfLiteParserPtr;

using TensorRawPtr = const tflite::TensorT *;
struct ParserFlatbuffersFixture
{
    ParserFlatbuffersFixture() :
            m_Runtime(armnn::IRuntime::Create(armnn::IRuntime::CreationOptions())),
            m_NetworkIdentifier(0),
            m_DynamicNetworkIdentifier(1)
    {
        ITfLiteParser::TfLiteParserOptions options;
        options.m_StandInLayerForUnsupported = true;
        options.m_InferAndValidate = true;

        m_Parser = std::make_unique<armnnTfLiteParser::TfLiteParserImpl>(
                        armnn::Optional<ITfLiteParser::TfLiteParserOptions>(options));
    }

    std::vector<uint8_t> m_GraphBinary;
    std::string          m_JsonString;
    armnn::IRuntimePtr   m_Runtime;
    armnn::NetworkId     m_NetworkIdentifier;
    armnn::NetworkId     m_DynamicNetworkIdentifier;
    bool                 m_TestDynamic;
    std::unique_ptr<armnnTfLiteParser::TfLiteParserImpl> m_Parser;

    /// If the single-input-single-output overload of Setup() is called, these will store the input and output name
    /// so they don't need to be passed to the single-input-single-output overload of RunTest().
    std::string m_SingleInputName;
    std::string m_SingleOutputName;

    void Setup(bool testDynamic = true)
    {
        m_TestDynamic = testDynamic;
        loadNetwork(m_NetworkIdentifier, false);

        if (m_TestDynamic)
        {
            loadNetwork(m_DynamicNetworkIdentifier, true);
        }
    }

    std::unique_ptr<tflite::ModelT> MakeModelDynamic(std::vector<uint8_t> graphBinary)
    {
        const uint8_t* binaryContent = graphBinary.data();
        const size_t len = graphBinary.size();
        if (binaryContent == nullptr)
        {
            throw armnn::InvalidArgumentException(fmt::format("Invalid (null) binary content {}",
                                                               CHECK_LOCATION().AsString()));
        }
        flatbuffers::Verifier verifier(binaryContent, len);
        if (verifier.VerifyBuffer<tflite::Model>() == false)
        {
            throw armnn::ParseException(fmt::format("Buffer doesn't conform to the expected Tensorflow Lite "
                                                    "flatbuffers format. size:{} {}",
                                                    len,
                                                    CHECK_LOCATION().AsString()));
        }
        auto model =  tflite::UnPackModel(binaryContent);

        for (auto const& subgraph : model->subgraphs)
        {
            std::vector<int32_t> inputIds = subgraph->inputs;
            for (unsigned int tensorIndex = 0; tensorIndex < subgraph->tensors.size(); ++tensorIndex)
            {
                if (std::find(inputIds.begin(), inputIds.end(), tensorIndex) != inputIds.end())
                {
                    continue;
                }
                for (auto const& tensor : subgraph->tensors)
                {
                    if (tensor->shape_signature.size() != 0)
                    {
                        continue;
                    }

                    for (unsigned int i = 0; i < tensor->shape.size(); ++i)
                    {
                        tensor->shape_signature.push_back(-1);
                    }
                }
            }
        }

        return model;
    }

    void loadNetwork(armnn::NetworkId networkId, bool loadDynamic)
    {
        if (!ReadStringToBinary())
        {
            throw armnn::Exception("LoadNetwork failed while reading binary input");
        }

        armnn::INetworkPtr network = loadDynamic ? m_Parser->LoadModel(MakeModelDynamic(m_GraphBinary))
                                                 : m_Parser->CreateNetworkFromBinary(m_GraphBinary);

        if (!network) {
            throw armnn::Exception("The parser failed to create an ArmNN network");
        }

        auto optimized = Optimize(*network, { armnn::Compute::CpuRef },
                                  m_Runtime->GetDeviceSpec());
        std::string errorMessage;

        armnn::Status ret = m_Runtime->LoadNetwork(networkId, move(optimized), errorMessage);

        if (ret != armnn::Status::Success)
        {
            throw armnn::Exception(
                fmt::format("The runtime failed to load the network. "
                            "Error was: {}. in {} [{}:{}]",
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
        std::string schemafile(g_TfLiteSchemaText, g_TfLiteSchemaText + g_TfLiteSchemaText_len);

        // parse schema first, so we can use it to parse the data after
        flatbuffers::Parser parser;

        bool ok = parser.Parse(schemafile.c_str());
        CHECK_MESSAGE(ok, std::string("Failed to parse schema file. Error was: " + parser.error_).c_str());

        ok = parser.Parse(m_JsonString.c_str());
        CHECK_MESSAGE(ok, std::string("Failed to parse json input. Error was: " + parser.error_).c_str());

        {
            const uint8_t * bufferPtr = parser.builder_.GetBufferPointer();
            size_t size = static_cast<size_t>(parser.builder_.GetSize());
            m_GraphBinary.assign(bufferPtr, bufferPtr+size);
        }
        return ok;
    }

    /// Executes the network with the given input tensor and checks the result against the given output tensor.
    /// This assumes the network has a single input and a single output.
    template <std::size_t NumOutputDimensions,
              armnn::DataType ArmnnType>
    void RunTest(size_t subgraphId,
                 const std::vector<armnn::ResolveType<ArmnnType>>& inputData,
                 const std::vector<armnn::ResolveType<ArmnnType>>& expectedOutputData);

    /// Executes the network with the given input tensors and checks the results against the given output tensors.
    /// This overload supports multiple inputs and multiple outputs, identified by name.
    template <std::size_t NumOutputDimensions,
              armnn::DataType ArmnnType>
    void RunTest(size_t subgraphId,
                 const std::map<std::string, std::vector<armnn::ResolveType<ArmnnType>>>& inputData,
                 const std::map<std::string, std::vector<armnn::ResolveType<ArmnnType>>>& expectedOutputData);

    /// Multiple Inputs, Multiple Outputs w/ Variable Datatypes and different dimension sizes.
    /// Executes the network with the given input tensors and checks the results against the given output tensors.
    /// This overload supports multiple inputs and multiple outputs, identified by name along with the allowance for
    /// the input datatype to be different to the output
    template <std::size_t NumOutputDimensions,
              armnn::DataType ArmnnType1,
              armnn::DataType ArmnnType2>
    void RunTest(size_t subgraphId,
                 const std::map<std::string, std::vector<armnn::ResolveType<ArmnnType1>>>& inputData,
                 const std::map<std::string, std::vector<armnn::ResolveType<ArmnnType2>>>& expectedOutputData,
                 bool isDynamic = false);

    /// Multiple Inputs with different DataTypes, Multiple Outputs w/ Variable DataTypes
    /// Executes the network with the given input tensors and checks the results against the given output tensors.
    /// This overload supports multiple inputs and multiple outputs, identified by name along with the allowance for
    /// the input datatype to be different to the output
    template <std::size_t NumOutputDimensions,
        armnn::DataType inputType1,
        armnn::DataType inputType2,
        armnn::DataType outputType>
    void RunTest(size_t subgraphId,
                 const std::map<std::string, std::vector<armnn::ResolveType<inputType1>>>& input1Data,
                 const std::map<std::string, std::vector<armnn::ResolveType<inputType2>>>& input2Data,
                 const std::map<std::string, std::vector<armnn::ResolveType<outputType>>>& expectedOutputData);

    /// Multiple Inputs, Multiple Outputs w/ Variable Datatypes and different dimension sizes.
    /// Executes the network with the given input tensors and checks the results against the given output tensors.
    /// This overload supports multiple inputs and multiple outputs, identified by name along with the allowance for
    /// the input datatype to be different to the output
    template<armnn::DataType ArmnnType1,
             armnn::DataType ArmnnType2>
    void RunTest(std::size_t subgraphId,
                 const std::map<std::string, std::vector<armnn::ResolveType<ArmnnType1>>>& inputData,
                 const std::map<std::string, std::vector<armnn::ResolveType<ArmnnType2>>>& expectedOutputData);

    static inline std::string GenerateDetectionPostProcessJsonString(
        const armnn::DetectionPostProcessDescriptor& descriptor)
    {
        flexbuffers::Builder detectPostProcess;
        detectPostProcess.Map([&]() {
            detectPostProcess.Bool("use_regular_nms", descriptor.m_UseRegularNms);
            detectPostProcess.Int("max_detections", descriptor.m_MaxDetections);
            detectPostProcess.Int("max_classes_per_detection", descriptor.m_MaxClassesPerDetection);
            detectPostProcess.Int("detections_per_class", descriptor.m_DetectionsPerClass);
            detectPostProcess.Int("num_classes", descriptor.m_NumClasses);
            detectPostProcess.Float("nms_score_threshold", descriptor.m_NmsScoreThreshold);
            detectPostProcess.Float("nms_iou_threshold", descriptor.m_NmsIouThreshold);
            detectPostProcess.Float("h_scale", descriptor.m_ScaleH);
            detectPostProcess.Float("w_scale", descriptor.m_ScaleW);
            detectPostProcess.Float("x_scale", descriptor.m_ScaleX);
            detectPostProcess.Float("y_scale", descriptor.m_ScaleY);
        });
        detectPostProcess.Finish();

        // Create JSON string
        std::stringstream strStream;
        std::vector<uint8_t> buffer = detectPostProcess.GetBuffer();
        std::copy(buffer.begin(), buffer.end(),std::ostream_iterator<int>(strStream,","));

        return strStream.str();
    }

    void CheckTensors(const TensorRawPtr& tensors, size_t shapeSize, const std::vector<int32_t>& shape,
                      tflite::TensorType tensorType, uint32_t buffer, const std::string& name,
                      const std::vector<float>& min, const std::vector<float>& max,
                      const std::vector<float>& scale, const std::vector<int64_t>& zeroPoint)
    {
        CHECK(tensors);
        CHECK_EQ(shapeSize, tensors->shape.size());
        CHECK(std::equal(shape.begin(), shape.end(), tensors->shape.begin(), tensors->shape.end()));
        CHECK_EQ(tensorType, tensors->type);
        CHECK_EQ(buffer, tensors->buffer);
        CHECK_EQ(name, tensors->name);
        CHECK(tensors->quantization);
        CHECK(std::equal(min.begin(), min.end(), tensors->quantization.get()->min.begin(),
                                      tensors->quantization.get()->min.end()));
        CHECK(std::equal(max.begin(), max.end(), tensors->quantization.get()->max.begin(),
                                      tensors->quantization.get()->max.end()));
        CHECK(std::equal(scale.begin(), scale.end(), tensors->quantization.get()->scale.begin(),
                                      tensors->quantization.get()->scale.end()));
        CHECK(std::equal(zeroPoint.begin(), zeroPoint.end(),
                                      tensors->quantization.get()->zero_point.begin(),
                                      tensors->quantization.get()->zero_point.end()));
    }

private:
    /// Fills the InputTensors with given input data
    template <armnn::DataType dataType>
    void FillInputTensors(armnn::InputTensors& inputTensors,
                          const std::map<std::string, std::vector<armnn::ResolveType<dataType>>>& inputData,
                          size_t subgraphId);
};

/// Fills the InputTensors with given input data
template <armnn::DataType dataType>
void ParserFlatbuffersFixture::FillInputTensors(
                  armnn::InputTensors& inputTensors,
                  const std::map<std::string, std::vector<armnn::ResolveType<dataType>>>& inputData,
                  size_t subgraphId)
{
    for (auto&& it : inputData)
    {
        armnn::BindingPointInfo bindingInfo = m_Parser->GetNetworkInputBindingInfo(subgraphId, it.first);
        bindingInfo.second.SetConstant(true);
        armnn::VerifyTensorInfoDataType(bindingInfo.second, dataType);
        inputTensors.push_back({ bindingInfo.first, armnn::ConstTensor(bindingInfo.second, it.second.data()) });
    }
}

/// Single Input, Single Output
/// Executes the network with the given input tensor and checks the result against the given output tensor.
/// This overload assumes the network has a single input and a single output.
template <std::size_t NumOutputDimensions,
          armnn::DataType armnnType>
void ParserFlatbuffersFixture::RunTest(size_t subgraphId,
                                       const std::vector<armnn::ResolveType<armnnType>>& inputData,
                                       const std::vector<armnn::ResolveType<armnnType>>& expectedOutputData)
{
    RunTest<NumOutputDimensions, armnnType>(subgraphId,
                                            { { m_SingleInputName, inputData } },
                                            { { m_SingleOutputName, expectedOutputData } });
}

/// Multiple Inputs, Multiple Outputs
/// Executes the network with the given input tensors and checks the results against the given output tensors.
/// This overload supports multiple inputs and multiple outputs, identified by name.
template <std::size_t NumOutputDimensions,
          armnn::DataType armnnType>
void ParserFlatbuffersFixture::RunTest(size_t subgraphId,
    const std::map<std::string, std::vector<armnn::ResolveType<armnnType>>>& inputData,
    const std::map<std::string, std::vector<armnn::ResolveType<armnnType>>>& expectedOutputData)
{
    RunTest<NumOutputDimensions, armnnType, armnnType>(subgraphId, inputData, expectedOutputData);
}

/// Multiple Inputs, Multiple Outputs w/ Variable Datatypes
/// Executes the network with the given input tensors and checks the results against the given output tensors.
/// This overload supports multiple inputs and multiple outputs, identified by name along with the allowance for
/// the input datatype to be different to the output
template <std::size_t NumOutputDimensions,
          armnn::DataType armnnType1,
          armnn::DataType armnnType2>
void ParserFlatbuffersFixture::RunTest(size_t subgraphId,
    const std::map<std::string, std::vector<armnn::ResolveType<armnnType1>>>& inputData,
    const std::map<std::string, std::vector<armnn::ResolveType<armnnType2>>>& expectedOutputData,
    bool isDynamic)
{
    using DataType2 = armnn::ResolveType<armnnType2>;

    // Setup the armnn input tensors from the given vectors.
    armnn::InputTensors inputTensors;
    FillInputTensors<armnnType1>(inputTensors, inputData, subgraphId);

    // Allocate storage for the output tensors to be written to and setup the armnn output tensors.
    std::map<std::string, std::vector<DataType2>> outputStorage;
    armnn::OutputTensors outputTensors;
    for (auto&& it : expectedOutputData)
    {
        armnn::LayerBindingId outputBindingId = m_Parser->GetNetworkOutputBindingInfo(subgraphId, it.first).first;
        armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkIdentifier, outputBindingId);

        // Check that output tensors have correct number of dimensions (NumOutputDimensions specified in test)
        auto outputNumDimensions = outputTensorInfo.GetNumDimensions();
        CHECK_MESSAGE((outputNumDimensions == NumOutputDimensions),
            fmt::format("Number of dimensions expected {}, but got {} for output layer {}",
                        NumOutputDimensions,
                        outputNumDimensions,
                        it.first));

        armnn::VerifyTensorInfoDataType(outputTensorInfo, armnnType2);
        outputStorage.emplace(it.first, std::vector<DataType2>(outputTensorInfo.GetNumElements()));
        outputTensors.push_back(
                { outputBindingId, armnn::Tensor(outputTensorInfo, outputStorage.at(it.first).data()) });
    }

    m_Runtime->EnqueueWorkload(m_NetworkIdentifier, inputTensors, outputTensors);

    // Set flag so that the correct comparison function is called if the output is boolean.
    bool isBoolean = armnnType2 == armnn::DataType::Boolean ? true : false;

    // Compare each output tensor to the expected values
    for (auto&& it : expectedOutputData)
    {
        armnn::BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(subgraphId, it.first);
        auto outputExpected = it.second;
        auto result = CompareTensors(outputExpected, outputStorage[it.first],
                                     bindingInfo.second.GetShape(), bindingInfo.second.GetShape(),
                                     isBoolean, isDynamic);
        CHECK_MESSAGE(result.m_Result, result.m_Message.str());
    }

    if (isDynamic)
    {
        m_Runtime->EnqueueWorkload(m_DynamicNetworkIdentifier, inputTensors, outputTensors);

        // Compare each output tensor to the expected values
        for (auto&& it : expectedOutputData)
        {
            armnn::BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(subgraphId, it.first);
            auto outputExpected = it.second;
            auto result = CompareTensors(outputExpected, outputStorage[it.first],
                                         bindingInfo.second.GetShape(), bindingInfo.second.GetShape(),
                                         false, isDynamic);
            CHECK_MESSAGE(result.m_Result, result.m_Message.str());
        }
    }
}

/// Multiple Inputs, Multiple Outputs w/ Variable Datatypes and different dimension sizes.
/// Executes the network with the given input tensors and checks the results against the given output tensors.
/// This overload supports multiple inputs and multiple outputs, identified by name along with the allowance for
/// the input datatype to be different to the output.
template <armnn::DataType armnnType1,
          armnn::DataType armnnType2>
void ParserFlatbuffersFixture::RunTest(std::size_t subgraphId,
    const std::map<std::string, std::vector<armnn::ResolveType<armnnType1>>>& inputData,
    const std::map<std::string, std::vector<armnn::ResolveType<armnnType2>>>& expectedOutputData)
{
    using DataType2 = armnn::ResolveType<armnnType2>;

    // Setup the armnn input tensors from the given vectors.
    armnn::InputTensors inputTensors;
    FillInputTensors<armnnType1>(inputTensors, inputData, subgraphId);

    armnn::OutputTensors outputTensors;
    outputTensors.reserve(expectedOutputData.size());
    std::map<std::string, std::vector<DataType2>> outputStorage;
    for (auto&& it : expectedOutputData)
    {
        armnn::BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(subgraphId, it.first);
        armnn::VerifyTensorInfoDataType(bindingInfo.second, armnnType2);

        std::vector<DataType2> out(it.second.size());
        outputStorage.emplace(it.first, out);
        outputTensors.push_back({ bindingInfo.first,
                                  armnn::Tensor(bindingInfo.second,
                                  outputStorage.at(it.first).data()) });
    }

    m_Runtime->EnqueueWorkload(m_NetworkIdentifier, inputTensors, outputTensors);

    // Checks the results.
    for (auto&& it : expectedOutputData)
    {
        std::vector<armnn::ResolveType<armnnType2>> out = outputStorage.at(it.first);
        {
            for (unsigned int i = 0; i < out.size(); ++i)
            {
                CHECK(doctest::Approx(it.second[i]).epsilon(0.000001f) == out[i]);
            }
        }
    }
}

/// Multiple Inputs with different DataTypes, Multiple Outputs w/ Variable DataTypes
/// Executes the network with the given input tensors and checks the results against the given output tensors.
/// This overload supports multiple inputs and multiple outputs, identified by name along with the allowance for
/// the input datatype to be different to the output
template <std::size_t NumOutputDimensions,
          armnn::DataType inputType1,
          armnn::DataType inputType2,
          armnn::DataType outputType>
void ParserFlatbuffersFixture::RunTest(size_t subgraphId,
    const std::map<std::string, std::vector<armnn::ResolveType<inputType1>>>& input1Data,
    const std::map<std::string, std::vector<armnn::ResolveType<inputType2>>>& input2Data,
    const std::map<std::string, std::vector<armnn::ResolveType<outputType>>>& expectedOutputData)
{
    using DataType2 = armnn::ResolveType<outputType>;

    // Setup the armnn input tensors from the given vectors.
    armnn::InputTensors inputTensors;
    FillInputTensors<inputType1>(inputTensors, input1Data, subgraphId);
    FillInputTensors<inputType2>(inputTensors, input2Data, subgraphId);

    // Allocate storage for the output tensors to be written to and setup the armnn output tensors.
    std::map<std::string, std::vector<DataType2>> outputStorage;
    armnn::OutputTensors outputTensors;
    for (auto&& it : expectedOutputData)
    {
        armnn::LayerBindingId outputBindingId = m_Parser->GetNetworkOutputBindingInfo(subgraphId, it.first).first;
        armnn::TensorInfo outputTensorInfo = m_Runtime->GetOutputTensorInfo(m_NetworkIdentifier, outputBindingId);

        // Check that output tensors have correct number of dimensions (NumOutputDimensions specified in test)
        auto outputNumDimensions = outputTensorInfo.GetNumDimensions();
        CHECK_MESSAGE((outputNumDimensions == NumOutputDimensions),
            fmt::format("Number of dimensions expected {}, but got {} for output layer {}",
                        NumOutputDimensions,
                        outputNumDimensions,
                        it.first));

        armnn::VerifyTensorInfoDataType(outputTensorInfo, outputType);
        outputStorage.emplace(it.first, std::vector<DataType2>(outputTensorInfo.GetNumElements()));
        outputTensors.push_back(
                { outputBindingId, armnn::Tensor(outputTensorInfo, outputStorage.at(it.first).data()) });
    }

    m_Runtime->EnqueueWorkload(m_NetworkIdentifier, inputTensors, outputTensors);

    // Set flag so that the correct comparison function is called if the output is boolean.
    bool isBoolean = outputType == armnn::DataType::Boolean ? true : false;

    // Compare each output tensor to the expected values
    for (auto&& it : expectedOutputData)
    {
        armnn::BindingPointInfo bindingInfo = m_Parser->GetNetworkOutputBindingInfo(subgraphId, it.first);
        auto outputExpected = it.second;
        auto result = CompareTensors(outputExpected, outputStorage[it.first],
                                     bindingInfo.second.GetShape(), bindingInfo.second.GetShape(),
                                     isBoolean);
        CHECK_MESSAGE(result.m_Result, result.m_Message.str());
    }
}