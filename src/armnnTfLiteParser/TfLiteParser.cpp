//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TfLiteParser.hpp"

#include <armnn/BackendOptions.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/Logging.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

// armnnUtils:
#include <armnnUtils/Permute.hpp>
#include <Filesystem.hpp>

#include <ParserHelper.hpp>
#include <VerificationHelpers.hpp>

// The generated code based on the Tf Lite schema:
#include <schema_generated.h>

#include <flatbuffers/flexbuffers.h>

#include <fmt/format.h>

#include <fstream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>

#define ARMNN_THROW_PARSE_EXCEPTION(msg) \
          { \
            throw armnn::ParseException( static_cast<const std::stringstream&>( std::stringstream() << msg \
               << ": " \
               << CHECK_LOCATION().AsString()).str()); \
          }

using namespace armnn;
using armnn::CheckLocation;
namespace armnnTfLiteParser
{
namespace
{

const uint32_t VIRTUAL_OPERATOR_ID = std::numeric_limits<uint32_t>::max();

void CheckSubgraph(const TfLiteParser::ModelPtr & model,
                   size_t subgraphIndex,
                   const CheckLocation & location)
{
    if (model.get() == nullptr)
    {
        throw ParseException(
            fmt::format("{} was called with invalid (null) model. "
                        "Possible reason is that the model is not yet loaded and Unpack(ed). "
                        "subgraph:{} at {}",
                        location.m_Function,
                        subgraphIndex,
                        location.FileLine()));
    }
    else if (subgraphIndex >= model->subgraphs.size())
    {
        throw ParseException(
            fmt::format("{} was called with an invalid subgraph index. "
                        "subgraph:{} at {}",
                        location.m_Function,
                        subgraphIndex,
                        location.FileLine()));
    }
}

#define CHECK_SUBGRAPH(MODEL, SUBGRAPH_INDEX) \
    CheckSubgraph(MODEL, SUBGRAPH_INDEX, CHECK_LOCATION())

void CheckModel(const TfLiteParser::ModelPtr & model,
                size_t subgraphIndex,
                size_t operatorIndex,
                const CheckLocation & location)
{
    if (model.get() == nullptr)
    {
        throw ParseException(
            fmt::format("{} was called with invalid (null) model. "
                        "Possible reason is that the model is not yet loaded and Unpack(ed). "
                        "subgraph:{} operator:{} at {}",
                        location.m_Function,
                        subgraphIndex,
                        operatorIndex,
                        location.FileLine()));
    }
    else if (subgraphIndex >= model->subgraphs.size())
    {
        throw ParseException(
            fmt::format("{} was called with an invalid subgraph index. "
                        "subgraph:{} operator:{} at {}",
                        location.m_Function,
                        subgraphIndex,
                        operatorIndex,
                        location.FileLine()));
    }
    else if (operatorIndex >= model->subgraphs[subgraphIndex]->operators.size() &&
             operatorIndex != VIRTUAL_OPERATOR_ID)
    {
        throw ParseException(
            fmt::format("{} was called with an invalid operator index. "
                        "subgraph:{} operator:{} at {}",
                        location.m_Function,
                        subgraphIndex,
                        operatorIndex,
                        location.FileLine()));
    }
}

#define CHECK_MODEL(MODEL, SUBGRAPH_INDEX, OPERATOR_INDEX) \
    CheckModel(MODEL, SUBGRAPH_INDEX, OPERATOR_INDEX, CHECK_LOCATION())

void CheckTensor(const TfLiteParser::ModelPtr & model,
                 size_t subgraphIndex,
                 size_t tensorIndex,
                 const CheckLocation & location)
{
    // not checking model, because I assume CHECK_MODEL already run
    // and checked that. An assert would do.
    ARMNN_ASSERT_MSG(model.get() != nullptr, "Expecting a valid model in this function");

    // also subgraph index should be checked by CHECK_MODEL so
    // I only add an assert here
    ARMNN_ASSERT_MSG(subgraphIndex < model->subgraphs.size(), "Expecting a valid subgraph index");

    // the tensor index is the only one to check here
    if (tensorIndex >= model->subgraphs[subgraphIndex]->tensors.size())
    {
        throw ParseException(
            fmt::format("{} was called with an invalid tensor index. "
                        "subgraph:{} tensor:{} at {}",
                        location.m_Function,
                        subgraphIndex,
                        tensorIndex,
                        location.FileLine()));
    }
}

#define CHECK_TENSOR(MODEL, SUBGRAPH_INDEX, TENSOR_INDEX) \
    CheckTensor(MODEL, SUBGRAPH_INDEX, TENSOR_INDEX, CHECK_LOCATION())

void CheckTensorPtr(TfLiteParser::TensorRawPtr rawPtr,
                    const CheckLocation & location)
{
    if (rawPtr == nullptr)
    {
        throw ParseException(
            fmt::format("{} was called with a null tensor pointer at {}", location.m_Function, location.FileLine()));
    }
}

#define CHECK_TENSOR_PTR(TENSOR_PTR) \
    CheckTensorPtr(TENSOR_PTR, CHECK_LOCATION())

void CheckBuffer(const TfLiteParser::ModelPtr & model,
                 size_t bufferIndex,
                 const CheckLocation & location)
{
    if (model.get() == nullptr)
    {
        throw ParseException(
            fmt::format("{} was called with invalid (null) model. "
                        "Possible reason is that the model is not yet loaded and Unpack(ed). "
                        "buffer:{} at {}",
                        location.m_Function,
                        bufferIndex,
                        location.FileLine()));
    }
    else if (bufferIndex >= model->buffers.size())
    {
        throw ParseException(
            fmt::format("{} was called with an invalid buffer index. "
                        "buffer index:{} at {}",
                        location.m_Function,
                        bufferIndex,
                        location.FileLine()));
    }
    else if (model->buffers[bufferIndex].get() == nullptr)
    {
        throw ParseException(
            fmt::format("The buffer #{} is null. {}",
                        bufferIndex,
                        location.AsString()));
    }
}

#define CHECK_BUFFER(MODEL, BUFFER_INDEX) \
    CheckBuffer(MODEL, BUFFER_INDEX, CHECK_LOCATION())

void CheckBufferSize(TfLiteParser::BufferRawPtr bufferPtr,
                     const armnn::TensorInfo & tensorInfo,
                     uint32_t bufferId,
                     const CheckLocation & location)
{
    if (bufferPtr == nullptr)
    {
        throw ParseException(
            fmt::format("BufferPtr is null for buffer:{}. {}",
                        bufferId,
                        location.AsString()));
    }
    else if(tensorInfo.GetNumElements() > bufferPtr->data.size() ||
            tensorInfo.GetNumBytes() > bufferPtr->data.size())
    {
        std::stringstream ss;
        ss << "Buffer #" << bufferId << " has " << bufferPtr->data.size() << " bytes. "
           << "For tensor: " << tensorInfo.GetShape()
           << " expecting: " << tensorInfo.GetNumBytes() << " bytes and "
           << tensorInfo.GetNumElements() << " elements. " << location.AsString();
        throw ParseException(ss.str());
    }
}

#define CHECK_BUFFER_SIZE(BUFFER_PTR, TENSOR_INFO, BUFFER_ID) \
    CheckBufferSize(BUFFER_PTR, TENSOR_INFO, BUFFER_ID, CHECK_LOCATION())

bool IsActivationSupported(tflite::ActivationFunctionType activationType)
{
    switch(activationType)
    {
        case tflite::ActivationFunctionType_NONE:
        case tflite::ActivationFunctionType_RELU:
        case tflite::ActivationFunctionType_RELU6:
        case tflite::ActivationFunctionType_TANH:
        {
            return true;
        }
        default:
        {
            return false;
        }
    }
}

#define CHECK_SUPPORTED_FUSED_ACTIVATION(OPTION, SUBGRAPH_INDEX, OPERATOR_INDEX) \
    do { \
        if (IsActivationSupported(OPTION->fused_activation_function) == false) \
        { \
            throw ParseException( \
                fmt::format("TfLite parser doesn't suppport fused activation: " \
                            "{}/{} in {} subgraph:{} operator:{} at {}", \
                            OPTION->fused_activation_function, \
                            tflite::EnumNameActivationFunctionType(\
                            OPTION->fused_activation_function), \
                            __func__, \
                            SUBGRAPH_INDEX, \
                            OPERATOR_INDEX, \
                            CHECK_LOCATION().FileLine())); \
        } \
    } while(false)


std::vector<unsigned int> AsUnsignedVector(const std::vector<int32_t> & in)
{
    std::vector<unsigned int> result;
    result.reserve(in.size());
    for (auto & i : in)
    {
        result.push_back(CHECKED_NON_NEGATIVE(i));
    }
    return result;
}

void CalcPadding(uint32_t inputSize,
                 uint32_t filterSize,
                 uint32_t stride,
                 uint32_t dilation,
                 uint32_t& paddingFront,
                 uint32_t& paddingBack,
                 tflite::Padding padding)
{
    paddingFront = 0;
    paddingBack = 0;
    if (padding == tflite::Padding_SAME)
    {
        uint32_t outputSize = (inputSize + stride - 1) / stride;
        uint32_t dilatedSize = filterSize + (dilation - 1) * (filterSize - 1);
        uint32_t temp = (outputSize - 1) * stride + dilatedSize;
        if (temp > inputSize)
        {
            paddingFront = (temp - inputSize) / 2;
            paddingBack = (temp - inputSize) - paddingFront;
        }
    }
}

armnn::TensorInfo ToTensorInfo(TfLiteParser::TensorRawPtr tensorPtr,
                               const std::vector<unsigned int>& shapes,
                               const armnn::PermutationVector& dimensionMappings = {0, 1, 2, 3},
                               const bool outputTensor = false)
{
    armnn::DataType type;
    CHECK_TENSOR_PTR(tensorPtr);

    switch (tensorPtr->type)
    {
        case tflite::TensorType_UINT8:
            type = armnn::DataType::QAsymmU8;
            break;
        case tflite::TensorType_FLOAT32:
            type = armnn::DataType::Float32;
            break;
        case tflite::TensorType_INT8:
            if (tensorPtr->quantization->zero_point.size() == 1)
            {
                // Per-tensor
                type = armnn::DataType::QAsymmS8;
            }
            else
            {
                // Per-channel
                type = armnn::DataType::QSymmS8;
            }
            break;
        case tflite::TensorType_INT16:
            type = armnn::DataType::QSymmS16;
            break;
        case tflite::TensorType_INT32:
            type = armnn::DataType::Signed32;
            break;
        case tflite::TensorType_INT64:
            type = armnn::DataType::Signed64;
            break;
        default:
        {
            CheckLocation location = CHECK_LOCATION();
            throw ParseException(
                fmt::format("Unsupported data type {} = {} for tensor: {}. {}",
                            tensorPtr->type,
                            tflite::EnumNameTensorType(tensorPtr->type),
                            tensorPtr->name,
                            location.AsString()));
        }
    }
    std::vector<unsigned int> safeShape = shapes;
    bool isDynamic = false;
    if (safeShape.size() == 0)
    {
        safeShape.push_back(1);
        if (outputTensor)
        {
            isDynamic = true;
        }
    }

    float quantizationScale = 0.0f;
    int32_t quantizationOffset = 0;

    if (tensorPtr->quantization.get())
    {
        if (tensorPtr->quantization->scale.size() <= 1)
        {
            CHECK_VALID_SIZE(tensorPtr->quantization->zero_point.size(), 0, 1);
            CHECK_VALID_SIZE(tensorPtr->quantization->zero_point.size(), 0, 1);

            if (tensorPtr->quantization->scale.size() == 1)
            {
                quantizationScale = tensorPtr->quantization->scale[0];
            }
            if (tensorPtr->quantization->zero_point.size() == 1)
            {
                // NOTE: we lose precision here when converting from 64 bit to 32
                //       but this is what we support at the moment in ArmNN
                quantizationOffset = armnn::numeric_cast<int32_t>(tensorPtr->quantization->zero_point[0]);
            }

            TensorShape tensorShape(armnn::numeric_cast<unsigned int>(safeShape.size()),
                                    safeShape.data());
            if (isDynamic)
            {
                tensorShape = TensorShape(1, false);
            }
            armnn::TensorInfo result(tensorShape,
                                     type,
                                     quantizationScale,
                                     quantizationOffset);
            return result;
        }
        else
        {
            std::vector<float> quantizationScales;
            std::vector<int32_t> quantizationOffsets;

            // Scale
            std::copy(tensorPtr->quantization->scale.begin(),
                      tensorPtr->quantization->scale.end(),
                      std::back_inserter(quantizationScales));

            // QSymmS8 Per-axis
            TensorShape tensorShape(armnn::numeric_cast<unsigned int>(safeShape.size()),
                                    safeShape.data());
            if (isDynamic)
            {
                tensorShape = TensorShape(1, false);
            }
            armnn::TensorInfo result(tensorShape,
                                     type,
                                     quantizationScales,
                                     dimensionMappings[armnn::numeric_cast<unsigned int>(
                                         tensorPtr->quantization->quantized_dimension)]);
            return result;
        }
    }
    else
    {
        TensorShape tensorShape(armnn::numeric_cast<unsigned int>(safeShape.size()),
                                safeShape.data());
        if (isDynamic)
        {
            tensorShape = TensorShape(1, false);
        }
        armnn::TensorInfo result(tensorShape,
                                 type,
                                 quantizationScale,
                                 quantizationOffset);
        return result;
    }
}

armnn::TensorInfo ToTensorInfo(TfLiteParser::TensorRawPtr tensorPtr,
                               const armnn::PermutationVector& dimensionMappings = {0, 1, 2, 3})
{
    auto const & dimensions = AsUnsignedVector(tensorPtr->shape);
    return ToTensorInfo(tensorPtr, dimensions, dimensionMappings);
}

armnn::TensorInfo ToTensorInfo(TfLiteParser::TensorRawPtr tensorPtr,
                               const bool outputTensor)
{
    auto const & dimensions = AsUnsignedVector(tensorPtr->shape);
    const armnn::PermutationVector& dimensionMappings = {0, 1, 2, 3};
    return ToTensorInfo(tensorPtr, dimensions, dimensionMappings, outputTensor);
}

template<typename T>
std::pair<armnn::ConstTensor, std::unique_ptr<T[]>>
CreateConstTensorImpl(TfLiteParser::BufferRawPtr bufferPtr,
                      TfLiteParser::TensorRawPtr tensorPtr,
                      armnn::TensorInfo& tensorInfo,
                      armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    IgnoreUnused(tensorPtr);
    ARMNN_ASSERT_MSG(tensorPtr != nullptr, "tensorPtr is null");
    ARMNN_ASSERT_MSG(bufferPtr != nullptr,
        fmt::format("Buffer for buffer:{} is null", tensorPtr->buffer).c_str());

    std::unique_ptr<T[]> data(new T[tensorInfo.GetNumElements()]);

    if (permutationVector.has_value() && permutationVector.value().GetSize() > 0)
    {
        tensorInfo = armnnUtils::Permuted(tensorInfo, permutationVector.value());
        armnnUtils::Permute(tensorInfo.GetShape(), permutationVector.value(),
                            reinterpret_cast<const T*>(bufferPtr->data.data()), data.get(), sizeof(T));
    }
    else
    {
        ::memcpy(data.get(), bufferPtr->data.data(), tensorInfo.GetNumBytes());
    }

    return std::make_pair(ConstTensor(tensorInfo, data.get()), std::move(data));
}

armnn::LayerBindingId GenerateLayerBindingId(size_t subgraphIndex, size_t tensorIndex)
{
    // generate the binding id by shifting the tensor id by 8 bit
    // and add the subgraph id, which allows 256 subgraphs
    return static_cast<armnn::LayerBindingId>((tensorIndex<<8)+subgraphIndex);
}

bool CheckShape(const armnn::TensorShape& actual, const std::vector<int32_t>& expected)
{
    const unsigned int actualSize = actual.GetNumDimensions();
    if (actualSize != expected.size())
    {
        return false;
    }

    for (unsigned int i = 0u; i < actualSize; i++)
    {
        if (expected[i] < 0 ||
            actual[i] != static_cast<unsigned int>(expected[i]))
        {
            return false;
        }
    }

    return true;
}

void CheckMatchingQuantization(const TensorInfo& first,
                               const TensorInfo& second,
                               const std::string& descName,
                               std::string const& firstName,
                               std::string const& secondName)
{
    if (!first.IsQuantized() ||
        !second.IsQuantized())
    {
        // Not a quantized type, ignore the validation
        return;
    }

    DataType firstDataType  = first.GetDataType();
    DataType secondDataType = second.GetDataType();

    if (firstDataType != secondDataType)
    {
        throw InvalidArgumentException(descName + ": " + firstName + " and " + secondName +
                                       " must be of the same quantized type, " +
                                       firstName + " is " + GetDataTypeName(firstDataType) + ", " +
                                       secondName + " is " + GetDataTypeName(secondDataType));
    }

    if (!first.IsTypeSpaceMatch(second))
    {
        throw InvalidArgumentException(descName + ": " + firstName + " and " + secondName +
                                       " must have the same quantization space, " +
                                       firstName + " has offset " + std::to_string(first.GetQuantizationOffset()) +
                                       " and scale " + std::to_string(first.GetQuantizationScale()) + ", " +
                                       secondName + " has offset " + std::to_string(second.GetQuantizationOffset()) +
                                       " and scale " + std::to_string(second.GetQuantizationScale()));
    }
}

} // <anonymous>

TfLiteParser::TfLiteParser(const Optional<ITfLiteParser::TfLiteParserOptions>& options)
: m_Options(options)
, m_Network(nullptr, nullptr)
, m_ParserFunctions(tflite::BuiltinOperator_MAX+1, &TfLiteParser::ParseUnsupportedOperator)
{
    // register supported operators
    m_ParserFunctions[tflite::BuiltinOperator_ADD]                     = &TfLiteParser::ParseAdd;
    m_ParserFunctions[tflite::BuiltinOperator_AVERAGE_POOL_2D]         = &TfLiteParser::ParseAveragePool2D;
    m_ParserFunctions[tflite::BuiltinOperator_BATCH_TO_SPACE_ND]       = &TfLiteParser::ParseBatchToSpaceND;
    m_ParserFunctions[tflite::BuiltinOperator_CONCATENATION]           = &TfLiteParser::ParseConcatenation;
    m_ParserFunctions[tflite::BuiltinOperator_CONV_2D]                 = &TfLiteParser::ParseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_CUSTOM]                  = &TfLiteParser::ParseCustomOperator;
    m_ParserFunctions[tflite::BuiltinOperator_DEPTHWISE_CONV_2D]       = &TfLiteParser::ParseDepthwiseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_DEQUANTIZE]              = &TfLiteParser::ParseDequantize;
    m_ParserFunctions[tflite::BuiltinOperator_EXP]                     = &TfLiteParser::ParseExp;
    m_ParserFunctions[tflite::BuiltinOperator_FULLY_CONNECTED]         = &TfLiteParser::ParseFullyConnected;
    m_ParserFunctions[tflite::BuiltinOperator_HARD_SWISH]              = &TfLiteParser::ParseHardSwish;
    m_ParserFunctions[tflite::BuiltinOperator_LEAKY_RELU]              = &TfLiteParser::ParseLeakyRelu;
    m_ParserFunctions[tflite::BuiltinOperator_LOGISTIC]                = &TfLiteParser::ParseLogistic;
    m_ParserFunctions[tflite::BuiltinOperator_L2_NORMALIZATION]        = &TfLiteParser::ParseL2Normalization;
    m_ParserFunctions[tflite::BuiltinOperator_MAX_POOL_2D]             = &TfLiteParser::ParseMaxPool2D;
    m_ParserFunctions[tflite::BuiltinOperator_MAXIMUM]                 = &TfLiteParser::ParseMaximum;
    m_ParserFunctions[tflite::BuiltinOperator_MEAN]                    = &TfLiteParser::ParseMean;
    m_ParserFunctions[tflite::BuiltinOperator_MINIMUM]                 = &TfLiteParser::ParseMinimum;
    m_ParserFunctions[tflite::BuiltinOperator_MUL]                     = &TfLiteParser::ParseMul;
    m_ParserFunctions[tflite::BuiltinOperator_NEG]                     = &TfLiteParser::ParseNeg;
    m_ParserFunctions[tflite::BuiltinOperator_PACK]                    = &TfLiteParser::ParsePack;
    m_ParserFunctions[tflite::BuiltinOperator_PAD]                     = &TfLiteParser::ParsePad;
    m_ParserFunctions[tflite::BuiltinOperator_QUANTIZE]                = &TfLiteParser::ParseQuantize;
    m_ParserFunctions[tflite::BuiltinOperator_RELU]                    = &TfLiteParser::ParseRelu;
    m_ParserFunctions[tflite::BuiltinOperator_RELU6]                   = &TfLiteParser::ParseRelu6;
    m_ParserFunctions[tflite::BuiltinOperator_RESHAPE]                 = &TfLiteParser::ParseReshape;
    m_ParserFunctions[tflite::BuiltinOperator_RESIZE_BILINEAR]         = &TfLiteParser::ParseResizeBilinear;
    m_ParserFunctions[tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR] = &TfLiteParser::ParseResizeNearestNeighbor;
    m_ParserFunctions[tflite::BuiltinOperator_SLICE]                   = &TfLiteParser::ParseSlice;
    m_ParserFunctions[tflite::BuiltinOperator_SOFTMAX]                 = &TfLiteParser::ParseSoftmax;
    m_ParserFunctions[tflite::BuiltinOperator_SPACE_TO_BATCH_ND]       = &TfLiteParser::ParseSpaceToBatchND;
    m_ParserFunctions[tflite::BuiltinOperator_SPLIT]                   = &TfLiteParser::ParseSplit;
    m_ParserFunctions[tflite::BuiltinOperator_SPLIT_V]                 = &TfLiteParser::ParseSplitV;
    m_ParserFunctions[tflite::BuiltinOperator_SQUEEZE]                 = &TfLiteParser::ParseSqueeze;
    m_ParserFunctions[tflite::BuiltinOperator_STRIDED_SLICE]           = &TfLiteParser::ParseStridedSlice;
    m_ParserFunctions[tflite::BuiltinOperator_SUB]                     = &TfLiteParser::ParseSub;
    m_ParserFunctions[tflite::BuiltinOperator_TANH]                    = &TfLiteParser::ParseTanH;
    m_ParserFunctions[tflite::BuiltinOperator_TRANSPOSE]               = &TfLiteParser::ParseTranspose;
    m_ParserFunctions[tflite::BuiltinOperator_TRANSPOSE_CONV]          = &TfLiteParser::ParseTransposeConv;
    m_ParserFunctions[tflite::BuiltinOperator_UNPACK]                  = &TfLiteParser::ParseUnpack;
    m_ParserFunctions[tflite::BuiltinOperator_DIV]                     = &TfLiteParser::ParseDiv;
    m_ParserFunctions[tflite::BuiltinOperator_ARG_MAX]                 = &TfLiteParser::ParseArgMax;
    // register supported custom operators
    m_CustomParserFunctions["TFLite_Detection_PostProcess"]      = &TfLiteParser::ParseDetectionPostProcess;
}

void TfLiteParser::ResetParser()
{
    m_Network = armnn::INetworkPtr(nullptr, nullptr);
    m_Model = nullptr;
    m_SubgraphConnections.clear();
}

INetworkPtr TfLiteParser::CreateNetworkFromBinaryFile(const char* graphFile)
{
    ResetParser();
    m_Model = LoadModelFromFile(graphFile);
    return CreateNetworkFromModel();
}

INetworkPtr TfLiteParser::CreateNetworkFromBinary(const std::vector<uint8_t> & binaryContent)
{
    ResetParser();
    m_Model = LoadModelFromBinary(binaryContent.data(), binaryContent.size());
    return CreateNetworkFromModel();
}

INetworkPtr TfLiteParser::CreateNetworkFromModel()
{

    using NetworkOptions = std::vector<BackendOptions>;
    NetworkOptions networkOptions = {};
    if (m_Options && m_Options.value().m_InferAndValidate)
    {
        BackendOptions shapeInferenceMethodOption("ShapeInferenceMethod",
                                                  {
                                                      { "InferAndValidate", true }
                                                  });

        networkOptions.push_back(shapeInferenceMethodOption);
    }

    m_Network = INetwork::Create(networkOptions);
    ARMNN_ASSERT(m_Model.get() != nullptr);

    if (m_Model->subgraphs.size() != 1)
    {
        throw ParseException(
                fmt::format("Current TfLite parser only supports 1 subgraph. Current one has: {} {}",
                            m_Model->subgraphs.size(),
                            CHECK_LOCATION().AsString()));
    }

    size_t subgraphIndex = 0;
    size_t operatorIndex = 0;
    try
    {
        for (SubgraphPtr const& subgraph : m_Model->subgraphs)
        {
            m_SubgraphConnections.emplace_back(subgraph->tensors.size());
            for (OperatorPtr const& op : subgraph->operators)
            {
                auto const& opCodePtr = m_Model->operator_codes[op->opcode_index];
                auto builtinCode = opCodePtr->builtin_code;

                if (builtinCode > tflite::BuiltinOperator_MAX)
                {
                    throw ParseException(fmt::format("Operator code {} is out of range 0-{}. "
                                                     "subgraph:{} operator idx:{}. {}",
                                                     builtinCode, tflite::BuiltinOperator_MAX, subgraphIndex,
                                                     operatorIndex, CHECK_LOCATION().AsString()));
                }

                // lookup and call the parser function
                auto& parserFunction = m_ParserFunctions[builtinCode];
                (this->*parserFunction)(subgraphIndex, operatorIndex);
                ++operatorIndex;
            }

            SetupInputLayers(subgraphIndex);
            SetupOutputLayers(subgraphIndex);
            SetupConstantLayers(subgraphIndex);

            ++subgraphIndex;
            operatorIndex = 0;
        }
    }
    catch (const ParseException& e)
    {
        std::stringstream errorString;
        errorString << "Failed to parse operator #" << operatorIndex << " within subgraph #"
                    << subgraphIndex << " error: " << e.what();
        ARMNN_LOG(error) << errorString.str();
        std::stringstream errors;
        errors << errorString.str() << "\n";
        throw ParseException(errors.str());
    }

    // establish the connections from the layer outputs to the inputs of the subsequent layers
    for (subgraphIndex = 0; subgraphIndex < m_SubgraphConnections.size(); ++subgraphIndex)
    {
        for (size_t tensorIndex = 0; tensorIndex < m_SubgraphConnections[subgraphIndex].size(); ++tensorIndex)
        {
            if (m_SubgraphConnections[subgraphIndex][tensorIndex].outputSlot != nullptr)
            {
                for (size_t inputSlotIdx = 0;
                    inputSlotIdx < m_SubgraphConnections[subgraphIndex][tensorIndex].inputSlots.size();
                    ++inputSlotIdx)
                {
                    m_SubgraphConnections[subgraphIndex][tensorIndex].outputSlot->Connect(
                        *(m_SubgraphConnections[subgraphIndex][tensorIndex].inputSlots[inputSlotIdx]));
                }
            }
        }
    }

    return std::move(m_Network);
}

void TfLiteParser::RegisterProducerOfTensor(size_t subgraphIndex,
                                            size_t tensorIndex,
                                            armnn::IOutputSlot* slot)
{
    CHECK_TENSOR(m_Model, subgraphIndex, tensorIndex);
    ARMNN_ASSERT(m_SubgraphConnections.size() > subgraphIndex);
    ARMNN_ASSERT(m_SubgraphConnections[subgraphIndex].size() > tensorIndex);

    TensorSlots & tensorSlots = m_SubgraphConnections[subgraphIndex][tensorIndex];

    // assuming there is only one producer for that tensor
    if (tensorSlots.outputSlot != nullptr)
    {
        throw ParseException(fmt::format("Another layer has already registered itself as the producer of "
                                         "subgraph:{} tensor:{} {}",
                                         subgraphIndex,
                                         tensorIndex,
                                         CHECK_LOCATION().AsString()));
    }

    tensorSlots.outputSlot = slot;
}

void TfLiteParser::RegisterConsumerOfTensor(size_t subgraphIndex,
                                            size_t tensorIndex,
                                            armnn::IInputSlot* slot)
{
    CHECK_TENSOR(m_Model, subgraphIndex, tensorIndex);
    ARMNN_ASSERT(m_SubgraphConnections.size() > subgraphIndex);
    ARMNN_ASSERT(m_SubgraphConnections[subgraphIndex].size() > tensorIndex);

    TensorSlots & tensorSlots = m_SubgraphConnections[subgraphIndex][tensorIndex];
    tensorSlots.inputSlots.push_back(slot);
}

void TfLiteParser::ParseCustomOperator(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    // NOTE: By default we presume the custom operator is not supported
    auto customParserFunction = &TfLiteParser::ParseUnsupportedOperator;

    // Identify custom code defined for custom operator
    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto& customCode  = m_Model->operator_codes[operatorPtr->opcode_index]->custom_code;

    // Find parser function that correspondes to custom code (if any)
    auto iterator = m_CustomParserFunctions.find(customCode);
    if (iterator != m_CustomParserFunctions.end())
    {
        customParserFunction = iterator->second;
    }

    // Run parser function
    (this->*customParserFunction)(subgraphIndex, operatorIndex);
}

void TfLiteParser::ParseUnsupportedOperator(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];

    auto opcodeIndex = operatorPtr->opcode_index;
    auto opcode      = m_Model->operator_codes[opcodeIndex]->builtin_code;

    if (!m_Options || !m_Options.value().m_StandInLayerForUnsupported)
    {
        // Do not add StandInLayer, throw ParseException instead
        throw ParseException(
            fmt::format("Operator not supported. "
                        "subgraph:{} operator:{} "
                        "opcode_index:{} opcode:{} / {} {}",
                        subgraphIndex,
                        operatorIndex,
                        opcodeIndex,
                        opcode,
                        tflite::EnumNameBuiltinOperator(opcode),
                        CHECK_LOCATION().AsString()));
    }

    auto inputs  = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);

    const unsigned int numInputs  = armnn::numeric_cast<unsigned int>(inputs.size());
    const unsigned int numOutputs = armnn::numeric_cast<unsigned int>(outputs.size());

    StandInDescriptor descriptor(numInputs, numOutputs);
    auto layerName = fmt::format("StandIn:{}:{}:{}", subgraphIndex, operatorIndex, opcode);

    // Add a non-executable StandInLayer as a placeholder for any unsupported operator
    IConnectableLayer* layer = m_Network->AddStandInLayer(descriptor, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    for (unsigned int i = 0u; i < numOutputs; ++i)
    {
        layer->GetOutputSlot(i).SetTensorInfo(ToTensorInfo(outputs[i], true));
    }

    auto inputTensorIds  = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    auto outputTensorIds = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));

    RegisterInputSlots(subgraphIndex, operatorIndex, layer, inputTensorIds);
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIds);
}

void TfLiteParser::ParseConv2D(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsConv2DOptions();

    CHECK_SUPPORTED_FUSED_ACTIVATION(options, subgraphIndex, operatorIndex);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX = CHECKED_NON_NEGATIVE(options->stride_w);
    desc.m_StrideY = CHECKED_NON_NEGATIVE(options->stride_h);
    desc.m_DataLayout = armnn::DataLayout::NHWC;
    desc.m_DilationX = CHECKED_NON_NEGATIVE(options->dilation_w_factor);
    desc.m_DilationY = CHECKED_NON_NEGATIVE(options->dilation_h_factor);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2, 3);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo filterTensorInfo = ToTensorInfo(inputs[1]);

    // assuming input is NHWC
    unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    // assuming the filter is OHWI : Output, H, W, Input
    // which is essentially the same as NHWC
    unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    CalcPadding(inputHeight, filterHeight, desc.m_StrideY,
                desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, desc.m_StrideX,
                desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, options->padding);

    auto filterTensorAndData = CreateConstTensor(inputs[1],
                                                 filterTensorInfo,
                                                 armnn::Optional<armnn::PermutationVector&>());
    armnn::IConnectableLayer* layer = nullptr;

    auto layerName = fmt::format("Conv2D:{}:{}", subgraphIndex, operatorIndex);

    if (inputs.size() == 3)
    {
        desc.m_BiasEnabled = true;
        armnn::TensorInfo biasTensorInfo = ToTensorInfo(inputs[2]);
        auto biasTensorAndData = CreateConstTensor(inputs[2],
                                                   biasTensorInfo,
                                                   armnn::Optional<armnn::PermutationVector&>());
        layer = m_Network->AddConvolution2dLayer(desc,
                                                 filterTensorAndData.first,
                                                 Optional<ConstTensor>(biasTensorAndData.first),
                                                 layerName.c_str());
    }
    else
    {
        layer = m_Network->AddConvolution2dLayer(desc,
                                                 filterTensorAndData.first,
                                                 EmptyOptional(),
                                                 layerName.c_str());
    }

    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);
    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseDepthwiseConv2D(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsDepthwiseConv2DOptions();

    CHECK_SUPPORTED_FUSED_ACTIVATION(options, subgraphIndex, operatorIndex);

    DepthwiseConvolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX = CHECKED_NON_NEGATIVE(options->stride_w);
    desc.m_StrideY = CHECKED_NON_NEGATIVE(options->stride_h);
    desc.m_DataLayout = armnn::DataLayout::NHWC;
    CHECKED_NON_NEGATIVE(options->depth_multiplier);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2, 3);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);
    desc.m_DilationX = CHECKED_NON_NEGATIVE(options->dilation_w_factor);
    desc.m_DilationY = CHECKED_NON_NEGATIVE(options->dilation_h_factor);

    // Mappings from TensorflowLite filter tensors to the ArmNN filter tensors (ArmNN weights have to be [M, I, H, W])
    PermutationVector permutationVector{ 2, 3, 1, 0 }; // [H, W, I, M] -> [M, I, H, W]

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo filterTensorInfo = ToTensorInfo(inputs[1], permutationVector);

    // Assuming input is NHWC
    unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    // TensorflowLite weights come in the format [1, H, W, I * M]
    unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    // Reshape weights as [ H, W, I, M ]
    filterTensorInfo.SetShape({ filterHeight,
                                filterWidth,
                                inputTensorInfo.GetShape()[3],
                                filterTensorInfo.GetShape()[3] / inputTensorInfo.GetShape()[3] });

    CalcPadding(inputHeight, filterHeight, desc.m_StrideY,
                desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, desc.m_StrideX,
                desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, options->padding);

    auto filterTensorAndData = CreateConstTensor(inputs[1], filterTensorInfo, permutationVector);
    armnn::IConnectableLayer* layer = nullptr;
    auto layerName = fmt::format("DepthwiseConv2D:{}:{}", subgraphIndex, operatorIndex);

    if (inputs.size() == 3)
    {
        desc.m_BiasEnabled = true;
        TensorInfo biasTensorInfo = ToTensorInfo(inputs[2]);
        auto biasTensorAndData = CreateConstTensor(inputs[2],
                                                   biasTensorInfo,
                                                   armnn::Optional<armnn::PermutationVector&>());
        layer = m_Network->AddDepthwiseConvolution2dLayer(desc,
                                                          filterTensorAndData.first,
                                                          Optional<ConstTensor>(biasTensorAndData.first),
                                                          layerName.c_str());
    }
    else
    {
        layer = m_Network->AddDepthwiseConvolution2dLayer(desc,
                                                          filterTensorAndData.first,
                                                          EmptyOptional(),
                                                          layerName.c_str());
    }
    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);
    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseDequantize(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Dequantize:{}:{}", subgraphIndex, operatorIndex);

    IConnectableLayer* layer = m_Network->AddDequantizeLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParser::ParseExp(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Exp:{}:{}", subgraphIndex, operatorIndex);

    ElementwiseUnaryDescriptor desc;
    desc.m_Operation = UnaryOperation::Exp;
    IConnectableLayer* layer = m_Network->AddElementwiseUnaryLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParser::ParseTranspose(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1, 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Transpose:{}:{}", subgraphIndex, operatorIndex);
    TransposeDescriptor desc;

    if (inputs.size() == 2)
    {
        armnn::TensorInfo permuteTensorInfo = ToTensorInfo(inputs[1]);
        BufferRawPtr permuteBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);
        auto numPermVecElements = permuteTensorInfo.GetNumElements();
        std::vector<unsigned int> permuteShape(numPermVecElements);
        ::memcpy(permuteShape.data(), permuteBufferPtr->data.data(), permuteTensorInfo.GetNumBytes());
        PermutationVector permutationVector(permuteShape.data(), permuteTensorInfo.GetNumElements());

        desc = TransposeDescriptor(permutationVector);
    }

    TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* layer = m_Network->AddTransposeLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseTransposeConv(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsTransposeConvOptions();

    TransposeConvolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX = CHECKED_NON_NEGATIVE(options->stride_w);
    desc.m_StrideY = CHECKED_NON_NEGATIVE(options->stride_h);
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 3);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    if (inputs[0])
    {
        armnn::TensorInfo tensorInfo = ToTensorInfo(inputs[0]);
        std::vector<int> output_shape(tensorInfo.GetNumElements());
        if (tensorInfo.GetDataType() == DataType::Signed32)
        {
            ::memcpy(output_shape.data(), GetBuffer(m_Model, inputs[0]->buffer)->data.data(), tensorInfo.GetNumBytes());
        }
        if (tensorInfo.GetDataType() == DataType::QAsymmU8)
        {
            for(unsigned int i=0; i < tensorInfo.GetNumElements(); i++)
            {
                output_shape[i] = GetBuffer(m_Model, inputs[0]->buffer)->data.data()[i];
            }
        }
        // Change from signed to unsigned int to store in TransposeConvolution2dDescriptor.
        for (int dimension : output_shape)
        {
            desc.m_OutputShape.push_back(static_cast<unsigned int>(dimension));
        }
        desc.m_OutputShapeEnabled = true;
    }
    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[2]);
    armnn::TensorInfo filterTensorInfo = ToTensorInfo(inputs[1]);

    // TfLite uses NHWC tensors
    const unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    const unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    const unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    const unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    CalcPadding(inputHeight,
                filterHeight,
                desc.m_StrideY,
                1, // DilationY
                desc.m_PadTop,
                desc.m_PadBottom,
                options->padding);

    CalcPadding(inputWidth,
                filterWidth,
                desc.m_StrideX,
                1, // DilationX
                desc.m_PadLeft,
                desc.m_PadRight,
                options->padding);

    auto filterTensorAndData = CreateConstTensor(inputs[1],
                                                 filterTensorInfo,
                                                 armnn::Optional<armnn::PermutationVector&>());

    armnn::IConnectableLayer* layer = nullptr;
    auto layerName = fmt::format("TransposeConv:{}:{}", subgraphIndex, operatorIndex);

    layer = m_Network->AddTransposeConvolution2dLayer(desc,
                                                      filterTensorAndData.first,
                                                      EmptyOptional(),
                                                      layerName.c_str());

    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // only the tensors for the inputs are relevant, exclude the const (filter) tensor
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[2]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseAveragePool2D(size_t subgraphIndex, size_t operatorIndex)
{
    ParsePool(subgraphIndex, operatorIndex, PoolingAlgorithm::Average);
}

void TfLiteParser::ParseBatchToSpaceND(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 3);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo blockShapeTensorInfo = ToTensorInfo(inputs[1]);
    BufferRawPtr blockShapeBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);

    armnn::TensorInfo cropsTensorInfo = ToTensorInfo(inputs[2]);
    BufferRawPtr cropsBufferPtr = GetBuffer(m_Model, inputs[2]->buffer);

    std::vector<unsigned int> blockShape(blockShapeTensorInfo.GetNumElements());
    ::memcpy(blockShape.data(), blockShapeBufferPtr->data.data(), blockShapeTensorInfo.GetNumBytes());

    std::vector<unsigned int> cropsVector(cropsTensorInfo.GetNumElements());
    ::memcpy(cropsVector.data(), cropsBufferPtr->data.data(), cropsTensorInfo.GetNumBytes());

    size_t step = 2;
    std::vector<std::pair<unsigned int, unsigned int>> crops;
    for (unsigned int i = 0; i < cropsTensorInfo.GetNumElements() / step; ++i)
    {
        crops.emplace_back(cropsVector[i * step], cropsVector[i * step + 1]);
    }

    armnn::BatchToSpaceNdDescriptor desc;
    desc.m_BlockShape = blockShape;
    desc.m_Crops = crops;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    auto layerName = fmt::format("BatchToSpaceND:{}:{}", subgraphIndex, operatorIndex);

    TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* layer = m_Network->AddBatchToSpaceNdLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseL2Normalization(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;
    auto layerName = fmt::format("L2Normalization:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddL2NormalizationLayer(desc, layerName.c_str());

    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseMaxPool2D(size_t subgraphIndex, size_t operatorIndex)
{
    ParsePool(subgraphIndex, operatorIndex, PoolingAlgorithm::Max);
}

void TfLiteParser::ParseMaximum(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Maximum:{}:{}", subgraphIndex, operatorIndex);

    TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);
    CheckMatchingQuantization(inputTensorInfo, input1TensorInfo, layerName, "Input 0", "Input 1");

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* layer = m_Network->AddMaximumLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseMinimum(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Minimum:{}:{}", subgraphIndex, operatorIndex);

    TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);
    CheckMatchingQuantization(inputTensorInfo, input1TensorInfo, layerName, "Input 0", "Input 1");

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* layer = m_Network->AddMinimumLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParsePool(size_t subgraphIndex,
                             size_t operatorIndex,
                             PoolingAlgorithm algorithm)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsPool2DOptions();

    CHECK_SUPPORTED_FUSED_ACTIVATION(options, subgraphIndex, operatorIndex);

    std::string layerName;

    switch (algorithm)
    {
        case PoolingAlgorithm::Average:
            layerName =
                fmt::format("AveragePool2D:{}:{}", subgraphIndex, operatorIndex);
            break;
        case PoolingAlgorithm::Max:
            layerName =
                fmt::format("MaxPool2D:{}:{}", subgraphIndex, operatorIndex);
            break;
        default:
            ARMNN_ASSERT_MSG(false, "Unsupported Pooling Algorithm");
    }

    Pooling2dDescriptor desc;

    desc.m_PoolType = algorithm;
    desc.m_StrideX = CHECKED_NON_NEGATIVE(options->stride_w);
    desc.m_StrideY = CHECKED_NON_NEGATIVE(options->stride_h);
    desc.m_PoolWidth = CHECKED_NON_NEGATIVE(options->filter_width);
    desc.m_PoolHeight = CHECKED_NON_NEGATIVE(options->filter_height);
    desc.m_PaddingMethod = PaddingMethod::Exclude;
    desc.m_OutputShapeRounding = OutputShapeRounding::Floor;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);
    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);

    // assuming input is NHWC
    unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    CalcPadding(inputHeight, desc.m_PoolHeight, desc.m_StrideY, 1u,
                desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, desc.m_PoolWidth, desc.m_StrideX, 1u,
                desc.m_PadLeft, desc.m_PadRight, options->padding);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* layer = m_Network->AddPooling2dLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);
    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseSlice(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 3);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    SliceDescriptor desc;

    // set begin tensor info for slice descriptor
    armnn::TensorInfo beginTensorInfo = ToTensorInfo(inputs[1]);
    BufferRawPtr beginBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);

    std::vector<unsigned int> begin(beginTensorInfo.GetNumElements());
    ::memcpy(begin.data(), beginBufferPtr->data.data(), beginTensorInfo.GetNumBytes());

    // set size tensor info for slice descriptor
    armnn::TensorInfo sizeTensorInfo = ToTensorInfo(inputs[2]);
    BufferRawPtr sizeBufferPtr = GetBuffer(m_Model, inputs[2]->buffer);

    std::vector<unsigned int> size(sizeTensorInfo.GetNumElements());
    ::memcpy(size.data(), sizeBufferPtr->data.data(), sizeTensorInfo.GetNumBytes());
    desc = SliceDescriptor(begin, size);

    auto layerName = fmt::format("Slice:{}:{}", subgraphIndex, operatorIndex);

    TensorInfo inputTensorInfo = ToTensorInfo(inputs[0]);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* const layer = m_Network->AddSliceLayer(desc, layerName.c_str());
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseSoftmax(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsSoftmaxOptions();

    SoftmaxDescriptor desc;
    desc.m_Beta = options->beta;

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Softmax:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* const layer = m_Network->AddSoftmaxLayer(desc, layerName.c_str());

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseSpaceToBatchND(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 3);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo blockShapeTensorInfo = ToTensorInfo(inputs[1]);
    BufferRawPtr blockShapeBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);

    armnn::TensorInfo padListTensorInfo = ToTensorInfo(inputs[2]);
    BufferRawPtr padListBufferPtr = GetBuffer(m_Model, inputs[2]->buffer);

    std::vector<unsigned int> blockShape(blockShapeTensorInfo.GetNumElements());
    ::memcpy(blockShape.data(), blockShapeBufferPtr->data.data(), blockShapeTensorInfo.GetNumBytes());

    std::vector<unsigned int> padListVector(padListTensorInfo.GetNumElements());
    ::memcpy(padListVector.data(), padListBufferPtr->data.data(), padListTensorInfo.GetNumBytes());

    size_t step = 2;
    std::vector<std::pair<unsigned int, unsigned int>> padList;
    for (unsigned int i = 0; i < padListTensorInfo.GetNumElements() / step; ++i)
    {
        padList.emplace_back(padListVector[i * step], padListVector[i * step + 1]);
    }

    armnn::SpaceToBatchNdDescriptor desc;
    desc.m_BlockShape = blockShape;
    desc.m_PadList = padList;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    auto layerName = fmt::format("SpaceToBatchND:{}:{}", subgraphIndex, operatorIndex);

    TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* layer = m_Network->AddSpaceToBatchNdLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

armnn::TensorInfo TfLiteParser::OutputShapeOfSqueeze(const std::vector<uint32_t> & squeezeDimsIn,
                                                     const armnn::TensorInfo & inputTensorInfo)
{
    CHECK_VALID_SIZE(squeezeDimsIn.size(), 0, 1, 2, 3, 4);
    std::vector<uint32_t> squeezeDims = squeezeDimsIn;
    static const uint32_t dimensionSequence[] = { 0, 1, 2, 3 };

    if (inputTensorInfo.GetNumDimensions() > 4)
    {
        std::stringstream ss;
        ss << "Input tensor has unexpected number of dimensions:" << inputTensorInfo.GetNumDimensions()
           << " shape:" << inputTensorInfo.GetShape() << " "
           << CHECK_LOCATION().AsString();
        throw ParseException(ss.str());
    }

    if (squeezeDims.empty())
    {
        squeezeDims.assign(dimensionSequence,
                           dimensionSequence+inputTensorInfo.GetNumDimensions());
    }

    std::vector<uint32_t> outputDims;
    for(unsigned int i = 0; i < inputTensorInfo.GetNumDimensions(); i++)
    {
        bool skipSqueeze = (std::find(squeezeDims.begin(), squeezeDims.end(), i) == squeezeDims.end());
        auto currentDimension = inputTensorInfo.GetShape()[i];
        if (skipSqueeze || currentDimension != 1)
        {
            outputDims.push_back(currentDimension);
        }
    }

    if (outputDims.size() > 4)
    {
        std::stringstream ss;
        ss << "Output tensor has unexpected number of dimensions:" << inputTensorInfo.GetNumDimensions()
           << " shape:" << inputTensorInfo.GetShape() << " "
           << CHECK_LOCATION().AsString();
        throw ParseException(ss.str());
    }

    TensorShape outShape = TensorShape(static_cast<unsigned int>(outputDims.size()),
                                       outputDims.data());

    // we need to preserve the tensor type and the quantization data as well
    TensorInfo outTensorInfo = inputTensorInfo;
    outTensorInfo.SetShape(outShape);

    return outTensorInfo;
}

void TfLiteParser::ParseSqueeze(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsSqueezeOptions();
    auto layerName = fmt::format("Squeeze:{}:{}", subgraphIndex, operatorIndex);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo outputTensorInfo =
        TfLiteParser::OutputShapeOfSqueeze(AsUnsignedVector(options->squeeze_dims),
                                           inputTensorInfo);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputTensorInfo.GetShape();

    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseStridedSlice(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 4);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsStridedSliceOptions();

    StridedSliceDescriptor desc;
    desc.m_BeginMask = options->begin_mask;
    desc.m_EllipsisMask = options->ellipsis_mask;
    desc.m_EndMask = options->end_mask;
    desc.m_NewAxisMask = options->new_axis_mask;
    desc.m_ShrinkAxisMask = options->shrink_axis_mask;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::TensorInfo beginTensorInfo = ToTensorInfo(inputs[1]);
    BufferRawPtr beginBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);

    std::vector<int> begin(beginTensorInfo.GetNumElements());
    ::memcpy(begin.data(), beginBufferPtr->data.data(), beginTensorInfo.GetNumBytes());

    armnn::TensorInfo endTensorInfo = ToTensorInfo(inputs[2]);
    BufferRawPtr endBufferPtr = GetBuffer(m_Model, inputs[2]->buffer);

    std::vector<int> end(endTensorInfo.GetNumElements());
    ::memcpy(end.data(), endBufferPtr->data.data(), endTensorInfo.GetNumBytes());

    armnn::TensorInfo strideTensorInfo = ToTensorInfo(inputs[3]);
    BufferRawPtr strideBufferPtr = GetBuffer(m_Model, inputs[3]->buffer);

    std::vector<int> stride(strideTensorInfo.GetNumElements());
    ::memcpy(stride.data(), strideBufferPtr->data.data(), strideTensorInfo.GetNumBytes());

    desc.m_Begin = begin;
    desc.m_End = end;
    desc.m_Stride = stride;

    auto layerName = fmt::format("StridedSlice:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddStridedSliceLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseSub(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsSubOptions();

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);

    auto layerName = fmt::format("Sub:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddSubtractionLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});

    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseDiv(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsDivOptions();

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);

    auto layerName = fmt::format("Div:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddDivisionLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});
    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseAdd(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsAddOptions();

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);

    auto layerName = fmt::format("Add:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddAdditionLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});
    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseMul(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsMulOptions();

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);

    auto layerName = fmt::format("Mul:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddMultiplicationLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});
    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseMean(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo dimTensorInfo = ToTensorInfo(inputs[1]);
    BufferRawPtr bufferPtr = GetBuffer(m_Model, inputs[1]->buffer);

    armnn::MeanDescriptor desc;
    std::vector<unsigned int> axis(dimTensorInfo.GetNumElements());
    ::memcpy(axis.data(), bufferPtr->data.data(), dimTensorInfo.GetNumBytes());
    desc.m_Axis = axis;

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);

    desc.m_KeepDims =
        inputTensorInfo.GetNumDimensions() == outputTensorInfo.GetNumDimensions() ?
            true : false;

    auto layerName = fmt::format("Mean:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddMeanLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseNeg(size_t subgraphIndex, size_t operatorIndex)
{
  CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

  auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
  CHECK_VALID_SIZE(inputs.size(), 1);

  auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
  CHECK_VALID_SIZE(outputs.size(), 1);

  auto layerName = fmt::format("Neg:{}:{}", subgraphIndex, operatorIndex);
  armnn::ElementwiseUnaryDescriptor descriptor(armnn::UnaryOperation::Neg);
  IConnectableLayer* layer = m_Network->AddElementwiseUnaryLayer(descriptor, layerName.c_str());
  ARMNN_ASSERT(layer != nullptr);

  TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
  layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

  auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
  RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

  auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
  RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParser::ParsePad(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    TfLiteParser::TensorRawPtrVector inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);

    TfLiteParser::TensorRawPtrVector outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo padTensorInfo = ToTensorInfo(inputs[1]);
    BufferRawPtr bufferPtr = GetBuffer(m_Model, inputs[1]->buffer);

    std::vector<unsigned int> padBuffer(padTensorInfo.GetNumElements());
    ::memcpy(padBuffer.data(), bufferPtr->data.data(), padTensorInfo.GetNumBytes());

    size_t step = 2;
    armnn::PadDescriptor desc;
    for (unsigned int i = 0; i < padTensorInfo.GetNumElements() / step; ++i)
    {
        desc.m_PadList.emplace_back(padBuffer[i * step], padBuffer[i * step + 1]);
    }

    auto layerName = fmt::format("Pad:{}:{}", subgraphIndex, operatorIndex);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);

    IConnectableLayer* layer = m_Network->AddPadLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseQuantize(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Quantize:{}:{}", subgraphIndex, operatorIndex);

    IConnectableLayer* layer = m_Network->AddQuantizeLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParser::ParseRelu(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex, ActivationFunction::ReLu);
}

void TfLiteParser::ParseRelu6(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex, ActivationFunction::BoundedReLu);
}

void TfLiteParser::ParseLeakyRelu(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex, operatorIndex, ActivationFunction::LeakyReLu);
}

void TfLiteParser::ParseLogistic(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex,ActivationFunction::Sigmoid);
}

void TfLiteParser::ParseTanH(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex,ActivationFunction::TanH);
}

void TfLiteParser::ParseHardSwish(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex, operatorIndex, ActivationFunction::HardSwish);
}

void TfLiteParser::ParseActivation(size_t subgraphIndex, size_t operatorIndex, ActivationFunction activationType)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    IgnoreUnused(operatorPtr);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Activation:");
    ActivationDescriptor activationDesc;
    activationDesc.m_Function = activationType;

    switch (activationType)
    {
        case ActivationFunction::ReLu:
        {
            layerName += fmt::format("RELU:{}:{}", subgraphIndex, operatorIndex);
            break;
        }
        case ActivationFunction::BoundedReLu:
        {
            layerName += fmt::format("RELU6:{}:{}", subgraphIndex, operatorIndex);
            activationDesc.m_A = 6.0f;
            activationDesc.m_B = 0.0f;
            break;
        }
        case ActivationFunction::Sigmoid:
        {
            layerName += fmt::format("SIGMOID:{}:{}", subgraphIndex, operatorIndex);
            break;
        }
        case ActivationFunction::TanH:
        {
            layerName += fmt::format("TANH:{}:{}", subgraphIndex, operatorIndex);
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            break;
        }
        case ActivationFunction::LeakyReLu:
        {
            layerName += fmt::format("LEAKYRELU:{}:{}", subgraphIndex, operatorIndex);
            const auto * options = operatorPtr->builtin_options.AsLeakyReluOptions();
            activationDesc.m_A = options->alpha;
            break;
        }
        case ActivationFunction::HardSwish:
            layerName += fmt::format("HARDSWISH:{}:{}", subgraphIndex, operatorIndex);
            break;
        default:
        {
            throw ParseException(
                fmt::format("Unexpected ActivationFunction[{}] when creating layerName {} ",
                            static_cast<int>(activationType), CHECK_LOCATION().AsString()));
        }
    }

    IConnectableLayer* const layer = m_Network->AddActivationLayer(activationDesc, layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}
armnn::TensorInfo TfLiteParser::OutputShapeOfReshape(const armnn::TensorInfo & inputTensorInfo,
                                                     const std::vector<int32_t> & targetDimsIn)
{
    std::vector<unsigned int> outputDims(targetDimsIn.begin(), targetDimsIn.end());
    const auto stretchDim = std::find(targetDimsIn.begin(), targetDimsIn.end(), -1);

    if (stretchDim != targetDimsIn.end())
    {
        if (std::find(std::next(stretchDim), targetDimsIn.end(), -1) != targetDimsIn.end())
        {
            throw ParseException(
                fmt::format("At most one component of shape can be -1 {}", CHECK_LOCATION().AsString()));
        }

        auto targetNumElements =
            armnn::numeric_cast<unsigned int>(
                std::accumulate(targetDimsIn.begin(), targetDimsIn.end(), -1, std::multiplies<int32_t>()));

        auto stretchIndex = static_cast<size_t>(std::distance(targetDimsIn.begin(), stretchDim));
        outputDims[stretchIndex] = inputTensorInfo.GetNumElements() / targetNumElements;
    }

    TensorShape outputShape = TensorShape(static_cast<unsigned int>(outputDims.size()), outputDims.data());

    TensorInfo reshapeInfo = inputTensorInfo;
    reshapeInfo.SetShape(outputShape);

    return reshapeInfo;
}

void TfLiteParser::ParseReshape(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsReshapeOptions();
    auto layerName = fmt::format("Reshape:{}:{}", subgraphIndex, operatorIndex);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo actualOutputTensorInfo  = ToTensorInfo(outputs[0]);
    CheckMatchingQuantization(inputTensorInfo, actualOutputTensorInfo, layerName, "Input 0", "Output 0");

    // Extracting new shape for the output
    // There are two ways it can be passed
    //  * First is to define the target shape in the operator built-in options
    //  * Second is to pass it as a second input tensor
    std::vector<int32_t> targetShape;
    bool targetShapeFound = false;
    // Check if built-in options were given
    if (options != nullptr)
    {
        // make sure the parameter is given
        if (options->new_shape.empty() == false)
        {
            targetShape = options->new_shape;
            targetShapeFound = true;
        }
    }

    // If there is no built-in option given or if the built-in new_shape parameter was empty
    if (!targetShapeFound)
    {
        // Check for a second input tensor
        if (inputs.size() > 1 && inputs[1] != nullptr)
        {
            if (inputs[1]->is_variable)
            {
                ARMNN_THROW_PARSE_EXCEPTION( "Target shapes defined in non-const input tensors is not supported");
            }

            if (inputs[1]->shape.size() != 1)
            {
                ARMNN_THROW_PARSE_EXCEPTION("Target 'shape' input is not a 1D tensor");
            }

            if (inputs[1]->type != tflite::TensorType_INT32)
            {
                ARMNN_THROW_PARSE_EXCEPTION("Target 'shape' input is not an int32 type");
            }

            // Extract target shape from input
            auto bufferPtr = GetBuffer(m_Model, inputs[1]->buffer);
            auto values = reinterpret_cast<const int32_t*>(bufferPtr->data.data());
            for (int i=0; i < inputs[1]->shape[0]; ++i)
            {
                targetShape.push_back(values[i]);
            }
        }
        else
        {
            ARMNN_THROW_PARSE_EXCEPTION("Target shape not defined in reshape parameters or input tensor. "
                                        "At least one method required");
        }
    }

    armnn::TensorInfo reshapeOutputTensorInfo =
        TfLiteParser::OutputShapeOfReshape(inputTensorInfo, targetShape);

    // Check for valid input size and that reshape parameters equal output shape
    const armnn::TensorShape& reshapeOutputTensorShape = reshapeOutputTensorInfo.GetShape();
    if (inputs.size() > 1 && !CheckShape(reshapeOutputTensorShape, outputs[0]->shape))
    {
        std::stringstream ss;
        ss << "New shape defined in reshape parameters "
           << reshapeOutputTensorShape
           << " does not equal output shape "
           << actualOutputTensorInfo.GetShape()
           << ": "
           << CHECK_LOCATION().AsString();
        throw ParseException(ss.str());
    }

    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = reshapeOutputTensorInfo.GetShape();

    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(reshapeOutputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseResizeBilinear(size_t subgraphIndex, size_t operatorIndex)
{
    ParseResize(subgraphIndex, operatorIndex, ResizeMethod::Bilinear);
}

void TfLiteParser::ParseResizeNearestNeighbor(size_t subgraphIndex, size_t operatorIndex)
{
    ParseResize(subgraphIndex, operatorIndex, ResizeMethod::NearestNeighbor);
}

void TfLiteParser::ParseResize(size_t subgraphIndex, size_t operatorIndex, ResizeMethod resizeMethod)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo sizeTensorInfo = ToTensorInfo(inputs[1]);

    // Data for the parsed tensor args (size) must be stored locally.
    std::vector<int32_t> sizeTensorData(sizeTensorInfo.GetNumElements());

    BufferRawPtr sizeBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);
    ::memcpy(sizeTensorData.data(), sizeBufferPtr->data.data(), sizeTensorInfo.GetNumBytes());

    ResizeDescriptor desc;
    desc.m_Method       = resizeMethod;
    desc.m_TargetHeight = static_cast<uint32_t> (sizeTensorData[0]);
    desc.m_TargetWidth  = static_cast<uint32_t> (sizeTensorData[1]);
    desc.m_DataLayout   = armnn::DataLayout::NHWC;

    auto layerName = fmt::format("Resize:");

    switch (resizeMethod)
    {
        case ResizeMethod::Bilinear:
        {
            layerName += fmt::format("BILINEAR:{}:{}", subgraphIndex, operatorIndex);

            const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
            const auto * options     = operatorPtr->builtin_options.AsResizeBilinearOptions();

            desc.m_AlignCorners = options->align_corners;
            break;
        }
        case ResizeMethod::NearestNeighbor:
        {
            layerName += fmt::format("NEARESTNEIGHBOR:{}:{}", subgraphIndex, operatorIndex);
            break;
        }
        default:
        {
            throw ParseException(
                fmt::format("Unexpected ResizeMethod[{}] when creating layerName {} ",
                            static_cast<int>(resizeMethod), CHECK_LOCATION().AsString()));
        }
    }

    TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* layer = m_Network->AddResizeLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParser::ParseConcatenation(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsConcatenationOptions();

    CHECK_SUPPORTED_FUSED_ACTIVATION(options, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    unsigned int numConcatView = static_cast<unsigned int>(inputs.size());
    uint32_t inputRank = ToTensorInfo(inputs[0]).GetNumDimensions();

    const unsigned int concatDimInput = static_cast<unsigned int>(
        (static_cast<int>(inputRank) + options->axis) % static_cast<int>(inputRank));

    OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatView), inputRank);
    concatDescriptor.SetConcatAxis(concatDimInput);

    unsigned int mergeDimOrigin = 0;

    for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
    {
        TensorInfo inputTensorInfo = ToTensorInfo(inputs[viewIndex]);

        // This set up concatDescriptor view origin
        armnnUtils::ProcessConcatInputTensorInfo(
            inputTensorInfo, concatDescriptor, concatDimInput, viewIndex, mergeDimOrigin);
    }

    auto layerName = fmt::format("Concatenation:{}:{}", subgraphIndex, operatorIndex);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);

    IConnectableLayer* layer = m_Network->AddConcatLayer(concatDescriptor, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes});

    // add fused activation layer
    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseFullyConnected(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorRfr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto options = operatorRfr->builtin_options.AsFullyConnectedOptions();

    CHECK_SUPPORTED_FUSED_ACTIVATION(options, subgraphIndex, operatorIndex);

    FullyConnectedDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_TransposeWeightMatrix = true;

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo filterTensorInfo = ToTensorInfo(inputs[1]);

    // Fully Connected Layer accepts two dimensional weights input
    int32_t weightsDimension = static_cast<int32_t>(filterTensorInfo.GetNumDimensions());
    if (weightsDimension != 2)
    {
        throw ParseException(
            fmt::format("Dimension {} for Fully Connected weights is not supported by Armnn. "
                        "Node {}",
                        weightsDimension,
                        CHECK_LOCATION().AsString()));
    }

    auto filterTensorAndData = CreateConstTensor(inputs[1],
                                                 filterTensorInfo,
                                                 armnn::Optional<armnn::PermutationVector&>());
    armnn::IConnectableLayer* layer = nullptr;
    auto layerName = fmt::format("FullyConnected:{}:{}", subgraphIndex, operatorIndex);

    if (inputs.size() == 3)
    {
        desc.m_BiasEnabled = true;
        TensorInfo biasTensorInfo = ToTensorInfo(inputs[2]);
        auto biasTensorAndData = CreateConstTensor(inputs[2],
                                                   biasTensorInfo,
                                                   armnn::Optional<armnn::PermutationVector&>());
        layer = m_Network->AddFullyConnectedLayer(desc,
                                                  filterTensorAndData.first,
                                                  Optional<ConstTensor>(biasTensorAndData.first),
                                                  layerName.c_str());
    }
    else
    {
        layer = m_Network->AddFullyConnectedLayer(desc,
                                                  filterTensorAndData.first,
                                                  EmptyOptional(),
                                                  layerName.c_str());
    }
    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));

    if (inputTensorInfo.GetNumDimensions() > 2)
    {
        // Add reshape to flatten to 2D [batch_size, input_size],
        // where "input_size" corresponds to the number of inputs to the layer,
        // matching the second dimension of weights,
        // and "batch_size" is calculated by dividing the number of elements by "input_size".
        std::vector<unsigned int> reshapedDimensions(2);
        reshapedDimensions[1] = filterTensorInfo.GetShape()[1];
        reshapedDimensions[0] = inputTensorInfo.GetNumElements() / reshapedDimensions[1];

        if (inputTensorInfo.GetNumElements() % reshapedDimensions[1] != 0)
        {
            throw ParseException(
                    fmt::format("Failed to deduce input tensor shape from filter size {} {}",
                                reshapedDimensions[1],
                                CHECK_LOCATION().AsString()));
        }

        armnn::TensorInfo reshapedTensorInfo = ToTensorInfo(inputs[0]);
        reshapedTensorInfo.SetShape(armnn::TensorShape{ 2, reshapedDimensions.data() });

        std::string reshapeLayerName = fmt::format("Reshape_for:{}", layer->GetName());
        armnn::ReshapeDescriptor desc;
        desc.m_TargetShape = reshapedTensorInfo.GetShape();
        armnn::IConnectableLayer* reshapeLayer = m_Network->AddReshapeLayer(desc, layerName.c_str());

        reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedTensorInfo);
        reshapeLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

        RegisterInputSlots(subgraphIndex, operatorIndex, reshapeLayer, {inputTensorIndexes[0]});
    }
    else
    {
        // register the input connection slot for the layer
        // only the tensors for the inputs are relevant, exclude the const tensors
        RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});
    }

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // we need to add the activation layer and fortunately we don't need to care about the data layout
    armnn::IConnectableLayer* fusedActivationLayer = AddFusedActivationLayer(layer, 0,
                                                                             options->fused_activation_function);

    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, fusedActivationLayer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseDetectionPostProcess(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 4);

    // Obtain custom options from flexbuffers
    auto custom_options = operatorPtr->custom_options;
    const flexbuffers::Map& m = flexbuffers::GetRoot(custom_options.data(), custom_options.size()).AsMap();

    // Obtain descriptor information from tf lite
    DetectionPostProcessDescriptor desc;
    desc.m_MaxDetections           = m["max_detections"].AsUInt32();
    desc.m_MaxClassesPerDetection  = m["max_classes_per_detection"].AsUInt32();
    desc.m_NmsScoreThreshold       = m["nms_score_threshold"].AsFloat();
    desc.m_NmsIouThreshold         = m["nms_iou_threshold"].AsFloat();
    desc.m_NumClasses              = m["num_classes"].AsUInt32();
    desc.m_ScaleH                  = m["h_scale"].AsFloat();
    desc.m_ScaleW                  = m["w_scale"].AsFloat();
    desc.m_ScaleX                  = m["x_scale"].AsFloat();
    desc.m_ScaleY                  = m["y_scale"].AsFloat();

    if (!(m["use_regular_nms"].IsNull()))
    {
        desc.m_UseRegularNms       = m["use_regular_nms"].AsBool();
    }
    if (!(m["detections_per_class"].IsNull()))
    {
        desc.m_DetectionsPerClass  = m["detections_per_class"].AsUInt32();
    }

    if (desc.m_NmsIouThreshold <= 0.0f || desc.m_NmsIouThreshold > 1.0f)
    {
        throw InvalidArgumentException("DetectionPostProcessTFLiteParser: Intersection over union threshold "
                                       "must be positive and less than or equal to 1.");
    }

    armnn::TensorInfo anchorTensorInfo = ToTensorInfo(inputs[2]);
    auto anchorTensorAndData = CreateConstTensor(inputs[2], anchorTensorInfo,
                                                 armnn::Optional<armnn::PermutationVector&>());

    auto layerName = fmt::format("DetectionPostProcess:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddDetectionPostProcessLayer(desc, anchorTensorAndData.first,
                                                                       layerName.c_str());

    ARMNN_ASSERT(layer != nullptr);

    // The model does not specify the output shapes.
    // The output shapes are calculated from the max_detection and max_classes_per_detection.
    unsigned int numDetectedBox = desc.m_MaxDetections * desc.m_MaxClassesPerDetection;
    m_OverridenOutputShapes.push_back({ 1, numDetectedBox, 4 });
    m_OverridenOutputShapes.push_back({ 1, numDetectedBox });
    m_OverridenOutputShapes.push_back({ 1, numDetectedBox });
    m_OverridenOutputShapes.push_back({ 1 });

    for (unsigned int i = 0 ; i < outputs.size() ; ++i)
    {
        armnn::TensorInfo detectionBoxOutputTensorInfo = ToTensorInfo(outputs[i], m_OverridenOutputShapes[i]);
        layer->GetOutputSlot(i).SetTensorInfo(detectionBoxOutputTensorInfo);
    }

    // Register the input connection slots for the layer, connections are made after all layers have been created
    // only the tensors for the inputs are relevant, exclude the const tensors
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});

    // Register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0],
                                                              outputTensorIndexes[1],
                                                              outputTensorIndexes[2],
                                                              outputTensorIndexes[3]});
}

/// The TfLite Pack operator is equivalent to the ArmNN Stack operator
void TfLiteParser::ParsePack(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    if (inputs.size() < 1)
    {
        throw ParseException("Pack must have at least one input.");
    }

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsPackOptions();

    StackDescriptor desc;
    desc.m_Axis = static_cast<uint32_t>(options->axis);
    desc.m_NumInputs = static_cast<uint32_t>(inputs.size());

    // Use the tensor shape of the first input as the "correct" input shape in the descriptor
    armnn::TensorInfo inputTensorInfo = ToTensorInfo(inputs[0]);
    desc.m_InputShape = inputTensorInfo.GetShape();

    auto layerName = fmt::format("Pack:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddStackLayer(desc, layerName.c_str());

    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseUnpack(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsUnpackOptions();

    // This unpackAxis indicates the axis to unpack
    const unsigned int unpackAxis = CHECKED_NON_NEGATIVE(options->axis);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);

    if (unpackAxis >= inputTensorInfo.GetNumDimensions())
    {
        throw ParseException(
                fmt::format("The unpack axis: {} cannot be greater than or equal to "
                            "the number of input dimension {} {}",
                            unpackAxis,
                            inputTensorInfo.GetNumDimensions(),
                            CHECK_LOCATION().AsString()));
    }

    unsigned int unpackNum = CHECKED_NON_NEGATIVE(options->num);
    // If num is not defined, automatically infer from the length of the dimension axis.
    if(unpackNum == 0)
    {
        unpackNum = inputTensorInfo.GetShape()[unpackAxis];
    }

    // If unpack number cannot be inferred and is still zero, throw ParseException.
    if(unpackNum == 0)
    {
        throw ParseException("Number to unpack must greater than zero.");
    }

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), unpackNum);

    auto inputDimSize = inputTensorInfo.GetNumDimensions();
    std::vector<unsigned int> unpackDimSizes(inputDimSize);

    // Add current input shape to unpackDimSizes
    for (unsigned int i = 0; i < inputDimSize; ++i)
    {
        unpackDimSizes[i] = inputTensorInfo.GetShape()[i];
    }

    if (unpackDimSizes[unpackAxis] != unpackNum)
    {
        throw ParseException("Number to unpack must be the same as length of the dimension to "
                             "unpack along.");
    }

    unpackDimSizes[unpackAxis] /= unpackNum;

    SplitterDescriptor splitDesc(unpackNum, static_cast<unsigned int>(unpackDimSizes.size()));
    for (unsigned int j = 0; j < unpackNum; ++j)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < unpackDimSizes.size(); ++dimIdx)
        {
            splitDesc.SetViewSize(j, dimIdx, unpackDimSizes[dimIdx]);
        }
        splitDesc.SetViewOriginCoord(j, unpackAxis, unpackDimSizes[unpackAxis] * j);
    }

    auto layerName = fmt::format("Unpack:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddSplitterLayer(splitDesc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorShape splitOutShape = TensorShape(static_cast<unsigned int>(unpackDimSizes.size()),
                                            unpackDimSizes.data());

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // Create reshape to remove the unpacked dimension for unpack operator of each output from Splitter.
    for (unsigned int k = 0; k < layer->GetNumOutputSlots(); ++k)
    {
        armnn::TensorInfo outputTensorInfo  = ToTensorInfo(outputs[k], true);
        std::string reshapeLayerName = fmt::format("Reshape_for:{}", layer->GetName());
        armnn::ReshapeDescriptor desc;
        desc.m_TargetShape = outputTensorInfo.GetShape();
        armnn::IConnectableLayer* reshapeLayer = m_Network->AddReshapeLayer(desc, layerName.c_str());

        layer->GetOutputSlot(k).SetTensorInfo(armnn::TensorInfo(splitOutShape,
                                                                outputTensorInfo.GetDataType(),
                                                                outputTensorInfo.GetQuantizationScale(),
                                                                outputTensorInfo.GetQuantizationOffset()));
        layer->GetOutputSlot(k).Connect(reshapeLayer->GetInputSlot(0));

        reshapeLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        uint32_t reshapedOutputId = CHECKED_NON_NEGATIVE(operatorPtr->outputs[k]);
        armnn::IOutputSlot* slot = &(reshapeLayer->GetOutputSlot(0));
        RegisterProducerOfTensor(subgraphIndex, reshapedOutputId, slot);
    }
}

void TfLiteParser::ParseSplit(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsSplitOptions();

    const unsigned int numSplits = CHECKED_NON_NEGATIVE(options->num_splits);

    // If number of splits cannot be inferred and is zero, throw ParseException.
    if(numSplits == 0)
    {
        throw ParseException("Number to splits must greater than zero.");
    }

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), numSplits);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[1]);
    armnn::TensorInfo axisTensorInfo = ToTensorInfo(inputs[0]);

    BufferRawPtr axisBufferPtr = GetBuffer(m_Model, inputs[0]->buffer);
    std::vector<unsigned int> axisData(axisTensorInfo.GetNumElements());
    ::memcpy(axisData.data(), axisBufferPtr->data.data(), axisTensorInfo.GetNumBytes());

    ARMNN_ASSERT(axisTensorInfo.GetNumElements() == 1);
    const unsigned int splitDim = axisData[0];

    auto inputDimSize = inputTensorInfo.GetNumDimensions();
    if (inputDimSize > MaxNumOfTensorDimensions)
    {
        throw ParseException(
            fmt::format("The number of dimensions: {} for input tensors of the split op cannot be greater than {} {}",
                        inputTensorInfo.GetNumDimensions(),
                        MaxNumOfTensorDimensions,
                        CHECK_LOCATION().AsString()));
    }

    std::vector<unsigned int> splitterDimSizes(inputDimSize);

    // Add current input shape to splitterDimSizes
    for (unsigned int i = 0; i < inputDimSize; ++i)
    {
        splitterDimSizes[i] = inputTensorInfo.GetShape()[i];
    }

    if (splitterDimSizes[splitDim] % numSplits != 0)
    {
        throw ParseException("Number of splits must evenly divide the dimension");
    }
    splitterDimSizes[splitDim] /= numSplits;

    SplitterDescriptor splitDesc(numSplits, inputDimSize);
    for (unsigned int j = 0; j < numSplits; ++j)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < splitterDimSizes.size(); ++dimIdx)
        {
            splitDesc.SetViewSize(j, dimIdx, splitterDimSizes[dimIdx]);
        }
        splitDesc.SetViewOriginCoord(j, splitDim, splitterDimSizes[splitDim] * j);
    }

    auto layerName = fmt::format("Split:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddSplitterLayer(splitDesc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[1]});

    for (unsigned int k = 0; k < layer->GetNumOutputSlots(); ++k)
    {
        armnn::TensorInfo tensorInfo = ToTensorInfo(outputs[k], true);
        layer->GetOutputSlot(k).SetTensorInfo(tensorInfo);
    }

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

unsigned int ComputeWrappedIndex(int idx, unsigned int numDimsIn)
{
    int numDims = armnn::numeric_cast<int>(numDimsIn);
    int v = idx < 0 ? numDims + idx : idx;
    ARMNN_ASSERT(v >= 0);
    ARMNN_ASSERT(v < numDims);

    return static_cast<unsigned int>(v);
}

void TfLiteParser::ParseSplitV(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsSplitVOptions();

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 3);

    auto& inputTensor = inputs[0];
    auto& splitsTensor = inputs[1];
    auto& axisTensor = inputs[2];

    armnn::TensorInfo inputTensorInfo = ToTensorInfo(inputTensor);
    armnn::TensorInfo splitsInfo      = ToTensorInfo(splitsTensor);
    armnn::TensorInfo axisTensorInfo  = ToTensorInfo(axisTensor);
    ARMNN_ASSERT(axisTensorInfo.GetNumElements() == 1);

    // Inputs
    auto inputDimSize = inputTensorInfo.GetNumDimensions();
    if (inputDimSize > MaxNumOfTensorDimensions)
    {
        throw ParseException(
            fmt::format("The number of dimensions: {} for input tensors of the "
                        "SplitV op cannot be greater than {} {}",
                        inputTensorInfo.GetNumDimensions(),
                        MaxNumOfTensorDimensions,
                        CHECK_LOCATION().AsString()));
    }

    // Get split axis
    BufferRawPtr axisBufferPtr = GetBuffer(m_Model, axisTensor->buffer);
    std::vector<int> axisData(axisTensorInfo.GetNumElements());
    ::memcpy(axisData.data(), axisBufferPtr->data.data(), axisTensorInfo.GetNumBytes());
    const unsigned int splitDim = ComputeWrappedIndex(axisData[0], inputTensorInfo.GetNumDimensions());

    // Set split sizes
    CHECK_VALID_SIZE(splitsInfo.GetNumDimensions(), 1);
    unsigned int numSplits{0};

    if(options)
    {
        numSplits = CHECKED_NON_NEGATIVE(options->num_splits);
    }
    else
    {
        numSplits = splitsInfo.GetNumElements();
    }

    if (numSplits <=0)
    {
        throw ParseException("SplitV has invalid number of splits");
    }

    std::vector<int> splitsData(numSplits);
    BufferRawPtr splitsBufferPtr = GetBuffer(m_Model, splitsTensor->buffer);
    ::memcpy(splitsData.data(), splitsBufferPtr->data.data(), splitsInfo.GetNumBytes());

    unsigned int idx = 0;
    int numInferred{0};
    unsigned int inferIdx{0};
    int splitSum{0};
    for (auto split : splitsData)
    {
        if (split < 0)
        {
            numInferred++;
            inferIdx = idx;
        }
        else
        {
            splitSum += split;
        }
        idx++;
    }
    // Check for inferred Axis
    if (numInferred == 0)
    {
        if (splitSum != armnn::numeric_cast<int>(inputTensorInfo.GetShape()[splitDim]))
        {
            throw ParseException("SplitV split_sizes does not sum to the dimension of value along split_dim.");
        }
    }
    else if (numInferred == 1)
    {
        splitsData[inferIdx] = armnn::numeric_cast<int>(inputTensorInfo.GetShape()[splitDim]) - splitSum;
    }
    else
    {
        throw ParseException("Cannot infer split size for more than one split");
    }

    //Ouput size validation
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), numSplits);

    // Setup Armnn descriptor
    SplitterDescriptor splitDesc(numSplits, inputDimSize);
    unsigned int accumSplit = 0;
    for (unsigned int j = 0; j < numSplits; ++j)
    {
        unsigned int splitSize = armnn::numeric_cast<unsigned int>(splitsData[j]);

        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < inputTensorInfo.GetNumDimensions(); ++dimIdx)
        {
            unsigned int dimSize = inputTensorInfo.GetShape()[dimIdx];
            if (dimIdx == splitDim)
            {
                dimSize = splitSize;
            }
            splitDesc.SetViewSize(j, dimIdx, dimSize);
        }

        splitDesc.SetViewOriginCoord(j, splitDim, accumSplit);
        accumSplit += splitSize;
    }

    auto layerName = fmt::format("SplitV:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddSplitterLayer(splitDesc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    for (unsigned int k = 0; k < layer->GetNumOutputSlots(); ++k)
    {
        armnn::TensorInfo tensorInfo = ToTensorInfo(outputs[k], true);
        layer->GetOutputSlot(k).SetTensorInfo(tensorInfo);
    }

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParser::ParseArgMax(size_t subgraphIndex, size_t operatorIndex)
{
    const auto &operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto *options = operatorPtr->builtin_options.AsArgMaxOptions();

    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("ArgMax:{}:{}", subgraphIndex, operatorIndex);

    armnn::TensorInfo sizeTensorInfo0 = ToTensorInfo(inputs[0]);
    armnn::TensorInfo sizeTensorInfo1 = ToTensorInfo(inputs[1]);

    // Get const axis value from model and set it to descriptor.
    BufferRawPtr axisBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);

    ArgMinMaxDescriptor desc;
    desc.m_Axis = axisBufferPtr->data.data()[0];
    // If output_type is int32 then set Signed32 else Signed64. Default type is Signed64.
    desc.m_Output_Type = options->output_type == 3 ? armnn::DataType::Signed32 : armnn::DataType::Signed64;
    desc.m_Function = ArgMinMaxFunction::Max;

    // Register a ArgMax layer.
    IConnectableLayer *layer = m_Network->AddArgMinMaxLayer(desc, layerName.c_str());

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Register input tensor to the layer.
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // Register output tensor to the layer.
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

armnn::IConnectableLayer* TfLiteParser::AddFusedActivationLayer(armnn::IConnectableLayer* prevLayer,
                                                                unsigned int outputSlot,
                                                                tflite::ActivationFunctionType activationType)
{
    ActivationDescriptor activationDesc;
    std::string layerName = prevLayer->GetName();

    switch(activationType)
    {
        case tflite::ActivationFunctionType_NONE:
        {
            // this is a no-op: return previous layer
            return prevLayer;
        }
        case tflite::ActivationFunctionType_RELU:
        {
            activationDesc.m_Function = ActivationFunction::ReLu;
            layerName += ":RELU";
            break;
        }
        case tflite::ActivationFunctionType_RELU6:
        {
            activationDesc.m_Function = ActivationFunction::BoundedReLu;
            activationDesc.m_A = 6.0f;
            activationDesc.m_B = 0.0f;
            layerName += ":RELU6";
            break;
        }
        case tflite::ActivationFunctionType_TANH:
        {
            activationDesc.m_Function = ActivationFunction::TanH;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            layerName += ":TANH";
            break;
        }

        // I only put these here as a reminder what others we could support
        case tflite::ActivationFunctionType_RELU_N1_TO_1:
        case tflite::ActivationFunctionType_SIGN_BIT:
        default:
        {
            throw ParseException(
                fmt::format("TfLite parser doesn't suppport fused activation: "
                            "{}/{} {} ",
                            activationType,
                            tflite::EnumNameActivationFunctionType(activationType),
                            CHECK_LOCATION().AsString()));

        }
    }

    IConnectableLayer* activationLayer =
        m_Network->AddActivationLayer(activationDesc, layerName.c_str());

    auto & prevOutputSlot = prevLayer->GetOutputSlot(outputSlot);
    prevOutputSlot.Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).SetTensorInfo(prevOutputSlot.GetTensorInfo());
    return activationLayer;
}

TfLiteParser::ModelPtr TfLiteParser::LoadModelFromFile(const char * fileName)
{
    if (fileName == nullptr)
    {
        throw InvalidArgumentException(fmt::format("Invalid (null) file name {}",
                                       CHECK_LOCATION().AsString()));
    }
    std::error_code errorCode;
    fs::path pathToFile(fileName);
    if (!fs::exists(pathToFile, errorCode))
    {
        //fmt::format() could not be used here (format error)
        std::stringstream msg;
        msg << "Cannot find the file (" << fileName << ") errorCode: " << errorCode
            << " " << CHECK_LOCATION().AsString();

        throw FileNotFoundException(msg.str());
    }
    std::ifstream file(fileName, std::ios::binary);
    std::string fileContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return LoadModelFromBinary(reinterpret_cast<const uint8_t *>(fileContent.c_str()),
                               fileContent.size());
}

TfLiteParser::ModelPtr TfLiteParser::LoadModelFromBinary(const uint8_t * binaryContent, size_t len)
{
    if (binaryContent == nullptr)
     {
        throw InvalidArgumentException(fmt::format("Invalid (null) binary content {}",
                                       CHECK_LOCATION().AsString()));
     }
    flatbuffers::Verifier verifier(binaryContent, len);
    if (verifier.VerifyBuffer<tflite::Model>() == false)
    {
        throw ParseException(
            fmt::format("Buffer doesn't conform to the expected Tensorflow Lite "
                        "flatbuffers format. size:{} {}",
                        len,
                        CHECK_LOCATION().AsString()));
    }
    return tflite::UnPackModel(binaryContent);
}

TfLiteParser::TensorRawPtrVector TfLiteParser::GetInputs(const ModelPtr & model,
                                                         size_t subgraphIndex,
                                                         size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);

    const auto & subgraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subgraphPtr->operators[operatorIndex];

    size_t inputCount = operatorPtr->inputs.size();
    TensorRawPtrVector result(inputCount);
    for (size_t i=0; i<inputCount; ++i)
    {
        uint32_t inputId = CHECKED_NON_NEGATIVE(operatorPtr->inputs[i]);
        result[i] = subgraphPtr->tensors[inputId].get();
    }
    return result;
}

TfLiteParser::TensorRawPtrVector TfLiteParser::GetOutputs(const ModelPtr & model,
                                                          size_t subgraphIndex,
                                                          size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);

    const auto & subgraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subgraphPtr->operators[operatorIndex];

    size_t outputCount = operatorPtr->outputs.size();
    TensorRawPtrVector result(outputCount);
    for (size_t i=0; i<outputCount; ++i)
    {
        uint32_t outputId = CHECKED_NON_NEGATIVE(operatorPtr->outputs[i]);
        CHECK_TENSOR(model, subgraphIndex, outputId);
        result[i] = subgraphPtr->tensors[outputId].get();
    }
    return result;
}

TfLiteParser::TensorIdRawPtrVector TfLiteParser::GetSubgraphInputs(const ModelPtr & model,
                                                                   size_t subgraphIndex)
{
    CHECK_SUBGRAPH(model, subgraphIndex);
    const auto & subgraphPtr = model->subgraphs[subgraphIndex];

    size_t inputCount = subgraphPtr->inputs.size();
    TensorIdRawPtrVector result(inputCount);
    for (size_t i=0; i<inputCount; ++i)
    {
        uint32_t inputId = CHECKED_NON_NEGATIVE(subgraphPtr->inputs[i]);
        CHECK_TENSOR(model, subgraphIndex, inputId);
        result[i] = std::make_pair(inputId, subgraphPtr->tensors[inputId].get());
    }
    return result;
}

TfLiteParser::TensorIdRawPtrVector TfLiteParser::GetSubgraphOutputs(const ModelPtr & model,
                                                                    size_t subgraphIndex)
{
    CHECK_SUBGRAPH(model, subgraphIndex);
    const auto & subgraphPtr = model->subgraphs[subgraphIndex];

    size_t outputCount = subgraphPtr->outputs.size();
    TensorIdRawPtrVector result(outputCount);
    for (size_t i=0; i<outputCount; ++i)
    {
        uint32_t outputId = CHECKED_NON_NEGATIVE(subgraphPtr->outputs[i]);
        result[i] = std::make_pair(outputId, subgraphPtr->tensors[outputId].get());
    }
    return result;
}

std::vector<int32_t>& TfLiteParser::GetInputTensorIds(const ModelPtr& model,
                                                      size_t subgraphIndex,
                                                      size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);
    const auto & subgraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subgraphPtr->operators[operatorIndex];
    return operatorPtr->inputs;
}

std::vector<int32_t>& TfLiteParser::GetOutputTensorIds(const ModelPtr& model,
                                                       size_t subgraphIndex,
                                                       size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);
    const auto & subgraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subgraphPtr->operators[operatorIndex];
    return operatorPtr->outputs;
}

void TfLiteParser::RegisterInputSlots(size_t subgraphIndex,
                                      size_t operatorIndex,
                                      IConnectableLayer* layer,
                                      const std::vector<unsigned int>& tensorIndexes)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    ARMNN_ASSERT(layer != nullptr);
    if (tensorIndexes.size() != layer->GetNumInputSlots())
    {
        throw ParseException(
            fmt::format("The number of tensor inputs ({}) does not match the number expected ({})"
                        " for subgraph:{} operator index:{} {}",
                        tensorIndexes.size(),
                        layer->GetNumInputSlots(),
                        subgraphIndex,
                        operatorIndex,
                        CHECK_LOCATION().AsString()));
    }

    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumInputSlots(); ++slotIndex)
    {
        unsigned int tensorIndex = tensorIndexes[slotIndex];
        armnn::IInputSlot* slot = &(layer->GetInputSlot(slotIndex));
        RegisterConsumerOfTensor(subgraphIndex, tensorIndex, slot);
    }
}

void TfLiteParser::RegisterOutputSlots(size_t subgraphIndex,
                                       size_t operatorIndex,
                                       IConnectableLayer* layer,
                                       const std::vector<unsigned int>& tensorIndexes)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    ARMNN_ASSERT(layer != nullptr);
    if (tensorIndexes.size() != layer->GetNumOutputSlots())
    {
        throw ParseException(
            fmt::format("The number of tensor outputs ({}) does not match the number expected ({})"
                        " for subgraph:{} operator index:{} {}",
                        tensorIndexes.size(),
                        layer->GetNumOutputSlots(),
                        subgraphIndex,
                        operatorIndex,
                        CHECK_LOCATION().AsString()));
    }

    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumOutputSlots(); ++slotIndex)
    {
        unsigned int tensorIndex = tensorIndexes[slotIndex];
        armnn::IOutputSlot* slot = &(layer->GetOutputSlot(slotIndex));
        RegisterProducerOfTensor(subgraphIndex, tensorIndex, slot);
    }
}

void TfLiteParser::SetupInputLayers(size_t subgraphIndex)
{
    CHECK_SUBGRAPH(m_Model, subgraphIndex);

    auto inputs = GetSubgraphInputs(m_Model, subgraphIndex);
    for (auto const & tensorIdAndPtr : inputs)
    {
        auto bindingId = GenerateLayerBindingId(subgraphIndex, tensorIdAndPtr.first);
        IConnectableLayer* layer =
            m_Network->AddInputLayer(bindingId, tensorIdAndPtr.second->name.c_str());

        auto tensorInfo = ToTensorInfo(tensorIdAndPtr.second);
        layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

        RegisterOutputSlots(subgraphIndex,
                            VIRTUAL_OPERATOR_ID,
                            layer,
                            { static_cast<uint32_t>(tensorIdAndPtr.first) });
    }
}

void TfLiteParser::SetupOutputLayers(size_t subgraphIndex)
{
    CHECK_SUBGRAPH(m_Model, subgraphIndex);

    auto outputs = GetSubgraphOutputs(m_Model, subgraphIndex);
    for (auto const & tensorIdAndPtr : outputs)
    {
        auto bindingId = GenerateLayerBindingId(subgraphIndex, tensorIdAndPtr.first);
        IConnectableLayer* layer =
            m_Network->AddOutputLayer(bindingId, tensorIdAndPtr.second->name.c_str());

        RegisterInputSlots(subgraphIndex,
                           VIRTUAL_OPERATOR_ID,
                           layer,
                           { static_cast<uint32_t>(tensorIdAndPtr.first) });
    }
}

void TfLiteParser::SetupConstantLayers(size_t subgraphIndex)
{
    CHECK_SUBGRAPH(m_Model, subgraphIndex);

    const auto & subgraphPtr = m_Model->subgraphs[subgraphIndex];
    for (unsigned int subgraphIndex = 0; subgraphIndex < m_SubgraphConnections.size(); ++subgraphIndex)
    {
        for (unsigned int tensorIndex = 0; tensorIndex < m_SubgraphConnections[subgraphIndex].size(); ++tensorIndex)
        {
            if (m_SubgraphConnections[subgraphIndex][tensorIndex].outputSlot == nullptr &&
                m_SubgraphConnections[subgraphIndex][tensorIndex].inputSlots.size() > 0)
            {
                TensorRawPtr tensorPtr = subgraphPtr->tensors[tensorIndex].get();
                armnn::TensorInfo tensorInfo = ToTensorInfo(tensorPtr);
                auto tensorAndData = CreateConstTensor(tensorPtr,
                                                       tensorInfo,
                                                       armnn::Optional<armnn::PermutationVector&>());

                std::string layerName = fmt::format("Constant:{}", tensorPtr->name);
                IConnectableLayer *layer =
                    m_Network->AddConstantLayer(tensorAndData.first, layerName.c_str());

                layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
                RegisterOutputSlots(subgraphIndex,
                                    VIRTUAL_OPERATOR_ID,
                                    layer,
                                    { tensorIndex });

            }
        }
    }
}

// example usage: BufferRawPtr bufferPtr = GetBuffer(m_Model, inputs[0]->buffer);
TfLiteParser::BufferRawPtr TfLiteParser::GetBuffer(const ModelPtr& model, size_t bufferIndex)
{
    CHECK_BUFFER(model, bufferIndex);
    return model->buffers[bufferIndex].get();
}

template<typename T>
std::pair<armnn::ConstTensor, TfLiteParser::SupportedDataStorage>
TfLiteParser::CreateConstTensorAndStoreData(TfLiteParser::BufferRawPtr bufferPtr,
                                            TfLiteParser::TensorRawPtr tensorPtr,
                                            armnn::TensorInfo& tensorInfo,
                                            armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    auto constData = CreateConstTensorImpl<T>(bufferPtr,
                                              tensorPtr,
                                              tensorInfo,
                                              permutationVector);
    TfLiteParser::SupportedDataStorage storage(std::move(constData.second));
    return std::make_pair(constData.first, std::move(storage));
}

std::pair<armnn::ConstTensor, TfLiteParser::SupportedDataStorage>
TfLiteParser::CreateConstTensor(TensorRawPtr tensorPtr,
                                armnn::TensorInfo& tensorInfo,
                                armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    CHECK_TENSOR_PTR(tensorPtr);
    auto bufferPtr = GetBuffer(m_Model, tensorPtr->buffer);
    CHECK_BUFFER_SIZE(bufferPtr, tensorInfo, tensorPtr->buffer);

    switch (tensorInfo.GetDataType())
    {
        case armnn::DataType::Float32:
            return CreateConstTensorAndStoreData<float>(bufferPtr,
                                                        tensorPtr,
                                                        tensorInfo,
                                                        permutationVector);
        case armnn::DataType::QAsymmU8:
            return CreateConstTensorAndStoreData<uint8_t>(bufferPtr,
                                                          tensorPtr,
                                                          tensorInfo,
                                                          permutationVector);
        case armnn::DataType::QSymmS8:
            return CreateConstTensorAndStoreData<int8_t>(bufferPtr,
                                                         tensorPtr,
                                                         tensorInfo,
                                                         permutationVector);
        case armnn::DataType::QAsymmS8:
            return CreateConstTensorAndStoreData<int8_t>(bufferPtr,
                                                         tensorPtr,
                                                         tensorInfo,
                                                         permutationVector);
        case armnn::DataType::Signed32:
            return CreateConstTensorAndStoreData<int32_t>(bufferPtr,
                                                          tensorPtr,
                                                          tensorInfo,
                                                          permutationVector);
        default:
        {
            std::stringstream errString;
            errString << "Unexpected datatype when creating const tensor: "
                        << armnn::GetDataTypeName(tensorInfo.GetDataType())
                        << " shape:" << tensorInfo.GetShape()
                        << CHECK_LOCATION().AsString();
            throw ParseException(errString.str());
        }
    }
}

BindingPointInfo TfLiteParser::GetNetworkInputBindingInfo(size_t subgraphId,
                                                          const std::string& name) const
{
    CHECK_SUBGRAPH(m_Model, subgraphId);
    auto inputs = GetSubgraphInputs(m_Model, subgraphId);
    for (auto const & input : inputs)
    {
        if (input.second->name == name)
        {
            auto bindingId = GenerateLayerBindingId(subgraphId, input.first);
            return std::make_pair(bindingId, ToTensorInfo(input.second));
        }
    }

    std::stringstream bindings;
    for (auto const & input : inputs)
    {
        bindings << "'" << input.second->name << "' ";
    }

    throw ParseException(
        fmt::format("No input binding found for subgraph:{} and name:{}. "
                    "Possible inputs are: [{}] {}",
                    subgraphId,
                    name,
                    bindings.str(),
                    CHECK_LOCATION().AsString()));
}

BindingPointInfo TfLiteParser::GetNetworkOutputBindingInfo(size_t subgraphId,
                                                           const std::string& name) const
{
    CHECK_SUBGRAPH(m_Model, subgraphId);
    auto outputs = GetSubgraphOutputs(m_Model, subgraphId);
    for (unsigned int i = 0; i < outputs.size(); ++i)
    {
        auto const output = outputs[i];
        if (output.second->name == name)
        {
            auto bindingId = GenerateLayerBindingId(subgraphId, output.first);
            std::vector<unsigned int> shape = m_OverridenOutputShapes.size() > 0 ?
                                                m_OverridenOutputShapes[i] : AsUnsignedVector(output.second->shape);
            return std::make_pair(bindingId, ToTensorInfo(output.second, shape));
        }
    }

    std::stringstream bindings;
    for (auto const & output : outputs)
    {
        bindings << "'" << output.second->name << "' ";
    }

    throw ParseException(
        fmt::format("No output binding found for subgraph:{} and name:{}. "
                    "Possible outputs are: [{}] {}",
                    subgraphId,
                    name,
                    bindings.str(),
                    CHECK_LOCATION().AsString()));
}

size_t TfLiteParser::GetSubgraphCount() const
{
    return m_Model->subgraphs.size();
}

std::vector<std::string> TfLiteParser::GetSubgraphInputTensorNames(size_t subgraphId) const
{
    CHECK_SUBGRAPH(m_Model, subgraphId);
    auto inputs = GetSubgraphInputs(m_Model, subgraphId);
    std::vector<std::string> result;
    result.reserve(inputs.size());
    for (auto const & input : inputs)
    {
        result.push_back(input.second->name);
    }
    return result;
}

std::vector<std::string> TfLiteParser::GetSubgraphOutputTensorNames(size_t subgraphId) const
{
    CHECK_SUBGRAPH(m_Model, subgraphId);
    auto outputs = GetSubgraphOutputs(m_Model, subgraphId);
    std::vector<std::string> result;
    result.reserve(outputs.size());
    for (auto const & output : outputs)
    {
        result.push_back(output.second->name);
    }
    return result;
}

ITfLiteParser* ITfLiteParser::CreateRaw(const Optional<ITfLiteParser::TfLiteParserOptions>& options)
{
    return new TfLiteParser(options);
}

ITfLiteParserPtr ITfLiteParser::Create(const Optional<ITfLiteParser::TfLiteParserOptions>& options)
{
    return ITfLiteParserPtr(CreateRaw(options), &ITfLiteParser::Destroy);
}

void ITfLiteParser::Destroy(ITfLiteParser* parser)
{
    delete parser;
}

TfLiteParser::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<float[]> && data)
: m_FloatData(std::move(data))
, m_Uint8Data(nullptr)
, m_Int8Data(nullptr)
, m_Int32Data(nullptr)
{
}

TfLiteParser::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<uint8_t[]> && data)
: m_FloatData(nullptr)
, m_Uint8Data(std::move(data))
, m_Int8Data(nullptr)
, m_Int32Data(nullptr)
{
}

TfLiteParser::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<int8_t[]> && data)
: m_FloatData(nullptr)
, m_Uint8Data(nullptr)
, m_Int8Data(std::move(data))
, m_Int32Data(nullptr)
{
}

TfLiteParser::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<int32_t[]> && data)
: m_FloatData(nullptr)
, m_Uint8Data(nullptr)
, m_Int8Data(nullptr)
, m_Int32Data(std::move(data))
{
}

} // armnnTfLiteParser
