//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TfLiteParser.hpp"

#include "armnnTfLiteParser/Version.hpp"

#include <armnn/BackendOptions.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/Logging.hpp>
#include <armnn/Tensor.hpp>
#include <armnnUtils/TensorUtils.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

// armnnUtils:
#include <armnnUtils/Permute.hpp>
#include <armnnUtils/Filesystem.hpp>

#include <ParserHelper.hpp>
#include <VerificationHelpers.hpp>

// The generated code based on the Tf Lite schema:
#include <schema_generated.h>

#include <flatbuffers/flexbuffers.h>

#include <fmt/format.h>

#include <algorithm>
#include <fstream>
#include <iostream>
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

ITfLiteParser::ITfLiteParser(const armnn::Optional<TfLiteParserOptions>& options) :
    pTfLiteParserImpl(new TfLiteParserImpl(options)) {}

ITfLiteParser::~ITfLiteParser() = default;

ITfLiteParser* ITfLiteParser::CreateRaw(const armnn::Optional<TfLiteParserOptions>& options)
{
    return new ITfLiteParser(options);
}

ITfLiteParserPtr ITfLiteParser::Create(const armnn::Optional<TfLiteParserOptions>& options)
{
    return ITfLiteParserPtr(CreateRaw(options), &ITfLiteParser::Destroy);
}

void ITfLiteParser::Destroy(ITfLiteParser* parser)
{
    delete parser;
}

armnn::INetworkPtr ITfLiteParser::CreateNetworkFromBinaryFile(const char* graphFile)
{
    return pTfLiteParserImpl->CreateNetworkFromBinaryFile(graphFile);
}

armnn::INetworkPtr ITfLiteParser::CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent)
{
    return pTfLiteParserImpl->CreateNetworkFromBinary(binaryContent);
}

BindingPointInfo ITfLiteParser::GetNetworkInputBindingInfo(size_t subgraphId,
                                                           const std::string& name) const
{
    return pTfLiteParserImpl->GetNetworkInputBindingInfo(subgraphId, name);
}

BindingPointInfo ITfLiteParser::GetNetworkOutputBindingInfo(size_t subgraphId,
                                                            const std::string& name) const
{
    return pTfLiteParserImpl->GetNetworkOutputBindingInfo(subgraphId, name);
}

size_t ITfLiteParser::GetSubgraphCount() const
{
    return pTfLiteParserImpl->GetSubgraphCount();
}

std::vector<std::string> ITfLiteParser::GetSubgraphInputTensorNames(size_t subgraphId) const
{
    return pTfLiteParserImpl->GetSubgraphInputTensorNames(subgraphId);
}

std::vector<std::string> ITfLiteParser::GetSubgraphOutputTensorNames(size_t subgraphId) const
{
    return pTfLiteParserImpl->GetSubgraphOutputTensorNames(subgraphId);
}

namespace
{

const uint32_t VIRTUAL_OPERATOR_ID = std::numeric_limits<uint32_t>::max();

void CheckSubgraph(const TfLiteParserImpl::ModelPtr& model,
                   size_t subgraphIndex,
                   const CheckLocation& location)
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

void CheckModel(const TfLiteParserImpl::ModelPtr& model,
                size_t subgraphIndex,
                size_t operatorIndex,
                const CheckLocation& location)
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

void CheckTensor(const TfLiteParserImpl::ModelPtr& model,
                 size_t subgraphIndex,
                 size_t tensorIndex,
                 const CheckLocation& location)
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

void CheckTensorPtr(TfLiteParserImpl::TensorRawPtr rawPtr,
                    const CheckLocation& location)
{
    if (rawPtr == nullptr)
    {
        throw ParseException(
            fmt::format("{} was called with a null tensor pointer at {}", location.m_Function, location.FileLine()));
    }
}

#define CHECK_TENSOR_PTR(TENSOR_PTR) \
    CheckTensorPtr(TENSOR_PTR, CHECK_LOCATION())

void CheckBuffer(const TfLiteParserImpl::ModelPtr& model,
                 size_t bufferIndex,
                 const CheckLocation& location)
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

void CheckBufferSize(TfLiteParserImpl::BufferRawPtr bufferPtr,
                     const armnn::TensorInfo& tensorInfo,
                     uint32_t bufferId,
                     const CheckLocation& location)
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


tflite::BuiltinOperator GetOpCode(const TfLiteParserImpl::ModelPtr& model, size_t subgraphIndex, size_t operatorIndex)
{
    const auto& operatorPtr = model->subgraphs[subgraphIndex]->operators[operatorIndex];
    auto opcodeIndex = operatorPtr->opcode_index;

// work around the introduction of the deprecated_builtin_code introduced in 2.4 in a backwards compatible manner
#if defined(ARMNN_POST_TFLITE_2_3)
    auto opcode = std::max(model->operator_codes[opcodeIndex]->builtin_code,
            static_cast<tflite::BuiltinOperator>(model->operator_codes[opcodeIndex]->deprecated_builtin_code));
#else
    auto opcode = model->operator_codes[opcodeIndex]->builtin_code;
#endif
    return opcode;
}

std::vector<unsigned int> GetUIntBuffer(armnn::TensorInfo info,
                                        const TfLiteParserImpl::ModelPtr& model,
                                        size_t bufferIndex)
{
    TfLiteParserImpl::BufferRawPtr bufferPtr = TfLiteParserImpl::GetBuffer(model, bufferIndex);
    std::vector<unsigned int> buffer(info.GetNumElements());

    if (info.GetDataType() == DataType::Signed32)
    {
        ::memcpy(buffer.data(), bufferPtr->data.data(), bufferPtr->data.size());
    }
    else if (info.GetDataType() == DataType::Signed64)
    {
        std::vector<uint64_t> uint64Buffer(info.GetNumElements());
        ::memcpy(uint64Buffer.data(), bufferPtr->data.data(), bufferPtr->data.size());
        buffer.assign(std::begin(uint64Buffer), std::end(uint64Buffer));
    }
    return buffer;
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


std::vector<unsigned int> AsUnsignedVector(const std::vector<int32_t>& in)
{
    std::vector<unsigned int> result;
    result.reserve(in.size());
    for (auto& i : in)
    {
        // If the location of the input data is -1 then the input should be ignored.
        if (i == -1)
        {
            continue;
        }
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

armnn::TensorInfo ToTensorInfo(TfLiteParserImpl::TensorRawPtr tensorPtr,
                               const std::vector<unsigned int>& shape,
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
        case tflite::TensorType_BOOL:
            type = armnn::DataType::Boolean;
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
    TensorShape tensorShape;

    std::vector<unsigned int> safeShape = shape;
    if (shape.size() == 0)
    {
        safeShape.push_back(1);
    }

    if (!outputTensor)
    {
        tensorShape = TensorShape(armnn::numeric_cast<unsigned int>(safeShape.size()), safeShape.data());
    }
    else
    {
        size_t shapeSignatureSize = tensorPtr->shape_signature.size();

        // If a shape signature exists we will use that to infer dynamic tensors
        if (shapeSignatureSize != 0)
        {
            // If the shape is incompatible with the shape signature override the shape
            if (shapeSignatureSize != shape.size())
            {
                safeShape = {};

                for (unsigned int i = 0; i < shapeSignatureSize; ++i)
                {
                    unsigned int dim = tensorPtr->shape_signature[i] > -1 ?
                                       static_cast<unsigned int>(tensorPtr->shape_signature[i]) : 0;
                    safeShape.push_back(dim);
                }
            }

            std::unique_ptr<bool[]> dimMask = std::make_unique<bool[]>(tensorPtr->shape_signature.size());
            for (unsigned int i = 0; i < tensorPtr->shape_signature.size(); ++i)
            {
                dimMask[i] = tensorPtr->shape_signature[i] == -1 ? false : true;
            }
            tensorShape = TensorShape(static_cast<unsigned int>(safeShape.size()), safeShape.data(), dimMask.get());
        }
        // If there is no shape signature treat the tensor as dynamic if the shape has a size of zero
        else if (shape.size() == 0)
        {
            tensorShape = TensorShape(1, false);
        }
        else
        {
            tensorShape = TensorShape(armnn::numeric_cast<unsigned int>(shape.size()), shape.data());
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
            armnn::TensorInfo result(tensorShape,
                                     type,
                                     quantizationScales,
                                     armnn::numeric_cast<unsigned int>(tensorPtr->quantization->quantized_dimension));
            return result;
        }
    }
    else
    {
        armnn::TensorInfo result(tensorShape,
                                 type,
                                 quantizationScale,
                                 quantizationOffset);
        return result;
    }
}

armnn::TensorInfo ToTensorInfo(TfLiteParserImpl::TensorRawPtr tensorPtr)
{
    auto const& dimensions = AsUnsignedVector(tensorPtr->shape);
    return ToTensorInfo(tensorPtr, dimensions);
}

armnn::TensorInfo ToTensorInfo(TfLiteParserImpl::TensorRawPtr tensorPtr,
                               const bool outputTensor)
{
    auto const& dimensions = AsUnsignedVector(tensorPtr->shape);
    return ToTensorInfo(tensorPtr, dimensions, outputTensor);
}

template<typename T>
std::pair<armnn::ConstTensor, std::unique_ptr<T[]>>
CreateConstTensorImpl(TfLiteParserImpl::BufferRawPtr bufferPtr,
                      TfLiteParserImpl::TensorRawPtr tensorPtr,
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

    // Make sure isConstant flag is set.
    tensorInfo.SetConstant();

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

TfLiteParserImpl::TfLiteParserImpl(const Optional<ITfLiteParser::TfLiteParserOptions>& options)
: m_Options(options)
, m_Network(nullptr, nullptr)
, m_ParserFunctions(tflite::BuiltinOperator_MAX+1, &TfLiteParserImpl::ParseUnsupportedOperator)
{
    // register supported operators
    m_ParserFunctions[tflite::BuiltinOperator_ABS]                     = &TfLiteParserImpl::ParseAbs;
    m_ParserFunctions[tflite::BuiltinOperator_ADD]                     = &TfLiteParserImpl::ParseAdd;
    m_ParserFunctions[tflite::BuiltinOperator_ARG_MIN]                 = &TfLiteParserImpl::ParseArgMin;
    m_ParserFunctions[tflite::BuiltinOperator_ARG_MAX]                 = &TfLiteParserImpl::ParseArgMax;
    m_ParserFunctions[tflite::BuiltinOperator_AVERAGE_POOL_2D]         = &TfLiteParserImpl::ParseAveragePool2D;
    m_ParserFunctions[tflite::BuiltinOperator_BATCH_TO_SPACE_ND]       = &TfLiteParserImpl::ParseBatchToSpaceND;
    m_ParserFunctions[tflite::BuiltinOperator_CAST]                    = &TfLiteParserImpl::ParseCast;
    m_ParserFunctions[tflite::BuiltinOperator_CONCATENATION]           = &TfLiteParserImpl::ParseConcatenation;
    m_ParserFunctions[tflite::BuiltinOperator_CONV_2D]                 = &TfLiteParserImpl::ParseConv2D;
    // Conv3D support was added in TF 2.5, so for backwards compatibility a hash define is needed.
    #if defined(ARMNN_POST_TFLITE_2_3)
    m_ParserFunctions[tflite::BuiltinOperator_CONV_3D]                 = &TfLiteParserImpl::ParseConv3D;
    #endif
    m_ParserFunctions[tflite::BuiltinOperator_CUSTOM]                  = &TfLiteParserImpl::ParseCustomOperator;
    m_ParserFunctions[tflite::BuiltinOperator_DEPTH_TO_SPACE]          = &TfLiteParserImpl::ParseDepthToSpace;
    m_ParserFunctions[tflite::BuiltinOperator_DEPTHWISE_CONV_2D]       = &TfLiteParserImpl::ParseDepthwiseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_DEQUANTIZE]              = &TfLiteParserImpl::ParseDequantize;
    m_ParserFunctions[tflite::BuiltinOperator_DIV]                     = &TfLiteParserImpl::ParseDiv;
    m_ParserFunctions[tflite::BuiltinOperator_ELU]                     = &TfLiteParserImpl::ParseElu;
    m_ParserFunctions[tflite::BuiltinOperator_EQUAL]                   = &TfLiteParserImpl::ParseEqual;
    m_ParserFunctions[tflite::BuiltinOperator_EXP]                     = &TfLiteParserImpl::ParseExp;
    m_ParserFunctions[tflite::BuiltinOperator_EXPAND_DIMS]             = &TfLiteParserImpl::ParseExpandDims;
    m_ParserFunctions[tflite::BuiltinOperator_FULLY_CONNECTED]         = &TfLiteParserImpl::ParseFullyConnected;
    m_ParserFunctions[tflite::BuiltinOperator_GATHER]                  = &TfLiteParserImpl::ParseGather;
    m_ParserFunctions[tflite::BuiltinOperator_GREATER]                 = &TfLiteParserImpl::ParseGreater;
    m_ParserFunctions[tflite::BuiltinOperator_GREATER_EQUAL]           = &TfLiteParserImpl::ParseGreaterOrEqual;
    m_ParserFunctions[tflite::BuiltinOperator_HARD_SWISH]              = &TfLiteParserImpl::ParseHardSwish;
    m_ParserFunctions[tflite::BuiltinOperator_LEAKY_RELU]              = &TfLiteParserImpl::ParseLeakyRelu;
    m_ParserFunctions[tflite::BuiltinOperator_LESS]                    = &TfLiteParserImpl::ParseLess;
    m_ParserFunctions[tflite::BuiltinOperator_LESS_EQUAL]              = &TfLiteParserImpl::ParseLessOrEqual;
    m_ParserFunctions[tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION]
            = &TfLiteParserImpl::ParseLocalResponseNormalization;
    m_ParserFunctions[tflite::BuiltinOperator_LOGICAL_NOT]             = &TfLiteParserImpl::ParseLogicalNot;
    m_ParserFunctions[tflite::BuiltinOperator_LOGISTIC]                = &TfLiteParserImpl::ParseLogistic;
    m_ParserFunctions[tflite::BuiltinOperator_L2_NORMALIZATION]        = &TfLiteParserImpl::ParseL2Normalization;
    m_ParserFunctions[tflite::BuiltinOperator_MAX_POOL_2D]             = &TfLiteParserImpl::ParseMaxPool2D;
    m_ParserFunctions[tflite::BuiltinOperator_MAXIMUM]                 = &TfLiteParserImpl::ParseMaximum;
    m_ParserFunctions[tflite::BuiltinOperator_MEAN]                    = &TfLiteParserImpl::ParseMean;
    m_ParserFunctions[tflite::BuiltinOperator_MINIMUM]                 = &TfLiteParserImpl::ParseMinimum;
    m_ParserFunctions[tflite::BuiltinOperator_MIRROR_PAD]              = &TfLiteParserImpl::ParseMirrorPad;
    m_ParserFunctions[tflite::BuiltinOperator_MUL]                     = &TfLiteParserImpl::ParseMul;
    m_ParserFunctions[tflite::BuiltinOperator_NEG]                     = &TfLiteParserImpl::ParseNeg;
    m_ParserFunctions[tflite::BuiltinOperator_NOT_EQUAL]               = &TfLiteParserImpl::ParseNotEqual;
    m_ParserFunctions[tflite::BuiltinOperator_PACK]                    = &TfLiteParserImpl::ParsePack;
    m_ParserFunctions[tflite::BuiltinOperator_PAD]                     = &TfLiteParserImpl::ParsePad;
    m_ParserFunctions[tflite::BuiltinOperator_PADV2]                   = &TfLiteParserImpl::ParsePad;
    m_ParserFunctions[tflite::BuiltinOperator_PRELU]                   = &TfLiteParserImpl::ParsePrelu;
    m_ParserFunctions[tflite::BuiltinOperator_QUANTIZE]                = &TfLiteParserImpl::ParseQuantize;
    m_ParserFunctions[tflite::BuiltinOperator_RELU]                    = &TfLiteParserImpl::ParseRelu;
    m_ParserFunctions[tflite::BuiltinOperator_RELU6]                   = &TfLiteParserImpl::ParseRelu6;
    m_ParserFunctions[tflite::BuiltinOperator_REDUCE_MAX]              = &TfLiteParserImpl::ParseReduceMax;
    m_ParserFunctions[tflite::BuiltinOperator_REDUCE_MIN]              = &TfLiteParserImpl::ParseReduceMin;
    m_ParserFunctions[tflite::BuiltinOperator_REDUCE_PROD]             = &TfLiteParserImpl::ParseReduceProd;
    m_ParserFunctions[tflite::BuiltinOperator_RESHAPE]                 = &TfLiteParserImpl::ParseReshape;
    m_ParserFunctions[tflite::BuiltinOperator_RESIZE_BILINEAR]         = &TfLiteParserImpl::ParseResizeBilinear;
    m_ParserFunctions[tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR] = &TfLiteParserImpl::ParseResizeNearestNeighbor;
    m_ParserFunctions[tflite::BuiltinOperator_RSQRT]                   = &TfLiteParserImpl::ParseRsqrt;
    m_ParserFunctions[tflite::BuiltinOperator_SHAPE]                   = &TfLiteParserImpl::ParseShape;
    m_ParserFunctions[tflite::BuiltinOperator_SLICE]                   = &TfLiteParserImpl::ParseSlice;
    m_ParserFunctions[tflite::BuiltinOperator_SOFTMAX]                 = &TfLiteParserImpl::ParseSoftmax;
    m_ParserFunctions[tflite::BuiltinOperator_SPACE_TO_BATCH_ND]       = &TfLiteParserImpl::ParseSpaceToBatchND;
    m_ParserFunctions[tflite::BuiltinOperator_SPLIT]                   = &TfLiteParserImpl::ParseSplit;
    m_ParserFunctions[tflite::BuiltinOperator_SPLIT_V]                 = &TfLiteParserImpl::ParseSplitV;
    m_ParserFunctions[tflite::BuiltinOperator_SQUEEZE]                 = &TfLiteParserImpl::ParseSqueeze;
    m_ParserFunctions[tflite::BuiltinOperator_STRIDED_SLICE]           = &TfLiteParserImpl::ParseStridedSlice;
    m_ParserFunctions[tflite::BuiltinOperator_SUB]                     = &TfLiteParserImpl::ParseSub;
    m_ParserFunctions[tflite::BuiltinOperator_SUM]                     = &TfLiteParserImpl::ParseSum;
    m_ParserFunctions[tflite::BuiltinOperator_TANH]                    = &TfLiteParserImpl::ParseTanH;
    m_ParserFunctions[tflite::BuiltinOperator_TRANSPOSE]               = &TfLiteParserImpl::ParseTranspose;
    m_ParserFunctions[tflite::BuiltinOperator_TRANSPOSE_CONV]          = &TfLiteParserImpl::ParseTransposeConv;
    m_ParserFunctions[tflite::BuiltinOperator_UNPACK]                  = &TfLiteParserImpl::ParseUnpack;

    // register supported custom operators
    m_CustomParserFunctions["TFLite_Detection_PostProcess"]      = &TfLiteParserImpl::ParseDetectionPostProcess;
}

void TfLiteParserImpl::ResetParser()
{
    m_Network = armnn::INetworkPtr(nullptr, nullptr);
    m_Model = nullptr;
    m_SubgraphConnections.clear();
}

INetworkPtr TfLiteParserImpl::CreateNetworkFromBinaryFile(const char* graphFile)
{
    ResetParser();
    m_Model = LoadModelFromFile(graphFile);
    return CreateNetworkFromModel();
}

INetworkPtr TfLiteParserImpl::CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent)
{
    ResetParser();
    m_Model = LoadModelFromBinary(binaryContent.data(), binaryContent.size());
    return CreateNetworkFromModel();
}


armnn::INetworkPtr TfLiteParserImpl::LoadModel(std::unique_ptr<tflite::ModelT> model)
{
    ResetParser();
    m_Model = std::move(model);

    return CreateNetworkFromModel();
}

INetworkPtr TfLiteParserImpl::CreateNetworkFromModel()
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

// work around the introduction of the deprecated_builtin_code introduced in 2.4 in a backwards compatible manner
#if defined(ARMNN_POST_TFLITE_2_3)
                auto builtinCode = std::max(opCodePtr->builtin_code,
                        static_cast<tflite::BuiltinOperator>(opCodePtr->deprecated_builtin_code));
#else
                auto builtinCode = opCodePtr->builtin_code;
#endif

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

void TfLiteParserImpl::RegisterProducerOfTensor(size_t subgraphIndex,
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

void TfLiteParserImpl::RegisterConsumerOfTensor(size_t subgraphIndex,
                                                size_t tensorIndex,
                                                armnn::IInputSlot* slot)
{
    CHECK_TENSOR(m_Model, subgraphIndex, tensorIndex);
    ARMNN_ASSERT(m_SubgraphConnections.size() > subgraphIndex);
    ARMNN_ASSERT(m_SubgraphConnections[subgraphIndex].size() > tensorIndex);

    TensorSlots& tensorSlots = m_SubgraphConnections[subgraphIndex][tensorIndex];
    tensorSlots.inputSlots.push_back(slot);
}

void TfLiteParserImpl::ParseCustomOperator(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    // NOTE: By default we presume the custom operator is not supported
    auto customParserFunction = &TfLiteParserImpl::ParseUnsupportedOperator;

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

void TfLiteParserImpl::ParseUnsupportedOperator(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];

    auto opcodeIndex = operatorPtr->opcode_index;

// work around the introduction of the deprecated_builtin_code introduced in 2.4 in a backwards compatible manner
#if defined(ARMNN_POST_TFLITE_2_3)
    auto opcode      = std::max(m_Model->operator_codes[opcodeIndex]->builtin_code,
            static_cast<tflite::BuiltinOperator>(m_Model->operator_codes[opcodeIndex]->deprecated_builtin_code));
#else
    auto opcode      = m_Model->operator_codes[opcodeIndex]->builtin_code;
#endif

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

void TfLiteParserImpl::ParseCast(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Cast:{}:{}", subgraphIndex, operatorIndex);

    IConnectableLayer* layer = m_Network->AddCastLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParserImpl::ParseConv2D(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsConv2DOptions();

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

    auto filterTensorAndData = CreateConstTensorNonPermuted(inputs[1], filterTensorInfo);
    armnn::IConnectableLayer* layer = nullptr;

    auto layerName = fmt::format("Conv2D:{}:{}", subgraphIndex, operatorIndex);

    if (inputs.size() == 3)
    {
        desc.m_BiasEnabled = true;
        armnn::TensorInfo biasTensorInfo = ToTensorInfo(inputs[2]);
        auto biasTensorAndData = CreateConstTensorNonPermuted(inputs[2], biasTensorInfo);
        layer = m_Network->AddConvolution2dLayer(desc,
                                                 filterTensorAndData,
                                                 Optional<ConstTensor>(biasTensorAndData),
                                                 layerName.c_str());
    }
    else
    {
        layer = m_Network->AddConvolution2dLayer(desc,
                                                 filterTensorAndData,
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

// Conv3D support was added in TF 2.5, so for backwards compatibility a hash define is needed.
#if defined(ARMNN_POST_TFLITE_2_3)
void TfLiteParserImpl::ParseConv3D(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsConv3DOptions();

    CHECK_SUPPORTED_FUSED_ACTIVATION(options, subgraphIndex, operatorIndex);

    Convolution3dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_DataLayout = armnn::DataLayout::NDHWC;
    desc.m_StrideX = CHECKED_NON_NEGATIVE(options->stride_w);
    desc.m_StrideY = CHECKED_NON_NEGATIVE(options->stride_h);
    desc.m_StrideZ = CHECKED_NON_NEGATIVE(options->stride_d);
    desc.m_DilationX = CHECKED_NON_NEGATIVE(options->dilation_w_factor);
    desc.m_DilationY = CHECKED_NON_NEGATIVE(options->dilation_h_factor);
    desc.m_DilationZ = CHECKED_NON_NEGATIVE(options->dilation_d_factor);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2, 3);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo filterTensorInfo = ToTensorInfo(inputs[1]);

    // Assuming input is NDHWC
    unsigned int inputDepth  = inputTensorInfo.GetShape()[1];
    unsigned int inputHeight = inputTensorInfo.GetShape()[2];
    unsigned int inputWidth  = inputTensorInfo.GetShape()[3];

    // Assuming the filter is DHWIO : Depth, Height, Width, OutputChannels, InputChannels
    unsigned int filterDepth  = filterTensorInfo.GetShape()[0];
    unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    CalcPadding(inputDepth, filterDepth, desc.m_StrideZ,
                desc.m_DilationY, desc.m_PadFront, desc.m_PadBack, options->padding);
    CalcPadding(inputHeight, filterHeight, desc.m_StrideY,
                desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, desc.m_StrideX,
                desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, options->padding);

    auto filterTensorAndData = CreateConstTensorNonPermuted(inputs[1], filterTensorInfo);

    auto layerName = fmt::format("Conv3D:{}:{}", subgraphIndex, operatorIndex);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    // Add the first input and weights tensor to the registration list.
    // The constant weights will be added by SetupConstantLayers.
    std::vector<unsigned int> tensorIndexesToRegister = {inputTensorIndexes[0], inputTensorIndexes[1]};

    if (inputs.size() == 3)
    {
        desc.m_BiasEnabled = true;

        // Add the biases input to the registration list, a constant layer will be added by SetupConstantLayers.
        tensorIndexesToRegister.emplace_back(inputTensorIndexes[2]);
    }

    armnn::IConnectableLayer* layer = m_Network->AddConvolution3dLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Register the input connection slots for the layer, connections are made after all layers have been created
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, tensorIndexesToRegister);

    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);
    // Register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}
#endif

void TfLiteParserImpl::ParseDepthwiseConv2D(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsDepthwiseConv2DOptions();

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

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo filterTensorInfo = ToTensorInfo(inputs[1]);

    // Assuming input is NHWC
    unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    // TensorflowLite weights come in the format [1, H, W, I * M]
    unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    CalcPadding(inputHeight, filterHeight, desc.m_StrideY,
                desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, desc.m_StrideX,
                desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, options->padding);

    // ArmNN uses the same filter tensor layout at TfLite [1, H, W, O] no need for any permutation
    auto filterTensor = CreateConstTensorNonPermuted(inputs[1], filterTensorInfo);
    armnn::IConnectableLayer* layer = nullptr;
    auto layerName = fmt::format("DepthwiseConv2D:{}:{}", subgraphIndex, operatorIndex);

    if (inputs.size() == 3)
    {
        desc.m_BiasEnabled = true;
        TensorInfo biasTensorInfo = ToTensorInfo(inputs[2]);
        auto biasTensorAndData = CreateConstTensorNonPermuted(inputs[2], biasTensorInfo);
        layer = m_Network->AddDepthwiseConvolution2dLayer(desc,
                                                          filterTensor,
                                                          Optional<ConstTensor>(biasTensorAndData),
                                                          layerName.c_str());
    }
    else
    {
        layer = m_Network->AddDepthwiseConvolution2dLayer(desc,
                                                          filterTensor,
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

void TfLiteParserImpl::ParseDequantize(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParseExpandDims(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("ExpandDims:{}:{}", subgraphIndex, operatorIndex);

    armnn::TensorInfo inputTensorInfo = ToTensorInfo(inputs[0]);
    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);

    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    ReshapeDescriptor reshapeDesc;

    if (outputTensorInfo.GetShape().AreAllDimensionsSpecified())
    {
        reshapeDesc.m_TargetShape = outputTensorInfo.GetShape();
    }
    else
    {
        int32_t axis = inputs[1]->shape[0];

        int32_t inputDimSize = static_cast<int32_t>(inputTensorInfo.GetShape().GetNumDimensions());

        if (axis > inputDimSize || axis < 0 - (inputDimSize + 1))
        {
            throw ParseException("axis must be in range [0 - (inputDimSize + 1), inputDimSize] inclusive");
        }

        if(axis < 0)
        {
            axis = inputDimSize + axis + 1;
        }

        std::vector<unsigned int> shape(static_cast<unsigned int>(inputDimSize) + 1);
        unsigned int inputShapeIndex = 0;
        for (unsigned int i = 0; i < static_cast<unsigned int>(inputDimSize + 1); ++i)
        {
            if (i == static_cast<unsigned int>(axis))
            {
                shape[i] = 1;
            }
            else
            {
                shape[i] = inputTensorInfo.GetShape()[inputShapeIndex];
                ++inputShapeIndex;
            }
        }

        reshapeDesc.m_TargetShape = TensorShape(static_cast<unsigned int>(inputDimSize + 1), shape.data());
    }

    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParserImpl::ParseTranspose(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParseTransposeConv(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsTransposeConvOptions();

    TransposeConvolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX = CHECKED_NON_NEGATIVE(options->stride_w);
    desc.m_StrideY = CHECKED_NON_NEGATIVE(options->stride_h);
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    if (inputs.size() == 4)
    {
        desc.m_BiasEnabled = true;
    }
    else
    {
        CHECK_VALID_SIZE(inputs.size(), 3);
    }

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

    auto filterTensorAndData = CreateConstTensorNonPermuted(inputs[1], filterTensorInfo);

    armnn::IConnectableLayer* layer = nullptr;
    auto layerName = fmt::format("TransposeConv:{}:{}", subgraphIndex, operatorIndex);

    if (desc.m_BiasEnabled)
    {
        auto biasTensorInfo = ToTensorInfo(inputs[3]);
        auto biasConstTensor = CreateConstTensorNonPermuted(inputs[3], biasTensorInfo);
        layer = m_Network->AddTransposeConvolution2dLayer(desc,
                                                          filterTensorAndData,
                                                          biasConstTensor,
                                                          layerName.c_str());
    }
    else
    {
        layer = m_Network->AddTransposeConvolution2dLayer(desc,
                                                          filterTensorAndData,
                                                          EmptyOptional(),
                                                          layerName.c_str());
    }

    ARMNN_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // only the tensors for the inputs are relevant, exclude the const (filter) tensor
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[2]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParserImpl::ParseAveragePool2D(size_t subgraphIndex, size_t operatorIndex)
{
    ParsePool(subgraphIndex, operatorIndex, PoolingAlgorithm::Average);
}

void TfLiteParserImpl::ParseBatchToSpaceND(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParseL2Normalization(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParseMaxPool2D(size_t subgraphIndex, size_t operatorIndex)
{
    ParsePool(subgraphIndex, operatorIndex, PoolingAlgorithm::Max);
}

void TfLiteParserImpl::ParseMaximum(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParseMinimum(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParsePool(size_t subgraphIndex,
                                 size_t operatorIndex,
                                 PoolingAlgorithm algorithm)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsPool2DOptions();

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

void TfLiteParserImpl::ParseSlice(size_t subgraphIndex, size_t operatorIndex)
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

    std::vector<int> signedSize(sizeTensorInfo.GetNumElements());
    ::memcpy(signedSize.data(), sizeBufferPtr->data.data(), sizeTensorInfo.GetNumBytes());
    std::vector<unsigned int> size(sizeTensorInfo.GetNumElements());
    TensorInfo inputTensorInfo = ToTensorInfo(inputs[0]);

    for (unsigned int i = 0; i < signedSize.size(); ++i)
    {
        int signedValue = signedSize[i];

        if (signedValue < -1 || signedValue > static_cast<int>(inputTensorInfo.GetShape()[i] - begin[i]))
        {
            throw ParseException(fmt::format("Invalid value for size {} size must be in range "
                                             "[-1, inputDimSize - begin] [-1, {}] inclusive {}",
                                             signedValue,
                                             inputTensorInfo.GetShape()[i] - begin[i],
                                             CHECK_LOCATION().AsString()));
        }

        if (signedValue == -1)
        {
            size[i] = inputTensorInfo.GetShape()[i] - begin[i];
        }
        else
        {
            size[i] = static_cast<unsigned int>(signedValue);
        }
    }

    desc = SliceDescriptor(begin, size);

    auto layerName = fmt::format("Slice:{}:{}", subgraphIndex, operatorIndex);

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

void TfLiteParserImpl::ParseSoftmax(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsSoftmaxOptions();

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

void TfLiteParserImpl::ParseSpaceToBatchND(size_t subgraphIndex, size_t operatorIndex)
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

armnn::TensorInfo TfLiteParserImpl::OutputShapeOfSqueeze(std::vector<uint32_t> squeezeDims,
                                                         const armnn::TensorInfo& inputTensorInfo)
{
    CHECK_VALID_SIZE(squeezeDims.size(), 0, 1, 2, 3, 4);
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

void TfLiteParserImpl::ParseShape(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Shape:{}:{}", subgraphIndex, operatorIndex);

    IConnectableLayer* layer = m_Network->AddShapeLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);


    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Check if output tensor type is Signed32 or Signed64
    if (outputTensorInfo.GetDataType() != armnn::DataType::Signed32 &&
        outputTensorInfo.GetDataType() != armnn::DataType::Signed64)
    {
        throw ParseException(
            fmt::format(
                "Output tensor data type is not supported. (Supported types: Signed32 & Signed64) {}",
                CHECK_LOCATION().AsString()));
    }

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParserImpl::ParseSqueeze(size_t subgraphIndex, size_t operatorIndex)
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

    std::vector<uint32_t> squeezeDim;
    // A single negative dim index is interpreted as a negative index in python
    // Meaning the index will be the shape size plus the negative index value
    if (options->squeeze_dims.size() == 1 && options->squeeze_dims[0] < 0)
    {
        int32_t dim = static_cast<int32_t>(inputTensorInfo.GetShape().GetNumDimensions()) + options->squeeze_dims[0];
        squeezeDim.push_back(static_cast<uint32_t>(dim));
    }
    else
    {
        squeezeDim = AsUnsignedVector(options->squeeze_dims);
    }

    armnn::TensorInfo outputTensorInfo = TfLiteParserImpl::OutputShapeOfSqueeze(squeezeDim, inputTensorInfo);

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

void TfLiteParserImpl::ParseStridedSlice(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 4);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsStridedSliceOptions();

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

void TfLiteParserImpl::ParseSub(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsSubOptions();

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

void TfLiteParserImpl::ParseDiv(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsDivOptions();

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

void TfLiteParserImpl::ParseAdd(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsAddOptions();

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

void TfLiteParserImpl::ParseMul(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsMulOptions();

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

void TfLiteParserImpl::ParseMean(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParsePad(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    TfLiteParserImpl::TensorRawPtrVector inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);

    TfLiteParserImpl::TensorRawPtrVector outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo = ToTensorInfo(inputs[0]);
    armnn::TensorInfo padTensorInfo = ToTensorInfo(inputs[1]);

    std::vector<unsigned int> padBuffer = GetUIntBuffer(padTensorInfo, m_Model, inputs[1]->buffer);

    size_t step = 2;
    armnn::PadDescriptor desc;
    auto opcode = GetOpCode(m_Model, subgraphIndex, operatorIndex);

    if (opcode == tflite::BuiltinOperator_PAD)
    {
        CHECK_VALID_SIZE(inputs.size(), 2);

        if (inputTensorInfo.IsQuantized())
        {
            desc.m_PadValue = static_cast<float>(inputTensorInfo.GetQuantizationOffset());
        }
    }
    else if (opcode == tflite::BuiltinOperator_PADV2)
    {
        CHECK_VALID_SIZE(inputs.size(), 3);

        armnn::TensorInfo padValueTensorInfo = ToTensorInfo(inputs[2]);

        if (padValueTensorInfo.GetNumElements() != 1)
        {
            ARMNN_THROW_PARSE_EXCEPTION("Multiple padding values are not supported in PADV2");
        }
        BufferRawPtr padValueBufferPtr = GetBuffer(m_Model, inputs[2]->buffer);

        // Get the pad value from the input tensor
        if (padValueBufferPtr->data.size() > 0)
        {
            switch (padValueTensorInfo.GetDataType())
            {
                case armnn::DataType::Float32:
                {
                    std::vector<float> padValueBuffer(padValueTensorInfo.GetNumElements());
                    ::memcpy(padValueBuffer.data(), padValueBufferPtr->data.data(), padValueBufferPtr->data.size());
                    desc.m_PadValue = padValueBuffer[0];
                    break;
                }
                case armnn::DataType::QAsymmU8:
                {
                    std::vector<uint8_t> padValueBuffer(padValueTensorInfo.GetNumElements());
                    ::memcpy(padValueBuffer.data(), padValueBufferPtr->data.data(), padValueBufferPtr->data.size());
                    desc.m_PadValue = armnn::Dequantize<uint8_t>(padValueBuffer[0],
                                                                 padValueTensorInfo.GetQuantizationScale(),
                                                                 padValueTensorInfo.GetQuantizationOffset());
                    break;
                }
                case armnn::DataType::QAsymmS8:
                case armnn::DataType::QSymmS8:
                {
                    std::vector<int8_t> padValueBuffer(padValueTensorInfo.GetNumElements());
                    ::memcpy(padValueBuffer.data(), padValueBufferPtr->data.data(), padValueBufferPtr->data.size());
                    desc.m_PadValue = armnn::Dequantize<int8_t>(padValueBuffer[0],
                                                                padValueTensorInfo.GetQuantizationScale(),
                                                                padValueTensorInfo.GetQuantizationOffset());
                    break;
                }
                default: ARMNN_THROW_PARSE_EXCEPTION("Unsupported DataType");
            }
        }
        else if (inputTensorInfo.IsQuantized())
        {
            desc.m_PadValue = static_cast<float>(inputTensorInfo.GetQuantizationOffset());
        }
    }

    for (unsigned int i = 0; i < padTensorInfo.GetNumElements() / step; ++i)
    {
        desc.m_PadList.emplace_back(padBuffer[i * step], padBuffer[i * step + 1]);
    }

    auto layerName = (opcode == tflite::BuiltinOperator_PAD) ? fmt::format("Pad:{}:{}", subgraphIndex, operatorIndex)
            : fmt::format("PadV2:{}:{}", subgraphIndex, operatorIndex);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);

    IConnectableLayer* layer = m_Network->AddPadLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParserImpl::ParseMirrorPad(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    TfLiteParserImpl::TensorRawPtrVector inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    TfLiteParserImpl::TensorRawPtrVector outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo = ToTensorInfo(inputs[0]);

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

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsMirrorPadOptions();

    if (options->mode == tflite::MirrorPadMode_REFLECT)
    {
        desc.m_PaddingMode = PaddingMode::Reflect;
    }
    else if (options->mode == tflite::MirrorPadMode_SYMMETRIC)
    {
        desc.m_PaddingMode = PaddingMode::Symmetric;
    }
    else
    {
        ARMNN_THROW_PARSE_EXCEPTION("PaddingMode must be either REFLECT or SYMMETRIC");
    }

    // If padding mode is Reflect then both paddings must be no greater than inputShape(i) - 1.
    // If padding mode is Symmetric then both paddings must be no greater than inputShape(i).
    auto inputShape = inputTensorInfo.GetShape();
    auto padList = desc.m_PadList;

    const unsigned int isReflect = static_cast<unsigned int>(desc.m_PaddingMode == PaddingMode::Reflect);
    for(unsigned int i = 0; i < padList.size(); ++i)
    {
        if(padList.at(i).first > (inputShape[i] - isReflect) ||
           padList.at(i).second > (inputShape[i] - isReflect))
        {
            ARMNN_THROW_PARSE_EXCEPTION("Padding values must be less (Reflect) or "
                                        "equal (Symmetric) to the dimension size.");
        }
    }

    auto layerName = fmt::format("MirrorPad:{}:{}", subgraphIndex, operatorIndex);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);

    IConnectableLayer* layer = m_Network->AddPadLayer(desc, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParserImpl::ParsePrelu(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Prelu:{}:{}", subgraphIndex, operatorIndex);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo alphaTensorInfo  = ToTensorInfo(inputs[1]);
    armnn::TensorInfo outputTensorInfo  = ToTensorInfo(outputs[0], true);
    CheckMatchingQuantization(inputTensorInfo, outputTensorInfo, layerName, "Input 0", "Output 0");

    IConnectableLayer* layer = m_Network->AddPreluLayer(layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    if (IsConstTensor(inputs[1]))
    {
        auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
        armnn::IInputSlot* slot = &(layer->GetInputSlot(0));
        RegisterConsumerOfTensor(subgraphIndex, inputTensorIndexes[0], slot);

        auto alphaTensorAndData = CreateConstTensorNonPermuted(inputs[1], alphaTensorInfo);
        std::string constLayerName = fmt::format("Constant:{}", inputs[1]->name);
        IConnectableLayer* constLayer =
                    m_Network->AddConstantLayer(alphaTensorAndData, constLayerName.c_str());
        ARMNN_ASSERT(constLayer != nullptr);

        constLayer->GetOutputSlot(0).SetTensorInfo(alphaTensorInfo);
        constLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));
        RegisterOutputSlots(subgraphIndex,
                            VIRTUAL_OPERATOR_ID,
                            constLayer,
                            { inputTensorIndexes[1] });
    }
    else
    {
        auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
        RegisterInputSlots(subgraphIndex, operatorIndex, layer, inputTensorIndexes);
    }

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParserImpl::ParseQuantize(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParseRelu(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex, ActivationFunction::ReLu);
}

void TfLiteParserImpl::ParseRelu6(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex, ActivationFunction::BoundedReLu);
}

void TfLiteParserImpl::ParseLeakyRelu(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex, operatorIndex, ActivationFunction::LeakyReLu);
}

void TfLiteParserImpl::ParseLogistic(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex,ActivationFunction::Sigmoid);
}

void TfLiteParserImpl::ParseTanH(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex,ActivationFunction::TanH);
}

void TfLiteParserImpl::ParseElu(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex, operatorIndex, ActivationFunction::Elu);
}

void TfLiteParserImpl::ParseHardSwish(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex, operatorIndex, ActivationFunction::HardSwish);
}

void TfLiteParserImpl::ParseActivation(size_t subgraphIndex, size_t operatorIndex, ActivationFunction activationType)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
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
            const auto* options = operatorPtr->builtin_options.AsLeakyReluOptions();
            activationDesc.m_A = options->alpha;
            break;
        }
        case ActivationFunction::Elu:
        {
            layerName += fmt::format("ELU:{}:{}", subgraphIndex, operatorIndex);
            activationDesc.m_A = 1.0f;
            break;
        }
        case ActivationFunction::HardSwish:
        {
            layerName += fmt::format("HARDSWISH:{}:{}", subgraphIndex, operatorIndex);
            break;
        }
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
armnn::TensorInfo TfLiteParserImpl::OutputShapeOfReshape(const armnn::TensorInfo& inputTensorInfo,
                                                         const std::vector<int32_t>& targetDimsIn)
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

void TfLiteParserImpl::ParseReshape(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsReshapeOptions();
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
            if (values)
            {
                for (int i = 0; i < inputs[1]->shape[0]; ++i)
                {
                    targetShape.push_back(values[i]);
                }
            }
            else
            {
                try
                {
                    // We attempt to infer during Runtime.
                    TensorShape reshapeShapes   = ToTensorInfo(inputs[1]).GetShape();
                    // The parser only supports shape (batch, -1) or (-1) for non-constant shape input.
                    if (reshapeShapes[0] > 2)
                    {
                        throw ParseException(fmt::format("Invalid input shape '{}' in Reshape layer '{}' {}. "
                                                         "When inferring during runtime, the parser only supports "
                                                         "shape (batch, -1) or (-1) for target shape input.",
                                                         reshapeShapes[0],
                                                         layerName,
                                                         CHECK_LOCATION().AsString()));
                    }

                    const int32_t numInputElements = inputTensorInfo.GetNumElements();
                    const int32_t inputTensorShape = inputTensorInfo.GetShape()[0];
                    if (reshapeShapes[0] == 1)
                    {
                        targetShape = {numInputElements};
                    }
                    else if (reshapeShapes[0] == 2)
                    {
                        targetShape = {inputTensorShape, numInputElements / inputTensorShape};
                    }
                }
                catch (const std::exception& exc)
                {
                    ARMNN_THROW_PARSE_EXCEPTION("Failed attempt to infer during runtime the target shape input for "
                                                "Reshape operation. Reshape operator target shape input buffer data "
                                                "is null. " << exc.what());
                }
            }
        }
        else
        {
            ARMNN_THROW_PARSE_EXCEPTION("Target shape not defined in reshape parameters or input tensor. "
                                        "At least one method required");
        }
    }

    armnn::TensorInfo reshapeOutputTensorInfo =
        TfLiteParserImpl::OutputShapeOfReshape(inputTensorInfo, targetShape);

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

void TfLiteParserImpl::ParseResizeBilinear(size_t subgraphIndex, size_t operatorIndex)
{
    ParseResize(subgraphIndex, operatorIndex, ResizeMethod::Bilinear);
}

void TfLiteParserImpl::ParseResizeNearestNeighbor(size_t subgraphIndex, size_t operatorIndex)
{
    ParseResize(subgraphIndex, operatorIndex, ResizeMethod::NearestNeighbor);
}

void TfLiteParserImpl::ParseResize(size_t subgraphIndex, size_t operatorIndex, ResizeMethod resizeMethod)
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

void TfLiteParserImpl::ParseConcatenation(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsConcatenationOptions();

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

void TfLiteParserImpl::ParseFullyConnected(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorRfr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
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

    armnn::IConnectableLayer* layer = nullptr;
    auto layerName = fmt::format("FullyConnected:{}:{}", subgraphIndex, operatorIndex);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    // Add the first input tensor to the registration list
    std::vector<unsigned int> tensorIndexesToRegister = {inputTensorIndexes[0]};
    std::vector<unsigned int> ignoreInputWhenRegister = {};

    desc.m_ConstantWeights = IsConstTensor(inputs[1]);

    // Add the weights input to the registration list, constant layers will be added by SetupConstantLayers if constant.
    tensorIndexesToRegister.emplace_back(inputTensorIndexes[1]);

    if (inputs.size() == 3)
    {
        desc.m_BiasEnabled = true;

        // Add the biases input to the registration list, constant layer will be added by SetupConstantLayers.
        tensorIndexesToRegister.emplace_back(inputTensorIndexes[2]);
    }

    // Filters and biases are always passed to fully connected as inputs
    layer = m_Network->AddFullyConnectedLayer(desc, layerName.c_str());

    ARMNN_ASSERT(layer != nullptr);
    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);

    unsigned int startingSlotIndex = 0;
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
        armnn::ReshapeDescriptor reshapeDescriptor;
        reshapeDescriptor.m_TargetShape = reshapedTensorInfo.GetShape();
        armnn::IConnectableLayer* reshapeLayer = m_Network->AddReshapeLayer(reshapeDescriptor, layerName.c_str());

        reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedTensorInfo);
        reshapeLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

        RegisterInputSlots(subgraphIndex, operatorIndex, reshapeLayer, {inputTensorIndexes[0]});
        // Fc layer connects to the reshape layer, so we skip the first input slot when registering fc's input slots
        tensorIndexesToRegister.erase(tensorIndexesToRegister.begin());
        startingSlotIndex = 1;
    }

    RegisterInputSlots(subgraphIndex, operatorIndex, layer, tensorIndexesToRegister, startingSlotIndex);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // we need to add the activation layer and fortunately we don't need to care about the data layout
    armnn::IConnectableLayer* fusedActivationLayer = AddFusedActivationLayer(layer, 0,
                                                                             options->fused_activation_function);

    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, fusedActivationLayer, {outputTensorIndexes[0]});
}

void TfLiteParserImpl::ParseDetectionPostProcess(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];

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
    auto anchorTensorAndData = CreateConstTensorNonPermuted(inputs[2], anchorTensorInfo);

    auto layerName = fmt::format("DetectionPostProcess:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddDetectionPostProcessLayer(desc, anchorTensorAndData,
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
void TfLiteParserImpl::ParsePack(size_t subgraphIndex, size_t operatorIndex)
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

void TfLiteParserImpl::ParseUnpack(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsUnpackOptions();

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

    std::vector<unsigned int> reshapeDims;
    for (unsigned int axis = 0; axis < splitOutShape.GetNumDimensions(); ++axis)
    {
        if (axis != unpackAxis)
        {
            reshapeDims.push_back(splitOutShape[axis]);
        }
    }

    TensorShape reshapeOutputShape(splitOutShape.GetNumDimensions() -1, reshapeDims.data());

    // Create reshape to remove the unpacked dimension for unpack operator of each output from Splitter.
    for (unsigned int k = 0; k < layer->GetNumOutputSlots(); ++k)
    {
        armnn::TensorInfo outputTensorInfo  = ToTensorInfo(outputs[k], true);
        std::string reshapeLayerName = fmt::format("Reshape_for:{}", layer->GetName());
        armnn::ReshapeDescriptor desc;
        desc.m_TargetShape = reshapeOutputShape;
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

void TfLiteParserImpl::ParseSplit(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsSplitOptions();

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

    armnn::TensorInfo inputTensorInfo = ToTensorInfo(inputs[1]);
    armnn::TensorInfo axisTensorInfo  = ToTensorInfo(inputs[0]);
    ARMNN_ASSERT(axisTensorInfo.GetNumElements() == 1);

    BufferRawPtr axisBufferPtr = GetBuffer(m_Model, inputs[0]->buffer);
    if (axisBufferPtr == nullptr)
    {
        throw ParseException(
                fmt::format("Operation has invalid inputs. Failed to read axis. {}",
                            CHECK_LOCATION().AsString()));
    }

    std::vector<int32_t> axisData(axisTensorInfo.GetNumElements());
    ::memcpy(axisData.data(), axisBufferPtr->data.data(), axisTensorInfo.GetNumBytes());
    int32_t axis = axisData[0];

    auto inputDimensions = static_cast<int32_t>(inputTensorInfo.GetNumDimensions());
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        // Square bracket denotes inclusive n while parenthesis denotes exclusive n
        // E.g. Rank 4 tensor can have axis in range [-4, 3)
        // -1 == 3, -2 == 2, -3 == 1, -4 == 0
        throw ParseException(
                fmt::format("Operation has invalid axis: {}. Axis must be in range [-n, n) {}",
                            axis,
                            CHECK_LOCATION().AsString()));
    }

    const unsigned int splitDim = armnnUtils::GetUnsignedAxis(inputTensorInfo.GetNumDimensions(), axis);

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

void TfLiteParserImpl::ParseSplitV(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsSplitVOptions();

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
    if (axisBufferPtr == nullptr)
    {
        throw ParseException(
                fmt::format("Operation has invalid inputs. Failed to read axis. {}",
                            CHECK_LOCATION().AsString()));
    }

    std::vector<int> axisData(axisTensorInfo.GetNumElements());
    ::memcpy(axisData.data(), axisBufferPtr->data.data(), axisTensorInfo.GetNumBytes());
    int32_t axis = axisData[0];

    auto inputDimensions = static_cast<int32_t>(inputTensorInfo.GetNumDimensions());
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        // Square bracket denotes inclusive n while parenthesis denotes exclusive n
        // E.g. Rank 4 tensor can have axis in range [-4, 3)
        // -1 == 3, -2 == 2, -3 == 1, -4 == 0
        throw ParseException(
                fmt::format("Operation has invalid axis: {}. Axis must be in range [-n, n) {}",
                            axis,
                            CHECK_LOCATION().AsString()));
    }
    const unsigned int splitDim = ComputeWrappedIndex(axis, inputTensorInfo.GetNumDimensions());

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

void TfLiteParserImpl::ParseArgMin(size_t subgraphIndex, size_t operatorIndex)
{
    ParseArgMinMax(subgraphIndex, operatorIndex, armnn::ArgMinMaxFunction::Min);
}

void TfLiteParserImpl::ParseArgMax(size_t subgraphIndex, size_t operatorIndex)
{
    ParseArgMinMax(subgraphIndex, operatorIndex, armnn::ArgMinMaxFunction::Max);
}

void TfLiteParserImpl::ParseArgMinMax(size_t subgraphIndex, size_t operatorIndex, ArgMinMaxFunction argMinMaxFunction)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo axisTensorInfo   = ToTensorInfo(inputs[1]);
    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    ARMNN_ASSERT(axisTensorInfo.GetNumElements() == 1);

    // Check if output tensor type is Signed32 or Signed64
    if (outputTensorInfo.GetDataType() != armnn::DataType::Signed32 &&
        outputTensorInfo.GetDataType() != armnn::DataType::Signed64)
    {
        throw ParseException(
                fmt::format(
                        "Output tensor data type is not supported. (Supported types: Signed32 & Signed64) {}",
                                CHECK_LOCATION().AsString()));
    }

    // Get const axis value from model and set it to descriptor.
    BufferRawPtr axisBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);
    if (axisBufferPtr == nullptr)
    {
        throw ParseException(
                fmt::format("Operation has invalid inputs. Failed to read axis. {}",
                            CHECK_LOCATION().AsString()));
    }

    std::vector<int32_t> axisData(axisTensorInfo.GetNumElements());
    ::memcpy(axisData.data(), axisBufferPtr->data.data(), axisTensorInfo.GetNumBytes());
    int32_t axis = axisData.front();

    auto inputDimensions = static_cast<int32_t>(inputTensorInfo.GetNumDimensions());
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        // Square bracket denotes inclusive n while parenthesis denotes exclusive n
        // E.g. Rank 4 tensor can have axis in range [-4, 3)
        // -1 == 3, -2 == 2, -3 == 1, -4 == 0
        throw ParseException(
                fmt::format("Operation has invalid axis: {}. Axis must be in range [-n, n) {}",
                                    axis,
                                    CHECK_LOCATION().AsString()));
    }

    ArgMinMaxDescriptor desc;
    desc.m_Axis = axis;
    desc.m_Function = argMinMaxFunction;

    // Register a ArgMin/ArgMax layer.
    auto layerName = argMinMaxFunction == ArgMinMaxFunction::Max ? "ArgMax:{}:{}" : "ArgMin:{}:{}";
    auto layerNameFormatted = fmt::format(layerName, subgraphIndex, operatorIndex);
    IConnectableLayer *layer = m_Network->AddArgMinMaxLayer(desc, layerNameFormatted.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Register input tensor to the layer.
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // Register output tensor to the layer.
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParserImpl::ParseGather(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    TfLiteParserImpl::TensorRawPtrVector inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);
    TfLiteParserImpl::TensorRawPtrVector outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::TensorInfo inputTensorInfo = ToTensorInfo(inputs[0]);
    armnn::TensorInfo indicesTensorInfo = ToTensorInfo(inputs[1]);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);

    armnn::GatherDescriptor gatherDescriptor;

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsGatherOptions();
    auto axis = options->axis;

    auto inputDimensions = static_cast<int32_t>(inputTensorInfo.GetNumDimensions());
    auto indicesDimensions = indicesTensorInfo.GetNumDimensions();
    auto outputDimensions = outputTensorInfo.GetNumDimensions();
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        throw ParseException(
            fmt::format("Operation has invalid axis: {} It is out of bounds [ -{}, {} ) {}",
                        axis,
                        inputDimensions, inputDimensions,
                        CHECK_LOCATION().AsString()));
    }
    if (outputDimensions != static_cast<unsigned int>(inputDimensions) + indicesDimensions - 1)
    {
        throw ParseException(
            fmt::format("Operation has invalid output dimensions: {} Output must be an ({} + {} - 1) -D tensor {}",
                        outputDimensions,
                        inputDimensions, indicesDimensions,
                        CHECK_LOCATION().AsString()));
    }

    gatherDescriptor.m_Axis = axis;

    auto layerName = fmt::format("Gather:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddGatherLayer(gatherDescriptor, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParserImpl::ParseDepthToSpace(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    TfLiteParserImpl::TensorRawPtrVector inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);
    TfLiteParserImpl::TensorRawPtrVector outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    armnn::DepthToSpaceDescriptor descriptor;

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsDepthToSpaceOptions();
    auto blockSize = options->block_size;
    if (blockSize < 2)
    {
        throw ParseException(
            fmt::format("Operation has invalid block size: {} Block size should be >= 2 {}",
                        blockSize,
                        CHECK_LOCATION().AsString()));
    }
    descriptor.m_BlockSize = armnn::numeric_cast<uint32_t>(blockSize);

    auto layerName = fmt::format("DepthToSpace:{}:{}", subgraphIndex, operatorIndex);
    IConnectableLayer* layer = m_Network->AddDepthToSpaceLayer(descriptor, layerName.c_str());
    ARMNN_ASSERT(layer != nullptr);
    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParserImpl::ParseSum(size_t subgraphIndex, size_t operatorIndex)
{
    ParseReduce(subgraphIndex, operatorIndex, armnn::ReduceOperation::Sum);
}

void TfLiteParserImpl::ParseReduceProd(size_t subgraphIndex, size_t operatorIndex)
{
    ParseReduce(subgraphIndex, operatorIndex, armnn::ReduceOperation::Prod);
}

void TfLiteParserImpl::ParseReduceMax(size_t subgraphIndex, size_t operatorIndex)
{
    ParseReduce(subgraphIndex, operatorIndex, armnn::ReduceOperation::Max);
}

void TfLiteParserImpl::ParseReduceMin(size_t subgraphIndex, size_t operatorIndex)
{
    ParseReduce(subgraphIndex, operatorIndex, armnn::ReduceOperation::Min);
}

void TfLiteParserImpl::ParseReduce(size_t subgraphIndex, size_t operatorIndex, ReduceOperation reduceOperation)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsReducerOptions();

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("Reduce:{}:{}", subgraphIndex, operatorIndex);

    armnn::TensorInfo inputTensorInfo0 = ToTensorInfo(inputs[0]);
    armnn::TensorInfo inputTensorInfo1 = ToTensorInfo(inputs[1]);

    ReduceDescriptor desc;
    BufferRawPtr axisBufferPtr = GetBuffer(m_Model, inputs[1]->buffer);
    // Get const axis value from model and set it to descriptor.
    if (axisBufferPtr != nullptr)
    {
        std::vector<int32_t> axisData(inputTensorInfo1.GetNumElements());
        ::memcpy(axisData.data(), axisBufferPtr->data.data(), inputTensorInfo1.GetNumBytes());

        // Convert the axis to unsigned int and remove duplicates.
        auto rank = static_cast<int32_t>(inputTensorInfo0.GetNumDimensions());
        std::set<unsigned int> uniqueAxis;
        std::transform(axisData.begin(),
                       axisData.end(),
                       std::inserter(uniqueAxis, uniqueAxis.begin()),
                       [rank](int i)->unsigned int{
                               return static_cast<uint32_t>(((i + rank) % rank)); });
        desc.m_vAxis.assign(uniqueAxis.begin(), uniqueAxis.end());
    }
    else
    {
        for (uint32_t i = 0; i < inputTensorInfo0.GetNumDimensions(); ++i)
        {
            desc.m_vAxis.push_back(i);
        }
    }

    desc.m_KeepDims        = options->keep_dims;
    desc.m_ReduceOperation = reduceOperation;

    // Register a new layer object, Sum.
    IConnectableLayer* layer = m_Network->AddReduceLayer(desc, layerName.c_str());

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Register input tensor to the layer.
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // Register output tensor to the layer.
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParserImpl::ParseAbs(size_t subgraphIndex, size_t operatorIndex)
{
    ParseElementwiseUnary(subgraphIndex, operatorIndex, armnn::UnaryOperation::Abs);
}

void TfLiteParserImpl::ParseExp(size_t subgraphIndex, size_t operatorIndex)
{
    ParseElementwiseUnary(subgraphIndex, operatorIndex, armnn::UnaryOperation::Exp);
}

void TfLiteParserImpl::ParseLocalResponseNormalization(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = fmt::format("LRN:{}:{}", subgraphIndex, operatorIndex);
    std::string layerNameFormatted = fmt::format(layerName, subgraphIndex, operatorIndex);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);

    const auto& operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto* options = operatorPtr->builtin_options.AsLocalResponseNormalizationOptions();

    armnn::NormalizationDescriptor descriptor;
    descriptor.m_DataLayout      = armnn::DataLayout::NHWC;
    descriptor.m_NormChannelType = armnn::NormalizationAlgorithmChannel::Across;
    descriptor.m_NormMethodType  = armnn::NormalizationAlgorithmMethod::LocalBrightness;
    descriptor.m_NormSize = static_cast<uint32_t>(options->radius);
    descriptor.m_K = options->bias;
    descriptor.m_Alpha = options->alpha;
    descriptor.m_Beta = options->beta;

    // ArmNN expects normSize to be the full size of the normalization
    // window rather than the radius as in TfLite.
    descriptor.m_NormSize = 1 + (2 * descriptor.m_NormSize);

    IConnectableLayer* layer = m_Network->AddNormalizationLayer(descriptor, layerNameFormatted.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParserImpl::ParseLogicalNot(size_t subgraphIndex, size_t operatorIndex)
{
    ParseElementwiseUnary(subgraphIndex, operatorIndex, armnn::UnaryOperation::LogicalNot);
}

void TfLiteParserImpl::ParseNeg(size_t subgraphIndex, size_t operatorIndex)
{
    ParseElementwiseUnary(subgraphIndex, operatorIndex, armnn::UnaryOperation::Neg);
}

void TfLiteParserImpl::ParseRsqrt(size_t subgraphIndex, size_t operatorIndex)
{
    ParseElementwiseUnary(subgraphIndex, operatorIndex, armnn::UnaryOperation::Rsqrt);
}

void TfLiteParserImpl::ParseElementwiseUnary(size_t subgraphIndex, size_t operatorIndex, UnaryOperation unaryOperation)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    std::string layerName = std::string(GetUnaryOperationAsCString(unaryOperation)) + ":{}:{}";
    std::string layerNameFormatted = fmt::format(layerName, subgraphIndex, operatorIndex);

    ElementwiseUnaryDescriptor desc;
    desc.m_Operation = unaryOperation;
    IConnectableLayer* layer = m_Network->AddElementwiseUnaryLayer(desc, layerNameFormatted.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, outputTensorIndexes);
}

void TfLiteParserImpl::ParseEqual(size_t subgraphIndex, size_t operatorIndex)
{
    ParseComparison(subgraphIndex, operatorIndex, armnn::ComparisonOperation::Equal);
}

void TfLiteParserImpl::ParseNotEqual(size_t subgraphIndex, size_t operatorIndex)
{
    ParseComparison(subgraphIndex, operatorIndex, armnn::ComparisonOperation::NotEqual);
}

void TfLiteParserImpl::ParseGreater(size_t subgraphIndex, size_t operatorIndex)
{
    ParseComparison(subgraphIndex, operatorIndex, armnn::ComparisonOperation::Greater);
}

void TfLiteParserImpl::ParseGreaterOrEqual(size_t subgraphIndex, size_t operatorIndex)
{
    ParseComparison(subgraphIndex, operatorIndex, armnn::ComparisonOperation::GreaterOrEqual);
}

void TfLiteParserImpl::ParseLess(size_t subgraphIndex, size_t operatorIndex)
{
    ParseComparison(subgraphIndex, operatorIndex, armnn::ComparisonOperation::Less);
}

void TfLiteParserImpl::ParseLessOrEqual(size_t subgraphIndex, size_t operatorIndex)
{
    ParseComparison(subgraphIndex, operatorIndex, armnn::ComparisonOperation::LessOrEqual);
}

void TfLiteParserImpl::ParseComparison(size_t subgraphIndex, size_t operatorIndex,
                                       ComparisonOperation comparisonOperation)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = std::string(GetComparisonOperationAsCString(comparisonOperation)) + ":{}:{}";
    std::string layerNameFormatted = fmt::format(layerName, subgraphIndex, operatorIndex);

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);
    CheckMatchingQuantization(inputTensorInfo, input1TensorInfo, layerNameFormatted, "Input 0", "Input 1");

    ComparisonDescriptor desc;
    desc.m_Operation = comparisonOperation;
    IConnectableLayer* layer = m_Network->AddComparisonLayer(desc, layerNameFormatted.c_str());
    ARMNN_ASSERT(layer != nullptr);

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0], true);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

armnn::IConnectableLayer* TfLiteParserImpl::AddFusedActivationLayer(armnn::IConnectableLayer* prevLayer,
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

TfLiteParserImpl::ModelPtr TfLiteParserImpl::LoadModelFromFile(const char* fileName)
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

TfLiteParserImpl::ModelPtr TfLiteParserImpl::LoadModelFromBinary(const uint8_t* binaryContent, size_t len)
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

TfLiteParserImpl::TensorRawPtrVector TfLiteParserImpl::GetInputs(const ModelPtr& model,
                                                                 size_t subgraphIndex,
                                                                 size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);

    const auto& subgraphPtr = model->subgraphs[subgraphIndex];
    const auto& operatorPtr = subgraphPtr->operators[operatorIndex];

    size_t inputCount = operatorPtr->inputs.size();
    TensorRawPtrVector result;
    for (size_t i = 0; i < inputCount; ++i)
    {
        // If the input location is -1 then assume input is turned off.
        if (operatorPtr->inputs[i] == -1)
        {
            continue;
        }
        else
        {
            uint32_t inputId = CHECKED_NON_NEGATIVE(operatorPtr->inputs[i]);
            result.push_back(subgraphPtr->tensors[inputId].get());
        }
    }
    return result;
}

TfLiteParserImpl::TensorRawPtrVector TfLiteParserImpl::GetOutputs(const ModelPtr& model,
                                                                  size_t subgraphIndex,
                                                                  size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);

    const auto& subgraphPtr = model->subgraphs[subgraphIndex];
    const auto& operatorPtr = subgraphPtr->operators[operatorIndex];

    size_t outputCount = operatorPtr->outputs.size();
    TensorRawPtrVector result(outputCount);
    for (size_t i = 0; i < outputCount; ++i)
    {
        uint32_t outputId = CHECKED_NON_NEGATIVE(operatorPtr->outputs[i]);
        CHECK_TENSOR(model, subgraphIndex, outputId);
        result[i] = subgraphPtr->tensors[outputId].get();
    }
    return result;
}

TfLiteParserImpl::TensorIdRawPtrVector TfLiteParserImpl::GetSubgraphInputs(const ModelPtr& model,
                                                                           size_t subgraphIndex)
{
    CHECK_SUBGRAPH(model, subgraphIndex);
    const auto& subgraphPtr = model->subgraphs[subgraphIndex];

    size_t inputCount = subgraphPtr->inputs.size();
    TensorIdRawPtrVector result(inputCount);
    for (size_t i = 0; i < inputCount; ++i)
    {
        uint32_t inputId = CHECKED_NON_NEGATIVE(subgraphPtr->inputs[i]);
        CHECK_TENSOR(model, subgraphIndex, inputId);
        result[i] = std::make_pair(inputId, subgraphPtr->tensors[inputId].get());
    }
    return result;
}

TfLiteParserImpl::TensorIdRawPtrVector TfLiteParserImpl::GetSubgraphOutputs(const ModelPtr& model,
                                                                            size_t subgraphIndex)
{
    CHECK_SUBGRAPH(model, subgraphIndex);
    const auto& subgraphPtr = model->subgraphs[subgraphIndex];

    size_t outputCount = subgraphPtr->outputs.size();
    TensorIdRawPtrVector result(outputCount);
    for (size_t i = 0; i < outputCount; ++i)
    {
        uint32_t outputId = CHECKED_NON_NEGATIVE(subgraphPtr->outputs[i]);
        result[i] = std::make_pair(outputId, subgraphPtr->tensors[outputId].get());
    }
    return result;
}

std::vector<int32_t>& TfLiteParserImpl::GetInputTensorIds(const ModelPtr& model,
                                                          size_t subgraphIndex,
                                                          size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);
    const auto& subgraphPtr = model->subgraphs[subgraphIndex];
    const auto& operatorPtr = subgraphPtr->operators[operatorIndex];
    return operatorPtr->inputs;
}

std::vector<int32_t>& TfLiteParserImpl::GetOutputTensorIds(const ModelPtr& model,
                                                           size_t subgraphIndex,
                                                           size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);
    const auto& subgraphPtr = model->subgraphs[subgraphIndex];
    const auto& operatorPtr = subgraphPtr->operators[operatorIndex];
    return operatorPtr->outputs;
}

void TfLiteParserImpl::RegisterInputSlots(size_t subgraphIndex,
                                          size_t operatorIndex,
                                          IConnectableLayer* layer,
                                          const std::vector<unsigned int>& tensorIndexes,
                                          unsigned int startingSlotIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    ARMNN_ASSERT(layer != nullptr);

    if (tensorIndexes.size() + startingSlotIndex != layer->GetNumInputSlots())
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

    for (unsigned int index = 0; index < tensorIndexes.size() ; ++index)
    {
        unsigned int tensorIndex = tensorIndexes[index];
        armnn::IInputSlot* slot = &(layer->GetInputSlot(startingSlotIndex + index));
        RegisterConsumerOfTensor(subgraphIndex, tensorIndex, slot);
    }
}

void TfLiteParserImpl::RegisterOutputSlots(size_t subgraphIndex,
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

void TfLiteParserImpl::SetupInputLayers(size_t subgraphIndex)
{
    CHECK_SUBGRAPH(m_Model, subgraphIndex);

    auto inputs = GetSubgraphInputs(m_Model, subgraphIndex);
    for (auto const& tensorIdAndPtr : inputs)
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

void TfLiteParserImpl::SetupOutputLayers(size_t subgraphIndex)
{
    CHECK_SUBGRAPH(m_Model, subgraphIndex);

    auto outputs = GetSubgraphOutputs(m_Model, subgraphIndex);
    for (auto const& tensorIdAndPtr : outputs)
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

void TfLiteParserImpl::SetupConstantLayers(size_t subgraphIndex)
{
    CHECK_SUBGRAPH(m_Model, subgraphIndex);

    const auto& subgraphPtr = m_Model->subgraphs[subgraphIndex];
    for (unsigned int subgraphIndex = 0; subgraphIndex < m_SubgraphConnections.size(); ++subgraphIndex)
    {
        for (unsigned int tensorIndex = 0; tensorIndex < m_SubgraphConnections[subgraphIndex].size(); ++tensorIndex)
        {
            if (m_SubgraphConnections[subgraphIndex][tensorIndex].outputSlot == nullptr &&
                m_SubgraphConnections[subgraphIndex][tensorIndex].inputSlots.size() > 0)
            {
                TensorRawPtr tensorPtr = subgraphPtr->tensors[tensorIndex].get();

                if(IsConstTensor(tensorPtr))
                {
                    armnn::TensorInfo tensorInfo = ToTensorInfo(tensorPtr);
                    auto tensorAndData = CreateConstTensorNonPermuted(tensorPtr, tensorInfo);

                    std::string layerName = fmt::format("Constant:{}", tensorPtr->name);
                    IConnectableLayer* layer = m_Network->AddConstantLayer(tensorAndData, layerName.c_str());

                    layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
                    RegisterOutputSlots(subgraphIndex,
                                        VIRTUAL_OPERATOR_ID,
                                        layer,
                                        { tensorIndex });
                }
                else
                {
                    throw ParseException(
                            fmt::format("Invalid Tensor: Tensor should be constant. {}",
                                        CHECK_LOCATION().AsString()));
                }
            }
        }
    }
}

// example usage: BufferRawPtr bufferPtr = GetBuffer(m_Model, inputs[0]->buffer);
TfLiteParserImpl::BufferRawPtr TfLiteParserImpl::GetBuffer(const ModelPtr& model, size_t bufferIndex)
{
    CHECK_BUFFER(model, bufferIndex);
    return model->buffers[bufferIndex].get();
}

template<typename T>
std::pair<armnn::ConstTensor, TfLiteParserImpl::SupportedDataStorage>
TfLiteParserImpl::CreateConstTensorAndStoreData(TfLiteParserImpl::BufferRawPtr bufferPtr,
                                            TfLiteParserImpl::TensorRawPtr tensorPtr,
                                            armnn::TensorInfo& tensorInfo,
                                            armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    // Make sure isConstant flag is set.
    tensorInfo.SetConstant();

    auto constData = CreateConstTensorImpl<T>(bufferPtr,
                                              tensorPtr,
                                              tensorInfo,
                                              permutationVector);
    TfLiteParserImpl::SupportedDataStorage storage(std::move(constData.second));
    return std::make_pair(constData.first, std::move(storage));
}

bool TfLiteParserImpl::IsConstTensor(TensorRawPtr tensorPtr)
{
    CHECK_TENSOR_PTR(tensorPtr);
    bool isConst = true;

    auto buffer = GetBuffer(m_Model, tensorPtr->buffer);
    if (buffer->data.size() == 0)
    {
        isConst = false;
    }

    return isConst;
}

std::pair<armnn::ConstTensor, TfLiteParserImpl::SupportedDataStorage>
TfLiteParserImpl::CreateConstTensorPermuted(TensorRawPtr tensorPtr,
                                            armnn::TensorInfo& tensorInfo,
                                            armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    CHECK_TENSOR_PTR(tensorPtr);
    auto bufferPtr = GetBuffer(m_Model, tensorPtr->buffer);
    CHECK_BUFFER_SIZE(bufferPtr, tensorInfo, tensorPtr->buffer);

    // Make sure isConstant flag is set.
    tensorInfo.SetConstant();

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

armnn::ConstTensor TfLiteParserImpl::CreateConstTensorNonPermuted(TensorRawPtr tensorPtr,
                                                                  armnn::TensorInfo& tensorInfo)
{
    CHECK_TENSOR_PTR(tensorPtr);
    auto bufferPtr = GetBuffer(m_Model, tensorPtr->buffer);
    CHECK_BUFFER_SIZE(bufferPtr, tensorInfo, tensorPtr->buffer);

    // Make sure isConstant flag is set.
    tensorInfo.SetConstant();

    return ConstTensor(tensorInfo, bufferPtr->data.data());
}

BindingPointInfo TfLiteParserImpl::GetNetworkInputBindingInfo(size_t subgraphId,
                                                              const std::string& name) const
{
    CHECK_SUBGRAPH(m_Model, subgraphId);
    auto inputs = GetSubgraphInputs(m_Model, subgraphId);
    for (auto const& input : inputs)
    {
        if (input.second->name == name)
        {
            auto bindingId = GenerateLayerBindingId(subgraphId, input.first);
            auto inputTensorInfo = ToTensorInfo(input.second);
            // Input tensors are always treated as constant tensors during network execution.
            inputTensorInfo.SetConstant(true);
            return std::make_pair(bindingId, inputTensorInfo);
        }
    }

    std::stringstream bindings;
    for (auto const& input : inputs)
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

BindingPointInfo TfLiteParserImpl::GetNetworkOutputBindingInfo(size_t subgraphId,
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
    for (auto const& output : outputs)
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

size_t TfLiteParserImpl::GetSubgraphCount() const
{
    return m_Model->subgraphs.size();
}

std::vector<std::string> TfLiteParserImpl::GetSubgraphInputTensorNames(size_t subgraphId) const
{
    CHECK_SUBGRAPH(m_Model, subgraphId);
    auto inputs = GetSubgraphInputs(m_Model, subgraphId);
    std::vector<std::string> result;
    result.reserve(inputs.size());
    for (auto const& input : inputs)
    {
        result.push_back(input.second->name);
    }
    return result;
}

std::vector<std::string> TfLiteParserImpl::GetSubgraphOutputTensorNames(size_t subgraphId) const
{
    CHECK_SUBGRAPH(m_Model, subgraphId);
    auto outputs = GetSubgraphOutputs(m_Model, subgraphId);
    std::vector<std::string> result;
    result.reserve(outputs.size());
    for (auto const& output : outputs)
    {
        result.push_back(output.second->name);
    }
    return result;
}

const std::string TfLiteParserImpl::GetVersion()
{
    return TFLITE_PARSER_VERSION;
}

TfLiteParserImpl::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<float[]>&& data)
: m_FloatData(std::move(data))
, m_Uint8Data(nullptr)
, m_Int8Data(nullptr)
, m_Int32Data(nullptr)
{
}

TfLiteParserImpl::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<uint8_t[]>&& data)
: m_FloatData(nullptr)
, m_Uint8Data(std::move(data))
, m_Int8Data(nullptr)
, m_Int32Data(nullptr)
{
}

TfLiteParserImpl::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<int8_t[]>&& data)
: m_FloatData(nullptr)
, m_Uint8Data(nullptr)
, m_Int8Data(std::move(data))
, m_Int32Data(nullptr)
{
}

TfLiteParserImpl::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<int32_t[]>&& data)
: m_FloatData(nullptr)
, m_Uint8Data(nullptr)
, m_Int8Data(nullptr)
, m_Int32Data(std::move(data))
{
}

} // armnnTfLiteParser
