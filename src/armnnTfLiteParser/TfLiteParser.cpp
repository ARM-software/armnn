//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "TfLiteParser.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/TypesUtils.hpp>
#include <boost/filesystem.hpp>

// armnnUtils:
#include <ParserHelper.hpp>
#include <Permute.hpp>
#include <VerificationHelpers.hpp>

// The generated code based on the Tf Lite schema:
#include <schema_generated.h>

#include <boost/core/ignore_unused.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/format.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fstream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <flatbuffers/flexbuffers.h>

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
            boost::str(
                boost::format("%1% was called with invalid (null) model. "
                              "Possible reason is that the model is not yet loaded and Unpack(ed). "
                              "subgraph:%2% at %3%") %
                              location.m_Function %
                              subgraphIndex %
                              location.FileLine()));
    }
    else if (subgraphIndex >= model->subgraphs.size())
    {
        throw ParseException(
            boost::str(
                boost::format("%1% was called with an invalid subgraph index. "
                              "subgraph:%2% at %3%") %
                              location.m_Function %
                              subgraphIndex %
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
            boost::str(
                boost::format("%1% was called with invalid (null) model. "
                                "Possible reason is that the model is not yet loaded and Unpack(ed). "
                                "subgraph:%2% operator:%3% at %4%") %
                                location.m_Function %
                                subgraphIndex %
                                operatorIndex %
                                location.FileLine()));
    }
    else if (subgraphIndex >= model->subgraphs.size())
    {
        throw ParseException(
            boost::str(
                boost::format("%1% was called with an invalid subgraph index. "
                                "subgraph:%2% operator:%3% at %4%") %
                                location.m_Function %
                                subgraphIndex %
                                operatorIndex %
                                location.FileLine()));
    }
    else if (operatorIndex >= model->subgraphs[subgraphIndex]->operators.size() &&
             operatorIndex != VIRTUAL_OPERATOR_ID)
    {
        throw ParseException(
            boost::str(
                boost::format("%1% was called with an invalid operator index. "
                                "subgraph:%2% operator:%3% at %4%") %
                                location.m_Function %
                                subgraphIndex %
                                operatorIndex %
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
    BOOST_ASSERT_MSG(model.get() != nullptr, "Expecting a valid model in this function");

    // also subgraph index should be checked by CHECK_MODEL so
    // I only add an assert here
    BOOST_ASSERT_MSG(subgraphIndex < model->subgraphs.size(), "Expecting a valid subgraph index");

    // the tensor index is the only one to check here
    if (tensorIndex >= model->subgraphs[subgraphIndex]->tensors.size())
    {
        throw ParseException(
            boost::str(
                boost::format("%1% was called with an invalid tensor index. "
                                "subgraph:%2% tensor:%3% at %4%") %
                                location.m_Function %
                                subgraphIndex %
                                tensorIndex %
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
            boost::str(
                boost::format("%1% was called with a null tensor pointer. "
                              "at %2%") %
                              location.m_Function %
                              location.FileLine()));

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
            boost::str(
                boost::format("%1% was called with invalid (null) model. "
                              "Possible reason is that the model is not yet loaded and Unpack(ed). "
                              "buffer:%2% at %3%") %
                              location.m_Function %
                              bufferIndex %
                              location.FileLine()));
    }
    else if (bufferIndex >= model->buffers.size())
    {
        throw ParseException(
            boost::str(
                boost::format("%1% was called with an invalid buffer index. "
                              "buffer index:%2% at %3%") %
                              location.m_Function %
                              bufferIndex %
                              location.FileLine()));
    }
    else if (model->buffers[bufferIndex].get() == nullptr)
    {
        throw ParseException(
            boost::str(
                boost::format("The buffer #%1% is null. %3%") %
                              bufferIndex %
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
            boost::str(
                boost::format("BufferPtr is null for buffer:%1%. %2%") %
                              bufferId %
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
                boost::str( \
                    boost::format("TfLite parser doesn't suppport fused activation: " \
                                  "%1%/%2% in %3% subgraph:%4% operator:%5% at %6%") % \
                                  OPTION->fused_activation_function % \
                                  tflite::EnumNameActivationFunctionType(\
                                      OPTION->fused_activation_function) % \
                                  __func__ % \
                                  SUBGRAPH_INDEX % \
                                  OPERATOR_INDEX % \
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

armnn::TensorInfo ToTensorInfo(TfLiteParser::TensorRawPtr tensorPtr, const std::vector<unsigned int>& shapes)
{
    armnn::DataType type;
    CHECK_TENSOR_PTR(tensorPtr);

    switch (tensorPtr->type)
    {
        case tflite::TensorType_UINT8:
            type = armnn::DataType::QuantisedAsymm8;
            break;
        case tflite::TensorType_FLOAT32:
            type = armnn::DataType::Float32;
            break;
        case tflite::TensorType_INT32:
            type = armnn::DataType::Signed32;
            break;

        default:
        {
            CheckLocation location = CHECK_LOCATION();
            throw ParseException(
                boost::str(
                    boost::format("Unsupported data type %1% = %2% for tensor: %3%. %4%") %
                                  tensorPtr->type %
                                  tflite::EnumNameTensorType(tensorPtr->type) %
                                  tensorPtr->name %
                                  location.AsString()));
        }
    }

    float quantizationScale = 0.0f;
    int32_t quantizationOffset = 0;

    if (tensorPtr->quantization.get())
    {
        CHECK_VALID_SIZE(tensorPtr->quantization->scale.size(), 0, 1);
        CHECK_VALID_SIZE(tensorPtr->quantization->zero_point.size(), 0, 1);

        if (tensorPtr->quantization->scale.size() == 1)
        {
            quantizationScale = tensorPtr->quantization->scale[0];
        }
        if (tensorPtr->quantization->zero_point.size() == 1)
        {
            // NOTE: we lose precision here when converting from 64 bit to 32
            //       but this is what we support at the monent in ArmNN
            quantizationOffset = static_cast<int32_t>(tensorPtr->quantization->zero_point[0]);
        }
    }

    std::vector<unsigned int> safeShape = shapes;
    if (safeShape.size() == 0)
    {
        safeShape.push_back(1);
    }

    // two statements (on purpose) for easier debugging:
    armnn::TensorInfo result(static_cast<unsigned int>(safeShape.size()),
                             safeShape.data(),
                             type,
                             quantizationScale,
                             quantizationOffset);
    return result;
}

armnn::TensorInfo ToTensorInfo(TfLiteParser::TensorRawPtr tensorPtr)
{
    auto const & dimensions = AsUnsignedVector(tensorPtr->shape);
    return ToTensorInfo(tensorPtr, dimensions);
}

template<typename T>
std::pair<armnn::ConstTensor, std::unique_ptr<T[]>>
CreateConstTensorImpl(TfLiteParser::BufferRawPtr bufferPtr,
                      TfLiteParser::TensorRawPtr tensorPtr,
                      armnn::TensorInfo& tensorInfo,
                      armnn::Optional<armnn::PermutationVector&> permutationVector)
{
    BOOST_ASSERT_MSG(tensorPtr != nullptr, "tensorPtr is null");
    BOOST_ASSERT_MSG(bufferPtr != nullptr,
        boost::str(
            boost::format("Buffer for buffer:%1% is null") % tensorPtr->buffer).c_str());

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

} // <anonymous>

TfLiteParser::TfLiteParser()
: m_Network(nullptr, nullptr)
, m_ParserFunctions(tflite::BuiltinOperator_MAX+1, &TfLiteParser::ParseUnsupportedOperator)
{
    // register supported operators
    m_ParserFunctions[tflite::BuiltinOperator_AVERAGE_POOL_2D]   =  &TfLiteParser::ParseAveragePool2D;
    m_ParserFunctions[tflite::BuiltinOperator_BATCH_TO_SPACE_ND] =  &TfLiteParser::ParseBatchToSpaceND;
    m_ParserFunctions[tflite::BuiltinOperator_CONCATENATION]     =  &TfLiteParser::ParseConcatenation;
    m_ParserFunctions[tflite::BuiltinOperator_CONV_2D]           =  &TfLiteParser::ParseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_DEPTHWISE_CONV_2D] =  &TfLiteParser::ParseDepthwiseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_CUSTOM]            =  &TfLiteParser::ParseDetectionPostProcess;
    m_ParserFunctions[tflite::BuiltinOperator_FULLY_CONNECTED]   =  &TfLiteParser::ParseFullyConnected;
    m_ParserFunctions[tflite::BuiltinOperator_LOGISTIC]          =  &TfLiteParser::ParseLogistic;
    m_ParserFunctions[tflite::BuiltinOperator_L2_NORMALIZATION]  =  &TfLiteParser::ParseL2Normalization;
    m_ParserFunctions[tflite::BuiltinOperator_MAX_POOL_2D]       =  &TfLiteParser::ParseMaxPool2D;
    m_ParserFunctions[tflite::BuiltinOperator_MAXIMUM]           =  &TfLiteParser::ParseMaximum;
    m_ParserFunctions[tflite::BuiltinOperator_MINIMUM]           =  &TfLiteParser::ParseMinimum;
    m_ParserFunctions[tflite::BuiltinOperator_RELU]              =  &TfLiteParser::ParseRelu;
    m_ParserFunctions[tflite::BuiltinOperator_RELU6]             =  &TfLiteParser::ParseRelu6;
    m_ParserFunctions[tflite::BuiltinOperator_RESHAPE]           =  &TfLiteParser::ParseReshape;
    m_ParserFunctions[tflite::BuiltinOperator_RESIZE_BILINEAR]   =  &TfLiteParser::ParseResizeBilinear;
    m_ParserFunctions[tflite::BuiltinOperator_SOFTMAX]           =  &TfLiteParser::ParseSoftmax;
    m_ParserFunctions[tflite::BuiltinOperator_SPACE_TO_BATCH_ND] =  &TfLiteParser::ParseSpaceToBatchND;
    m_ParserFunctions[tflite::BuiltinOperator_SQUEEZE]           =  &TfLiteParser::ParseSqueeze;
    m_ParserFunctions[tflite::BuiltinOperator_STRIDED_SLICE]     =  &TfLiteParser::ParseStridedSlice;
    m_ParserFunctions[tflite::BuiltinOperator_SUB]               =  &TfLiteParser::ParseSub;
    m_ParserFunctions[tflite::BuiltinOperator_ADD]               =  &TfLiteParser::ParseAdd;
    m_ParserFunctions[tflite::BuiltinOperator_MUL]               =  &TfLiteParser::ParseMul;
    m_ParserFunctions[tflite::BuiltinOperator_MEAN]              =  &TfLiteParser::ParseMean;
    m_ParserFunctions[tflite::BuiltinOperator_PACK]              =  &TfLiteParser::ParsePack;
    m_ParserFunctions[tflite::BuiltinOperator_PAD]               =  &TfLiteParser::ParsePad;
    m_ParserFunctions[tflite::BuiltinOperator_SPLIT]             =  &TfLiteParser::ParseSplit;
    m_ParserFunctions[tflite::BuiltinOperator_TANH]              =  &TfLiteParser::ParseTanH;
    m_ParserFunctions[tflite::BuiltinOperator_TRANSPOSE_CONV]    =  &TfLiteParser::ParseTransposeConv;
    m_ParserFunctions[tflite::BuiltinOperator_UNPACK]            =  &TfLiteParser::ParseUnpack;
}

void TfLiteParser::ResetParser()
{
    m_Network = armnn::INetworkPtr(nullptr, nullptr);
    m_Model = nullptr;
    m_SubgraphConnections.clear();
}

void TfLiteParser::AddBroadcastReshapeLayer(size_t subgraphIndex,
                                            size_t operatorIndex,
                                            IConnectableLayer *layer)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    BOOST_ASSERT(layer != nullptr);

    const auto & subgraphPtr = m_Model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subgraphPtr->operators[operatorIndex];

    BOOST_ASSERT(operatorPtr->inputs.size() > 1);

    uint32_t reshapedInputId = CHECKED_NON_NEGATIVE(operatorPtr->inputs[0]);
    TensorRawPtr tensorPtr = subgraphPtr->tensors[reshapedInputId].get();
    uint32_t inputId = CHECKED_NON_NEGATIVE(operatorPtr->inputs[1]);
    TensorRawPtr tensorPtr1 = subgraphPtr->tensors[inputId].get();

    armnn::TensorInfo reshapedTensorInfo = ToTensorInfo(tensorPtr);
    armnn::TensorInfo inputTensorInfo    = ToTensorInfo(tensorPtr1);

    if (inputTensorInfo.GetNumDimensions() < reshapedTensorInfo.GetNumDimensions())
    {
        uint32_t id = reshapedInputId;
        reshapedInputId = inputId;
        inputId = id;

        reshapedTensorInfo = ToTensorInfo(tensorPtr1);
        inputTensorInfo = ToTensorInfo(tensorPtr);
    }

    uint32_t numDimensions = inputTensorInfo.GetNumDimensions();

    std::vector<unsigned> reshapedDim;
    for (unsigned int i = 0; i < reshapedTensorInfo.GetNumDimensions(); ++i)
    {
        reshapedDim.push_back(reshapedTensorInfo.GetShape()[i]);
    }

    std::vector<unsigned int> reshapedDimensions(numDimensions, 1);
    std::copy_backward (reshapedDim.begin(), reshapedDim.end(), reshapedDimensions.end());

    reshapedTensorInfo.SetShape(armnn::TensorShape{ numDimensions, reshapedDimensions.data() });

    std::string layerName = boost::str(boost::format("Reshape_for:%1%") % layer->GetName());
    armnn::ReshapeDescriptor desc;
    desc.m_TargetShape = reshapedTensorInfo.GetShape();
    armnn::IConnectableLayer* reshapeLayer = m_Network->AddReshapeLayer(desc, layerName.c_str());

    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedTensorInfo);
    reshapeLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

    RegisterInputSlots(subgraphIndex, operatorIndex, reshapeLayer, {reshapedInputId});

    armnn::IInputSlot* input1Slot = &(layer->GetInputSlot(1));
    RegisterConsumerOfTensor(subgraphIndex, inputId, input1Slot);
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
    m_Network = INetwork::Create();
    BOOST_ASSERT(m_Model.get() != nullptr);

    bool failedToCreate = false;
    std::stringstream errors;

    if (m_Model->subgraphs.size() != 1)
    {
        throw ParseException(
                boost::str(
                        boost::format("Current TfLite parser only supports 1 subgraph. Current one has: %1% %2%") %
                        m_Model->subgraphs.size() %
                        CHECK_LOCATION().AsString()));
    }

    size_t subgraphIndex = 0;
    for (SubgraphPtr const & subgraph : m_Model->subgraphs)
    {
        m_SubgraphConnections.emplace_back(subgraph->tensors.size());

        size_t operatorIndex = 0;
        for (OperatorPtr const & op : subgraph->operators)
        {
            try
            {
                auto const & opCodePtr = m_Model->operator_codes[op->opcode_index];
                auto builtinCode = opCodePtr->builtin_code;

                if (builtinCode > tflite::BuiltinOperator_MAX)
                {
                    throw ParseException(
                            boost::str(
                                    boost::format("Operator code %1% is out of range 0-%2%. "
                                                  "subgraph:%3% operator idx:%4%. %5%") %
                                                  builtinCode %
                                                  tflite::BuiltinOperator_MAX %
                                                  subgraphIndex %
                                                  operatorIndex %
                                                  CHECK_LOCATION().AsString()));
                }

                // lookup and call the parser function
                auto & parserFunction = m_ParserFunctions[builtinCode];
                (this->*parserFunction)(subgraphIndex, operatorIndex);
            }
            catch (const ParseException& e)
            {
                failedToCreate = true;
                std::stringstream errorString;

                errorString << "Failed to parse operator #" << operatorIndex
                            << " within subgraph #" << subgraphIndex
                            << " error: " << e.what();
                BOOST_LOG_TRIVIAL(error) << errorString.str();

                errors << errorString.str() << "\n";
            }
            ++operatorIndex;
        }

        SetupInputLayers(subgraphIndex);
        SetupOutputLayers(subgraphIndex);
        SetupConstantLayers(subgraphIndex);

        ++subgraphIndex;
    }

    if (failedToCreate)
    {
        // we can skip everything and let the outer exception handler deal with the error
        throw ParseException(errors.str());
    }

    // establish the connections from the layer outputs to the inputs of the subsequent layers
    for (size_t subgraphIndex = 0; subgraphIndex < m_SubgraphConnections.size(); ++subgraphIndex)
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
    BOOST_ASSERT(m_SubgraphConnections.size() > subgraphIndex);
    BOOST_ASSERT(m_SubgraphConnections[subgraphIndex].size() > tensorIndex);

    TensorSlots & tensorSlots = m_SubgraphConnections[subgraphIndex][tensorIndex];

    // assuming there is only one producer for that tensor
    if (tensorSlots.outputSlot != nullptr)
    {
        throw ParseException(boost::str(
                boost::format("Another layer has already registered itself as the producer of "
                              "subgraph:%1% tensor:%2% %3%") %
                               subgraphIndex %
                               tensorIndex %
                               CHECK_LOCATION().AsString()));
    }

    tensorSlots.outputSlot = slot;
}

void TfLiteParser::RegisterConsumerOfTensor(size_t subgraphIndex,
                                            size_t tensorIndex,
                                            armnn::IInputSlot* slot)
{
    CHECK_TENSOR(m_Model, subgraphIndex, tensorIndex);
    BOOST_ASSERT(m_SubgraphConnections.size() > subgraphIndex);
    BOOST_ASSERT(m_SubgraphConnections[subgraphIndex].size() > tensorIndex);

    TensorSlots & tensorSlots = m_SubgraphConnections[subgraphIndex][tensorIndex];
    tensorSlots.inputSlots.push_back(slot);
}

void TfLiteParser::ParseUnsupportedOperator(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    //
    auto opcodeIndex = operatorPtr->opcode_index;
    auto opcode = m_Model->operator_codes[opcodeIndex]->builtin_code;

    throw ParseException(
        boost::str(
            boost::format("Operator not supported. "
                          "subgraph:%1% operator:%2% "
                          "opcode_index:%3% opcode:%4% / %5% %6%") %
                          subgraphIndex %
                          operatorIndex %
                          opcodeIndex %
                          opcode %
                          tflite::EnumNameBuiltinOperator(opcode) %
                          CHECK_LOCATION().AsString()));
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

    auto layerName = boost::str(boost::format("Conv2D:%1%:%2%") % subgraphIndex % operatorIndex);

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

    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo filterTensorInfo = ToTensorInfo(inputs[1]);

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

    // Mappings from TensorflowLite filter tensors to the ArmNN filter tensors (ArmNN weights have to be [M, I, H, W])
    PermutationVector permutationVector{ 2, 3, 1, 0 }; // [H, W, I, M] -> [M, I, H, W]

    CalcPadding(inputHeight, filterHeight, desc.m_StrideY,
                desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, desc.m_StrideX,
                desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, options->padding);

    auto filterTensorAndData = CreateConstTensor(inputs[1], filterTensorInfo, permutationVector);
    armnn::IConnectableLayer* layer = nullptr;
    auto layerName = boost::str(boost::format("DepthwiseConv2D:%1%:%2%") % subgraphIndex % operatorIndex);

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
    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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
    auto layerName = boost::str(boost::format("TransposeConv:%1%:%2%") % subgraphIndex % operatorIndex);

    layer = m_Network->AddTransposeConvolution2dLayer(desc,
                                                      filterTensorAndData.first,
                                                      EmptyOptional(),
                                                      layerName.c_str());

    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);

    auto layerName = boost::str(boost::format("BatchToSpaceND:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddBatchToSpaceNdLayer(desc, layerName.c_str());

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
    auto layerName = boost::str(boost::format("L2Normalization:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddL2NormalizationLayer(desc, layerName.c_str());

    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);

    auto layerName = boost::str(boost::format("Maximum:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddMaximumLayer(layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    if (inputTensorInfo.GetNumDimensions() != input1TensorInfo.GetNumDimensions())
    {
        AddBroadcastReshapeLayer(subgraphIndex, operatorIndex, layer);
    }
    else
    {
        RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});
    }

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

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo input1TensorInfo = ToTensorInfo(inputs[1]);

    auto layerName = boost::str(boost::format("Minimum:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddMinimumLayer(layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    if (inputTensorInfo.GetNumDimensions() != input1TensorInfo.GetNumDimensions())
    {
        AddBroadcastReshapeLayer(subgraphIndex, operatorIndex, layer);
    }
    else
    {
        RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});
    }

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
                boost::str(boost::format("AveragePool2D:%1%:%2%") % subgraphIndex % operatorIndex);
            break;
        case PoolingAlgorithm::Max:
            layerName =
                boost::str(boost::format("MaxPool2D:%1%:%2%") % subgraphIndex % operatorIndex);
            break;
        default:
            BOOST_ASSERT_MSG(false, "Unsupported Pooling Algorithm");
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

    IConnectableLayer* layer = m_Network->AddPooling2dLayer(desc, layerName.c_str());

    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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

    auto layerName = boost::str(boost::format("Softmax:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* const layer = m_Network->AddSoftmaxLayer(desc, layerName.c_str());

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);

    auto layerName = boost::str(boost::format("SpaceToBatchND:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddSpaceToBatchNdLayer(desc, layerName.c_str());

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

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo outputTensorInfo =
        TfLiteParser::OutputShapeOfSqueeze(AsUnsignedVector(options->squeeze_dims),
                                           inputTensorInfo);

    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputTensorInfo.GetShape();

    auto layerName = boost::str(boost::format("Squeeze:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
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

    auto layerName = boost::str(boost::format("StridedSlice:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddStridedSliceLayer(desc, layerName.c_str());

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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

    auto layerName = boost::str(boost::format("Sub:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddSubtractionLayer(layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    if (inputTensorInfo.GetNumDimensions() != input1TensorInfo.GetNumDimensions())
    {
        AddBroadcastReshapeLayer(subgraphIndex, operatorIndex, layer);
    }
    else
    {
        RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});
    }

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

    auto layerName = boost::str(boost::format("Add:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddAdditionLayer(layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    if (inputTensorInfo.GetNumDimensions() != input1TensorInfo.GetNumDimensions())
    {
        AddBroadcastReshapeLayer(subgraphIndex, operatorIndex, layer);
    }
    else
    {
        RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});
    }

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

    auto layerName = boost::str(boost::format("Mul:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddMultiplicationLayer(layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    if (inputTensorInfo.GetNumDimensions() != input1TensorInfo.GetNumDimensions())
    {
        AddBroadcastReshapeLayer(subgraphIndex, operatorIndex, layer);
    }
    else
    {
        RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});
    }

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
    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);

    desc.m_KeepDims =
        inputTensorInfo.GetNumDimensions() == outputTensorInfo.GetNumDimensions() ?
            true : false;

    auto layerName = boost::str(boost::format("Mean:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddMeanLayer(desc, layerName.c_str());

    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
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

    auto layerName = boost::str(boost::format("Pad:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddPadLayer(desc, layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}


void TfLiteParser::ParseRelu(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex, ActivationFunction::ReLu);
}

void TfLiteParser::ParseRelu6(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex, ActivationFunction::BoundedReLu);
}

void TfLiteParser::ParseLogistic(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex,ActivationFunction::Sigmoid);
}

void TfLiteParser::ParseTanH(size_t subgraphIndex, size_t operatorIndex)
{
    ParseActivation(subgraphIndex,operatorIndex,ActivationFunction::TanH);
}


void TfLiteParser::ParseActivation(size_t subgraphIndex, size_t operatorIndex, ActivationFunction activationType)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    boost::ignore_unused(operatorPtr);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = str(boost::format("Activation:"));
    ActivationDescriptor activationDesc;
    activationDesc.m_Function = activationType;

    switch (activationType)
    {
        case ActivationFunction::ReLu:
        {
            layerName += str(boost::format("RELU:%1%:%2%") % subgraphIndex % operatorIndex);
            break;
        }
        case ActivationFunction::BoundedReLu:
        {
            layerName += str(boost::format("RELU6:%1%:%2%") % subgraphIndex % operatorIndex);
            activationDesc.m_A = 6.0f;
            activationDesc.m_B = 0.0f;
            break;
        }
        case ActivationFunction::Sigmoid:
        {
            layerName += str(boost::format("SIGMOID:%1%:%2%") % subgraphIndex % operatorIndex);
            break;
        }
        case ActivationFunction::TanH:
        {
            layerName += str(boost::format("TANH:%1%:%2%") % subgraphIndex % operatorIndex);
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            break;
        }
        default:
        {
            throw ParseException(
                boost::str(boost::format("Unexpected ActivationFunction[%1%] when creating layerName "
                                         " %2% ") %static_cast<int>(activationType)% CHECK_LOCATION().AsString()));
        }
    }

    IConnectableLayer* const layer = m_Network->AddActivationLayer(activationDesc, layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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
                boost::str(
                    boost::format("At most one component of shape can be -1 %1%") % CHECK_LOCATION().AsString()));
        }

        auto targetNumElements =
            boost::numeric_cast<unsigned int>(
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

    armnn::TensorInfo inputTensorInfo  = ToTensorInfo(inputs[0]);
    armnn::TensorInfo actualOutputTensorInfo  = ToTensorInfo(outputs[0]);
    armnn::TensorInfo reshapeOutputTensorInfo =
        TfLiteParser::OutputShapeOfReshape(inputTensorInfo, options->new_shape);

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

    auto layerName = boost::str(boost::format("Reshape:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
    layer->GetOutputSlot(0).SetTensorInfo(reshapeOutputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseResizeBilinear(size_t subgraphIndex, size_t operatorIndex)
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
    desc.m_Method       = armnn::ResizeMethod::Bilinear;
    desc.m_TargetHeight = static_cast<uint32_t> (sizeTensorData[0]);
    desc.m_TargetWidth  = static_cast<uint32_t> (sizeTensorData[1]);
    desc.m_DataLayout   = armnn::DataLayout::NHWC;

    auto layerName = boost::str(boost::format("ResizeBilinear:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddResizeLayer(desc, layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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

    auto layerName = boost::str(boost::format("Concatenation:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddConcatLayer(concatDescriptor, layerName.c_str());

    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));

    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

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
            boost::str(
                boost::format(
                    "Dimension %1% for Fully Connected weights is not supported by Armnn. "
                    "Node %2%")
                % weightsDimension
                % CHECK_LOCATION().AsString()));
    }

    auto filterTensorAndData = CreateConstTensor(inputs[1],
                                                 filterTensorInfo,
                                                 armnn::Optional<armnn::PermutationVector&>());
    armnn::IConnectableLayer* layer = nullptr;
    auto layerName = boost::str(boost::format("FullyConnected:%1%:%2%") % subgraphIndex % operatorIndex);

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
    BOOST_ASSERT(layer != nullptr);

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
                    boost::str(
                            boost::format(
                                    "Failed to deduce input tensor shape from filter size %1%")
                            % reshapedDimensions[1]
                            % CHECK_LOCATION().AsString()));
        }

        armnn::TensorInfo reshapedTensorInfo = ToTensorInfo(inputs[0]);
        reshapedTensorInfo.SetShape(armnn::TensorShape{ 2, reshapedDimensions.data() });

        std::string reshapeLayerName = boost::str(boost::format("Reshape_for:%1%") % layer->GetName());
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

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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

    auto layerName = boost::str(boost::format("DetectionPostProcess:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddDetectionPostProcessLayer(desc, anchorTensorAndData.first,
                                                                       layerName.c_str());

    BOOST_ASSERT(layer != nullptr);

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

    auto layerName = boost::str(boost::format("Pack:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddStackLayer(desc, layerName.c_str());

    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
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
                boost::str(
                        boost::format(
                                "The unpack axis: %1% cannot be greater than or equal to "
                                "the number of input dimension %2% %3%")
                        % unpackAxis
                        % inputTensorInfo.GetNumDimensions()
                        % CHECK_LOCATION().AsString()));
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

    auto layerName = boost::str(boost::format("Unpack:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddSplitterLayer(splitDesc, layerName.c_str());

    TensorShape splitOutShape = TensorShape(static_cast<unsigned int>(unpackDimSizes.size()),
                                            unpackDimSizes.data());

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // Reshape to remove unpacked dimension
    unsigned int reshapedNumDimensions = inputDimSize - 1;
    std::vector<unsigned int> reshapedDimensions(reshapedNumDimensions);

    unsigned int reshapeIndex = 0;
    for (unsigned int i = 0; i < inputDimSize; ++i)
    {
        if (i == unpackAxis)
        {
            continue;
        }
        reshapedDimensions[reshapeIndex++] = unpackDimSizes[i];
    }

    // Create reshape to remove the unpacked dimension for unpack operator of each output from Splitter.
    for (unsigned int k = 0; k < layer->GetNumOutputSlots(); ++k)
    {
        armnn::TensorInfo reshapedTensorInfo = inputTensorInfo;
        reshapedTensorInfo.SetShape(armnn::TensorShape{ reshapedNumDimensions, reshapedDimensions.data() });

        std::string reshapeLayerName = boost::str(boost::format("Reshape_for:%1%") % layer->GetName());
        armnn::ReshapeDescriptor desc;
        desc.m_TargetShape = reshapedTensorInfo.GetShape();
        armnn::IConnectableLayer* reshapeLayer = m_Network->AddReshapeLayer(desc, layerName.c_str());

        layer->GetOutputSlot(k).SetTensorInfo(armnn::TensorInfo(splitOutShape, inputTensorInfo.GetDataType()));
        layer->GetOutputSlot(k).Connect(reshapeLayer->GetInputSlot(0));

        reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedTensorInfo);

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

    BOOST_ASSERT(axisTensorInfo.GetNumElements() == 1);
    const unsigned int splitDim = axisData[0];

    // Armnn supports split along the channel dimension for data formats NHWC and NCHW.
    if (splitDim == 0 || splitDim == 2)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Dimension %1% for split is not supported by Armnn. %2%")
                    % splitDim
                    % CHECK_LOCATION().AsString()));
    }

    auto inputDimSize = inputTensorInfo.GetNumDimensions();
    if (inputDimSize > MaxNumOfTensorDimensions)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "The number of dimensions: %1% for input tensors of the "
                    "split op cannot be greater than %2% %3%")
                    % inputTensorInfo.GetNumDimensions()
                    % MaxNumOfTensorDimensions
                    % CHECK_LOCATION().AsString()));
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

    auto layerName = boost::str(boost::format("Split:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddSplitterLayer(splitDesc, layerName.c_str());

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[1]});

    TensorShape outShape = TensorShape(static_cast<unsigned int>(splitterDimSizes.size()),
                                       splitterDimSizes.data());

    for (unsigned int k = 0; k < layer->GetNumOutputSlots(); ++k)
    {
        layer->GetOutputSlot(k).SetTensorInfo(armnn::TensorInfo(outShape,
                                                                inputTensorInfo.GetDataType()));
    }

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
                boost::str(
                    boost::format("TfLite parser doesn't suppport fused activation: "
                                  "%1%/%2% %3% ") %
                                  activationType %
                                  tflite::EnumNameActivationFunctionType(activationType) %
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
        throw InvalidArgumentException(boost::str(boost::format("Invalid (null) file name %1%") %
                                       CHECK_LOCATION().AsString()));
    }
    boost::system::error_code errorCode;
    boost::filesystem::path pathToFile(fileName);
    if (!boost::filesystem::exists(pathToFile, errorCode))
    {
        throw FileNotFoundException(boost::str(boost::format("Cannot find the file (%1%) errorCode: %2% %3%") %
                                    fileName %
                                    errorCode %
                                    CHECK_LOCATION().AsString()));
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
        throw InvalidArgumentException(boost::str(boost::format("Invalid (null) binary content %1%") %
                                       CHECK_LOCATION().AsString()));
     }
    flatbuffers::Verifier verifier(binaryContent, len);
    if (verifier.VerifyBuffer<tflite::Model>() == false)
    {
        throw ParseException(
            boost::str(boost::format("Buffer doesn't conform to the expected Tensorflow Lite "
                                     "flatbuffers format. size:%1% %2%") %
                       len %
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
    BOOST_ASSERT(layer != nullptr);
    if (tensorIndexes.size() != layer->GetNumInputSlots())
    {
        throw ParseException(
            boost::str(boost::format("The number of tensor inputs (%1%) does not match the number expected (%2%)"
                                     " for subgraph:%3% operator index:%4% %5%") %
                       tensorIndexes.size() %
                       layer->GetNumInputSlots() %
                       subgraphIndex %
                       operatorIndex %
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
    BOOST_ASSERT(layer != nullptr);
    if (tensorIndexes.size() != layer->GetNumOutputSlots())
    {
        throw ParseException(
            boost::str(boost::format("The number of tensor outputs (%1%) does not match the number expected (%2%)"
                                     " for subgraph:%3% operator index:%4% %5%") %
                       tensorIndexes.size() %
                       layer->GetNumOutputSlots() %
                       subgraphIndex %
                       operatorIndex %
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

                std::string layerName = boost::str(boost::format("Constant:%1%") % tensorPtr->name);
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
        case armnn::DataType::QuantisedAsymm8:
            return CreateConstTensorAndStoreData<uint8_t>(bufferPtr,
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
        boost::str(
            boost::format("No input binding found for subgraph:%1% and name:%2%. "
                          "Possible inputs are: [%3%] %4%") %
            subgraphId %
            name %
            bindings.str() %
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
        boost::str(
            boost::format("No output binding found for subgraph:%1% and name:%2%. "
                          "Possible outputs are: [%3%] %4%") %
            subgraphId %
            name %
            bindings.str() %
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

ITfLiteParser* ITfLiteParser::CreateRaw()
{
    return new TfLiteParser();
}

ITfLiteParserPtr ITfLiteParser::Create()
{
    return ITfLiteParserPtr(CreateRaw(), &ITfLiteParser::Destroy);
}

void ITfLiteParser::Destroy(ITfLiteParser* parser)
{
    delete parser;
}

TfLiteParser::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<float[]> && data)
: m_FloatData(std::move(data))
, m_Uint8Data(nullptr)
, m_Int32Data(nullptr)
{
}

TfLiteParser::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<uint8_t[]> && data)
: m_FloatData(nullptr)
, m_Uint8Data(std::move(data))
, m_Int32Data(nullptr)
{
}

TfLiteParser::SupportedDataStorage::SupportedDataStorage(std::unique_ptr<int32_t[]> && data)
: m_FloatData(nullptr)
, m_Uint8Data(nullptr)
, m_Int32Data(std::move(data))
{
}

} // armnnTfLiteParser
