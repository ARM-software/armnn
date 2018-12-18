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

#include <fstream>
#include <algorithm>
#include <limits>
#include <numeric>

using namespace armnn;
using armnn::CheckLocation;
namespace armnnTfLiteParser
{
namespace
{
const PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
const PermutationVector ArmNNToNHWC = { 0, 3, 1, 2 };

IConnectableLayer* SwizzleIn(INetwork& network,
                             IConnectableLayer* layer,
                             unsigned int inputSlotIndex,
                             const TensorInfo & inputInfo)
{
    BOOST_ASSERT(layer != nullptr);
    // Add swizzle layer
    std::stringstream name;
    name << "swizzle_for-" << layer->GetName() << ":in" << inputSlotIndex;
    IConnectableLayer* const swizzleLayer = network.AddPermuteLayer(NHWCToArmNN, name.str().c_str());
    // Set swizzled output shape
    const TensorInfo swizzleOutInfo = armnnUtils::Permuted(inputInfo, NHWCToArmNN);
    swizzleLayer->GetOutputSlot(0).SetTensorInfo(swizzleOutInfo);
    // Connect the swizzle layer to the actual layer
    swizzleLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(inputSlotIndex));

    return swizzleLayer;
}

IConnectableLayer* DeswizzleOut(INetwork& network,
                                IConnectableLayer* layer,
                                unsigned int outputSlotIndex,
                                const TensorInfo & outputInfo)
{
    BOOST_ASSERT(layer != nullptr);
    // Add deswizzle layer
    std::stringstream name;
    name << "deswizzle_for-" << layer->GetName() << ":out" << outputSlotIndex;
    IConnectableLayer* const deswizzleLayer = network.AddPermuteLayer(ArmNNToNHWC, name.str().c_str());
    // Set deswizzled output shape
    deswizzleLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    // Set original layer output shape
    const TensorInfo deswizzleOutInfo = armnnUtils::Permuted(outputInfo, NHWCToArmNN);
    layer->GetOutputSlot(outputSlotIndex).SetTensorInfo(deswizzleOutInfo);
    // Connect the actual layer to the deswizzle layer
    layer->GetOutputSlot(outputSlotIndex).Connect(deswizzleLayer->GetInputSlot(0));

    return deswizzleLayer;
}

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
                 uint32_t& paddingFront,
                 uint32_t& paddingBack,
                 tflite::Padding padding)
{
    paddingFront = 0;
    paddingBack = 0;
    if (padding == tflite::Padding_SAME)
    {
        uint32_t outputSize = (inputSize + stride - 1) / stride;
        uint32_t temp = (outputSize - 1) * stride + filterSize;
        if (temp > inputSize)
        {
            paddingFront = (temp - inputSize) / 2;
            paddingBack = (temp - inputSize) - paddingFront;
        }
    }
}

armnn::TensorInfo ToTensorInfo(TfLiteParser::TensorRawPtr tensorPtr)
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

    auto const & dimensions = AsUnsignedVector(tensorPtr->shape);

    // two statements (on purpose) for easier debugging:
    armnn::TensorInfo result(static_cast<unsigned int>(tensorPtr->shape.size()),
                             dimensions.data(),
                             type,
                             quantizationScale,
                             quantizationOffset);
    return result;
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

} // <anonymous>

TfLiteParser::TfLiteParser()
: m_Network(nullptr, nullptr)
, m_ParserFunctions(tflite::BuiltinOperator_MAX+1, &TfLiteParser::ParseUnsupportedOperator)
{
    // register supported operators
    m_ParserFunctions[tflite::BuiltinOperator_AVERAGE_POOL_2D]   =  &TfLiteParser::ParseAveragePool2D;
    m_ParserFunctions[tflite::BuiltinOperator_CONCATENATION]     =  &TfLiteParser::ParseConcatenation;
    m_ParserFunctions[tflite::BuiltinOperator_CONV_2D]           =  &TfLiteParser::ParseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_DEPTHWISE_CONV_2D] =  &TfLiteParser::ParseDepthwiseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_FULLY_CONNECTED]   =  &TfLiteParser::ParseFullyConnected;
    m_ParserFunctions[tflite::BuiltinOperator_MAX_POOL_2D]       =  &TfLiteParser::ParseMaxPool2D;
    m_ParserFunctions[tflite::BuiltinOperator_RELU]              =  &TfLiteParser::ParseRelu;
    m_ParserFunctions[tflite::BuiltinOperator_RELU6]             =  &TfLiteParser::ParseRelu6;
    m_ParserFunctions[tflite::BuiltinOperator_RESHAPE]           =  &TfLiteParser::ParseReshape;
    m_ParserFunctions[tflite::BuiltinOperator_SOFTMAX]           =  &TfLiteParser::ParseSoftmax;
    m_ParserFunctions[tflite::BuiltinOperator_SQUEEZE]           =  &TfLiteParser::ParseSqueeze;
    m_ParserFunctions[tflite::BuiltinOperator_ADD]               =  &TfLiteParser::ParseAdd;
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
    for (SubGraphPtr const & subgraph : m_Model->subgraphs)
    {
        m_SubgraphConnections.emplace_back(subgraph->tensors.size());

        size_t operatorIndex = 0;
        for (OperatorPtr const & op : subgraph->operators)
        {
            try
            {
                if (op->custom_options.size() > 0)
                {
                    throw ParseException(
                            boost::str(
                                    boost::format("Custom options for op: %1% is not supported. "
                                                  "It has %2% bytes of custom options. %3%") %
                                                  op->opcode_index %
                                                  op->custom_options.size() %
                                                  CHECK_LOCATION().AsString()));
                }

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

    CalcPadding(inputHeight, filterHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, options->padding);

    auto filterTensorAndData = CreateConstTensor(inputs[1],
                                                 filterTensorInfo,
                                                 armnn::Optional<armnn::PermutationVector&>());
    armnn::IConnectableLayer* layer;

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
                                                 biasTensorAndData.first,
                                                 layerName.c_str());
    }
    else
    {
        layer = m_Network->AddConvolution2dLayer(desc,
                                                 filterTensorAndData.first,
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
    // ACL only supports a depth (channel) multiplier of 1, it is not currently stored in the descriptor
    CHECK_VALID_SIZE(CHECKED_NON_NEGATIVE(options->depth_multiplier), 1);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2, 3);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

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

    CalcPadding(inputHeight, filterHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, options->padding);

    auto filterTensorAndData = CreateConstTensor(inputs[1], filterTensorInfo, permutationVector);
    armnn::IConnectableLayer* layer;
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
                                                          biasTensorAndData.first,
                                                          layerName.c_str());
    }
    else
    {
        layer = m_Network->AddDepthwiseConvolution2dLayer(desc,
                                                          filterTensorAndData.first,
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

void TfLiteParser::ParseAveragePool2D(size_t subgraphIndex, size_t operatorIndex)
{
    ParsePool(subgraphIndex, operatorIndex, PoolingAlgorithm::Average);
}

void TfLiteParser::ParseMaxPool2D(size_t subgraphIndex, size_t operatorIndex)
{
    ParsePool(subgraphIndex, operatorIndex, PoolingAlgorithm::Max);
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

    CalcPadding(inputHeight, desc.m_PoolHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, options->padding);
    CalcPadding(inputWidth, desc.m_PoolWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, options->padding);

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

void TfLiteParser::ParseAdd(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsAddOptions();

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = boost::str(boost::format("Add:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddAdditionLayer(layerName.c_str());

    TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0], inputTensorIndexes[1]});

    layer = AddFusedActivationLayer(layer, 0, options->fused_activation_function);

    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, layer, {outputTensorIndexes[0]});
}

void TfLiteParser::ParseRelu(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    boost::ignore_unused(operatorPtr);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = str(boost::format("Activation:RELU:%1%:%2%") % subgraphIndex % operatorIndex);
    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::ReLu;
    IConnectableLayer* const layer  =
        m_Network->AddActivationLayer(activationDesc, layerName.c_str());

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

void TfLiteParser::ParseRelu6(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    boost::ignore_unused(operatorPtr);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(inputs.size(), 1);

    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = str(boost::format("Activation:RELU6:%1%:%2%") % subgraphIndex % operatorIndex);
    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::BoundedReLu;
    activationDesc.m_A = 6.0f;
    activationDesc.m_B = 0.0f;
    IConnectableLayer* const layer  =
        m_Network->AddActivationLayer(activationDesc, layerName.c_str());

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
    if (inputs.size() > 1 && (options->new_shape != outputs[0]->shape))
    {
        std::stringstream ss;
        ss << "New shape defined in reshape parameters "
           << reshapeOutputTensorInfo.GetShape()
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

void TfLiteParser::ParseConcatenation(size_t subgraphIndex, size_t operatorIndex)
{
    CHECK_MODEL(m_Model, subgraphIndex, operatorIndex);

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsConcatenationOptions();

    CHECK_SUPPORTED_FUSED_ACTIVATION(options, subgraphIndex, operatorIndex);

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    unsigned int numInputs = static_cast<unsigned int>(inputs.size());
    unsigned int numConcatView = numInputs;

    OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatView), MaxNumOfTensorDimensions);
    std::vector<unsigned int>mergeDimSizes(MaxNumOfTensorDimensions, 0u);

    unsigned int mergeDim = 0;

    // This concatDim indicates the data format: 3 is the NHWC, 1 is the NCHW.
    // axis could also be negative numbers. Negative axis are interpreted as counting from the end of the rank,
    // i.e., axis + rank(values)-th dimension.
    int32_t inputRank = static_cast<int32_t>(ToTensorInfo(inputs[0]).GetNumDimensions());
    const unsigned int concatDimInput = static_cast<unsigned int>((inputRank + options->axis) % inputRank);

    // ArmNN supports concatenation along the channel dimension for data formats NHWC and NCHW.
    if (concatDimInput == 0 || concatDimInput == 2)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Dimension %1% for concatenation is not supported by Armnn. "
                    "Node %2%")
                % concatDimInput
                % CHECK_LOCATION().AsString()));
    }

    for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
    {
        TensorInfo inputTensorInfo = ToTensorInfo(inputs[viewIndex]);

        // process the input tensor info
        armnnUtils::ProcessConcatInputTensorInfo(inputTensorInfo, concatDescriptor,
                                                 concatDimInput, viewIndex, mergeDimSizes, mergeDim);
    }

    auto layerName = boost::str(boost::format("Concatenation:%1%:%2%") % subgraphIndex % operatorIndex);
    IConnectableLayer* layer = m_Network->AddMergerLayer(concatDescriptor, layerName.c_str());

    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    if (concatDimInput == 3)
    {
        // Adding Fused Activation Layer after this moment....
        for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
        {
            // add permute layers to swizzle the inputs
            armnn::TensorInfo inputTensorInfo = ToTensorInfo(inputs[viewIndex]);
            IConnectableLayer* const swizzleLayer = SwizzleIn(*m_Network, layer, viewIndex, inputTensorInfo);

            BOOST_ASSERT(swizzleLayer != nullptr);

            // register the input connection slots for the layer
            // only the tensors for the inputs are relevant, exclude the const tensors
            RegisterInputSlots(subgraphIndex, operatorIndex, swizzleLayer, {inputTensorIndexes[viewIndex]});
        }

        // add permute layer to deswizzle the output
        IConnectableLayer* const deswizzleLayer = DeswizzleOut(*m_Network, layer, 0, outputTensorInfo);

        // add fused activation layer after the trailing swizzle layer
        layer = AddFusedActivationLayer(deswizzleLayer, 0, options->fused_activation_function);
    }
    else
    {
        // set the layer output tensor info
        layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        // register the input connection slots for the layer, connections are made after all layers have been created
        // only the tensors for the inputs are relevant, exclude the const tensors
        RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes});
    }

    // register the output connection slots for the layer, connections are made after all layers have been created
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
    armnn::IConnectableLayer* layer;
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
                                                  biasTensorAndData.first,
                                                  layerName.c_str());
    }
    else
    {
        layer = m_Network->AddFullyConnectedLayer(desc,
                                                  filterTensorAndData.first,
                                                  layerName.c_str());
    }
    BOOST_ASSERT(layer != nullptr);

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // register the input connection slot for the layer
    // only the tensors for the inputs are relevant, exclude the const tensors
    auto inputTensorIndexes = AsUnsignedVector(GetInputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterInputSlots(subgraphIndex, operatorIndex, layer, {inputTensorIndexes[0]});

    // we need to add the activation layer and fortunately we don't need to care about the data layout
    armnn::IConnectableLayer* fusedActivationLayer = AddFusedActivationLayer(layer, 0,
                                                                             options->fused_activation_function);
    // register the output connection slots for the layer, connections are made after all layers have been created
    auto outputTensorIndexes = AsUnsignedVector(GetOutputTensorIds(m_Model, subgraphIndex, operatorIndex));
    RegisterOutputSlots(subgraphIndex, operatorIndex, fusedActivationLayer, {outputTensorIndexes[0]});
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

    const auto & subGraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subGraphPtr->operators[operatorIndex];

    size_t inputCount = operatorPtr->inputs.size();
    TensorRawPtrVector result(inputCount);
    for (size_t i=0; i<inputCount; ++i)
    {
        uint32_t inputId = CHECKED_NON_NEGATIVE(operatorPtr->inputs[i]);
        result[i] = subGraphPtr->tensors[inputId].get();
    }
    return result;
}

TfLiteParser::TensorRawPtrVector TfLiteParser::GetOutputs(const ModelPtr & model,
                                                          size_t subgraphIndex,
                                                          size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);

    const auto & subGraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subGraphPtr->operators[operatorIndex];

    size_t outputCount = operatorPtr->outputs.size();
    TensorRawPtrVector result(outputCount);
    for (size_t i=0; i<outputCount; ++i)
    {
        uint32_t outputId = CHECKED_NON_NEGATIVE(operatorPtr->outputs[i]);
        CHECK_TENSOR(model, subgraphIndex, outputId);
        result[i] = subGraphPtr->tensors[outputId].get();
    }
    return result;
}

TfLiteParser::TensorIdRawPtrVector TfLiteParser::GetSubgraphInputs(const ModelPtr & model,
                                                                   size_t subgraphIndex)
{
    CHECK_SUBGRAPH(model, subgraphIndex);
    const auto & subGraphPtr = model->subgraphs[subgraphIndex];

    size_t inputCount = subGraphPtr->inputs.size();
    TensorIdRawPtrVector result(inputCount);
    for (size_t i=0; i<inputCount; ++i)
    {
        uint32_t inputId = CHECKED_NON_NEGATIVE(subGraphPtr->inputs[i]);
        CHECK_TENSOR(model, subgraphIndex, inputId);
        result[i] = std::make_pair(inputId, subGraphPtr->tensors[inputId].get());
    }
    return result;
}

TfLiteParser::TensorIdRawPtrVector TfLiteParser::GetSubgraphOutputs(const ModelPtr & model,
                                                                    size_t subgraphIndex)
{
    CHECK_SUBGRAPH(model, subgraphIndex);
    const auto & subGraphPtr = model->subgraphs[subgraphIndex];

    size_t outputCount = subGraphPtr->outputs.size();
    TensorIdRawPtrVector result(outputCount);
    for (size_t i=0; i<outputCount; ++i)
    {
        uint32_t outputId = CHECKED_NON_NEGATIVE(subGraphPtr->outputs[i]);
        result[i] = std::make_pair(outputId, subGraphPtr->tensors[outputId].get());
    }
    return result;
}

std::vector<int32_t>& TfLiteParser::GetInputTensorIds(const ModelPtr& model,
                                                      size_t subgraphIndex,
                                                      size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);
    const auto & subGraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subGraphPtr->operators[operatorIndex];
    return operatorPtr->inputs;
}

std::vector<int32_t>& TfLiteParser::GetOutputTensorIds(const ModelPtr& model,
                                                       size_t subgraphIndex,
                                                       size_t operatorIndex)
{
    CHECK_MODEL(model, subgraphIndex, operatorIndex);
    const auto & subGraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subGraphPtr->operators[operatorIndex];
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
    for (auto const & output : outputs)
    {
        if (output.second->name == name)
        {
            auto bindingId = GenerateLayerBindingId(subgraphId, output.first);
            return std::make_pair(bindingId, ToTensorInfo(output.second));
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
