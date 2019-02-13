//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DeserializeParser.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Exceptions.hpp>

#include <ParserHelper.hpp>
#include <Permute.hpp>
#include <VerificationHelpers.hpp>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>

// The generated code based on the Serialize schema:
#include <Schema_generated.h>

#include <fstream>

using armnn::ParseException;
using namespace armnn;
using namespace armnn::armnnSerializer;

namespace armnnDeserializeParser {

namespace {

const uint32_t VIRTUAL_LAYER_ID = std::numeric_limits<uint32_t>::max();

 void CheckGraph(const DeserializeParser::GraphPtr& graph,
                 unsigned int layersIndex,
                 const CheckLocation& location)
{
    if (graph->layers() == nullptr)
    {
        throw ParseException(
                boost::str(
                        boost::format("%1% was called with invalid (null) graph. "
                                      "Possible reason is that the graph is not yet loaded and Unpack(ed). "
                                      "layers:%2% at %3%") %
                        location.m_Function %
                        layersIndex %
                        location.FileLine()));
    }
    else if (layersIndex >= graph->layers()->size())
    {
        throw ParseException(
                boost::str(
                        boost::format("%1% was called with an invalid layers index. "
                                      "layers:%2% at %3%") %
                        location.m_Function %
                        layersIndex %
                        location.FileLine()));
    }
}

void CheckLayers(const DeserializeParser::GraphPtr& graph,
                 unsigned int layersIndex,
                 unsigned int layerIndex,
                 const CheckLocation& location)
{
    if (graph->layers() == nullptr)
    {
        throw ParseException(
            boost::str(
                boost::format("%1% was called with invalid (null) graph. "
                              "Possible reason is that the graph is not yet loaded and Unpack(ed). "
                              "layers:%2% at %3%") %
                location.m_Function %
                layersIndex %
                location.FileLine()));
    }
    else if (layersIndex >= graph->layers()->size())
    {
        throw ParseException(
            boost::str(
                boost::format("%1% was called with an invalid layers index. "
                              "layers:%2% at %3%") %
                location.m_Function %
                layersIndex %
                location.FileLine()));
    }
    else if (layerIndex >= graph->layers()[layersIndex].size()
            && layerIndex != VIRTUAL_LAYER_ID)
    {
        throw ParseException(
            boost::str(
                boost::format("%1% was called with an invalid layer index. "
                              "layers:%2% layer:%3% at %4%") %
                location.m_Function %
                layersIndex %
                layerIndex %
                location.FileLine()));
    }
}

void CheckTensorPtr(DeserializeParser::TensorRawPtr rawPtr,
                    const CheckLocation& location)
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

#define CHECK_LAYERS(GRAPH, LAYERS_INDEX, LAYER_INDEX) \
    CheckLayers(GRAPH, LAYERS_INDEX, LAYER_INDEX, CHECK_LOCATION())

#define CHECK_GRAPH(GRAPH, LAYERS_INDEX) \
    CheckGraph(GRAPH, LAYERS_INDEX, CHECK_LOCATION())
}

DeserializeParser::DeserializeParser()
: m_Network(nullptr, nullptr),
//May require LayerType_Max to be included
m_ParserFunctions(Layer_MAX+1, &DeserializeParser::ParseUnsupportedLayer)
{
    // register supported layers
    m_ParserFunctions[Layer_AdditionLayer]         =  &DeserializeParser::ParseAdd;
    m_ParserFunctions[Layer_MultiplicationLayer]   =  &DeserializeParser::ParseMultiplication;
}

DeserializeParser::LayerBaseRawPtr DeserializeParser::GetBaseLayer(const GraphPtr& graphPtr, unsigned int layerIndex)
{
    auto layerType = graphPtr->layers()->Get(layerIndex)->layer_type();

    switch(layerType)
    {
        case Layer::Layer_AdditionLayer:
            return graphPtr->layers()->Get(layerIndex)->layer_as_AdditionLayer()->base();
        case Layer::Layer_InputLayer:
           return graphPtr->layers()->Get(layerIndex)->layer_as_InputLayer()->base()->base();
        case Layer::Layer_MultiplicationLayer:
            return graphPtr->layers()->Get(layerIndex)->layer_as_MultiplicationLayer()->base();
        case Layer::Layer_OutputLayer:
            return graphPtr->layers()->Get(layerIndex)->layer_as_OutputLayer()->base()->base();
        case Layer::Layer_NONE:
        default:
            throw ParseException(boost::str(
                  boost::format("Layer must have a type %1%") %
                  Layer::Layer_NONE));
    }
}

int32_t DeserializeParser::GetBindingLayerInfo(const GraphPtr& graphPtr, unsigned int layerIndex)
{
    auto layerType = graphPtr->layers()->Get(layerIndex)->layer_type();

    if (layerType == Layer::Layer_InputLayer)
    {
        return graphPtr->layers()->Get(layerIndex)->layer_as_InputLayer()->base()->layerBindingId();
    }
    else if ( layerType == Layer::Layer_OutputLayer )
    {
        return graphPtr->layers()->Get(layerIndex)->layer_as_OutputLayer()->base()->layerBindingId();
    }
    return 0;
}

armnn::TensorInfo ToTensorInfo(DeserializeParser::TensorRawPtr tensorPtr)
{
    armnn::DataType type;
    CHECK_TENSOR_PTR(tensorPtr);

    switch (tensorPtr->dataType())
    {
        case DataType_QuantisedAsymm8:
            type = armnn::DataType::QuantisedAsymm8;
            break;
        case DataType_Float32:
            type = armnn::DataType::Float32;
            break;
        case DataType_Float16:
            type = armnn::DataType::Float16;
            break;
        case DataType_Boolean:
            type = armnn::DataType::Boolean;
            break;
        default:
        {
            CheckLocation location = CHECK_LOCATION();
            throw ParseException(
                    boost::str(
                            boost::format("Unsupported data type %1% = %2%. %3%") %
                            tensorPtr->dataType() %
                            EnumNameDataType(tensorPtr->dataType()) %
                            location.AsString()));
        }
    }
    float quantizationScale = tensorPtr->quantizationScale();
    int32_t quantizationOffset = tensorPtr->quantizationOffset();

    auto dimensions = tensorPtr->dimensions();
    unsigned int size = dimensions->size();
    std::vector<unsigned int> outputDims(dimensions->begin(), dimensions->begin() + size);

    // two statements (on purpose) for easier debugging:
    armnn::TensorInfo result(size,
                             outputDims.data(),
                             type,
                             quantizationScale,
                             quantizationOffset);
    return result;
}

DeserializeParser::LayerBaseRawPtrVector DeserializeParser::GetGraphInputs(const GraphPtr& graphPtr)
{

    CHECK_GRAPH(graphPtr, 0);
    const auto& numInputs = graphPtr->inputIds()->size();

    LayerBaseRawPtrVector result(numInputs);

    for (unsigned int i=0; i<numInputs; ++i)
    {
        uint32_t inputId = graphPtr->inputIds()->Get(i);
        result[i] = GetBaseLayer(graphPtr, static_cast<uint32_t>(inputId));
    }
    return result;
}

DeserializeParser::LayerBaseRawPtrVector DeserializeParser::GetGraphOutputs(const GraphPtr& graphPtr)
{
    CHECK_GRAPH(graphPtr, 0);
    const auto& numOutputs = graphPtr->outputIds()->size();

    LayerBaseRawPtrVector result(numOutputs);

    for (unsigned int i=0; i<numOutputs; ++i)
    {
        uint32_t outputId = graphPtr->outputIds()->Get(i);
        result[i] = GetBaseLayer(graphPtr, static_cast<uint32_t>(outputId));
    }
    return result;
}

DeserializeParser::TensorRawPtrVector DeserializeParser::GetInputs(const GraphPtr& graphPtr,
                                                                   unsigned int layerIndex)
{
    CHECK_LAYERS(graphPtr, 0, layerIndex);
    auto layer = GetBaseLayer(graphPtr, layerIndex);
    const auto& numInputs = layer->inputSlots()->size();

    TensorRawPtrVector result(numInputs);

   for (unsigned int i=0; i<numInputs; ++i)
   {
       auto inputId = CHECKED_NON_NEGATIVE(static_cast<int32_t>
                                          (layer->inputSlots()->Get(i)->connection()->sourceLayerIndex()));
       result[i] = GetBaseLayer(graphPtr, inputId)->outputSlots()->Get(0)->tensorInfo();
   }
   return result;
}

DeserializeParser::TensorRawPtrVector DeserializeParser::GetOutputs(const GraphPtr& graphPtr,
                                                                    unsigned int layerIndex)
{
    CHECK_LAYERS(graphPtr, 0, layerIndex);
    auto layer = GetBaseLayer(graphPtr, layerIndex);
    const auto& numOutputs = layer->outputSlots()->size();

    TensorRawPtrVector result(numOutputs);

    for (unsigned int i=0; i<numOutputs; ++i)
    {
        result[i] = layer->outputSlots()->Get(i)->tensorInfo();
    }
    return result;
}

void DeserializeParser::ParseUnsupportedLayer(unsigned int layerIndex)
{
    CHECK_LAYERS(m_Graph, 0, layerIndex);
    const auto layerName = GetBaseLayer(m_Graph, layerIndex)->layerName()->c_str();
    throw ParseException(
        boost::str(
            boost::format("Layer not supported. "
                          "layerIndex: %1% "
                          "layerName: %2% / %3%") %
            layerIndex %
            layerName %
            CHECK_LOCATION().AsString()));
}

void DeserializeParser::ResetParser()
{
    m_Network = armnn::INetworkPtr(nullptr, nullptr);
    m_Graph = nullptr;
}

IDeserializeParser* IDeserializeParser::CreateRaw()
{
    return new DeserializeParser();
}

IDeserializeParserPtr IDeserializeParser::Create()
{
    return IDeserializeParserPtr(CreateRaw(), &IDeserializeParser::Destroy);
}

void IDeserializeParser::Destroy(IDeserializeParser* parser)
{
    delete parser;
}

INetworkPtr DeserializeParser::CreateNetworkFromBinaryFile(const char* graphFile)
{
    ResetParser();
    m_Graph = LoadGraphFromFile(graphFile, m_FileContent);
    return CreateNetworkFromGraph();
}

INetworkPtr DeserializeParser::CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent)
{
     ResetParser();
     m_Graph = LoadGraphFromBinary(binaryContent.data(), binaryContent.size());
     return CreateNetworkFromGraph();
}

DeserializeParser::GraphPtr DeserializeParser::LoadGraphFromFile(const char* fileName, std::string& fileContent)
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
    fileContent = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    return LoadGraphFromBinary(reinterpret_cast<const uint8_t*>(fileContent.c_str()), fileContent.size());
}

DeserializeParser::GraphPtr DeserializeParser::LoadGraphFromBinary(const uint8_t* binaryContent, size_t len)
{
    if (binaryContent == nullptr)
    {
        throw InvalidArgumentException(boost::str(boost::format("Invalid (null) binary content %1%") %
                                                  CHECK_LOCATION().AsString()));
    }
    flatbuffers::Verifier verifier(binaryContent, len);
    if (verifier.VerifyBuffer<SerializedGraph>() == false)
    {
        throw ParseException(
                boost::str(boost::format("Buffer doesn't conform to the expected Armnn "
                                         "flatbuffers format. size:%1% %2%") %
                           len %
                           CHECK_LOCATION().AsString()));
    }
    return GetSerializedGraph(binaryContent);
}

INetworkPtr DeserializeParser::CreateNetworkFromGraph()
{
    m_Network = INetwork::Create();
    BOOST_ASSERT(m_Graph != nullptr);
    unsigned int layerIndex = 0;
    m_GraphConnections.emplace_back(m_Graph->layers()->size());
    for (AnyLayer const* layer : *m_Graph->layers())
    {
        if (layer->layer_type() != Layer_InputLayer &&
            layer->layer_type() != Layer_OutputLayer)
        {
            // lookup and call the parser function
            auto& parserFunction = m_ParserFunctions[layer->layer_type()];
            (this->*parserFunction)(layerIndex);
        }
        ++layerIndex;
    }

    SetupInputLayers();
    SetupOutputLayers();

    // establish the connections from the layer outputs to the inputs of the subsequent layers
    for (size_t connectionIndex = 0; connectionIndex < m_GraphConnections[0].size(); ++connectionIndex)
    {
        if (m_GraphConnections[0][connectionIndex].outputSlot != nullptr)
        {
            for (size_t inputSlotIdx = 0;
                 inputSlotIdx < m_GraphConnections[0][connectionIndex].inputSlots.size();
                 ++inputSlotIdx)
            {
                m_GraphConnections[0][connectionIndex].outputSlot->Connect(
                        *(m_GraphConnections[0][connectionIndex].inputSlots[inputSlotIdx]));
            }
        }
    }

    return std::move(m_Network);
}

BindingPointInfo DeserializeParser::GetNetworkInputBindingInfo(unsigned int layerIndex,
                                                               const std::string& name) const
{
    CHECK_LAYERS(m_Graph, 0, layerIndex);
    auto inputs = GetGraphInputs(m_Graph);

    for (auto const& input : inputs)
    {
        if (input->layerName()->c_str() == name)
        {
            int bindingId = reinterpret_cast<armnn::LayerBindingId>(GetBindingLayerInfo(m_Graph, input->index()));
            auto layerBase = GetBaseLayer(m_Graph,input->index())->outputSlots()->Get(layerIndex);
            return std::make_pair(bindingId, ToTensorInfo(layerBase->tensorInfo()));
        }
    }
    throw ParseException(
            boost::str(
                    boost::format("No input binding found for layer:%1% / %2%") %
                    name %
                    CHECK_LOCATION().AsString()));
}

BindingPointInfo DeserializeParser::GetNetworkOutputBindingInfo(unsigned int layerIndex,
                                                                const std::string& name) const
{
    CHECK_LAYERS(m_Graph, 0, layerIndex);
    auto outputs = GetGraphOutputs(m_Graph);

    for (auto const& output : outputs)
    {
        if (output->layerName()->c_str() == name)
        {
            int bindingId = reinterpret_cast<armnn::LayerBindingId>(GetBindingLayerInfo(m_Graph, output->index()));
            auto layer = GetBaseLayer(m_Graph, output->index());
            auto sourceLayerIndex = layer->inputSlots()->Get(0)->connection()->sourceLayerIndex();
            auto sourceLayer = GetBaseLayer(m_Graph, sourceLayerIndex);
            return std::make_pair(bindingId, ToTensorInfo(sourceLayer->outputSlots()->Get(0)->tensorInfo()));
        }
    }
    throw ParseException(
        boost::str(
            boost::format("No output binding found for layer:%1% / %2%") %
            name %
            CHECK_LOCATION().AsString()));
}

void DeserializeParser::SetupInputLayers()
{
    CHECK_GRAPH(m_Graph, 0);
    auto inputs = GetGraphInputs(m_Graph);
    for (auto const& input : inputs)
    {
        IConnectableLayer* layer =
            m_Network->AddInputLayer(GetBindingLayerInfo(m_Graph, input->index()), input->layerName()->c_str());

        auto tensorInfo = ToTensorInfo(input->outputSlots()->Get(0)->tensorInfo());
        layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

        RegisterOutputSlots(input->index(), layer);
    }
}

void DeserializeParser::SetupOutputLayers()
{
    CHECK_GRAPH(m_Graph, 0);
    auto outputs = GetGraphOutputs(m_Graph);
    for (auto const& output : outputs)
    {
        IConnectableLayer* layer =
            m_Network->AddOutputLayer(GetBindingLayerInfo(m_Graph, output->index()), output->layerName()->c_str());

        RegisterInputSlots(output->index(), layer);
    }
}

void DeserializeParser::RegisterOutputSlots(uint32_t layerIndex,
                                            IConnectableLayer* layer)
{
    CHECK_LAYERS(m_Graph, 0, layerIndex);
    BOOST_ASSERT(layer != nullptr);
    auto parsedLayer = GetBaseLayer(m_Graph, layerIndex);
    if (parsedLayer->outputSlots()->size() != layer->GetNumOutputSlots())
    {
        throw ParseException(
            boost::str(boost::format("The number of outputslots (%1%) does not match the number expected (%2%)"
                                     " for layer index: %3% %4%") %
                       parsedLayer->outputSlots()->size() %
                       layer->GetNumOutputSlots() %
                       layerIndex %
                       CHECK_LOCATION().AsString()));
    }

    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumOutputSlots(); ++slotIndex)
    {
        armnn::IOutputSlot* slot = &(layer->GetOutputSlot(slotIndex));
        RegisterOutputSlotOfConnection(layerIndex, slot);
    }
}

void DeserializeParser::RegisterInputSlots(uint32_t layerIndex,
                                           armnn::IConnectableLayer* layer)
{
    CHECK_LAYERS(m_Graph, 0, layerIndex);
    BOOST_ASSERT(layer != nullptr);
    auto parsedLayer = GetBaseLayer(m_Graph, layerIndex);
    if (parsedLayer->inputSlots()->size() != layer->GetNumInputSlots())
    {
        throw ParseException(
            boost::str(boost::format("The number of inputslots (%1%) does not match the number expected (%2%)"
                                     " for layer index:%3% %4%") %
                       parsedLayer->inputSlots()->size() %
                       layer->GetNumInputSlots() %
                       layerIndex %
                       CHECK_LOCATION().AsString()));
    }

    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumInputSlots(); ++slotIndex)
    {
        armnn::IInputSlot* slot = &(layer->GetInputSlot(slotIndex));
        uint32_t sourceLayerIndex = parsedLayer->inputSlots()->Get(slotIndex)->connection()->sourceLayerIndex();
        RegisterInputSlotOfConnection(sourceLayerIndex, slot);
    }
}

void DeserializeParser::RegisterInputSlotOfConnection(uint32_t connectionIndex,
                                                      armnn::IInputSlot* slot)
{
    BOOST_ASSERT(m_GraphConnections[0].size() > connectionIndex);

    Slots& slots = m_GraphConnections[0][connectionIndex];
    slots.inputSlots.push_back(slot);
}

void DeserializeParser::RegisterOutputSlotOfConnection(uint32_t connectionIndex,
                                                       armnn::IOutputSlot* slot)
{
    BOOST_ASSERT(m_GraphConnections[0].size() > connectionIndex);

    Slots& slots = m_GraphConnections[0][connectionIndex];

    // assuming there is only one producer for that tensor
    if (slots.outputSlot != nullptr)
    {
        throw ParseException(boost::str(
            boost::format("Another layer has already registered itself as the producer of "
                          "connection:%1% / %2%") %
            connectionIndex %
            CHECK_LOCATION().AsString()));
    }

    slots.outputSlot = slot;
}

void DeserializeParser::ParseAdd(unsigned int layerIndex)
{
    CHECK_LAYERS(m_Graph, 0, layerIndex);
    auto inputs = GetInputs(m_Graph, layerIndex);
    CHECK_LOCATION();
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Graph, layerIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = boost::str(boost::format("Addition:%1%") % layerIndex);
    IConnectableLayer* layer = m_Network->AddAdditionLayer(layerName.c_str());

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    RegisterInputSlots(layerIndex, layer);
    RegisterOutputSlots(layerIndex, layer);
}

void DeserializeParser::ParseMultiplication(unsigned int layerIndex)
{
    CHECK_LAYERS(m_Graph, 0, layerIndex);
    auto inputs = GetInputs(m_Graph, layerIndex);
    CHECK_LOCATION();
    CHECK_VALID_SIZE(inputs.size(), 2);

    auto outputs = GetOutputs(m_Graph, layerIndex);
    CHECK_VALID_SIZE(outputs.size(), 1);

    auto layerName = boost::str(boost::format("Multiplication:%1%") % layerIndex);
    IConnectableLayer* layer = m_Network->AddMultiplicationLayer(layerName.c_str());

    armnn::TensorInfo outputTensorInfo = ToTensorInfo(outputs[0]);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    RegisterInputSlots(layerIndex, layer);
    RegisterOutputSlots(layerIndex, layer);
}

}
