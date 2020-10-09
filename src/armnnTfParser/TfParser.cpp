//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TfParser.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/Descriptors.hpp>

#include <armnnUtils/Permute.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnnUtils/Transpose.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <GraphTopologicalSort.hpp>
#include <ParserHelper.hpp>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <tensorflow/core/framework/graph.pb.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <numeric>

using namespace armnnUtils;
using namespace armnn;

namespace armnnTfParser
{
namespace
{

const PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
const PermutationVector ArmNNToNHWC = { 0, 3, 1, 2 };


template <typename Callable>
void ReadMandatoryNodeAttributeImpl(const tensorflow::NodeDef& nodeDef,
    const std::string& attribName,
    tensorflow::AttrValue::ValueCase expectedValueCase,
    Callable callable)
{
    auto iter = nodeDef.attr().find(attribName);
    if (iter != nodeDef.attr().end())
    {
        const auto& attrValue = iter->second;
        if (attrValue.value_case() == expectedValueCase)
        {
            callable(attrValue);
        }
        else
        {
            throw ParseException(
                fmt::format("Attribute {} of node {} expected to have {} as tensorflow::AttrValue::ValueCase, "
                            "but found {} instead {}",
                            attribName,
                            nodeDef.name(),
                            static_cast<int>(expectedValueCase),
                            static_cast<int>(attrValue.value_case()),
                            CHECK_LOCATION().AsString()));
        }
    }
    else
    {
        throw ParseException(
            fmt::format("Could not find required attribute {} in node {} {}",
                        attribName,
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
}

template <typename Callable>
void ReadOptionalNodeAttributeImpl(const tensorflow::NodeDef& nodeDef,
    const std::string& attribName,
    tensorflow::AttrValue::ValueCase expectedValueCase,
    Callable callable)
{
    auto iter = nodeDef.attr().find(attribName);
    if (iter != nodeDef.attr().end())
    {
        const auto& attrValue = iter->second;
        if (attrValue.value_case() == expectedValueCase)
        {
            callable(attrValue);
        }
        else
        {
            throw ParseException(
                fmt::format("Attribute {} of node {} expected to have {} as tensorflow::AttrValue::ValueCase, "
                            "but found {} instead {}",
                            attribName,
                            nodeDef.name(),
                            static_cast<int>(expectedValueCase),
                            static_cast<int>(attrValue.value_case()),
                            CHECK_LOCATION().AsString()));
        }
    }
}

float ReadMandatoryNodeFloatAttribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    float attribValue = 0.0f;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kF,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.f();
    });
    return attribValue;
}

int32_t ReadMandatoryNodeInt32Attribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    int32_t attribValue = 0u;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kI,
                                   [&attribValue](const tensorflow::AttrValue& attrValue)
                                   {
                                       attribValue = static_cast<int32_t>(attrValue.i());
                                   });
    return attribValue;
}

bool ReadMandatoryNodeBoolAttribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    bool attribValue = false;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kB,
                                   [&attribValue](const tensorflow::AttrValue& attrValue)
                                   {
                                       attribValue = static_cast<bool>(attrValue.b());
                                   });
    return attribValue;
}

uint32_t ReadMandatoryNodeUint32Attribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    uint32_t attribValue = 0u;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kI,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = static_cast<uint32_t>(attrValue.i());
    });
    return attribValue;
}

std::string ReadMandatoryNodeStringAttribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    std::string attribValue = "";
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kS,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.s();
    });
    return attribValue;
}

std::vector<uint32_t> ReadMandatoryNodeUint32ListAttribute(const tensorflow::NodeDef& nodeDef,
    const std::string& name)
{
    std::vector<uint32_t> attriList;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kList,
        [&attriList](const tensorflow::AttrValue& attrValue)
    {
        for (int attriNum = 0; attriNum < attrValue.list().i_size(); ++attriNum)
        {
            attriList.push_back(static_cast<uint32_t>(attrValue.list().i(attriNum)));
        }
    });

    return attriList;
}

std::vector<uint32_t> ReadOptionalNodeUint32ListAttribute(const tensorflow::NodeDef& nodeDef,
    const std::string& name)
{
    std::vector<uint32_t> attriList;
    ReadOptionalNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kList,
        [&attriList](const tensorflow::AttrValue& attrValue)
    {
        for (int attriNum = 0; attriNum < attrValue.list().i_size(); ++attriNum)
        {
            attriList.push_back(static_cast<uint32_t>(attrValue.list().i(attriNum)));
        }
    });

    return attriList;
}

std::string ReadOptionalNodeStringAttribute(const tensorflow::NodeDef& nodeDef,
                                            const std::string& name,
                                            const std::string& defaultValue = "")
{
    std::string attribValue = defaultValue;
    ReadOptionalNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kS,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.s();
    });
    return attribValue;
}

bool ReadOptionalNodeBoolAttribute(const tensorflow::NodeDef& nodeDef,
    const std::string& name,
    bool defaultValue = false)
{
    bool attribValue = defaultValue;
    ReadOptionalNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kB,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.b();
    });
    return attribValue;
}

tensorflow::DataType ReadMandatoryNodeTypeAttribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    tensorflow::DataType attribValue = tensorflow::DT_INVALID;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kType,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.type();
    });
    return attribValue;
}

TensorInfo PrepareReshape(const TensorInfo& input, const std::vector<int32_t>& targetDims)
{
    std::vector<unsigned int> outDims(targetDims.begin(), targetDims.end());
    const auto stretchDim = std::find(targetDims.begin(), targetDims.end(), -1);

    if (stretchDim != targetDims.end())
    {
        if (std::find(std::next(stretchDim), targetDims.end(), -1) != targetDims.end())
        {
            throw ParseException(
                fmt::format("At most one component of shape can be -1 {}",
                            CHECK_LOCATION().AsString()));
        }

        auto targetNumElements =
            armnn::numeric_cast<unsigned int>(
                std::accumulate(targetDims.begin(), targetDims.end(), -1, std::multiplies<int32_t>()));
        auto stretchIndex = static_cast<size_t>(std::distance(targetDims.begin(), stretchDim));
        outDims[stretchIndex] = input.GetNumElements() / targetNumElements;
    }

    TensorInfo reshapeInfo = input;
    reshapeInfo.SetShape(TensorShape{ static_cast<unsigned int>(outDims.size()), outDims.data() });

    return reshapeInfo;
}

// We need the input0Slot to guide the reshape for input1Slot.
IOutputSlot* AddBroadcastReshapeLayer(IOutputSlot* input0Slot, IOutputSlot* input1Slot, bool isNHWC,
                                         INetwork& m_Network, const tensorflow::NodeDef& nodeDef)
{
    const TensorInfo& input1Info = input1Slot->GetTensorInfo();
    const TensorInfo inputTensorInfo = input0Slot->GetTensorInfo();
    const unsigned int matchDim = inputTensorInfo.GetNumDimensions() - (isNHWC ? 1 : 3);
    std::array<unsigned int, MaxNumOfTensorDimensions> reshapedDimensions;
    std::fill_n(reshapedDimensions.begin(), inputTensorInfo.GetNumDimensions(), 1);
    reshapedDimensions[matchDim] = input1Info.GetShape()[0];

    armnn::TensorInfo reshapedInfo = input1Info;
    reshapedInfo.SetShape(TensorShape{ inputTensorInfo.GetNumDimensions(), reshapedDimensions.data() });

    const std::string reshapeLayerName = "reshape_for-" + nodeDef.name();
    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = reshapedInfo.GetShape();
    IConnectableLayer* const reshapeLayer = m_Network.AddReshapeLayer(reshapeDesc, reshapeLayerName.c_str());

    input1Slot->Connect(reshapeLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);

    input1Slot = &reshapeLayer->GetOutputSlot(0);

    return input1Slot;
}

OutputId ParseOutputId(const std::string & name)
{
    unsigned int outputNum = 0;
    size_t colonPos = name.find_last_of(":");
    if (colonPos != std::string::npos)
    {
        int n = std::stoi(name.substr(colonPos+1));
        if (n<0 || n>100)
        {
            throw ParseException(
                fmt::format("Output tensor id is out of range for {} {}",
                            name,
                            CHECK_LOCATION().AsString()));
        }
        outputNum = static_cast<unsigned int>(n);
    }
    return OutputId(name.substr(0,colonPos),outputNum);
}

#define CHECK_DATA_FORMAT(NODE_DEF, FORMAT, NODE_TYPE) \
    if( FORMAT != "NHWC" && FORMAT != "NCHW" ) \
    { \
        throw ParseException( \
            fmt::format("Unsupported data format {} passed for {} node {}. " \
                        "Only NHWC and NCHW supported {}", \
                        FORMAT, \
                        NODE_TYPE, \
                        NODE_DEF.name(), \
                        CHECK_LOCATION().AsString())); \
    }

#define CHECK_PADDING_TYPE(NODE_DEF, PADDING) \
    if(PADDING != "SAME" && PADDING != "VALID" ) \
    { \
        throw ParseException( \
            fmt::format("Only 'SAME' and 'VALID' padding supported. Got {} for {} {}", \
                        PADDING, \
                        NODE_DEF.name(), \
                        CHECK_LOCATION().AsString())); \
    } \

} // namespace

const std::map<std::string, TfParser::OperationParsingFunction> TfParser::ms_OperationNameToParsingFunctions = {
    { "Const",                 &TfParser::ParseConst },
    { "Add",                   &TfParser::ParseAdd },
    { "AddN",                  &TfParser::ParseAddN },
    { "BiasAdd",               &TfParser::ParseBiasAdd },
    { "Identity",              &TfParser::ParseIdentity },
    { "Conv2D",                &TfParser::ParseConv2D },
    { "DepthwiseConv2dNative", &TfParser::ParseDepthwiseConv2D },
    { "ExpandDims",            &TfParser::ParseExpandDims },
    { "FusedBatchNorm",        &TfParser::ParseFusedBatchNorm },
    { "Gather",                &TfParser::ParseGather},
    { "Greater",               &TfParser::ParseGreater},
    { "ConcatV2",              &TfParser::ParseConcat },
    { "LRN",                   &TfParser::ParseLrn },
    { "MatMul",                &TfParser::ParseMatMul },
    { "Mean",                  &TfParser::ParseMean },
    { "Mul",                   &TfParser::ParseMul },
    { "Placeholder",           &TfParser::ParsePlaceholder },
    { "RealDiv",               &TfParser::ParseRealDiv },
    { "Relu",                  &TfParser::ParseRelu },
    { "Relu6",                 &TfParser::ParseRelu6 },
    { "Reshape",               &TfParser::ParseReshape },
    { "ResizeBilinear",        &TfParser::ParseResizeBilinear },
    { "Rsqrt",                 &TfParser::ParseRsqrt },
    { "Shape",                 &TfParser::ParseShape },
    { "Squeeze",               &TfParser::ParseSqueeze },
    { "Sigmoid",               &TfParser::ParseSigmoid },
    { "Softmax",               &TfParser::ParseSoftmax },
    { "Softplus",              &TfParser::ParseSoftplus },
    { "Split",                 &TfParser::ParseSplit },
    { "StridedSlice",          &TfParser::ParseStridedSlice },
    { "Tanh",                  &TfParser::ParseTanh },
    { "MaxPool",               &TfParser::ParseMaxPool },
    { "AvgPool",               &TfParser::ParseAvgPool },
    { "Maximum",               &TfParser::ParseMaximum },
    { "Minimum",               &TfParser::ParseMinimum },
    { "Equal",                 &TfParser::ParseEqual },
    { "Pad",                   &TfParser::ParsePad },
    { "Sub",                   &TfParser::ParseSub },
    { "Pack" ,                 &TfParser::ParseStack },
    { "Stack",                 &TfParser::ParseStack },
    { "Transpose",             &TfParser::ParseTranspose },
};

const std::list<std::string> TfParser::m_ControlInputs = {
    "Assert"
};

ITfParser* ITfParser::CreateRaw()
{
    return new TfParser();
}

ITfParserPtr ITfParser::Create()
{
    return ITfParserPtr(CreateRaw(), &ITfParser::Destroy);
}

void ITfParser::Destroy(ITfParser* parser)
{
    delete parser;
}

inline void CalculateSamePadding(uint32_t inputSize, uint32_t stride,
                                 uint32_t filterSize, bool samePadding,
                                 uint32_t* paddingFront, uint32_t* paddingBack) {
    *paddingFront = 0;
    *paddingBack = 0;

    if (samePadding) {
        uint32_t outputSize = (inputSize + stride - 1) / stride;
        uint32_t temp = (outputSize - 1) * stride + filterSize;
        if (temp > inputSize) {
            *paddingFront = (temp - inputSize) / 2;
            *paddingBack = (temp - inputSize) - *paddingFront;
        }
    }
}

void CalcPadding(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t& outPadHead, uint32_t& outPadTail,
                 bool samePadding)
{
    CalculateSamePadding(input, stride, kernel, samePadding, &outPadHead, &outPadTail);
}

/// An Abstract base class which represents a single tensorflow operation (node)
/// that has been (potentially partially) converted to Armnn.
/// It may not yet have been fully converted into actual Armnn layers.
class ParsedTfOperation
{
public:
    ParsedTfOperation(TfParser* parser, const tensorflow::NodeDef& node)
    : m_Parser(parser)
    , m_Node(node)
    {
    }

    virtual ~ParsedTfOperation() {};

    const tensorflow::NodeDef& GetNode() const { return m_Node; }

    /// Gets the ArmNN IOutputSlot corresponding to the given output index of the Tensorflow operation.
    /// This may result in the creation of Armnn layers if this was deferred (e.g. see ParsedConstTfOperation).
    virtual IOutputSlot& ResolveArmnnOutputSlot(unsigned int tfOutputIndex) = 0;

    /// If this operation is an Identity then this will follow return the 'parent' operation (recursively).
    virtual ParsedTfOperation* ResolveIdentityOperations()
    {
        return this;
    }

protected:
    TfParser* m_Parser;
    const tensorflow::NodeDef& m_Node;
};

/// An ParsedTfOperation where the Armnn equivalent is a single layer,
/// with output slots that correspond directly to the Tf node outputs.
class SingleLayerParsedTfOperation : public ParsedTfOperation
{
public:
    SingleLayerParsedTfOperation(TfParser* parser, const tensorflow::NodeDef& node, IConnectableLayer* layer)
    : ParsedTfOperation(parser, node)
    , m_Layer(layer)
    {
    }

    IOutputSlot& ResolveArmnnOutputSlot(unsigned int tfOutputIndex) override
    {
        ARMNN_ASSERT(m_Layer);
        // Assumes one-to-one mapping between Tf and armnn output slots.
        unsigned int armnnOutputSlotIdx = tfOutputIndex;
        if (armnnOutputSlotIdx >= m_Layer->GetNumOutputSlots())
        {
            throw ParseException(
                fmt::format("The requested output slot #{} "
                            "for {} does not exist {}",
                            armnnOutputSlotIdx,
                            m_Layer->GetName(),
                            CHECK_LOCATION().AsString()));
        }
        return m_Layer->GetOutputSlot(armnnOutputSlotIdx);
    }

protected:
    IConnectableLayer* m_Layer;
};

/// A SingleLayerParsedTfOperation for deferred layer creation.
class DeferredSingleLayerParsedTfOperation : public SingleLayerParsedTfOperation
{
public:
    DeferredSingleLayerParsedTfOperation(TfParser* parser, const tensorflow::NodeDef& node)
    : SingleLayerParsedTfOperation(parser, node, nullptr)
    {
    }

    IOutputSlot& ResolveArmnnOutputSlot(unsigned int tfOutputIndex) override
    {
        if (!m_Layer)
        {
            CreateLayerDeferred();
        }
        return SingleLayerParsedTfOperation::ResolveArmnnOutputSlot(tfOutputIndex);
    }

private:
    virtual void CreateLayerDeferred() = 0;
};


TfParser::TfParser()
    : m_Network(nullptr, nullptr)
{
}


const tensorflow::NodeDef* TfParser::ResolveIdentityNode(const tensorflow::NodeDef* nodeDef)
{
    if (nodeDef->op() != "Identity")
    {
        return nodeDef;
    }

    if (nodeDef->input_size() != 1)
    {
        throw ParseException(
            fmt::format("Identity node should have a single input! {} has {} inputs {}",
                        nodeDef->name(),
                        nodeDef->input_size(),
                        CHECK_LOCATION().AsString()));
    }

    auto it = m_NodesByName.find(nodeDef->input(0));
    if (it != m_NodesByName.end())
    {
        const tensorflow::NodeDef* inputNode = it->second;
        return ResolveIdentityNode(inputNode);
    }
    else
    {
        throw ParseException(
            fmt::format("Cannot find what the Identity node {} is linked to! {}",
                        nodeDef->name(),
                        CHECK_LOCATION().AsString()));
    }
}

std::vector<OutputOfConstNodeDef>
TfParser::GetTfInputNodes(const tensorflow::NodeDef& nodeDef) const
{
    std::vector<OutputOfConstNodeDef> ret;

    if (nodeDef.op() == "Const")
    {
        // For some reason const node can have "Control Inputs". We ignore them for now.
        return ret;
    }

    ret.reserve(armnn::numeric_cast<size_t>(nodeDef.input_size()));
    for (int j = 0; j < nodeDef.input_size(); ++j)
    {
        OutputId outputId = ParseOutputId(nodeDef.input(j));

        if (nodeDef.input(j)[0] == '^') // I couldn't find a better test for control inputs.
        {
            // We currently allow Control Input from TensorFlow graph but we ignore them from ArmNN graph.
            continue;
        }

        auto inputIt = m_NodesByName.find(outputId.m_IndexedValue);
        if (inputIt == m_NodesByName.end())
        {
            throw ParseException(
                fmt::format("Can't find node '{}', which is listed as an input of '{}' {}",
                            nodeDef.input(j),
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
        }
        ret.push_back(OutputOfConstNodeDef(inputIt->second,outputId.m_Index));
    }

    return ret;
}

std::vector<OutputOfParsedTfOperation>
TfParser::GetInputParsedTfOperationsChecked(const tensorflow::NodeDef& nodeDef,
                                            std::size_t expectedNumInputs)
{
    // Fetches the tensorflow nodes connected as inputs and validate the size.
    std::vector<OutputOfConstNodeDef> nodes = GetTfInputNodes(nodeDef);
    const std::size_t numInputs = nodes.size();
    if (numInputs != expectedNumInputs)
    {
        throw ParseException(
            fmt::format("Unexpected number of inputs for node {}. Expected {}, found {} {}",
                        nodeDef.name(),
                        expectedNumInputs,
                        numInputs,
                        CHECK_LOCATION().AsString()));
    }
    // Fetches the corresponding ParsedTfOperation operations
    std::vector<OutputOfParsedTfOperation> result;
    for (auto&& node : nodes)
    {
        auto it = m_ParsedTfOperations.find(node.m_IndexedValue->name());
        if (it == m_ParsedTfOperations.end())
        {
            throw ParseException(
                fmt::format("Node with name '{}' has not been parsed {}",
                            node.m_IndexedValue->name(),
                            CHECK_LOCATION().AsString()));
        }
        ParsedTfOperation* parsedOp = it->second.get();
        // Transparently 'skip' any Identity operations. This simplifies the logic inside the ParseXXX() functions.
        parsedOp = parsedOp->ResolveIdentityOperations();
        result.push_back(OutputOfParsedTfOperation(parsedOp,node.m_Index));
    }
    return result;
}

IConnectableLayer* TfParser::CreateAdditionLayer(
            const tensorflow::NodeDef& nodeDef,
            IOutputSlot* input0Slot,
            IOutputSlot* input1Slot,
            const std::string& layerName)
{
    const TensorInfo& input0Info = input0Slot->GetTensorInfo();
    const TensorInfo& input1Info = input1Slot->GetTensorInfo();

    const unsigned int input0Dim = input0Info.GetNumDimensions();
    const unsigned int input1Dim = input1Info.GetNumDimensions();
    if (input0Dim != input1Dim)
    {
        // broadcasting where input0 and input1 have different number of dimensions
        // is only supported for 1D and 4D tensors pair
        if (input0Dim == 1 && input1Dim == 4)
        {
            input0Slot = AddBroadcastReshapeLayer(input1Slot, input0Slot, true, *m_Network, nodeDef);
        }
        else if (input0Dim == 4 && input1Dim == 1)
        {
            input1Slot = AddBroadcastReshapeLayer(input0Slot, input1Slot, true, *m_Network, nodeDef);
        }
        else
        {
            throw ParseException(
                    fmt::format("Unsupported broadcast configuration for {} operation {} {}",
                                layerName,
                                nodeDef.name(),
                                CHECK_LOCATION().AsString()));
        }
    }
    IConnectableLayer* const layer = m_Network->AddAdditionLayer(layerName.c_str());

    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    // Ensure the output tensor has the correct dimensions even if a broadcast has been done
    TensorInfo outputInfo = input0Slot->GetTensorInfo();
    std::vector<unsigned int> outputShape;

    const TensorShape& input0Shape = input0Slot->GetTensorInfo().GetShape();
    const TensorShape& input1Shape = input1Slot->GetTensorInfo().GetShape();

    for (unsigned int i = 0; i < input0Shape.GetNumDimensions(); i++)
    {
        outputShape.push_back(std::max(input0Shape[i], input1Shape[i]));
    }

    outputInfo.SetShape(TensorShape(input0Shape.GetNumDimensions(), outputShape.data()));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return layer;
}

IConnectableLayer* TfParser::CreateAdditionLayer(
            const tensorflow::NodeDef& nodeDef,
            IConnectableLayer* layerOne,
            IConnectableLayer* layerTwo,
            unsigned int numberOfAddition,
            unsigned long numberOfLayersToConnect,
            bool isOdd)
{
    IOutputSlot* input0Slot = &layerOne->GetOutputSlot(0);
    IOutputSlot* input1Slot = &layerTwo->GetOutputSlot(0);
    std::string layerName(nodeDef.name());
    if (isOdd || numberOfLayersToConnect != 2)
    {
        // we are not connecting the final layer
        layerName.append("_addN_").append(std::to_string(numberOfAddition));
    }
    return CreateAdditionLayer(nodeDef, input0Slot, input1Slot, layerName);
}

IConnectableLayer* TfParser::CreateAdditionLayer(
        const tensorflow::NodeDef& nodeDef,
        const OutputOfParsedTfOperation& opOne,
        const OutputOfParsedTfOperation& opTwo,
        unsigned int numberOfAddition)
{
    IOutputSlot* input0Slot = &opOne.m_IndexedValue->ResolveArmnnOutputSlot(opOne.m_Index);
    IOutputSlot* input1Slot = &opTwo.m_IndexedValue->ResolveArmnnOutputSlot(opTwo.m_Index);
    std::string layerName(nodeDef.name());
    layerName.append("_addN_").append(std::to_string(numberOfAddition));
    return CreateAdditionLayer(nodeDef, input0Slot, input1Slot, layerName);
}

IConnectableLayer* TfParser::CreateAdditionLayer(
            const tensorflow::NodeDef& nodeDef,
            const OutputOfParsedTfOperation& op,
            IConnectableLayer* layer)
{
    IOutputSlot* input0Slot = &op.m_IndexedValue->ResolveArmnnOutputSlot(op.m_Index);
    IOutputSlot* input1Slot = &layer->GetOutputSlot(0);
    return CreateAdditionLayer(nodeDef, input0Slot, input1Slot, nodeDef.name());
}

ParsedTfOperationPtr TfParser::ParseAddN(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    uint32_t numberOfInputs = ReadMandatoryNodeUint32Attribute(nodeDef, "N");
    if (numberOfInputs < 2)
    {
        // should never happen
        throw ParseException(
                fmt::format("AddN Node with name '{}' has less than two ({}) inputs {}",
                            nodeDef.name(),
                            std::to_string(numberOfInputs),
                            CHECK_LOCATION().AsString()));
    }
    else if (numberOfInputs == 2)
    {
        //this is the same as a simple Add operation
        return AddAdditionLayer(nodeDef, false);
    }
    else
    {
        // build a binary tree of Add layers and return the final Add as the return from the function
        // if we have an odd number of inputs then the final Add will consist of a layer connecting to an
        // OutputOfParsedTfOperation, otherwise it will be two layers being added together
        std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, numberOfInputs);
        unsigned int numberOfAdditions = 0;
        std::vector<IConnectableLayer*> layers;
        // NOTE: at this point we will have a minimum of three inputs
        for (unsigned int i = 0; i < numberOfInputs; ++i)
        {
            // every time i is odd we have two inputs to process.
            bool onSecondItem = i % 2;
            if (onSecondItem)
            {
                ++numberOfAdditions;
                IConnectableLayer* newLayer = CreateAdditionLayer(
                        nodeDef, inputs[ i - 1], inputs[i], numberOfAdditions);
                layers.push_back(newLayer);
            }
        }

        std::vector<IConnectableLayer*> layersToConnect(layers);
        unsigned long numberOfLayersToConnect = layersToConnect.size();
        bool isOdd = numberOfInputs % 2;

        while (numberOfLayersToConnect > 1)
        {
            layers.clear();
            for (unsigned long i = 0; i < numberOfLayersToConnect; ++i) {
                bool onSecondItem = i % 2;
                if (onSecondItem) {
                    ++numberOfAdditions;
                    IConnectableLayer* newLayer = CreateAdditionLayer(
                        nodeDef,
                        layersToConnect[i - 1],
                        layersToConnect[i],
                        numberOfAdditions,
                        numberOfLayersToConnect,
                        isOdd);
                    layers.push_back(newLayer);
                }
            }
            //OK... need to go again... maybe
            layersToConnect = layers;
            numberOfLayersToConnect = layersToConnect.size();
        }
        IConnectableLayer* finalLayer = layersToConnect[0];
        // if we had an odd number of inputs we need to connect the final layer to the
        // last OutputOfParsedTfOperation in order to create the last Add layer we will
        // be handing back.
        if (isOdd)
        {
            // connect the final layer to the last op
            finalLayer = CreateAdditionLayer(nodeDef, inputs[numberOfInputs - 1], finalLayer);
        }
        return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, finalLayer);
    }
}

ParsedTfOperationPtr TfParser::ParseAdd(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    // If one of the inputs is a MatMul and the other is a const, then we handle both nodes
    // together as FullyConnected.
    if (inputs[0].m_IndexedValue->GetNode().op() == "MatMul" &&
        HasParsedConstTensor<float>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        IConnectableLayer* layer =
            AddFullyConnectedLayer(inputs[0].m_IndexedValue->GetNode(),
                                   &nodeDef,nodeDef.name().c_str());
        return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
    }
    else if (HasParsedConstTensor<float>(inputs[0].m_IndexedValue->GetNode().name()) &&
                                         inputs[1].m_IndexedValue->GetNode().op() == "MatMul")
    {
        IConnectableLayer* layer =
            AddFullyConnectedLayer(inputs[1].m_IndexedValue->GetNode(),
                                   &nodeDef,nodeDef.name().c_str());
        return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
    }
    else
    {
        // Otherwise it's just a regular addition.
        return AddAdditionLayer(nodeDef);
    }
}

ParsedTfOperationPtr TfParser::ParseBiasAdd(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    return AddAdditionLayer(nodeDef, true);
}

/// An ParsedTfOperation which forwards to another (used for Identity nodes).
class ParsedIdentityTfOperation : public ParsedTfOperation
{
public:
    ParsedIdentityTfOperation(TfParser* parser, const tensorflow::NodeDef& node, ParsedTfOperation* representative)
        : ParsedTfOperation(parser, node)
        , m_Representative(representative)
    {
    }

    virtual IOutputSlot& ResolveArmnnOutputSlot(unsigned int tfOutputIndex) override
    {
        ARMNN_ASSERT(m_Representative);
        return m_Representative->ResolveArmnnOutputSlot(tfOutputIndex);
    }

    virtual ParsedTfOperation* ResolveIdentityOperations() override
    {
        return m_Representative->ResolveIdentityOperations();
    }

private:
    ParsedTfOperation* m_Representative;
};

ParsedTfOperationPtr TfParser::ParseIdentity(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);
    // Any requests for the output slots of this node should be forwarded to the node connected as input.
    return std::make_unique<ParsedIdentityTfOperation>(this, nodeDef, inputs[0].m_IndexedValue);
}

/// An ParsedTfOperation for a Const node.
/// Creation of the armnn ConstLayer is deferred until it is actually needed, because Const nodes are mostly used
/// for weight inputs to MatMul/Conv2D nodes and in these cases armnn doesn't need a ConstLayer.
template <typename T>
class ParsedConstTfOperation : public DeferredSingleLayerParsedTfOperation
{
public:
    ParsedConstTfOperation(TfParser* parser, const tensorflow::NodeDef& node,
        const T* tensorData, const TensorInfo& tensorInfo)
        : DeferredSingleLayerParsedTfOperation(parser, node),
        m_Storage(tensorData, tensorData + tensorInfo.GetNumElements()),
        m_TensorInfo(tensorInfo)
    {
        ARMNN_ASSERT(GetDataTypeSize(tensorInfo.GetDataType()) == sizeof(T));
    }

    void CreateLayerDeferred() override
    {
        ARMNN_ASSERT(m_Layer == nullptr);
        m_Layer = m_Parser->m_Network->AddConstantLayer(ConstTensor(m_TensorInfo, m_Storage), m_Node.name().c_str());
        m_Layer->GetOutputSlot(0).SetTensorInfo(m_TensorInfo);
    }

    ConstTensor GetConstTensor(std::vector<T>& outputTensorData) const
    {
        outputTensorData.resize(m_TensorInfo.GetNumElements());

        memcpy(outputTensorData.data(), m_Storage.data(), m_TensorInfo.GetNumBytes());

        // Updates the result to point to the user provided storage.
        ConstTensor constTensor(m_TensorInfo, outputTensorData);
        return constTensor;
    }

    const T* GetStorage() const
    {
        return m_Storage.data();
    }

    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

private:
    ///< Manages the lifetime of the tensor data.
    std::vector<T> m_Storage;
    ///< Describes the layout of the tensor and points to the data in m_Storage.
    TensorInfo m_TensorInfo;
};

DataType ConvertTfTensorDataType(const tensorflow::DataType tfDataType,
                                 const tensorflow::NodeDef& nodeDef)
{
    switch (tfDataType)
    {
    case tensorflow::DT_FLOAT:
        return DataType::Float32;
        break;
    case tensorflow::DT_INT32:
        return DataType::Signed32;
        break;
    default:
        throw ParseException(
            fmt::format("Unknown DataType {} for node {} {}",
                        tensorflow::DataType_Name(tfDataType),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
}

struct ParseTfTensorValueList
{
    template<typename DataType>
    static void Parse(
        const tensorflow::TensorProto& tfTensor,
        unsigned int dstElements,
        std::vector<int8_t>& outputData);

    template <typename DataType>
    static void ReadData(const void* srcData, unsigned int numSrcElements,
        std::vector<int8_t>& dstData, unsigned int numDstElements)
    {
        // If there are no entries in the list, perform no action.
        if (numSrcElements == 0)
        {
            return;
        }

        // If no size was provided, use the length of the value list.
        if (numDstElements == 0)
        {
            numDstElements = numSrcElements;
        }

        // Allocates memory.
        dstData.resize(std::max(numSrcElements, numDstElements) * sizeof(DataType));

        const DataType* srcTensor = reinterpret_cast<const DataType*>(srcData);
        DataType* dstTensor = reinterpret_cast<DataType*>(dstData.data());

        // Copies the value list entries into the destination.
        std::copy(srcTensor, srcTensor + numSrcElements, dstTensor);

        if (numDstElements > numSrcElements)
        {
            // Uses the last element in the list to fill the remaining entries.
            std::fill(dstTensor + numSrcElements, dstTensor + numDstElements, srcTensor[numSrcElements - 1]);
        }
    }

};

template <>
void ParseTfTensorValueList::Parse<float>(const tensorflow::TensorProto& tfTensor,
    unsigned int dstElements, std::vector<int8_t>& outputData)
{
    ReadData<float>(tfTensor.float_val().data(), static_cast<unsigned int>(tfTensor.float_val_size()),
        outputData, dstElements);
}

template <>
void ParseTfTensorValueList::Parse<int32_t>(const tensorflow::TensorProto& tfTensor,
    unsigned int dstElements, std::vector<int8_t>& outputData)
{
    ReadData<int32_t>(tfTensor.int_val().data(), static_cast<unsigned int>(tfTensor.int_val_size()),
        outputData, dstElements);
}

template <template<typename> class OperatorType, typename T = int8_t>
struct MakeTfOperation
{
    template<typename DataType, class... Args>
    inline static std::unique_ptr<OperatorType<DataType>> Parse(TfParser* parser, const tensorflow::NodeDef& node,
        Args&&... args)
    {
        return std::make_unique<OperatorType<DataType>>(parser, node, std::forward<Args>(args)...);
    }
};

template <>
struct MakeTfOperation<ParsedConstTfOperation>
{
    template<typename DataType, class... Args>
    inline static std::unique_ptr<ParsedConstTfOperation<DataType>> Parse(TfParser* parser,
        const tensorflow::NodeDef& node, const std::vector<int8_t>& tensorData, const TensorInfo& tensorInfo)
    {
        return std::make_unique<ParsedConstTfOperation<DataType>>(parser, node,
            reinterpret_cast<const DataType*>(tensorData.data()), tensorInfo);
    }
};

template <class FuncType>
struct InvokeParseFunction
{
    template<class ResType, class... Args>
    inline static ResType Result(DataType dataType, Args&&... args)
    {
        if (dataType == DataType::Float32)
        {
            return FuncType::template Parse<float>(std::forward<Args>(args)...);
        }
        else if (dataType == DataType::Signed32)
        {
            return FuncType::template Parse<int32_t>(std::forward<Args>(args)...);
        }

        return ResType();
    }

    template<class... Args>
    inline static void Result(DataType dataType, Args&&... args)
    {
        if (dataType == DataType::Float32)
        {
            FuncType::template Parse<float>(std::forward<Args>(args)...);
        }
        else if (dataType == DataType::Signed32)
        {
            FuncType::template Parse<int32_t>(std::forward<Args>(args)...);
        }
    }
};

ParsedTfOperationPtr TfParser::ParseConst(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    ARMNN_ASSERT(nodeDef.op() == "Const");

    if (nodeDef.attr().count("value") == 0)
    {
        throw ParseException(
            fmt::format("Value not found for Const node - {} {}",
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    const tensorflow::TensorProto& tfTensor = nodeDef.attr().at("value").tensor();
    const tensorflow::TensorShapeProto& tfTensorShape = tfTensor.tensor_shape();
    const tensorflow::DataType tfDataType = ReadMandatoryNodeTypeAttribute(nodeDef, "dtype");

    const auto GetDimensionSize = [](auto& d) { return d.size(); };

    std::vector<unsigned int> dimensionSizes;
    std::transform(tfTensorShape.dim().begin(), tfTensorShape.dim().end(),
        std::back_inserter(dimensionSizes), GetDimensionSize);

    // Calculates number of elements.
    const DataType dataType = ConvertTfTensorDataType(tfDataType, nodeDef);
    unsigned int numElements = 0U;

    if (!dimensionSizes.empty())
    {
        numElements = std::accumulate(dimensionSizes.begin(), dimensionSizes.end(),
                                      1U, std::multiplies<unsigned int>());
    }

    std::vector<int8_t> tensorData;

    // Get tensor data from the list of values attribute.
    if (tfTensor.tensor_content().empty())
    {
        InvokeParseFunction<ParseTfTensorValueList>::Result<void>(dataType, tfTensor, numElements, tensorData);

        // If the tensor shape is not defined, but there is a value list, then interpret the data as a 1D
        // tensor of the provided number of elements.
        if (numElements == 0)
        {
            const unsigned int tfNumElements =
                static_cast<unsigned int>(tensorData.size()) / GetDataTypeSize(dataType);
            dimensionSizes.push_back(tfNumElements);
        }
    }
    // Gets tensor data from tensor content attribute.
    else
    {
        tensorData.assign(tfTensor.tensor_content().begin(), tfTensor.tensor_content().end());

        // Checks if a tensor shape is defined for the tensor content.
        if (numElements == 0)
        {
            throw ParseException(
                fmt::format("No tensor shape found for Const node - {} {}",
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
        }
    }

    // Const node requires at least a list of values or a content attribute.
    if (tensorData.empty())
    {
        throw ParseException(
            fmt::format("No tensor data found for Const node - {} {}",
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    const TensorInfo tensorInfo(static_cast<unsigned int>(dimensionSizes.size()),
                                dimensionSizes.data(),
                                dataType);

    // If we have a list of values, then the length of the list must be
    // less than or equal to the number of elements implied by the shape argument.
    if (tensorData.size() > tensorInfo.GetNumBytes())
    {
        throw ParseException(
            fmt::format("Number of elements ({}) should be less than or equal "
                        "to the number of elements implied by the shape argument ({}) for Const node - {} {}",
                        (tensorData.size() / GetDataTypeSize(dataType)),
                        tensorInfo.GetNumElements(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    return InvokeParseFunction<MakeTfOperation<ParsedConstTfOperation>>::Result<ParsedTfOperationPtr>(
        dataType, this, nodeDef, tensorData, tensorInfo);
}

template<typename Type>
bool TfParser::HasParsedConstTensor(const std::string & nodeName) const
{
    auto it = m_ParsedTfOperations.find(nodeName);
    if (it == m_ParsedTfOperations.end())
    {
        return false;
    }
    return dynamic_cast<ParsedConstTfOperation<Type>*>(it->second.get()) != nullptr;
}

template<typename Type>
bool TfParser::HasParsedConstTensor(ParsedTfOperation* parsedTfOpPtr) const
{
    return dynamic_cast<ParsedConstTfOperation<Type>*>(parsedTfOpPtr) != nullptr;
}

unsigned int TfParser::GetConstInputIndex(const std::vector<OutputOfParsedTfOperation>& inputs)
{
    for (unsigned int i = 0; i < inputs.size(); i++)
    {
        if (HasParsedConstTensor<int32_t>(inputs[i].m_IndexedValue->GetNode().name()))
        {
            return i;
        }
    }
    throw ParseException(
            fmt::format("ArmNN only supports operators with constant axis. {}",
                        CHECK_LOCATION().AsString()));

}

ParsedTfOperationPtr TfParser::ParseConv2D(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

    if (!HasParsedConstTensor<float>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException(
            fmt::format("ArmNN only supports Convolution layers with constant weights for {}, input {} {}",
                        nodeDef.name(),
                        inputs[1].m_IndexedValue->GetNode().name(),
                        CHECK_LOCATION().AsString()));
    }
    ParsedConstTfOperation<float>* weightNode =
        PolymorphicDowncast<ParsedConstTfOperation<float> *>(inputs[1].m_IndexedValue);

    std::string paddingString = ReadMandatoryNodeStringAttribute(nodeDef, "padding");
    std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");
    std::vector<uint32_t> strides = ReadMandatoryNodeUint32ListAttribute(nodeDef, "strides");

    // Read the dilations, if present - only [1,1,1,1] (the default) is supported.
    std::vector<uint32_t> dilations = ReadOptionalNodeUint32ListAttribute(nodeDef, "dilations");
    if (!dilations.empty())
    {
        for (auto dilation : dilations)
        {
            if (dilation != 1u)
            {
                throw ParseException(
                    fmt::format("ArmNN only supports Convolution layers with dilations [1,1,1,1] for {} {}",
                                nodeDef.name(),
                                CHECK_LOCATION().AsString()));
            }
        }
    }

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = false;

    CHECK_DATA_FORMAT(nodeDef, dataFormat, "Conv2D");

    DataLayout dataLayout = dataFormat == "NHWC" ? DataLayout::NHWC : DataLayout::NCHW;

    desc.m_DataLayout = dataLayout;

    DataLayoutIndexed dataLayoutIndexed(dataLayout);

    desc.m_StrideX = strides[dataLayoutIndexed.GetWidthIndex()];
    desc.m_StrideY = strides[dataLayoutIndexed.GetHeightIndex()];

    uint32_t inputHeight = inputTensorInfo.GetShape()[dataLayoutIndexed.GetHeightIndex()];
    uint32_t inputWidth  = inputTensorInfo.GetShape()[dataLayoutIndexed.GetWidthIndex()];

    // Mappings from TensorFlow filter tensors to the ArmNN filter tensors.
    // Tensorflow weights are [H, W, In, Out].
    // ArmNN weights have to be [Out, H, W, In] when the data layout is NHWC,
    // and [Out, In, H, W] when the data layout is NCHW.
    PermutationVector permutationVector =
            dataLayout == DataLayout::NHWC ?
                std::initializer_list<unsigned int>{ 1, 2, 3, 0 } : // NHWC: [H, W, In, Out] -> [Out, H, W, In]
                std::initializer_list<unsigned int>{ 2, 3, 1, 0 };  // NCHW: [H, W, In, Out] -> [Out, In, H, W]

    // Swizzle the tensor using the given permutation vector.
    const TensorInfo& weightTensorInfo = weightNode->GetTensorInfo();
    const TensorInfo weightTensorSwizzledInfo = armnnUtils::Permuted(weightTensorInfo, permutationVector);

    // Swizzles the content of the tensor's permanent storage into a local storage.
    std::vector<float> weightTensorSwizzledData(weightTensorInfo.GetNumElements());
    armnnUtils::Permute(weightTensorSwizzledInfo.GetShape(), permutationVector,
                        weightNode->GetStorage(), weightTensorSwizzledData.data(), sizeof(float));

    // Create a weight tensor with the newly swizzled data.
    ConstTensor weightTensor(weightTensorSwizzledInfo, weightTensorSwizzledData);

    uint32_t weightHeight = weightTensor.GetShape()[dataLayoutIndexed.GetHeightIndex()];
    uint32_t weightWidth  = weightTensor.GetShape()[dataLayoutIndexed.GetWidthIndex()];

    bool padding = false;
    TensorInfo outputInfo;
    unsigned int outputHeight = 0;
    unsigned int outputWidth = 0;

    CHECK_PADDING_TYPE(nodeDef, paddingString);

    if (paddingString == "SAME")
    {
        padding = true;

        outputHeight = static_cast<uint32_t>(ceil(static_cast<float>(inputHeight) /
                                                  static_cast<float>(desc.m_StrideY)));
        outputWidth  = static_cast<uint32_t>(ceil(static_cast<float>(inputWidth) /
                                                  static_cast<float>(desc.m_StrideX)));
    }
    else if (paddingString == "VALID")
    {
        padding = false;

        outputHeight = static_cast<uint32_t>(ceil(static_cast<float>(inputHeight - weightHeight + 1) /
                                                  static_cast<float>(desc.m_StrideY)));
        outputWidth  = static_cast<uint32_t>(ceil(static_cast<float>(inputWidth - weightWidth + 1) /
                                                  static_cast<float>(desc.m_StrideX)));
    }

    switch (dataLayout)
    {
    case DataLayout::NHWC:
        outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                  outputHeight,
                                  outputWidth,
                                  weightTensor.GetShape()[0] },
                                DataType::Float32);
        break;
    case DataLayout::NCHW:
    default:
        outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                  weightTensor.GetShape()[0],
                                  outputHeight,
                                  outputWidth },
                                DataType::Float32);
        break;
    }

    CalcPadding(inputHeight, weightHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, padding);
    CalcPadding(inputWidth, weightWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, padding);

    IConnectableLayer* layer = m_Network->AddConvolution2dLayer(desc,
                                                                weightTensor,
                                                                EmptyOptional(),
                                                                nodeDef.name().c_str());
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    inputSlot.Connect(layer->GetInputSlot(0));

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseDepthwiseConv2D(const tensorflow::NodeDef& nodeDef,
                                                    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

    if (!HasParsedConstTensor<float>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException(
            fmt::format("ArmNN only supports Depthwise Convolution layer with constant weights. "
                        "Non const input found {} for node {} {}",
                        inputs[1].m_IndexedValue->GetNode().name(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    ParsedConstTfOperation<float>* weightNode =
        PolymorphicDowncast<ParsedConstTfOperation<float> *>(inputs[1].m_IndexedValue);

    std::string paddingString = ReadMandatoryNodeStringAttribute(nodeDef, "padding");
    std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");
    std::vector<uint32_t> strides = ReadMandatoryNodeUint32ListAttribute(nodeDef, "strides");

    DepthwiseConvolution2dDescriptor desc;
    desc.m_BiasEnabled = false;

    CHECK_DATA_FORMAT(nodeDef, dataFormat, "DepthwiseConv2dNative");

    DataLayout dataLayout = dataFormat == "NHWC" ? DataLayout::NHWC : DataLayout::NCHW;

    desc.m_DataLayout = dataLayout;

    DataLayoutIndexed dataLayoutIndexed(dataLayout);

    desc.m_StrideX = strides[dataLayoutIndexed.GetWidthIndex()];
    desc.m_StrideY = strides[dataLayoutIndexed.GetHeightIndex()];

    uint32_t inputHeight = inputTensorInfo.GetShape()[dataLayoutIndexed.GetHeightIndex()];
    uint32_t inputWidth  = inputTensorInfo.GetShape()[dataLayoutIndexed.GetWidthIndex()];

    // Mappings from TensorFlow filter tensors to the ArmNN filter tensors.
    // Tensorflow weights come in the format [H, W, I, M].
    // ArmNN weights have to be [M, I, H, W].
    PermutationVector permutationVector{ 2, 3, 1, 0 }; // [H, W, I, M] -> [M, I, H, W]

    // Swizzle the tensor using the given permutation vector.
    const TensorInfo& weightTensorInfo = weightNode->GetTensorInfo();
    const TensorInfo weightTensorSwizzledInfo = armnnUtils::Permuted(weightTensorInfo, permutationVector);

    // Swizzles the content of the tensor's permanent storage into a local storage.
    std::vector<float> weightTensorSwizzledData(weightTensorInfo.GetNumElements());
    armnnUtils::Permute(weightTensorSwizzledInfo.GetShape(), permutationVector,
                        weightNode->GetStorage(), weightTensorSwizzledData.data(), sizeof(float));

    // Create a weight tensor with the newly swizzled data.
    ConstTensor weightTensor(weightTensorSwizzledInfo, weightTensorSwizzledData);

    uint32_t weightHeight = weightTensor.GetShape()[2];
    uint32_t weightWidth  = weightTensor.GetShape()[3];

    bool padding = false;
    TensorInfo outputInfo;
    unsigned int outputHeight = 0;
    unsigned int outputWidth = 0;

    CHECK_PADDING_TYPE(nodeDef, paddingString);

    if (paddingString == "SAME")
    {
        padding = true;

        outputHeight = static_cast<uint32_t>(ceil(static_cast<float>(inputHeight) /
                                                  static_cast<float>(desc.m_StrideY)));
        outputWidth  = static_cast<uint32_t>(ceil(static_cast<float>(inputWidth) /
                                                  static_cast<float>(desc.m_StrideX)));
    }
    else if (paddingString == "VALID")
    {
        padding = false;

        outputHeight = static_cast<uint32_t>(ceil(static_cast<float>(inputHeight - weightHeight + 1) /
                                                  static_cast<float>(desc.m_StrideY)));
        outputWidth  = static_cast<uint32_t>(ceil(static_cast<float>(inputWidth - weightWidth + 1) /
                                                  static_cast<float>(desc.m_StrideX)));
    }

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                      outputHeight,
                                      outputWidth,
                                      weightTensor.GetShape()[0] * weightTensor.GetShape()[1]},
                                    DataType::Float32);
            break;
        case DataLayout::NCHW:
        default:
            outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                      weightTensor.GetShape()[0] * weightTensor.GetShape()[1],
                                      outputHeight,
                                      outputWidth },
                                    DataType::Float32);
            break;
    }

    CalcPadding(inputHeight, weightHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, padding);
    CalcPadding(inputWidth, weightWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, padding);

    IConnectableLayer* layer = m_Network->AddDepthwiseConvolution2dLayer(desc,
                                                                         weightTensor,
                                                                         EmptyOptional(),
                                                                         nodeDef.name().c_str());
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    inputSlot.Connect(layer->GetInputSlot(0));

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

TensorInfo OutputShapeOfExpandDims(const tensorflow::NodeDef& nodeDef,
                                   TensorInfo inputTensorInfo,
                                   std::int32_t expandDim)
{
    ARMNN_ASSERT(nodeDef.op() == "ExpandDims");

    if (inputTensorInfo.GetNumDimensions() > 4) {
        throw ParseException(
                fmt::format("Unsupported number of dimensions: {} for input shape for ExpandDims {} {}",
                            inputTensorInfo.GetNumDimensions(),
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
    }

    std::int32_t inputDimSize = armnn::numeric_cast<int32_t>(inputTensorInfo.GetNumDimensions());
    std::vector<uint32_t> outputDims;

    // expandDim operation requires: -1-input.dims() <= dim <= input.dims()
    if (expandDim >= -1 - inputDimSize && expandDim <= inputDimSize)
    {
        // add current input shape to outputDims
        for (unsigned int i = 0; i < inputTensorInfo.GetNumDimensions(); ++i) {
            auto currentDimension = inputTensorInfo.GetShape()[i];
            outputDims.push_back(currentDimension);
        }

        // insert a dimension of 1 at index 'expandDim' of inputs shape
        if (expandDim >= 0)
        {
            auto getPosition = std::next(outputDims.begin() + 0, expandDim);
            outputDims.insert(getPosition, 1);
        }

        // if negative number for 'expandDim' then count backwards from the last element
        // and insert 1 dimension at index 'expandDim'
        if (expandDim < 0)
        {
            int outputDimSize = armnn::numeric_cast<int>(outputDims.size() + 1);
            auto getPosition = std::next(outputDims.begin() + outputDimSize, expandDim);
            outputDims.insert(getPosition, 1);
        }
    }
    else
    {
        throw InvalidArgumentException(
                fmt::format("Cannot expand dimension {} in input tensor with {} dimension {}",
                            expandDim,
                            inputDimSize,
                            CHECK_LOCATION().AsString()));
    }

    if (outputDims.size() > 4)
    {
        throw ParseException(
                fmt::format("Unsupported number of dimensions: {} for output shape for ExpandDims {} {}",
                            outputDims.size(),
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
    }

    TensorShape outShape = TensorShape(static_cast<unsigned int>(outputDims.size()),
                                       outputDims.data());

    TensorInfo outTensorInfo = inputTensorInfo;
    outTensorInfo.SetShape(outShape);

    return outTensorInfo;
}

ParsedTfOperationPtr TfParser::ParseExpandDims(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    // Number of inputs can either
    // be 1 - that indicates that the axis parameter is passed as an attribute of the operation
    // or 2 - which means that the axis parameter is passed as a second input
    std::vector<OutputOfConstNodeDef> nodes = GetTfInputNodes(nodeDef);
    const std::size_t numInputs = nodes.size();
    std::vector<OutputOfParsedTfOperation> inputs;
    std::int32_t expandDim; // axis or dim parameter. Describes which dimension to expand.
    if (numInputs == 1)
    {
        inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);
        expandDim = ReadMandatoryNodeInt32Attribute(nodeDef, "Tdim");
    }
    else
    {
        inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

        // make sure data type is int32
        IOutputSlot& prevLayerOutputSlot = inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);
        TensorInfo inputTensorInfo = prevLayerOutputSlot.GetTensorInfo();

        if (inputTensorInfo.GetDataType()!=armnn::DataType::Signed32)
        {
            throw ParseException(
                    fmt::format("The axis parameter of ExpandDims operation given as second input is not of type int32."
                                " Input {0} Node {1} {2}",
                                inputs[1].m_IndexedValue->GetNode().name(),
                                nodeDef.name(),
                                CHECK_LOCATION().AsString()));
        }

        // ensure the second input is a constant value
        if (!HasParsedConstTensor<int32_t>(inputs[1].m_IndexedValue->GetNode().name()))
        {
            throw ParseException(
                    fmt::format("ArmNN only supports ExpandDims layers with constant axis/dim parameter. "
                                "Input {0} Node {1} {2}",
                                inputs[1].m_IndexedValue->GetNode().name(),
                                nodeDef.name(),
                                CHECK_LOCATION().AsString()));
        }

        // make sure the second input is scalar or contains only a single value
        // (we don't support expand dims for multiple axis but we don't care what shape the
        //  given tensor has as long as there is only a single value in it
        //  e.g. a tensor like this [[[1]]] is completely fine)
        if (inputTensorInfo.GetNumElements() != 1)
        {
            throw ParseException(
                    fmt::format("The axis parameter of ExpandDims operation given as second input is not "
                                "allowed to hold more than one value. "
                                "Input {0} Node {1} {2}",
                                inputs[1].m_IndexedValue->GetNode().name(),
                                nodeDef.name(),
                                CHECK_LOCATION().AsString()));
        }

        ParsedConstTfOperation<int32_t>* expandDimsNode =
                PolymorphicDowncast<ParsedConstTfOperation<int32_t>*>(inputs[1].m_IndexedValue);

        memcpy(&expandDim, expandDimsNode->GetStorage(), sizeof(expandDim));
    }

    // First input is the vector that should be expanded by another dimension
    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = prevLayerOutputSlot.GetTensorInfo();

    TensorInfo outputInfo;
    outputInfo = OutputShapeOfExpandDims(nodeDef, inputTensorInfo, expandDim);

    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputInfo.GetShape();
    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, nodeDef.name().c_str());
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseFusedBatchNorm(const tensorflow::NodeDef& nodeDef,
                                                   const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 5);

    if (!HasParsedConstTensor<float>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException(
            fmt::format("ArmNN only supports FusedBatchNormalization layers with constant scale. "
                        "Input {}. Node {} {}",
                        inputs[1].m_IndexedValue->GetNode().name(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
    ParsedConstTfOperation<float>* scaleNode =
        PolymorphicDowncast<ParsedConstTfOperation<float> *>(inputs[1].m_IndexedValue);

    if (!HasParsedConstTensor<float>(inputs[2].m_IndexedValue->GetNode().name()))
    {
        throw ParseException(
            fmt::format("ArmNN only supports FusedBatchNormalization layers with constant offset. "
                        "Input {}. Node {} {}",
                        inputs[2].m_IndexedValue->GetNode().name(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
    ParsedConstTfOperation<float>* offsetNode =
        PolymorphicDowncast<ParsedConstTfOperation<float> *>(inputs[2].m_IndexedValue);

    if (!HasParsedConstTensor<float>(inputs[3].m_IndexedValue->GetNode().name()))
    {
        throw ParseException(
            fmt::format("ArmNN only supports FusedBatchNormalization layers with constant mean. "
                        "Input {}. Node {} {}",
                        inputs[3].m_IndexedValue->GetNode().name(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
    ParsedConstTfOperation<float>* meanNode =
        PolymorphicDowncast<ParsedConstTfOperation<float> *>(inputs[3].m_IndexedValue);

    if (!HasParsedConstTensor<float>(inputs[4].m_IndexedValue->GetNode().name()))
    {
        throw ParseException(
            fmt::format("ArmNN only supports FusedBatchNormalization layers with constant variance. "
                        "Input {}. Node {} {}",
                        inputs[4].m_IndexedValue->GetNode().name(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
    ParsedConstTfOperation<float>* varianceNode =
        PolymorphicDowncast<ParsedConstTfOperation<float> *>(inputs[4].m_IndexedValue);

    const std::string dataFormat = ReadOptionalNodeStringAttribute(nodeDef, "data_format", "NHWC");
    CHECK_DATA_FORMAT(nodeDef, dataFormat, "FusedBatchNorm");

    // The descriptor only has the epsilon attribute.
    BatchNormalizationDescriptor desc;
    desc.m_Eps = ReadMandatoryNodeFloatAttribute(nodeDef, "epsilon");
    desc.m_DataLayout = dataFormat == "NHWC" ? DataLayout::NHWC : DataLayout::NCHW;

    // Data for the parsed tensor args (scale, offset, mean, variance) must be stored
    // locally until the layer is added.
    std::vector<float> scaleTensorData;
    ConstTensor scaleTensor = scaleNode->GetConstTensor(scaleTensorData);

    std::vector<float> offsetTensorData;
    ConstTensor offsetTensor = offsetNode->GetConstTensor(offsetTensorData);

    std::vector<float> meanTensorData;
    ConstTensor meanTensor = meanNode->GetConstTensor(meanTensorData);

    std::vector<float> varianceTensorData;
    ConstTensor varianceTensor = varianceNode->GetConstTensor(varianceTensorData);

    IConnectableLayer* layer = m_Network->AddBatchNormalizationLayer(desc,
                                                                     meanTensor,
                                                                     varianceTensor,
                                                                     offsetTensor,
                                                                     scaleTensor,
                                                                     nodeDef.name().c_str());

    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);

    layer->GetOutputSlot(0).SetTensorInfo(inputSlot.GetTensorInfo());
    inputSlot.Connect(layer->GetInputSlot(0));

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

bool TfParser::IsSupportedLeakyReluPattern(const tensorflow::NodeDef& mulNodeDef,
                                           size_t alphaLayerIndex,
                                           const OutputOfParsedTfOperation& otherOp,
                                           armnn::IOutputSlot** outputOfLeakyRelu,
                                           armnn::ActivationDescriptor & desc)
{
    const tensorflow::NodeDef& otherNodeDef = otherOp.m_IndexedValue->GetNode();

    // Verifying all these assumptions hold:
    //
    // 1, the mulNodeDef is an elementwise multiplication node "Mul"
    // 2, the alphaLayerIndex selects a constant node from the inputs of the "Mul" node
    // 3, the inputLayerIndex selects a layer which has the same name as otherNodeDef
    //

    if (mulNodeDef.op() == "Mul")
    {
        size_t otherLayerIndex = (alphaLayerIndex == 0 ? 1 : 0);
        std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(mulNodeDef, 2);

        ARMNN_ASSERT(inputs.size() == 2);
        ARMNN_ASSERT((otherLayerIndex == 0 || alphaLayerIndex == 0));
        ARMNN_ASSERT((otherLayerIndex == 1 || alphaLayerIndex == 1));
        ARMNN_ASSERT(((otherLayerIndex + alphaLayerIndex) == 1));

        if (inputs[otherLayerIndex].m_IndexedValue->GetNode().name() == otherNodeDef.name())
        {
            if (HasParsedConstTensor<float>(inputs[alphaLayerIndex].m_IndexedValue->GetNode().name()))
            {
                ParsedConstTfOperation<float>* alpha =
                    PolymorphicDowncast<ParsedConstTfOperation<float> *>(
                        inputs[alphaLayerIndex].m_IndexedValue);

                std::vector<float> const_data;
                ConstTensor const_tensor = alpha->GetConstTensor(const_data);

                if (const_data.size() == 1)
                {
                    desc.m_Function = ActivationFunction::LeakyReLu;
                    desc.m_A = const_data[0];

                    *outputOfLeakyRelu = &(otherOp.m_IndexedValue->ResolveArmnnOutputSlot(otherOp.m_Index));
                    return true;
                }
            }
        }
    }
    return false;
}

ParsedTfOperationPtr TfParser::ParseMaximum(const tensorflow::NodeDef& nodeDef,
                                            const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    if (inputs.size() != 2)
    {
        throw ParseException(
            fmt::format("Maximum expects two inputs!. Got {} for Node {} {}",
                        inputs.size(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    auto inputNode0 = inputs[0].m_IndexedValue->GetNode();
    auto inputNode1 = inputs[1].m_IndexedValue->GetNode();
    IOutputSlot* outputOfLeakyRelu = nullptr;

    ActivationDescriptor desc;

    // A max node may be part of a LeakyRelu, with one input as a multiplication with a scalar constant,
    // i.e. one of the four possible scenarios:
    //  1, max(mul(a, x), x)
    //  2, max(mul(x, a), x)
    //  3, max(x, mul(a, x))
    //  4, max(x, mul(x, a))
    // These are handled by an activation layer.

    if (IsSupportedLeakyReluPattern(inputNode0, 0, inputs[1], &outputOfLeakyRelu, desc) ||
        IsSupportedLeakyReluPattern(inputNode0, 1, inputs[1], &outputOfLeakyRelu, desc) ||
        IsSupportedLeakyReluPattern(inputNode1, 0, inputs[0], &outputOfLeakyRelu, desc) ||
        IsSupportedLeakyReluPattern(inputNode1, 1, inputs[0], &outputOfLeakyRelu, desc))
    {
        ARMNN_ASSERT(outputOfLeakyRelu != nullptr);

        IConnectableLayer* const layer = m_Network->AddActivationLayer(desc, nodeDef.name().c_str());
        outputOfLeakyRelu->Connect(layer->GetInputSlot(0));
        layer->GetOutputSlot(0).SetTensorInfo(outputOfLeakyRelu->GetTensorInfo());
        return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
    }
    else
    {
        // Anything else is just a maximum layer.

        return AddMaximumLayer(nodeDef);
    }
}

std::pair<armnn::IOutputSlot*, armnn::IOutputSlot*> TfParser::ProcessElementwiseInputSlots(
            const tensorflow::NodeDef& nodeDef, const std::string& layerName)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);
    const unsigned int input0Dim = input0Slot->GetTensorInfo().GetNumDimensions();
    const unsigned int input1Dim = input1Slot->GetTensorInfo().GetNumDimensions();

    if (input0Dim != input1Dim)
    {
        // broadcasting where input0 and input1 have different number of dimensions
        // is only supported for 1D and 4D tensors pair
        if (input0Dim == 1 && input1Dim == 4)
        {
            input0Slot = AddBroadcastReshapeLayer(input1Slot, input0Slot, true, *m_Network, nodeDef);
        }
        else if (input0Dim == 4 && input1Dim == 1)
        {
            input1Slot = AddBroadcastReshapeLayer(input0Slot, input1Slot, true, *m_Network, nodeDef);
        }
        else
        {
            throw ParseException(
                    fmt::format("Unsupported broadcast configuration for {} operation {} {}",
                                layerName,
                                nodeDef.name(),
                                CHECK_LOCATION().AsString()));
        }
    }
    return {input0Slot, input1Slot};
}

ParsedTfOperationPtr TfParser::ProcessComparisonLayer(
    IOutputSlot* input0Slot,
    IOutputSlot* input1Slot,
    IConnectableLayer* const layer,
    const tensorflow::NodeDef& nodeDef)
{
    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    TensorInfo outputInfo = input0Slot->GetTensorInfo();
    outputInfo.SetDataType(DataType::Boolean);
    std::vector<unsigned int> outputShape;

    const TensorShape& input0Shape = input0Slot->GetTensorInfo().GetShape();
    const TensorShape& input1Shape = input1Slot->GetTensorInfo().GetShape();

    for (unsigned int i = 0; i < input0Shape.GetNumDimensions(); i++)
    {
        outputShape.push_back(std::max(input0Shape[i], input1Shape[i]));
    }

    outputInfo.SetShape(TensorShape(input0Shape.GetNumDimensions(), outputShape.data()));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ProcessElementwiseLayer(
        IOutputSlot* input0Slot,
        IOutputSlot* input1Slot,
        IConnectableLayer* const layer,
        const tensorflow::NodeDef& nodeDef)
{
    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    TensorInfo outputInfo = input0Slot->GetTensorInfo();
    std::vector<unsigned int> outputShape;

    const TensorShape& input0Shape = input0Slot->GetTensorInfo().GetShape();
    const TensorShape& input1Shape = input1Slot->GetTensorInfo().GetShape();

    for (unsigned int i = 0; i < input0Shape.GetNumDimensions(); i++)
    {
        outputShape.push_back(std::max(input0Shape[i], input1Shape[i]));
    }

    outputInfo.SetShape(TensorShape(input0Shape.GetNumDimensions(), outputShape.data()));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseGather(const tensorflow::NodeDef& nodeDef,
                                           const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    IOutputSlot& params = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot& indices = inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);
    GatherDescriptor descriptor;
    descriptor.m_Axis = ReadMandatoryNodeInt32Attribute(nodeDef, "axis");

    // Infer shape of output tensor
    unsigned int paramsDim = params.GetTensorInfo().GetNumDimensions();
    unsigned int indicesDim = indices.GetTensorInfo().GetNumDimensions();
    unsigned int outputDim = paramsDim - 1 + indicesDim;

    std::vector<unsigned int> dimSizes;

    for (unsigned int i = 0; i < indicesDim; ++i)
    {
        dimSizes.push_back(indices.GetTensorInfo().GetShape()[i]);
    }
    for (unsigned int i = 1; i < paramsDim; ++i)
    {
        dimSizes.push_back(params.GetTensorInfo().GetShape()[i]);
    }

    const TensorShape& inferredShape = TensorShape(outputDim, dimSizes.data());

    const TensorInfo inferredOutputInfo(inferredShape, params.GetTensorInfo().GetDataType());

    IConnectableLayer* const layer = m_Network->AddGatherLayer(descriptor, nodeDef.name().c_str());
    layer->GetOutputSlot(0).SetTensorInfo(inferredOutputInfo);

    params.Connect(layer->GetInputSlot(0));
    indices.Connect(layer->GetInputSlot(1));

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseGreater(const tensorflow::NodeDef& nodeDef,
                                            const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::pair<armnn::IOutputSlot*, armnn::IOutputSlot*> inputLayers = ProcessElementwiseInputSlots(nodeDef, "Greater");
    IOutputSlot* input0Slot = inputLayers.first;
    IOutputSlot* input1Slot = inputLayers.second;

    ComparisonDescriptor descriptor(ComparisonOperation::Greater);
    IConnectableLayer* const layer = m_Network->AddComparisonLayer(descriptor, nodeDef.name().c_str());

    return ProcessComparisonLayer(input0Slot, input1Slot, layer, nodeDef);
}

ParsedTfOperationPtr TfParser::ParseEqual(const tensorflow::NodeDef& nodeDef,
                                          const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::pair<armnn::IOutputSlot*, armnn::IOutputSlot*> inputLayers = ProcessElementwiseInputSlots(nodeDef, "Equal");
    IOutputSlot* input0Slot = inputLayers.first;
    IOutputSlot* input1Slot = inputLayers.second;

    ComparisonDescriptor descriptor(ComparisonOperation::Equal);
    IConnectableLayer* const layer = m_Network->AddComparisonLayer(descriptor, nodeDef.name().c_str());

    return ProcessComparisonLayer(input0Slot, input1Slot, layer, nodeDef);
}

ParsedTfOperationPtr TfParser::ParseMinimum(const tensorflow::NodeDef& nodeDef,
                                            const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::pair<armnn::IOutputSlot*, armnn::IOutputSlot*> inputLayers = ProcessElementwiseInputSlots(nodeDef, "Minimum");
    IOutputSlot* input0Slot = inputLayers.first;
    IOutputSlot* input1Slot = inputLayers.second;

    IConnectableLayer* const layer = m_Network->AddMinimumLayer(nodeDef.name().c_str());

    return ProcessElementwiseLayer(input0Slot, input1Slot, layer, nodeDef);
}

ParsedTfOperationPtr TfParser::ParseSub(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

    const TensorInfo& input0Info = input0Slot->GetTensorInfo();
    const TensorInfo& input1Info = input1Slot->GetTensorInfo();

    if (input0Info.GetNumDimensions() == 1)
    {
        const bool isNHWC = true;
        input0Slot = AddBroadcastReshapeLayer(input1Slot, input0Slot, isNHWC, *m_Network, nodeDef);
    }

    if (input1Info.GetNumDimensions() == 1)
    {
        const bool isNHWC = true;
        input1Slot = AddBroadcastReshapeLayer(input0Slot, input1Slot, isNHWC, *m_Network, nodeDef);
    }

    IConnectableLayer* const layer = m_Network->AddSubtractionLayer(nodeDef.name().c_str());

    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    if (input0Info.GetNumDimensions() == 1)
    {
        layer->GetOutputSlot(0).SetTensorInfo(input1Slot->GetTensorInfo());
    }
    else
    {
        layer->GetOutputSlot(0).SetTensorInfo(input0Slot->GetTensorInfo());
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseStack(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfConstNodeDef> nodes = GetTfInputNodes(nodeDef);

    unsigned int numInputs = static_cast<unsigned int>(nodes.size());
    if (numInputs < 1)
    {
        throw ParseException(
            fmt::format("Pack/Stack expects at least one input. Got {} for Node {} {}",
                        numInputs,
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, numInputs);
    // Use the tensor shape of the first input as the "correct" input shape in the descriptor
    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    const TensorInfo& inputTensorInfo = input0Slot->GetTensorInfo();
    auto numDimensions = inputTensorInfo.GetShape().GetNumDimensions();

    // validate axis
    int32_t axis = ReadMandatoryNodeInt32Attribute(nodeDef, "axis");
    const int sNumDimensions = (static_cast<int>(numDimensions) + 1);
    if (!(axis < sNumDimensions && axis >= -sNumDimensions))
    {
        throw ParseException(
            fmt::format("Axis index is not in range. Got {} for Node {} {}",
                        axis,
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    if (axis < 0)
    {
        axis = static_cast<int32_t>(numDimensions) + axis + 1;
    }

    StackDescriptor stackDescriptor;
    stackDescriptor.m_Axis = static_cast<uint32_t>(axis);
    stackDescriptor.m_NumInputs = static_cast<uint32_t>(numInputs);
    stackDescriptor.m_InputShape = inputTensorInfo.GetShape();

    const unsigned int supportedNumDims = 4;
    for (unsigned int viewIndex = 0; viewIndex < numInputs; ++viewIndex)
    {
        IOutputSlot& inputSlot = inputs[viewIndex].m_IndexedValue->ResolveArmnnOutputSlot(inputs[viewIndex].m_Index);
        TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

        // Double check dimensions of the tensors
        if (inputTensorInfo.GetNumDimensions() >= supportedNumDims)
        {
            throw armnn::ParseException(
                    fmt::format("The number of dimensions: {} for input tensors of the "
                                "Pack/Stack op. Number of dimensions should be less than {} {}",
                                inputTensorInfo.GetNumDimensions(),
                                supportedNumDims,
                                CHECK_LOCATION().AsString()));
        }
    }

    std::vector<unsigned int> outputDimensions;
    for (unsigned int i = 0; i < stackDescriptor.m_InputShape.GetNumDimensions(); ++i)
    {
        outputDimensions.push_back(stackDescriptor.m_InputShape[i]);
    }
    outputDimensions.insert(outputDimensions.begin() + axis, numInputs);

    // add Stack Layer
    IConnectableLayer* const layer = m_Network->AddStackLayer(stackDescriptor, nodeDef.name().c_str());

    for (unsigned int viewIndex = 0; viewIndex < numInputs; ++viewIndex)
    {
        IOutputSlot& inputSlot = inputs[viewIndex].m_IndexedValue->ResolveArmnnOutputSlot(inputs[viewIndex].m_Index);
        inputSlot.Connect(layer->GetInputSlot(viewIndex));
    }

    layer->GetOutputSlot(0).SetTensorInfo(
        armnn::TensorInfo(static_cast<uint32_t>(outputDimensions.size()),
                          outputDimensions.data(),
                          inputTensorInfo.GetDataType()));

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseTranspose(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    auto inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    const auto inputCount = inputs.size();

    if (inputCount != 2)
    {
        throw ParseException(
                fmt::format("The number of given input is {}. It should be two for Transpose op."
                            "Node {} {}",
                            inputCount,
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
    }

    auto* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);

    const auto  constInput         = inputs[GetConstInputIndex(inputs)];
    auto*       permuteVectorInput =
        PolymorphicDowncast<ParsedConstTfOperation<int32_t>*>(constInput.m_IndexedValue);
    const auto& permuteVectorInfo  = permuteVectorInput->GetTensorInfo();

    std::vector<int32_t> permuteVectorData;
    permuteVectorInput->GetConstTensor(permuteVectorData);

    std::vector<unsigned int>      armnnPermuteVectorData(permuteVectorData.begin(), permuteVectorData.end());

    const auto permutationVector = PermutationVector(armnnPermuteVectorData.data(), permuteVectorInfo.GetNumElements());
    const auto desc              = TransposeDescriptor(permutationVector);

    auto* layer = m_Network->AddTransposeLayer(desc, nodeDef.name().c_str());
    ARMNN_ASSERT(layer);

    input0Slot->Connect(layer->GetInputSlot(0));

    const auto&       input0Info = input0Slot->GetTensorInfo();
    armnn::TensorInfo outputInfo {input0Info};
    outputInfo.SetShape(armnnUtils::TransposeTensorShape(input0Info.GetShape(), desc.m_DimMappings));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

unsigned int CheckPaddingTensor(const ConstTensor& paddingTensor,
                                const TensorInfo& inputTensorInfo,
                                const std::string& nodeName)
{
    unsigned int rank = paddingTensor.GetShape()[0];
    unsigned int expectedRank = inputTensorInfo.GetNumDimensions();
    if (rank != expectedRank)
    {
        throw ParseException(
                fmt::format("Expected the padding tensor to be of rank {} not {} on Node {} {}.",
                            expectedRank,
                            rank,
                            nodeName,
                            CHECK_LOCATION().AsString()));
    }
    unsigned int second = paddingTensor.GetShape()[1];
    if (second != 2)
    {
        throw ParseException(
                fmt::format("Expected the padding tensor to be of dimensions "
                            "[{1}, 2] not [{1}, {2}] on Node {3} {4}.",
                            rank,
                            second,
                            nodeName,
                            CHECK_LOCATION().AsString()));
    }
    return rank;
}

TensorInfo CalculatePaddedOutputTensorInfo(const TensorInfo& inputTensorInfo,
                                           const std::vector<std::pair<unsigned int, unsigned int>>& padList)
{
    unsigned int numDims = inputTensorInfo.GetNumDimensions();
    std::vector<unsigned int> outDims;
    for (unsigned int i = 0; i < numDims; ++i)
    {
        unsigned int dimSize = inputTensorInfo.GetShape()[i];
        const std::pair<unsigned int, unsigned int>& dimPadding = padList[i];
        dimSize += dimPadding.first;
        dimSize += dimPadding.second;
        outDims.push_back(dimSize);
    }
    TensorInfo paddedTensorInfo = inputTensorInfo;
    unsigned int outDimsSize = static_cast<unsigned int>(outDims.size());
    paddedTensorInfo.SetShape(TensorShape{ outDimsSize, outDims.data() });
    return paddedTensorInfo;
}

ParsedTfOperationPtr TfParser::ParsePad(const tensorflow::NodeDef& nodeDef,
                                        const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    // input consists of:
    // input[0] the tensor which will be padded
    // input[1] the tensor holding the padding values
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    IOutputSlot& previousLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = previousLayerOutputSlot.GetTensorInfo();
    if (!HasParsedConstTensor<int32_t>(inputs[1].m_IndexedValue))
    {
        throw ParseException(
                fmt::format("ArmNN only supports Pad with constant padding. "
                            "Input {}. Node {} {}",
                            inputs[1].m_IndexedValue->GetNode().name(),
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));

    }
    ParsedConstTfOperation<int32_t>* paddingTensorOp =
            PolymorphicDowncast<ParsedConstTfOperation<int32_t>*>(inputs[1].m_IndexedValue);

    std::vector<int32_t> paddingTensorData;
    ConstTensor paddingTensor = paddingTensorOp->GetConstTensor(paddingTensorData);
    // paddings is an integer tensor with shape [n, 2], where n is the rank of tensor
    // and should match the rank of the input tensor that is being padded.
    // For each dimension D of input, paddings[D, 0] indicates how many values to add
    // before the contents of tensor in that dimension, and paddings[D, 1] indicates how
    // many values to add after the contents of tensor in that dimension
    // This needs to be translated into a padList for ACL
    std::vector<std::pair<unsigned int, unsigned int>> padList;
    unsigned int rank = CheckPaddingTensor(paddingTensor, inputTensorInfo, nodeDef.name());
    for (unsigned int i = 0; i < rank; ++i)
    {
        std::pair<unsigned int, unsigned int> paddingForDim;
        for (unsigned int j = 0; j < 2; j++)
        {
            unsigned int index = (i * 2) + j;
            int paddingAmount = paddingTensorData[index];
            // make sure we can cast to an unsigned value
            if (paddingAmount < 0)
            {
                throw ParseException(
                        fmt::format("Negative amount {} specified at [{}, {}] of padding tensor on Node {} {}.",
                                    paddingAmount,
                                    i,
                                    j,
                                    nodeDef.name(),
                                    CHECK_LOCATION().AsString()));
            }
            if (j == 0)
            {
                paddingForDim.first = static_cast<unsigned int>(paddingAmount);
            }
            else
            {
                paddingForDim.second = static_cast<unsigned int>(paddingAmount);
            }
        }
        padList.push_back(paddingForDim);
    }
    PadDescriptor padDescriptor(padList);
    IConnectableLayer* layer = m_Network->AddPadLayer(padDescriptor, nodeDef.name().c_str());
    previousLayerOutputSlot.Connect(layer->GetInputSlot(0));
    // Use the padding to calculate the new output tensor shape
    TensorInfo outputTensorInfo = CalculatePaddedOutputTensorInfo(inputTensorInfo, padList);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseConcat(const tensorflow::NodeDef& nodeDef,
                                           const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfConstNodeDef> nodes = GetTfInputNodes(nodeDef);

    // In tensorflow, we have the last input of the Concat layer as the axis for concatenation.
    unsigned int numInputs = static_cast<unsigned int>(nodes.size());

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, numInputs);

    // Constant tensor index
    unsigned int index = GetConstInputIndex(inputs);
    // Get the axis tensor data
    ParsedConstTfOperation<int32_t>* shapeNode =
            PolymorphicDowncast<ParsedConstTfOperation<int32_t>*>(inputs[index].m_IndexedValue);

    std::vector<int32_t> axisTensorData;
    shapeNode->GetConstTensor(axisTensorData);

    // This concatDim indicates the data format: 3 is the NHWC, 1 is the NCHW.
    const unsigned int concatDim = static_cast<unsigned int>(axisTensorData[0]);

    // Armnn supports concatenation along the channel dimension for data formats NHWC and NCHW.
    if (concatDim == 0 || concatDim == 2)
    {
        throw ParseException(
                fmt::format("Dimension {} for concatenation is not supported by Armnn. "
                            "Node {} {}",
                            concatDim,
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
    }

    const unsigned int supportedNumDims = 4;
    unsigned int numConcatViews = numInputs - 1;
    OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatViews), supportedNumDims);
    concatDescriptor.SetConcatAxis(concatDim);
    TensorShape mergeDims(supportedNumDims);
    unsigned int mergeDim = 0;
    for (unsigned int viewIndex = 0; viewIndex < numConcatViews; ++viewIndex)
    {
        // Need to double check whether it should be
        IOutputSlot& inputSlot = inputs[viewIndex].m_IndexedValue->ResolveArmnnOutputSlot(inputs[viewIndex].m_Index);
        TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

        // Double check dimensions of the tensors
        if (inputTensorInfo.GetNumDimensions() != supportedNumDims)
        {
            throw armnn::ParseException(
                    fmt::format("The number of dimensions: {} for input tensors of the "
                                "concatenation op should be {} {}",
                                inputTensorInfo.GetNumDimensions(),
                                supportedNumDims,
                                CHECK_LOCATION().AsString()));
        }

        // Copy the input tensor shape to mergeDimSizes and initialize the view origin coordinates for the current input
        mergeDims = inputTensorInfo.GetShape();
        unsigned int* viewOrigin = const_cast<unsigned int*>(concatDescriptor.GetViewOrigin(viewIndex));
        std::fill(viewOrigin, viewOrigin + supportedNumDims, 0);

        // Update the view origin coordinates and the merge dimension value
        concatDescriptor.SetViewOriginCoord(viewIndex, concatDim, mergeDim);
        mergeDim += mergeDims[concatDim];
    }

    // Update the output shape
    mergeDims[concatDim] = mergeDim;
    armnn::IConnectableLayer *layer = m_Network->AddConcatLayer(concatDescriptor, nodeDef.name().c_str());

    layer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(mergeDims, DataType::Float32));

    for (unsigned int viewIndex = 0; viewIndex < numConcatViews; ++viewIndex)
    {
        IOutputSlot& inputSlot = inputs[viewIndex].m_IndexedValue->ResolveArmnnOutputSlot(inputs[viewIndex].m_Index);
        inputSlot.Connect(layer->GetInputSlot(viewIndex));
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseShape(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    // Note: the Shape layer is handled in a special way, because:
    //        1. ARMNN doesn't support int32 tensors which it outputs.
    //        2. ARMNN works with statically shaped tensors which are known at parse time.
    //        3. because of 1. and 2. we treat the output of Shape as a temporary const int32
    //           tensor which may be used as an input to other ops, most likely a Reshape.

    const tensorflow::DataType tfDataType = ReadMandatoryNodeTypeAttribute(nodeDef, "out_type");
    if (tfDataType != tensorflow::DT_INT32)
    {
        throw ParseException(
            fmt::format("Armnn only supports DT_INT32 as out_type. Got {} for Node {} {}",
                        tensorflow::DataType_Name(tfDataType),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    const std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);
    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    const TensorInfo& prevLayerTensorInfo = prevLayerOutputSlot.GetTensorInfo();
    unsigned int prevLayerDimensions = prevLayerTensorInfo.GetNumDimensions();

    std::vector<int32_t> shapeTensorData;
    shapeTensorData.reserve(prevLayerDimensions);

    for (unsigned int i=0; i<prevLayerDimensions; ++i)
    {
        shapeTensorData.push_back(static_cast<int32_t>(prevLayerTensorInfo.GetShape()[i]));
    }

    TensorInfo shapeTensorInfo(1, &prevLayerDimensions, DataType::Signed32);

    return std::make_unique<ParsedConstTfOperation<int32_t>>(this,
                                                             nodeDef,
                                                             &shapeTensorData[0],
                                                             shapeTensorInfo);
}

ParsedTfOperationPtr TfParser::ParseReshape(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    ParsedTfOperation* inputNode = inputs[0].m_IndexedValue;

    if (!HasParsedConstTensor<int32_t>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException(
            fmt::format("ArmNN only supports Reshape layers with constant shapes. "
                        "Input {} Node {} {}",
                        inputs[1].m_IndexedValue->GetNode().name(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
    ParsedConstTfOperation<int32_t>* shapeNode =
        PolymorphicDowncast<ParsedConstTfOperation<int32_t>*>(inputs[1].m_IndexedValue);

    armnn::IOutputSlot& prevLayerOutputSlot = inputNode->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = prevLayerOutputSlot.GetTensorInfo();

    std::vector<int32_t> shapeTensorData;
    ConstTensor shapeTensor = shapeNode->GetConstTensor(shapeTensorData);
    const TensorInfo outputTensorInfo = PrepareReshape(inputTensorInfo, shapeTensorData);

    TensorShape targetShape = outputTensorInfo.GetShape();
    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = targetShape;

    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, nodeDef.name().c_str());
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseResizeBilinear(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    if (!HasParsedConstTensor<int32_t>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException(
            fmt::format("ArmNN only supports ResizeBilinear layers with constant sizes. "
                        "Input {}. Node {} {}",
                        inputs[1].m_IndexedValue->GetNode().name(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
    ParsedConstTfOperation<int32_t>* sizeNode =
        PolymorphicDowncast<ParsedConstTfOperation<int32_t>*>(inputs[1].m_IndexedValue);

    // Checks the align_corners attribute is not set.
    if (ReadOptionalNodeBoolAttribute(nodeDef, "align_corners", false))
    {
        throw ParseException(
            fmt::format("ArmNN only supports ResizeBilinear layers with align_corners set to false. "
                        "Node {} {}",
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    // Data for the parsed tensor args (size) must be stored locally.
    std::vector<int32_t> sizeTensorData;
    ConstTensor sizeTensor = sizeNode->GetConstTensor(sizeTensorData);

    // The descriptor only has target height and width attributes, which we get from the size tensor.
    ResizeDescriptor desc;
    desc.m_Method       = armnn::ResizeMethod::Bilinear;
    desc.m_TargetHeight = static_cast<uint32_t> (sizeTensorData[0]);
    desc.m_TargetWidth  = static_cast<uint32_t> (sizeTensorData[1]);
    desc.m_DataLayout   = armnn::DataLayout::NHWC;

    IConnectableLayer* layer = m_Network->AddResizeLayer(desc, nodeDef.name().c_str());

    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();
    // The input shape is always in BHWC format, this will be swizzled below; for now,
    // get the batch and channels to make up the ArmNN output shape with the target size.
    unsigned int outBatch = inputTensorInfo.GetShape()[0];
    unsigned int outChannels = inputTensorInfo.GetShape()[3];
    unsigned int outHeight = desc.m_TargetHeight;
    unsigned int outWidth = desc.m_TargetWidth;
    TensorShape outShape({outBatch, outHeight, outWidth, outChannels });
    // The output DataType is always Float32, regardless of the input DataType.
    const TensorInfo outputTensorInfo(outShape, armnn::DataType::Float32);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    inputSlot.Connect(layer->GetInputSlot(0));

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

TensorInfo OutputShapeOfSqueeze(const tensorflow::NodeDef& nodeDef, TensorInfo inputTensorInfo)
{
    ARMNN_ASSERT(nodeDef.op() == "Squeeze");
    tensorflow::DataType tfDataType = ReadMandatoryNodeTypeAttribute(nodeDef, "T");

    DataType type;
    if (tfDataType == tensorflow::DT_FLOAT)
    {
        type = DataType::Float32;
    }
    else if (tfDataType == tensorflow::DT_INT32)
    {
        type = DataType::Signed32;
    }
    else
    {
        throw ParseException(
            fmt::format("Unsupported DataType {} for Squeeze operation {} {}",
                        tensorflow::DataType_Name(tfDataType),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }


    if (inputTensorInfo.GetNumDimensions() > 4)
    {
        throw ParseException(
            fmt::format("Unsupported number of dimensions: {} for input shape for Squeeze {} {}",
                        inputTensorInfo.GetNumDimensions(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    std::vector<uint32_t> squeezeDims = ReadOptionalNodeUint32ListAttribute(nodeDef, "squeeze_dims");
    static const uint32_t dimensionSequence[] = { 0, 1, 2, 3 };

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
        throw ParseException(
            fmt::format("Unsupported number of dimensions: {} for output shape for Squeeze {} {}",
                        outputDims.size(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    TensorShape outShape = TensorShape(static_cast<unsigned int>(outputDims.size()),
                                       outputDims.data());

    TensorInfo outTensorInfo = inputTensorInfo;
    outTensorInfo.SetShape(outShape);
    outTensorInfo.SetDataType(type);

    return outTensorInfo;
}

ParsedTfOperationPtr TfParser::ParseSqueeze(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = prevLayerOutputSlot.GetTensorInfo();

    TensorInfo outputInfo;
    outputInfo = OutputShapeOfSqueeze(nodeDef, inputTensorInfo);

    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputInfo.GetShape();
    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, nodeDef.name().c_str());
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseLrn(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    NormalizationDescriptor normalizationDescriptor;
    normalizationDescriptor.m_NormMethodType = NormalizationAlgorithmMethod::LocalBrightness;
    normalizationDescriptor.m_NormChannelType = NormalizationAlgorithmChannel::Across;
    normalizationDescriptor.m_Alpha = ReadMandatoryNodeFloatAttribute(nodeDef, "alpha");
    normalizationDescriptor.m_Beta = ReadMandatoryNodeFloatAttribute(nodeDef, "beta");
    normalizationDescriptor.m_K = ReadMandatoryNodeFloatAttribute(nodeDef, "bias");
    normalizationDescriptor.m_NormSize = ReadMandatoryNodeUint32Attribute(nodeDef, "depth_radius");
    normalizationDescriptor.m_DataLayout = armnn::DataLayout::NHWC;

    // The window size must be an odd value. For a window size of (2 * n + 1), TensorFlow defines depth_radius = n.
    normalizationDescriptor.m_NormSize = normalizationDescriptor.m_NormSize * 2 + 1;

    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IConnectableLayer* layer = m_Network->AddNormalizationLayer(normalizationDescriptor,
        nodeDef.name().c_str());
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(prevLayerOutputSlot.GetTensorInfo());

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

/// An ParsedTfOperation for a MatMul node.
/// Creation of the armnn FullyConnected layer is deferred until it is actually needed, because
/// MatMul nodes are often used for the first part of a biased FullyConnected (MatMul followed
/// by Add) and in these cases armnn doesn't need a separate layer for the MatMul.
///
class ParsedMatMulTfOperation : public DeferredSingleLayerParsedTfOperation
{
public:
    ParsedMatMulTfOperation(TfParser* parser, const tensorflow::NodeDef& node)
        : DeferredSingleLayerParsedTfOperation(parser, node)
    {
    }

    void CreateLayerDeferred() override
    {
        ARMNN_ASSERT(m_Layer == nullptr);
        m_Layer = m_Parser->AddFullyConnectedLayer(m_Node, nullptr, m_Node.name().c_str());
    }
};

ParsedTfOperationPtr TfParser::ParseMatMul(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    // Defers the creation of the layer (see ParsedMatMulTfOperation).
    return std::make_unique<ParsedMatMulTfOperation>(this, nodeDef);
}

ParsedTfOperationPtr TfParser::ParseMean(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

    if (inputs.size() != 2)
    {
        throw ParseException(
                fmt::format("Mean expects two inputs!. Got {} for Node {} {}",
                            inputs.size(),
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
    }

    bool keepDims = ReadMandatoryNodeBoolAttribute(nodeDef, "keep_dims");

    ParsedConstTfOperation<int32_t>* axisNode =
             PolymorphicDowncast<ParsedConstTfOperation<int32_t>*>(inputs[1].m_IndexedValue);

    const TensorInfo& axisTensorInfo = axisNode->GetTensorInfo();

    ConstTensor axisTensor(axisTensorInfo, axisNode->GetStorage());
    const int* axisData = static_cast<const int*>(axisTensor.GetMemoryArea());

    TensorInfo outputTensorInfo;
    MeanDescriptor meanDescriptor;
    meanDescriptor.m_KeepDims = keepDims;

    // Negative axis values are supported so that the process requires
    // to convert them into the corresponding positive ones.
    // Duplicate values are also removed.
    std::vector<int> rawAxisVector(axisData, axisData + axisTensorInfo.GetNumElements());
    std::set<unsigned int> positiveAxisSet;
    int rank = static_cast<int>(inputTensorInfo.GetNumDimensions());

    std::transform(rawAxisVector.begin(), rawAxisVector.end(),
                   std::inserter(positiveAxisSet, positiveAxisSet.begin()),
                   [rank](int i) -> unsigned int { return static_cast<unsigned int>((i + rank) % rank); });

    CalculateReducedOutputTensoInfo(inputTensorInfo, positiveAxisSet, keepDims, outputTensorInfo);

    if (inputTensorInfo.GetNumDimensions() > positiveAxisSet.size())
    {
        meanDescriptor.m_Axis.assign(positiveAxisSet.begin(), positiveAxisSet.end());
    }

    IConnectableLayer* layer = m_Network->AddMeanLayer(meanDescriptor, nodeDef.name().c_str());
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    inputSlot.Connect(layer->GetInputSlot(0));

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

/// An ParsedTfOperation for a Mul node.
/// Creation of the armnn Mul layer is deferred until it is actually needed, because Mul nodes
/// are also used for the first part of a leaky relu activation function (Mul followed by Maximum)
/// and in these cases armnn doesn't need a separate layer for the Mul.
///
class ParsedMulTfOperation : public DeferredSingleLayerParsedTfOperation
{
public:
    ParsedMulTfOperation(TfParser* parser, const tensorflow::NodeDef& node)
        : DeferredSingleLayerParsedTfOperation(parser, node)
    {
    }

    void CreateLayerDeferred() override
    {
        ARMNN_ASSERT(m_Layer == nullptr);
        m_Layer = m_Parser->AddMultiplicationLayer(m_Node);
    }
};

ParsedTfOperationPtr TfParser::ParseMul(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    return std::make_unique<ParsedMulTfOperation>(this, nodeDef);
}

ParsedTfOperationPtr TfParser::ParsePlaceholder(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 0);

    const LayerBindingId layerId = armnn::numeric_cast<LayerBindingId>(m_NetworkInputsBindingInfo.size());

    auto it = m_InputShapes.find(nodeDef.name());
    if (it == m_InputShapes.end())
    {
        throw ParseException(
            fmt::format("Missing input shape for Placeholder '{}' {}",
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }
    TensorInfo tensorInfo(it->second, DataType::Float32);

    IConnectableLayer* const layer = m_Network->AddInputLayer(layerId, nodeDef.name().c_str());

    layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    TrackInputBinding(layer, layerId, tensorInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseRealDiv(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);
    return AddRealDivLayer(nodeDef);
}

ParsedTfOperationPtr TfParser::ParseRelu(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::ReLu;
    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::ParseRelu6(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::BoundedReLu;
    activationDesc.m_A = 6.0f;
    activationDesc.m_B = 0.0f;

    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::ParseSigmoid(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::Sigmoid;

    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::ParseRsqrt(const tensorflow::NodeDef &nodeDef,
    const tensorflow::GraphDef &graphDef)
{
    IgnoreUnused(graphDef);

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    ElementwiseUnaryDescriptor descriptor(UnaryOperation::Rsqrt);
    IConnectableLayer* const layer = m_Network->AddElementwiseUnaryLayer(descriptor, nodeDef.name().c_str());

    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(prevLayerOutputSlot.GetTensorInfo());

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseSoftmax(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    SoftmaxDescriptor softmaxDescriptor;
    IConnectableLayer* const layer = m_Network->AddSoftmaxLayer(softmaxDescriptor, nodeDef.name().c_str());

    IOutputSlot& prevLayerSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    prevLayerSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(prevLayerSlot.GetTensorInfo());

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseSplit(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    std::vector<OutputOfConstNodeDef> nodes = GetTfInputNodes(nodeDef);
    unsigned int numInputs = static_cast<unsigned int>(nodes.size());
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, numInputs);

    // Constant tensor index
    unsigned int index = GetConstInputIndex(inputs);
    // Get the axis tensor data
    ParsedConstTfOperation<int32_t>* shapeNode =
                PolymorphicDowncast<ParsedConstTfOperation<int32_t>*>(inputs[index].m_IndexedValue);

    std::vector<int32_t> axisTensorData;
    shapeNode->GetConstTensor(axisTensorData);

    // This splitDim indicates the data format: 3 is the NHWC, 1 is the NCHW.
    const unsigned int splitDim = static_cast<unsigned int>(axisTensorData[0]);

    // Armnn supports split along the channel dimension for data formats NHWC and NCHW.
    if (splitDim == 0 || splitDim == 2)
    {
        throw armnn::ParseException(
                fmt::format("Dimension {} for split is not supported by Armnn. "
                            "Node {} {}",
                            splitDim,
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
    }

    // As Armnn only supports splitter outputs of the same shape, therefore num_split will be limited to an integer.
    uint32_t num_split = ReadMandatoryNodeUint32Attribute(nodeDef, "num_split");

    IOutputSlot& inputSlot = inputs[1 - index].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1 - index].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

    const unsigned int supportedNumDims = 4;
    auto inputDimSize = inputTensorInfo.GetNumDimensions();

    if (inputDimSize != supportedNumDims)
    {
        throw armnn::ParseException(
                fmt::format("The number of dimensions: {} for input tensors of the "
                            "split op should be {} {}",
                            inputTensorInfo.GetNumDimensions(),
                            supportedNumDims,
                            CHECK_LOCATION().AsString()));
    }

    std::vector<unsigned int> splitterDimSizes(inputDimSize);

    // Add current input shape to splitterDimSizes
    for (unsigned int i = 0; i < inputDimSize; ++i)
    {
        splitterDimSizes[i] = inputTensorInfo.GetShape()[i];
    }

    if (splitterDimSizes[splitDim] % num_split != 0)
    {
        throw ParseException("Number of splits must evenly divide the dimension");
    }
    splitterDimSizes[splitDim] /= num_split;

    SplitterDescriptor splitDesc(num_split);
    for (unsigned int g = 0; g < num_split; ++g)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < splitterDimSizes.size(); ++dimIdx)
        {
            splitDesc.SetViewSize(g, dimIdx, splitterDimSizes[dimIdx]);
        }
        splitDesc.SetViewOriginCoord(g, splitDim, splitterDimSizes[splitDim] * g);
    }

    IConnectableLayer *layer = m_Network->AddSplitterLayer(splitDesc, nodeDef.name().c_str());

    inputSlot.Connect(layer->GetInputSlot(0));

    TensorShape outShape = TensorShape(static_cast<unsigned int>(splitterDimSizes.size()),
                                       splitterDimSizes.data());

    for (unsigned int i = 0; i < layer->GetNumOutputSlots(); ++i)
    {
        layer->GetOutputSlot(i).SetTensorInfo(armnn::TensorInfo(outShape, inputTensorInfo.GetDataType()));
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseSoftplus(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::SoftReLu;

    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::ParseStridedSlice(const tensorflow::NodeDef& nodeDef,
                                                 const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    std::vector<OutputOfConstNodeDef> nodes = GetTfInputNodes(nodeDef);
    unsigned int numInputs = static_cast<unsigned int>(nodes.size());
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, numInputs);

    ParsedConstTfOperation<int32_t>* beginNode =
            PolymorphicDowncast<ParsedConstTfOperation<int32_t> *>(inputs[1].m_IndexedValue);
    std::vector<int32_t> beginTensorData;
    beginNode->GetConstTensor(beginTensorData);

    ParsedConstTfOperation<int32_t>* endNode =
            PolymorphicDowncast<ParsedConstTfOperation<int32_t> *>(inputs[2].m_IndexedValue);
    std::vector<int32_t> endTensorData;
    endNode->GetConstTensor(endTensorData);

    ParsedConstTfOperation<int32_t>* stridesNode =
            PolymorphicDowncast<ParsedConstTfOperation<int32_t> *>(inputs[3].m_IndexedValue);
    std::vector<int32_t> stridesTensorData;
    stridesNode->GetConstTensor(stridesTensorData);

    StridedSliceDescriptor desc;
    desc.m_Begin = beginTensorData;
    desc.m_End = endTensorData;
    desc.m_Stride = stridesTensorData;
    desc.m_BeginMask = ReadMandatoryNodeInt32Attribute(nodeDef, "begin_mask");
    desc.m_EndMask = ReadMandatoryNodeInt32Attribute(nodeDef, "end_mask");
    desc.m_EllipsisMask = ReadMandatoryNodeInt32Attribute(nodeDef, "ellipsis_mask");
    desc.m_NewAxisMask = ReadMandatoryNodeInt32Attribute(nodeDef, "new_axis_mask");
    desc.m_ShrinkAxisMask = ReadMandatoryNodeInt32Attribute(nodeDef, "shrink_axis_mask");
    desc.m_DataLayout = armnn::DataLayout::NHWC;
    IConnectableLayer* const layer = m_Network->AddStridedSliceLayer(desc, nodeDef.name().c_str());

    IOutputSlot& prevLayerSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = prevLayerSlot.GetTensorInfo();

    TensorInfo outputTensorInfo;
    CalculateStridedSliceOutputTensorInfo(inputTensorInfo, desc, outputTensorInfo);

    prevLayerSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseTanh(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    IgnoreUnused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::TanH;
    activationDesc.m_A = 1.0f;
    activationDesc.m_B = 1.0f;

    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::AddActivationLayer(const tensorflow::NodeDef& nodeDef,
    ActivationDescriptor& activationDesc)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    IConnectableLayer* const layer = m_Network->AddActivationLayer(activationDesc, nodeDef.name().c_str());

    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(prevLayerOutputSlot.GetTensorInfo());
    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseMaxPool(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    return ParsePooling2d(nodeDef, graphDef, PoolingAlgorithm::Max);
}

ParsedTfOperationPtr TfParser::ParseAvgPool(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    return ParsePooling2d(nodeDef, graphDef, PoolingAlgorithm::Average);
}

ParsedTfOperationPtr TfParser::ParsePooling2d(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef, PoolingAlgorithm pooltype)
{
    IgnoreUnused(graphDef);

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);
    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

    if (inputs.size() != 1)
    {
        throw ParseException(
            fmt::format("2D Pooling expects one input!. Got {} for Node {} {}",
                        inputs.size(),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    std::string paddingString = ReadMandatoryNodeStringAttribute(nodeDef, "padding");
    std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");
    std::vector<uint32_t> strides = ReadMandatoryNodeUint32ListAttribute(nodeDef, "strides");
    std::vector<uint32_t> ksize = ReadMandatoryNodeUint32ListAttribute(nodeDef, "ksize"); // size of pool windows

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType            = pooltype;
    pooling2dDescriptor.m_PaddingMethod       = PaddingMethod::Exclude;
    pooling2dDescriptor.m_OutputShapeRounding = OutputShapeRounding::Floor;

    CHECK_DATA_FORMAT(nodeDef, dataFormat, "Pooling2D");
    DataLayout dataLayout = dataFormat == "NHWC" ? DataLayout::NHWC : DataLayout::NCHW;
    pooling2dDescriptor.m_DataLayout = dataLayout;
    DataLayoutIndexed dataLayoutIndexed(dataLayout);

    pooling2dDescriptor.m_StrideX    = strides[dataLayoutIndexed.GetWidthIndex()];
    pooling2dDescriptor.m_StrideY    = strides[dataLayoutIndexed.GetHeightIndex()];
    pooling2dDescriptor.m_PoolWidth  = ksize[dataLayoutIndexed.GetWidthIndex()];
    pooling2dDescriptor.m_PoolHeight = ksize[dataLayoutIndexed.GetHeightIndex()];

    uint32_t inputHeight = inputTensorInfo.GetShape()[dataLayoutIndexed.GetHeightIndex()];
    uint32_t inputWidth  = inputTensorInfo.GetShape()[dataLayoutIndexed.GetWidthIndex()];

    bool padding = false;
    TensorInfo outputInfo;
    unsigned int outputHeight = 0;
    unsigned int outputWidth  = 0;

    CHECK_PADDING_TYPE(nodeDef, paddingString);

    if (paddingString == "SAME")
    {
        padding = true;

        outputHeight = static_cast<uint32_t>(ceil(static_cast<float>(inputHeight) /
                                                  static_cast<float>(pooling2dDescriptor.m_StrideY)));
        outputWidth  = static_cast<uint32_t>(ceil(static_cast<float>(inputWidth) /
                                                  static_cast<float>(pooling2dDescriptor.m_StrideX)));
    }
    else if (paddingString == "VALID")
    {
        padding = false;

        outputHeight = static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputHeight - pooling2dDescriptor.m_PoolHeight + 1) /
                                              static_cast<float>(pooling2dDescriptor.m_StrideY)));
        outputWidth  = static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputWidth - pooling2dDescriptor.m_PoolWidth + 1) /
                                              static_cast<float>(pooling2dDescriptor.m_StrideX)));
    }

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                      outputHeight,
                                      outputWidth,
                                      inputTensorInfo.GetShape()[3] },
                                    DataType::Float32);
            break;
        case DataLayout::NCHW:
            outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                      inputTensorInfo.GetShape()[1],
                                      outputHeight,
                                      outputWidth },
                                    DataType::Float32);
            break;
    }

    CalcPadding(inputWidth, pooling2dDescriptor.m_PoolWidth, pooling2dDescriptor.m_StrideX,
                pooling2dDescriptor.m_PadLeft, pooling2dDescriptor.m_PadRight, padding);
    CalcPadding(inputHeight, pooling2dDescriptor.m_PoolHeight, pooling2dDescriptor.m_StrideY,
                pooling2dDescriptor.m_PadTop, pooling2dDescriptor.m_PadBottom, padding);


    IConnectableLayer* layer = m_Network->AddPooling2dLayer(pooling2dDescriptor, nodeDef.name().c_str());
    if (layer == nullptr)
    {
        throw ParseException(
            fmt::format("Failed to add pooling2d layer for {} {}",
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    inputSlot.Connect(layer->GetInputSlot(0));

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::AddAdditionLayer(const tensorflow::NodeDef& nodeDef, bool isBiasAdd)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

    const TensorInfo& input0Info = input0Slot->GetTensorInfo();
    const TensorInfo& input1Info = input1Slot->GetTensorInfo();

    if (isBiasAdd)
    {
        // BiasAdd takes bias as a 1D tensor. We need to add a reshape layer to create a 4D tensor
        // with the same data in the correct dimension for broadcast in addition.
        if(input1Info.GetNumDimensions() != 1)
        {
            throw ParseException(
                fmt::format("Unsupported bias for BiasAdd. It should be a 1D vector. "
                            "Got {} dimensions for input {}. Node {} {}",
                            input1Info.GetNumDimensions(),
                            inputs[1].m_IndexedValue->GetNode().name(),
                            nodeDef.name(),
                            CHECK_LOCATION().AsString()));
        }

        const std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");

        CHECK_DATA_FORMAT(nodeDef, dataFormat, "BiasAdd");
        input1Slot = AddBroadcastReshapeLayer(input0Slot, input1Slot, dataFormat == "NHWC", *m_Network, nodeDef);
    }
    else
    {
        if (input0Info.GetNumDimensions() == 1)
        {
            const bool isNHWC = true;
            input0Slot = AddBroadcastReshapeLayer(input1Slot, input0Slot, isNHWC, *m_Network, nodeDef);
        }

        if (input1Info.GetNumDimensions() == 1)
        {
            const bool isNHWC = true;
            input1Slot = AddBroadcastReshapeLayer(input0Slot, input1Slot, isNHWC, *m_Network, nodeDef);
        }
    }

    IConnectableLayer* const layer = m_Network->AddAdditionLayer(nodeDef.name().c_str());

    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    if (input0Info.GetNumDimensions() == input1Info.GetNumDimensions())
    {
        const TensorShape& input0Shape = input0Info.GetShape();
        const TensorShape& input1Shape = input1Info.GetShape();

        std::vector<unsigned int> outputShape;
        outputShape.reserve(input0Shape.GetNumDimensions());
        TensorInfo outputInfo(input0Info);

        for (unsigned int i = 0; i < input0Shape.GetNumDimensions(); i++)
        {
            outputShape.push_back(std::max(input0Shape[i], input1Shape[i]));
        }

        outputInfo.SetShape(TensorShape(input0Shape.GetNumDimensions(), outputShape.data()));

        layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    }
    else if (input0Info.GetNumDimensions() == 1 && isBiasAdd == false)
    {
        layer->GetOutputSlot(0).SetTensorInfo(input1Slot->GetTensorInfo());
    }
    else
    {
        layer->GetOutputSlot(0).SetTensorInfo(input0Slot->GetTensorInfo());
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::AddRealDivLayer(const tensorflow::NodeDef& nodeDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    IConnectableLayer* const layer = m_Network->AddDivisionLayer(nodeDef.name().c_str());
    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

    auto const input0NumDims = input0Slot->GetTensorInfo().GetNumDimensions();
    auto const input1NumDims = input1Slot->GetTensorInfo().GetNumDimensions();


    if (input0NumDims < input1NumDims)
    {
        const bool isNHWC = true;
        input0Slot = AddBroadcastReshapeLayer(input1Slot, input0Slot, isNHWC, *m_Network, nodeDef);
    }
    if (input1NumDims < input0NumDims)
    {
        const bool isNHWC = true;
        input1Slot = AddBroadcastReshapeLayer(input0Slot, input1Slot, isNHWC, *m_Network, nodeDef);
    }

    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    if (input0NumDims < input1NumDims)
    {
        layer->GetOutputSlot(0).SetTensorInfo(input1Slot->GetTensorInfo());
    }
    else
    {
        layer->GetOutputSlot(0).SetTensorInfo(input0Slot->GetTensorInfo());

    }
    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::AddMaximumLayer(const tensorflow::NodeDef& nodeDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

    auto const input0NumDims = input0Slot->GetTensorInfo().GetNumDimensions();
    auto const input1NumDims = input1Slot->GetTensorInfo().GetNumDimensions();

    if (input0NumDims < input1NumDims)
    {
        const bool isNHWC = true;
        input0Slot = AddBroadcastReshapeLayer(input1Slot, input0Slot, isNHWC, *m_Network, nodeDef);
    }
    if (input1NumDims < input0NumDims)
    {
        const bool isNHWC = true;
        input1Slot = AddBroadcastReshapeLayer(input0Slot, input1Slot, isNHWC, *m_Network, nodeDef);
    }

    IConnectableLayer* const layer = m_Network->AddMaximumLayer(nodeDef.name().c_str());

    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    TensorInfo outputInfo = input0Slot->GetTensorInfo();
    std::vector<unsigned int> outputShape;

    const TensorShape& input0Shape = input0Slot->GetTensorInfo().GetShape();
    const TensorShape& input1Shape = input1Slot->GetTensorInfo().GetShape();

    for (unsigned int i = 0; i < input0Shape.GetNumDimensions(); i++)
    {
        outputShape.push_back(std::max(input0Shape[i], input1Shape[i]));
    }

    outputInfo.SetShape(TensorShape(input0Shape.GetNumDimensions(), outputShape.data()));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

IConnectableLayer* TfParser::AddMultiplicationLayer(const tensorflow::NodeDef& nodeDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    IConnectableLayer* const layer = m_Network->AddMultiplicationLayer(nodeDef.name().c_str());
    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

    auto const input0NumDims = input0Slot->GetTensorInfo().GetNumDimensions();
    auto const input1NumDims = input1Slot->GetTensorInfo().GetNumDimensions();

    if (input0NumDims < input1NumDims)
    {
        const bool isNHWC = true;
        input0Slot = AddBroadcastReshapeLayer(input1Slot, input0Slot, isNHWC, *m_Network, nodeDef);
    }
    if (input1NumDims < input0NumDims)
    {
        const bool isNHWC = true;
        input1Slot = AddBroadcastReshapeLayer(input0Slot, input1Slot, isNHWC, *m_Network, nodeDef);
    }

    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    if (input0NumDims < input1NumDims)
    {
        layer->GetOutputSlot(0).SetTensorInfo(input1Slot->GetTensorInfo());
    }
    else
    {
        layer->GetOutputSlot(0).SetTensorInfo(input0Slot->GetTensorInfo());
    }
    return layer;
}

IConnectableLayer* TfParser::AddFullyConnectedLayer(const tensorflow::NodeDef& matMulNodeDef,
    const tensorflow::NodeDef* addNodeDef, const char* armnnLayerName)
{
    // Finds bias const (if applicable).
    ParsedConstTfOperation<float>* biasNode = nullptr;
    if (addNodeDef != nullptr)
    {
        std::vector<OutputOfParsedTfOperation> addInputs = GetInputParsedTfOperationsChecked(*addNodeDef, 2);
        // Finds our inputs.
        if (HasParsedConstTensor<float>(addInputs[0].m_IndexedValue->GetNode().name()))
        {
            biasNode = PolymorphicDowncast<ParsedConstTfOperation<float>*>(addInputs[0].m_IndexedValue);
        }
        else if (HasParsedConstTensor<float>(addInputs[1].m_IndexedValue->GetNode().name()))
        {
            biasNode = PolymorphicDowncast<ParsedConstTfOperation<float>*>(addInputs[1].m_IndexedValue);
        }
        else
        {
            throw ParseException(
                fmt::format("ArmNN only supports fully connected layers with constant bias. "
                            "Inputs {} and {}. AddNode {}. MatMulNode {} {}",
                            addInputs[0].m_IndexedValue->GetNode().name(),
                            addInputs[1].m_IndexedValue->GetNode().name(),
                            addNodeDef->name(),
                            matMulNodeDef.name(),
                            CHECK_LOCATION().AsString()));
        }
    }

    // Finds matmul inputs.
    ParsedConstTfOperation<float>* weightNode = nullptr;
    ParsedTfOperation* inputNode  = nullptr;
    unsigned int inputIdx = 0;
    std::vector<OutputOfParsedTfOperation> mulInputs = GetInputParsedTfOperationsChecked(matMulNodeDef, 2);
    if (HasParsedConstTensor<float>(mulInputs[0].m_IndexedValue->GetNode().name()))
    {
        weightNode = PolymorphicDowncast<ParsedConstTfOperation<float>*>(mulInputs[0].m_IndexedValue);
        inputNode = mulInputs[1].m_IndexedValue;
        inputIdx = mulInputs[1].m_Index;
    }
    else if (HasParsedConstTensor<float>(mulInputs[1].m_IndexedValue->GetNode().name()))
    {
        weightNode = PolymorphicDowncast<ParsedConstTfOperation<float>*>(mulInputs[1].m_IndexedValue);
        inputNode = mulInputs[0].m_IndexedValue;
        inputIdx = mulInputs[0].m_Index;
    }
    else
    {
        throw ParseException(
            fmt::format("ArmNN only supports fully connected layers with constant weights. "
                        "Inputs {} and {}. MatMulNode {} {}",
                        mulInputs[0].m_IndexedValue->GetNode().name(),
                        mulInputs[1].m_IndexedValue->GetNode().name(),
                        matMulNodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    std::vector<float> weightTensorData;
    // Handles weight.
    ConstTensor weights = weightNode->GetConstTensor(weightTensorData);

    FullyConnectedDescriptor desc;
    desc.m_BiasEnabled = addNodeDef != nullptr;

    IConnectableLayer* layer = nullptr;
    Optional<ConstTensor> optionalBiases;
    std::vector<float> biasTensorData;
    // Makes the layer.
    if (addNodeDef != nullptr)
    {
        ConstTensor biases = biasNode->GetConstTensor(biasTensorData);

        if (weights.GetShape()[1] != biases.GetShape()[0])
        {
            throw ParseException(
                fmt::format("Shape of matmul weights and bias do not match. "
                            "AddNode {}. MatMulNode {} {}",
                            addNodeDef->name(),
                            matMulNodeDef.name(),
                            CHECK_LOCATION().AsString()));
        }

        optionalBiases = Optional<ConstTensor>(biases);
    }
    layer = m_Network->AddFullyConnectedLayer(desc, weights, optionalBiases, armnnLayerName);

    ARMNN_ASSERT(layer != nullptr);

    inputNode->ResolveArmnnOutputSlot(inputIdx).Connect(layer->GetInputSlot(0));
    unsigned int batches = inputNode->ResolveArmnnOutputSlot(inputIdx).GetTensorInfo().GetShape()[0];

    // Handles output.
    TensorInfo outputInfo({ batches, weights.GetShape()[1] }, DataType::Float32);
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    return layer;
}

void TfParser::LoadNodeDef(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    // Gets the type of the node (assume float).
    tensorflow::DataType type = tensorflow::DT_FLOAT;
    if (nodeDef.attr().count("T") != 0)
    {
        auto attr = nodeDef.attr().at("T");
        type      = attr.type();
    }
    else if (nodeDef.attr().count("dtype") != 0)
    {
        auto attr = nodeDef.attr().at("dtype");
        type      = attr.type();
    }

    if ((type != tensorflow::DT_FLOAT && type != tensorflow::DT_INT32) && nodeDef.op() != "Const")
    {
        throw ParseException(
            fmt::format("Currently only FLOAT and INT32 are supported for tensorflow nodes (apart from Const). "
                        "Got {} for Node {} {}",
                        tensorflow::DataType_Name(type),
                        nodeDef.name(),
                        CHECK_LOCATION().AsString()));
    }

    const std::string& operation = nodeDef.op();
    auto itControlInput = std::find(m_ControlInputs.begin(), m_ControlInputs.end(), operation);
    if (itControlInput != m_ControlInputs.end())
    {
        // We currently allow Control Input from TensorFlow graph but we ignore them from ArmNN graph.
        return;
    }
    auto it = ms_OperationNameToParsingFunctions.find(operation);
    if (it != ms_OperationNameToParsingFunctions.end())
    {
        auto func = it->second;
        ParsedTfOperationPtr parsedTfOperation = (this->*func)(nodeDef, graphDef);
        ParsedTfOperation* parsedTfOperationRaw = parsedTfOperation.get();

        // Stores the parsed operation so that dependent layers can connect to it.
        auto it = m_ParsedTfOperations.find(nodeDef.name());
        if (it != m_ParsedTfOperations.end())
        {
            throw ParseException(fmt::format("Name {} used by more than one node", nodeDef.name()));
        }
        m_ParsedTfOperations[nodeDef.name()] = std::move(parsedTfOperation);

        // If this node was requested as an output from the network, then adds an ArmNN output layer.
        if (std::find(m_RequestedOutputs.begin(), m_RequestedOutputs.end(), nodeDef.name()) !=
            m_RequestedOutputs.end())
        {
            auto outId = ParseOutputId(nodeDef.name());
            const LayerBindingId layerId = armnn::numeric_cast<LayerBindingId>(m_NetworkOutputsBindingInfo.size());
            IOutputSlot& prevSlot = parsedTfOperationRaw->ResolveArmnnOutputSlot(outId.m_Index);

            TensorInfo tensorInfo = prevSlot.GetTensorInfo();

            IConnectableLayer* outputLayer = m_Network->AddOutputLayer(layerId, nodeDef.name().c_str());

            prevSlot.Connect(outputLayer->GetInputSlot(0));

            TrackOutputBinding(outputLayer, layerId, tensorInfo);
        }
    }
    else
    {
        throw ParseException(
            fmt::format("Unsupported operation {} in tensorflow::GraphDef {}",
                        operation,
                        CHECK_LOCATION().AsString()));
    }
}

void TfParser::LoadGraphDef(const tensorflow::GraphDef& graphDef)
{
    // Adds all nodes to our map.
    m_NodesByName.clear();
    m_NetworkInputsBindingInfo.clear();
    m_NetworkOutputsBindingInfo.clear();

    for (int i = 0; i < graphDef.node_size(); ++i)
    {
        const tensorflow::NodeDef& node = graphDef.node(i);
        m_NodesByName[node.name()]      = &node;
    }

    // Checks that the input nodes the user has requested exist.
    for (const auto& pair : m_InputShapes)
    {
        const std::string& requestedInputName = pair.first;
        auto nodeIt = m_NodesByName.find(requestedInputName);
        if (nodeIt == m_NodesByName.end())
        {
            throw ParseException(
                    fmt::format("Couldn't find requested input node '{}' in graph {}",
                                requestedInputName,
                                CHECK_LOCATION().AsString()));
        }
    }

    // Finds the output nodes the user requested.
    std::vector<const tensorflow::NodeDef*> targetNodes;
    for (const std::string& requestedOutputName : m_RequestedOutputs)
    {
        auto nodeIt = m_NodesByName.find(requestedOutputName);
        if (nodeIt == m_NodesByName.end())
        {
            throw ParseException(
                fmt::format("Couldn't find requested output node '{}' in graph {}",
                            requestedOutputName,
                            CHECK_LOCATION().AsString()));
        }
        targetNodes.push_back(nodeIt->second);
    }

    // Sorts them into a linear ordering such that all inputs of a node are before the node itself.
    std::vector<const tensorflow::NodeDef*> sortedNodes;
    if (!armnnUtils::GraphTopologicalSort<const tensorflow::NodeDef*>(
        targetNodes,
        [this](const tensorflow::NodeDef* node)
        {
            auto outputs = GetTfInputNodes(*node);
            std::vector<const tensorflow::NodeDef*> nodesOnly;
            for (const auto & o : outputs) {
                nodesOnly.push_back(o.m_IndexedValue);
            }
            return nodesOnly;
        },
        sortedNodes))
    {
        throw ParseException(
            fmt::format("Cycle detected in graph {}",
                        CHECK_LOCATION().AsString()));
    }

    // Parses each node in order, knowing that all inputs of a node will be processed before the node itself.
    for (const auto& it : sortedNodes)
    {
        const tensorflow::NodeDef& currentNode = *it;
        LoadNodeDef(currentNode, graphDef);
    }
}

INetworkPtr TfParser::CreateNetworkFromTextFile(const char* graphFile,
    const std::map<std::string, TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    FILE* fd = fopen(graphFile, "r");

    if (fd == nullptr)
    {
        throw FileNotFoundException(
            fmt::format("Graph file {} failed to open {}",
                        graphFile,
                        CHECK_LOCATION().AsString()));
    }

    // Parses the file into a message.
    tensorflow::GraphDef graphDef;
    auto                 input   = new google::protobuf::io::FileInputStream(fileno(fd));
    bool                 success = google::protobuf::TextFormat::Parse(input, &graphDef);
    delete input;
    fclose(fd);

    if (!success)
    {
        throw ParseException(
            fmt::format("Failed to parse graph file {}",
                        CHECK_LOCATION().AsString()));
    }

    return CreateNetworkFromGraphDef(graphDef, inputShapes, requestedOutputs);
}

INetworkPtr TfParser::CreateNetworkFromString(const char* protoText,
    const std::map<std::string, TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    // Parses the string into a message.
    tensorflow::GraphDef graphDef;
    bool success = google::protobuf::TextFormat::ParseFromString(protoText, &graphDef);

    if (!success)
    {
        throw ParseException(
            fmt::format("Failed to parse graph file {}",
                        CHECK_LOCATION().AsString()));
    }

    return CreateNetworkFromGraphDef(graphDef, inputShapes, requestedOutputs);
}

INetworkPtr TfParser::CreateNetworkFromBinaryFile(const char* graphFile,
    const std::map<std::string, TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    FILE* fd = fopen(graphFile, "rb");

    if (fd == nullptr)
    {
        throw FileNotFoundException(
            fmt::format("Graph file {} failed to open {}",
                        graphFile,
                        CHECK_LOCATION().AsString()));
    }

    // Parses the file into a message.
    tensorflow::GraphDef graphDef;

    google::protobuf::io::FileInputStream  inStream(fileno(fd));
    google::protobuf::io::CodedInputStream codedStream(&inStream);
    codedStream.SetTotalBytesLimit(INT_MAX);
    bool success = graphDef.ParseFromCodedStream(&codedStream);
    fclose(fd);

    if (!success)
    {
        throw ParseException(
            fmt::format("Failed to parse protobuf file {} {}",
                        graphFile,
                        CHECK_LOCATION().AsString()));
    }

    return CreateNetworkFromGraphDef(graphDef, inputShapes, requestedOutputs);
}

INetworkPtr TfParser::CreateNetworkFromGraphDef(const tensorflow::GraphDef& graphDef,
    const std::map<std::string, TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    m_Network = INetwork::Create();

    m_InputShapes = inputShapes;
    if (requestedOutputs.size() == 0)
    {
        throw ParseException(
            fmt::format("requestedOutputs must have at least one entry {}",
                        CHECK_LOCATION().AsString()));
    }
    m_RequestedOutputs = requestedOutputs;

    try
    {
        LoadGraphDef(graphDef);
    }
    catch (const ParseException& e)
    {
        Cleanup();
        throw e;
    }

    Cleanup();

    return std::move(m_Network);
}

void TfParser::Cleanup()
{
    // Cleanup, in case we reuse this parser.
    m_InputShapes.clear();
    m_RequestedOutputs.clear();
    m_NodesByName.clear();
    m_ParsedTfOperations.clear();
}

BindingPointInfo TfParser::GetNetworkInputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "input", m_NetworkInputsBindingInfo);
}

BindingPointInfo TfParser::GetNetworkOutputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "output", m_NetworkOutputsBindingInfo);
}

std::pair<LayerBindingId, TensorInfo> TfParser::GetBindingInfo(const std::string& layerName,
    const char* bindingPointDesc,
    const std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    auto it = nameToBindingInfo.find(layerName);
    if (it == nameToBindingInfo.end())
    {
        throw InvalidArgumentException(
            fmt::format("Unknown {} '{}' {}",
                        bindingPointDesc,
                        layerName,
                        CHECK_LOCATION().AsString()));
    }
    return it->second;
}

void TfParser::TrackInputBinding(IConnectableLayer* layer, LayerBindingId id, const TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, "input", m_NetworkInputsBindingInfo);
}

void TfParser::TrackOutputBinding(IConnectableLayer* layer, LayerBindingId id, const TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, "output", m_NetworkOutputsBindingInfo);
}

void TfParser::TrackBindingPoint(IConnectableLayer* layer,
    LayerBindingId id,
    const TensorInfo& tensorInfo,
    const char* bindingPointDesc,
    std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    const std::string layerName = layer->GetName();
    auto it = nameToBindingInfo.find(layerName);
    if (it == nameToBindingInfo.end())
    {
        nameToBindingInfo[layerName] = std::make_pair(id, tensorInfo);
    }
    else
    {
        throw ParseException(
            fmt::format("Id {} used by more than one {} layer {}",
                        id,
                        bindingPointDesc,
                        CHECK_LOCATION().AsString()));
    }
}

} // namespace armnnTfParser
