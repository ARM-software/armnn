//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "CaffeParser.hpp"

#include "armnn/Descriptors.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Utils.hpp"
#include "armnn/Exceptions.hpp"

#include "GraphTopologicalSort.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>

// Caffe
#include "caffe/proto/caffe.pb.h"

// ProtoBuf
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>

#include <cmath>
#include <sstream>
#include <queue>
#include <fcntl.h>

/// Caffe networks are loaded from protobuf files (binary or text) using the protobuf library and the generated
/// code from caffe.pb.h. This gives us a caffe::NetParameter which is an in-memory version of the file.
/// This contains a flat list of Caffe 'layers' (e.g. convolution, pooling etc.).
/// Each layer has inputs (called "bottoms") and outputs (called "tops"). Data flows from bottom to top.
/// The bottoms of a layer refer to the tops of other layers, not their names.
/// The names of layers seem to be arbitrary (you could rename a layer and the network wouldn't need any other changes).
///
/// Some layers (e.g. Relu) can be configured so that their top and bottom are both the same. This is called an
/// "in-place" layer and is a Caffe runtime feature used to reduce memory usage by modifying tensors in-place.
/// This isn't relevant to the parser and so we preprocess these layers to convert them to regular layers, to result
/// in a consistent graph structure.

namespace armnnCaffeParser
{

using namespace armnn;
using namespace caffe;
using namespace std;
using namespace google::protobuf::io;

const std::map<std::string, CaffeParser::OperationParsingFunction> CaffeParser::ms_CaffeLayerNameToParsingFunctions = {
    { "Input",        &CaffeParser::ParseInputLayer },
    { "Convolution",  &CaffeParser::ParseConvLayer },
    { "Pooling",      &CaffeParser::ParsePoolingLayer },
    { "ReLU",         &CaffeParser::ParseReluLayer },
    { "LRN",          &CaffeParser::ParseLRNLayer },
    { "InnerProduct", &CaffeParser::ParseInnerProductLayer },
    { "Softmax",      &CaffeParser::ParseSoftmaxLayer },
    { "Eltwise",      &CaffeParser::ParseEltwiseLayer },
    { "Concat",       &CaffeParser::ParseConcatLayer },
    { "BatchNorm",    &CaffeParser::ParseBatchNormLayer },
    { "Scale",        &CaffeParser::ParseScaleLayer },
    { "Split",        &CaffeParser::ParseSplitLayer },
    { "Dropout",      &CaffeParser::ParseDropoutLayer},
    { "Reorg",        &CaffeParser::ParseReorgLayer },
    { "DetectionOutput", &CaffeParser::ParseDetectionOutputLayer},
};

ICaffeParser* ICaffeParser::CreateRaw()
{
    return new CaffeParser();
}

ICaffeParserPtr ICaffeParser::Create()
{
    return ICaffeParserPtr(CreateRaw(), &ICaffeParser::Destroy);
}

void ICaffeParser::Destroy(ICaffeParser* parser)
{
    delete parser;
}

CaffeParser::CaffeParser()
: m_Network(nullptr, nullptr)
{

}

void GetDataFromBlob(const LayerParameter& layerParam, vector<float>& outData, unsigned int blobIndex)
{
    if (blobIndex >= boost::numeric_cast<unsigned int>(layerParam.blobs_size()))
    {
        throw ParseException(boost::str(boost::format("Expected data blob at index %1% in layer %2% not found")
            % blobIndex % layerParam.name()));
    }

    const BlobProto& blob = layerParam.blobs(boost::numeric_cast<int>(blobIndex));

    if (boost::numeric_cast<size_t>(blob.data_size()) != outData.size())
    {
        throw ParseException(boost::str(boost::format(
            "Data blob at index %1% in layer %2% has an unexpected size. Expected %3% elements but got %4% elements")
            % blobIndex % layerParam.name() % outData.size() % blob.data_size()));
    }

    for (unsigned int i = 0; i < outData.size(); ++i)
    {
        outData[i] = blob.data(boost::numeric_cast<int>(i));
    }
}

bool IsInRange(unsigned int value, unsigned int min, unsigned int max)
{
    return (value >= min && value <= max) ? true : false;
}

template <typename T>
size_t SizeOfVectorData(const vector<T>& vec)
{
    return vec.size() * sizeof(T);
}

void ValidateNumInputsOutputs(const caffe::LayerParameter& layerParameter,
                              unsigned int                 numInputs,
                              unsigned int                 numOutputs)
{
    int numInputsActual = layerParameter.bottom_size();
    if (numInputs != boost::numeric_cast<unsigned int>(numInputsActual))
    {
        throw ParseException("Loading layer: invalid number of inputs");
    }

    int numOutputsActual = layerParameter.top_size();
    if (numOutputs != boost::numeric_cast<unsigned int>(numOutputsActual))
    {
        throw ParseException("Loading layer: invalid number of outputs");
    }
}

BindingPointInfo CaffeParser::GetNetworkInputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "input", m_NetworkInputsBindingInfo);
}

BindingPointInfo CaffeParser::GetNetworkOutputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "output", m_NetworkOutputsBindingInfo);
}

std::pair<armnn::LayerBindingId, armnn::TensorInfo> CaffeParser::GetBindingInfo(const std::string& layerName,
    const char* bindingPointDesc,
    const std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    auto it = nameToBindingInfo.find(layerName);
    if (it == nameToBindingInfo.end())
    {
        throw InvalidArgumentException(boost::str(boost::format("Unknown %1% '%2%'") % bindingPointDesc % layerName));
    }
    return it->second;
}

TensorInfo CaffeParser::BlobShapeToTensorInfo(const caffe::BlobShape& blobShape) const
{
    std::vector<unsigned int> shape;
    for (int j = 0; j < blobShape.dim_size(); ++j)
    {
        shape.push_back(static_cast<unsigned int>(blobShape.dim(j)));
    }

    return TensorInfo(boost::numeric_cast<unsigned int>(shape.size()), shape.data(), DataType::Float32);
}

BlobShape TensorDescToBlobShape(const TensorInfo& desc)
{
    BlobShape ret;
    for (unsigned int i = 0; i < desc.GetNumDimensions(); ++i)
    {
        ret.add_dim(i);
        ret.set_dim(boost::numeric_cast<int>(i), desc.GetShape()[i]);
    }

    return ret;
}

vector<const LayerParameter*> CaffeParser::GetInputs(const LayerParameter& layerParam)
{
    std::vector<const caffe::LayerParameter*> ret;
    ret.reserve(boost::numeric_cast<size_t>(layerParam.bottom_size()));
    for (int j = 0; j < layerParam.bottom_size(); ++j)
    {
        std::string inputName = layerParam.bottom(j);
        auto inputIt = m_CaffeLayersByTopName.find(inputName);
        if (inputIt == m_CaffeLayersByTopName.end())
        {
            throw ParseException(
                "Can't find Caffe layer with top called '" + inputName + "', which is listed as an input of '" +
                layerParam.name() + "'");
        }
        ret.push_back(inputIt->second);
    }

    return ret;
}

void CaffeParser::ParseInputLayer(const LayerParameter& layerParam)
{
    BOOST_ASSERT(layerParam.type() == "Input");
    ValidateNumInputsOutputs(layerParam, 0, 1);

    const InputParameter& param = layerParam.input_param();

    const armnn::LayerBindingId inputId = boost::numeric_cast<armnn::LayerBindingId>(m_NetworkInputsBindingInfo.size());
    armnn::IConnectableLayer* const inputLayer = m_Network->AddInputLayer(inputId, layerParam.name().c_str());

    // Decide on the tensor info for this input. This can be specified in the Caffe network but can also
    // be overriden by user input (m_inputShapes).
    armnn::TensorInfo inputTensorInfo;

    const BlobShape* originalShape = param.shape_size() > 0 && param.shape(0).dim_size() > 0 ?
        &param.shape(0) : nullptr;
    if (originalShape)
    {
        inputTensorInfo = BlobShapeToTensorInfo(*originalShape);
    }

    auto overrideIt = m_InputShapes.find(layerParam.name());
    if (overrideIt != m_InputShapes.end())
    {
        const TensorShape& overrideShape = overrideIt->second;
        if (originalShape &&
            (    originalShape->dim(1) != overrideShape[1]
              || originalShape->dim(2) != overrideShape[2]
              || originalShape->dim(3) != overrideShape[3]))
        {
            throw ParseException("Parsed input shape for '" + layerParam.name() +
                "' is incompatible with the override provided");
        }
        inputTensorInfo.SetShape(overrideShape);
    }
    else if (!originalShape)
    {
        throw ParseException("No input descriptor given for '" + layerParam.name() +
            "' and no input shape found in caffe model");
    }

    TrackInputBinding(inputLayer, inputId, inputTensorInfo);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), inputLayer->GetOutputSlot(0));
}

void CaffeParser::ParseConvLayer(const LayerParameter& layerParam)
{
    BOOST_ASSERT(layerParam.type() == "Convolution");
    ValidateNumInputsOutputs(layerParam, 1, 1);

    ConvolutionParameter convParam      = layerParam.convolution_param();
    BlobShape inputShape = TensorDescToBlobShape(GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo());

    unsigned int kernelH = 0;
    unsigned int kernelW = 0;
    if (convParam.has_kernel_h() && convParam.has_kernel_w())
    {
        kernelH = convParam.kernel_h();
        kernelW = convParam.kernel_w();
    }
    else if (convParam.kernel_size_size() > 0)
    {
        kernelH = (convParam.kernel_size()).Get(0);
        kernelW = (convParam.kernel_size()).Get(0);
    }
    else
    {
        throw ParseException("Loading Convolution Layer: Kernel Size defined Illegally");
    }

    if (!IsInRange(kernelH, 0, 11) || !IsInRange(kernelW, 0, 11) || (kernelH != kernelW))
    {
        throw ParseException("Loading Convolution Layer: Kernel has invalid size");
    }

    unsigned int strideH = 0;
    unsigned int strideW = 0;

    if (convParam.has_stride_h() && convParam.has_stride_w())
    {
        strideH = convParam.stride_h();
        strideW = convParam.stride_w();
    }
    else if (convParam.stride_size() > 0)
    {
        strideH = (convParam.stride()).Get(0);
        strideW = (convParam.stride()).Get(0);
    }
    else
    {
        // Caffe stride default is 1
        strideH = strideW = 1;
    }

    if (!IsInRange(strideH, 0, 11) || !IsInRange(strideW, 0, 11) || (strideH != strideW))
    {
        throw ParseException("Loading Convolution Layer: stride has invalid size");
    }

    unsigned int padH = 0;
    unsigned int padW = 0;

    if (convParam.has_pad_h() && convParam.has_pad_w())
    {
        padH = convParam.pad_h();
        padW = convParam.pad_w();
    }
    else if (convParam.pad_size() > 0)
    {
        padH = (convParam.pad()).Get(0);
        padW = (convParam.pad()).Get(0);
    }
    else
    {
        padH = 0;
        padW = 0;
    }

    if (!IsInRange(padH, 0, 11) || !IsInRange(padW, 0, 11) || (padH != padW))
    {
        throw ParseException("Loading Convolution Layer: pad has invalid size");
    }

    // Handle grouping
    const unsigned int numGroups = convParam.has_group() ? convParam.group() : 1;
    armnn::IOutputSlot& inputConnection = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0));

    vector<string> convLayerNames(numGroups);
    vector<armnn::IConnectableLayer*> convLayers(numGroups);
    convLayerNames[0] = layerParam.name();

    armnn::IConnectableLayer* splitterLayer = nullptr;
    if (numGroups > 1)
    {
        // This convolution is to be applied to chunks of the input data so add a splitter layer

        // Redirect the convolution input to the splitter
        unsigned int splitterDimSizes[4] = {static_cast<unsigned int>(inputShape.dim(0)),
                                            static_cast<unsigned int>(inputShape.dim(1)),
                                            static_cast<unsigned int>(inputShape.dim(2)),
                                            static_cast<unsigned int>(inputShape.dim(3))};

        // Split dimension 1 of the splitter output shape and conv input shapes
        // according to the number of groups
        splitterDimSizes[1] /= numGroups;
        inputShape.set_dim(1, splitterDimSizes[1]);

        // This is used to describe how the input is to be split
        ViewsDescriptor splitterDesc(numGroups);

        // Create an output node for each group, giving each a unique name
        for (unsigned int g = 0; g < numGroups; ++g)
        {
            // Work out the names of the splitter layers child convolutions
            stringstream ss;
            ss << layerParam.name() << "_" << g;
            convLayerNames[g] = ss.str();

            splitterDesc.SetViewOriginCoord(g, 1, splitterDimSizes[1] * g);

            // Set the size of the views.
            for (unsigned int dimIdx=0; dimIdx < 4; dimIdx++)
            {
                splitterDesc.SetViewSize(g, dimIdx, splitterDimSizes[dimIdx]);
            }
        }

        const std::string splitterLayerName = std::string("splitter_") + layerParam.bottom(0);

        // Add the splitter layer
        splitterLayer = m_Network->AddSplitterLayer(splitterDesc,
            splitterLayerName.c_str());

        inputConnection.Connect(splitterLayer->GetInputSlot(0));
        for (unsigned int i = 0; i < splitterLayer->GetNumOutputSlots(); i++)
        {
            splitterLayer->GetOutputSlot(i).SetTensorInfo(BlobShapeToTensorInfo(inputShape));
        }
    }

    // Ignored Caffe Parameters
    // * Dilation Size
    // * Weight Filler
    // * Bias Filler
    // * Engine
    // * Force nd_im2col
    // * Axis

    // Not Available ArmNN Interface Parameters
    // * Rounding policy;

    Convolution2dDescriptor convolution2dDescriptor;
    convolution2dDescriptor.m_PadLeft        = padW;
    convolution2dDescriptor.m_PadRight       = padW;
    convolution2dDescriptor.m_PadTop         = padH;
    convolution2dDescriptor.m_PadBottom      = padH;
    convolution2dDescriptor.m_StrideX        = strideW;
    convolution2dDescriptor.m_StrideY        = strideH;

    unsigned int numFilters = convParam.num_output();

    // Populate convolution output tensor descriptor dimensions
    BlobShape outputShape;
    outputShape.add_dim(0);
    outputShape.set_dim(0, inputShape.dim(0));
    outputShape.add_dim(1);
    // Ensure that dimension 1 of the convolution output is split according to the number of groups.
    outputShape.set_dim(1, numFilters / numGroups);
    outputShape.add_dim(2);
    outputShape.set_dim(
        2, (static_cast<int>(static_cast<float>(inputShape.dim(2) + 2 * padH - kernelH) /
            boost::numeric_cast<float>(strideH)) + 1));
    outputShape.add_dim(3);
    outputShape.set_dim(
        3, (static_cast<int>(static_cast<float>(inputShape.dim(3) + 2 * padW - kernelW) /
            boost::numeric_cast<float>(strideW)) + 1));

    // Load the weight data for ALL groups
    vector<float> weightData(boost::numeric_cast<size_t>(numGroups * inputShape.dim(1) * outputShape.dim(1) *
        kernelH * kernelW));
    GetDataFromBlob(layerParam, weightData, 0);

    const unsigned int weightDimSizes[4] = {
        static_cast<unsigned int>(outputShape.dim(1)), static_cast<unsigned int>(inputShape.dim(1)), kernelH, kernelW};

    // Bias data - This defaults to true in Caffe
    TensorInfo biasInfo;
    vector<float> biasData;
    convolution2dDescriptor.m_BiasEnabled = convParam.has_bias_term() ? convParam.bias_term() : true;
    if (convolution2dDescriptor.m_BiasEnabled)
    {
        biasData.resize(boost::numeric_cast<size_t>(numGroups * outputShape.dim(1)), 1.f);
        GetDataFromBlob(layerParam, biasData, 1);

        const unsigned int biasDimSizes[1] = {static_cast<unsigned int>(outputShape.dim(1))};
        biasInfo = TensorInfo(1, biasDimSizes, DataType::Float32);
    }

    const unsigned int numWeightsPerGroup = boost::numeric_cast<unsigned int>(weightData.size()) / numGroups;
    const unsigned int numBiasesPerGroup  = boost::numeric_cast<unsigned int>(biasData.size()) / numGroups;

    armnn::IConnectableLayer* returnLayer = nullptr;

    for (unsigned int g = 0; g < numGroups; ++g)
    {
        // set the slot index, group 0 should be connected to the 0th output of the splitter
        // group 1 should be connected to the 1st output of the splitter

        // Pull out the weights for this group from that loaded from the model file earlier
        ConstTensor weights(TensorInfo(4, weightDimSizes, DataType::Float32),
                            weightData.data() + numWeightsPerGroup * g);

        IConnectableLayer* convLayer = nullptr;
        if (convolution2dDescriptor.m_BiasEnabled)
        {
            // Pull out the biases for this group from that loaded from the model file earlier
            ConstTensor biases(biasInfo, biasData.data() + numBiasesPerGroup * g);

            convLayer = m_Network->AddConvolution2dLayer(convolution2dDescriptor,
                weights, biases, convLayerNames[g].c_str());
        }
        else
        {
            convLayer = m_Network->AddConvolution2dLayer(convolution2dDescriptor,
                weights, convLayerNames[g].c_str());
        }
        convLayers[g] = convLayer;

        // If we have more than one group then the input to the nth convolution the splitter layer's nth output,
        // otherwise it's the regular input to this layer.
        armnn::IOutputSlot& splitterInputConnection = splitterLayer ? splitterLayer->GetOutputSlot(g) : inputConnection;
        splitterInputConnection.Connect(convLayer->GetInputSlot(0));
        convLayer->GetOutputSlot(0).SetTensorInfo(BlobShapeToTensorInfo(outputShape));

        returnLayer = convLayer;
    }

    if (numGroups > 1)
    {
        // If the convolution was performed in chunks, add a layer to merge the results

        // The merge input shape matches that of the convolution output
        unsigned int mergeDimSizes[4] = {static_cast<unsigned int>(outputShape.dim(0)),
                                         static_cast<unsigned int>(outputShape.dim(1)),
                                         static_cast<unsigned int>(outputShape.dim(2)),
                                         static_cast<unsigned int>(outputShape.dim(3))};

        // This is used to describe how the input is to be merged
        OriginsDescriptor mergeDesc(numGroups);

        // Now create an input node for each group, using the name from
        // the output of the corresponding convolution
        for (unsigned int g = 0; g < numGroups; ++g)
        {
            mergeDesc.SetViewOriginCoord(g, 1, mergeDimSizes[1] * g);
        }

        // Make sure the output from the merge is the correct size to hold the data for all groups
        mergeDimSizes[1] *= numGroups;
        outputShape.set_dim(1, mergeDimSizes[1]);

        // The merge layer just assumes the name of the original convolution
        // layer so the following layer connection "just works"
        const string mergeOutputName = layerParam.name();

        // Finally add the merge layer
        IConnectableLayer* layer = m_Network->AddMergerLayer(mergeDesc, mergeOutputName.c_str());

        for (unsigned int g = 0; g < numGroups; ++g)
        {
            convLayers[g]->GetOutputSlot(0).Connect(layer->GetInputSlot(g));
        }
        layer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(4, mergeDimSizes, DataType::Float32));

        returnLayer = layer;
    }

    BOOST_ASSERT(returnLayer);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), returnLayer->GetOutputSlot(0));
}

void CaffeParser::ParsePoolingLayer(const LayerParameter& layerParam)
{
    ValidateNumInputsOutputs(layerParam, 1, 1);

    PoolingParameter param = layerParam.pooling_param();

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo();

    // Kernel size
    unsigned int kernel_h = 0;
    unsigned int kernel_w = 0;
    if (param.has_kernel_h() && param.has_kernel_w())
    {
        kernel_h = param.kernel_h();
        kernel_w = param.kernel_w();
    }
    else if (param.kernel_size() > 0)
    {
        kernel_h = param.kernel_size();
        kernel_w = param.kernel_size();
    }
    else if (param.has_global_pooling())
    {
        kernel_h = inputInfo.GetShape()[2];
        kernel_w = inputInfo.GetShape()[3];
    }
    else
    {
        throw ParseException("Loading Pooling Layer: Kernel Size defined Illegally");
    }

    if (!IsInRange(kernel_h, 0, 11) || !IsInRange(kernel_w, 0, 11) || (kernel_h != kernel_w))
    {
        throw ParseException(boost::str(
            boost::format("Loading Pooling Layer: kernel has invalid size: %1% x %2%") % kernel_h % kernel_w));
    }

    // Strides
    // Default to a valid value for the case of global pooling (where the strides don't have to be explicitly set)
    unsigned int stride_h = 1;
    unsigned int stride_w = 1;
    if (param.has_stride_h() && param.has_stride_w())
    {
        stride_h = param.stride_h();
        stride_w = param.stride_w();
    }
    else if (param.has_stride())
    {
        stride_h = param.stride();
        stride_w = param.stride();
    }
    else if (!param.has_global_pooling())
    {
        throw ParseException("Loading Pooling Layer: Stride Size defined Illegally");
    }

    if (!IsInRange(stride_h, 0, 11) || !IsInRange(stride_w, 0, 11) || (stride_h != stride_w))
    {
        throw ParseException("Loading Pooling Layer: stride has invalid size");
    }

    // Padding
    unsigned int pad_h = 0;
    unsigned int pad_w = 0;
    if (param.has_pad_h() && param.has_pad_w())
    {
        pad_h = param.pad_h();
        pad_w = param.pad_w();
    }
    else if (param.has_pad())
    {
        pad_h = param.pad();
        pad_w = param.pad();
    }
    else
    {
        pad_h = 0;
        pad_w = 0;
    }

    if (!IsInRange(pad_h, 0, 11) || !IsInRange(pad_w, 0, 11) || (pad_h != pad_w))
    {
        throw ParseException("Loading Pooling Layer: pad has invalid size");
    }

    // Ignored Caffe Parameters
    //      Stochastic Pooling
    //      Engine

    // Populate Weight and Bias Filter Descriptor
    Pooling2dDescriptor pooling2dDescriptor;
    if (param.has_pool())
    {
        PoolingParameter_PoolMethod p = param.pool();
        switch (p)
        {
            case PoolingParameter_PoolMethod_MAX:
            {
                pooling2dDescriptor.m_PoolType = PoolingAlgorithm::Max;
                break;
            }
            case PoolingParameter_PoolMethod_AVE:
            {
                pooling2dDescriptor.m_PoolType = PoolingAlgorithm::Average;
                break;
            }
            case PoolingParameter_PoolMethod_STOCHASTIC:
            {
                throw ParseException("Loading Pooling Layer: Stochastic Pooling Not Supported");
            }
            default:
            {
                throw ParseException("Loading Pooling Layer: Mode Not Supported");
            }
        }
    }
    else
    {
        throw ParseException("Loading Pooling Layer: No Pooling Method Defined");
    }

    pooling2dDescriptor.m_PadLeft     = pad_w;
    pooling2dDescriptor.m_PadRight    = pad_w;
    pooling2dDescriptor.m_PadTop      = pad_h;
    pooling2dDescriptor.m_PadBottom   = pad_h;
    pooling2dDescriptor.m_StrideX     = stride_w;
    pooling2dDescriptor.m_StrideY     = stride_h;
    pooling2dDescriptor.m_PoolWidth   = kernel_w;
    pooling2dDescriptor.m_PoolHeight  = kernel_h;

    pooling2dDescriptor.m_OutputShapeRounding = OutputShapeRounding::Ceiling;
    pooling2dDescriptor.m_PaddingMethod  = PaddingMethod::IgnoreValue;

    armnn::IConnectableLayer* poolingLayer = m_Network->AddPooling2dLayer(pooling2dDescriptor,
        layerParam.name().c_str());


    TensorInfo outputInfo(
        { inputInfo.GetShape()[0],
          inputInfo.GetShape()[1],
          static_cast<unsigned int>(ceil(
              static_cast<float>(inputInfo.GetShape()[2] + 2 * pad_h - kernel_h) /
              boost::numeric_cast<float>(stride_h))) + 1,
          static_cast<unsigned int>(ceil(
              static_cast<float>(inputInfo.GetShape()[3] + 2 * pad_w - kernel_w) /
              boost::numeric_cast<float>(stride_w))) + 1 },
        DataType::Float32);

    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).Connect(poolingLayer->GetInputSlot(0));
    poolingLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), poolingLayer->GetOutputSlot(0));
}

void CaffeParser::ParseReluLayer(const LayerParameter& layerParam)
{
    ValidateNumInputsOutputs(layerParam, 1, 1);

    const string& name = layerParam.name();
    const ReLUParameter& param = layerParam.relu_param();

    ActivationDescriptor activationDescriptor;
    const float negativeSlope = param.negative_slope();
    if (negativeSlope == 0.0f)
    {
        activationDescriptor.m_Function = ActivationFunction::ReLu;
    }
    else
    {
        activationDescriptor.m_Function = ActivationFunction::LeakyReLu;
        activationDescriptor.m_A = negativeSlope;
    }

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo();
    IConnectableLayer* const activationLayer = m_Network->AddActivationLayer(activationDescriptor, name.c_str());
    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), activationLayer->GetOutputSlot(0));
}

void CaffeParser::ParseLRNLayer(const LayerParameter& layerParam)
{
    ValidateNumInputsOutputs(layerParam, 1, 1);

    LRNParameter param = layerParam.lrn_param();

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo();

    // Ignored BATCH NORMALIZATION Caffe Parameters
    // Ignored MVN Caffe Parameters
    // Ignored LRN Caffe Parameters
    //      Engine

    NormalizationDescriptor normalizationDescriptor;
    if (param.has_norm_region())
    {
        LRNParameter_NormRegion n = param.norm_region();
        switch (n)
        {
            case LRNParameter_NormRegion_ACROSS_CHANNELS:
            {
                normalizationDescriptor.m_NormChannelType = NormalizationAlgorithmChannel::Across;
                break;
            }
            case LRNParameter_NormRegion_WITHIN_CHANNEL:
            {
                normalizationDescriptor.m_NormChannelType = NormalizationAlgorithmChannel::Within;
                break;
            }
            default:
                throw ParseException("Loading LRN Layer: Mode Not Supported");
        }
    }
    else
    {
        // Caffe defaults to normalization across channels
        normalizationDescriptor.m_NormChannelType = NormalizationAlgorithmChannel::Across;
    }

    normalizationDescriptor.m_NormMethodType = NormalizationAlgorithmMethod::LocalBrightness;
    if (param.has_local_size())
    {
        normalizationDescriptor.m_NormSize = param.local_size();
    }
    else
    {
        throw ParseException("Loading LRN Layer: Local_size not defined");
    }

    if (param.has_alpha())
    {
        normalizationDescriptor.m_Alpha = param.alpha();
        normalizationDescriptor.m_Alpha /= boost::numeric_cast<float>(param.local_size());
    }
    else
    {
        throw ParseException("Loading LRN Layer: Alpha not defined");
    }
    if (param.has_beta())
    {
        normalizationDescriptor.m_Beta = param.beta();
    }
    else
    {
        throw ParseException("Loading LRN Layer: Beta not defined");
    }
    if (param.has_k())
    {
        normalizationDescriptor.m_K = param.k();
    }
    else
        normalizationDescriptor.m_K = 1;

    IConnectableLayer* const normLayer = m_Network->AddNormalizationLayer(normalizationDescriptor,
        layerParam.name().c_str());
    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).Connect(normLayer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), normLayer->GetOutputSlot(0));
}

void CaffeParser::ParseInnerProductLayer(const LayerParameter& layerParam)
{
    InnerProductParameter param = layerParam.inner_product_param();

    ValidateNumInputsOutputs(layerParam, 1, 1);

    unsigned int outputSize = param.num_output();

    // Ignored Caffe Parameters
    // Weight Filler
    // Bias Filler
    // Engine
    // Axis

    FullyConnectedDescriptor tensorFullyConnectedDescriptor;

    if (param.has_transpose())
    {
        // If true assume transposed weights
        tensorFullyConnectedDescriptor.m_TransposeWeightMatrix = param.transpose();
    }
    else
    {
        // caffe defaults to transposed
        tensorFullyConnectedDescriptor.m_TransposeWeightMatrix = true;
    }

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo();

    TensorInfo weightInfo;
    TensorInfo biasInfo;

    // allow implicit flattening of extra dimensions
    unsigned int inputSize = inputInfo.GetShape()[1];
    for (unsigned int i = 2; i < inputInfo.GetNumDimensions(); ++i)
    {
        inputSize *= inputInfo.GetShape()[i];
    }

    vector<float> weightData(inputSize * outputSize);

    GetDataFromBlob(layerParam, weightData, 0);
    const unsigned int swTD[2] = { outputSize, inputSize };
    ConstTensor weights(TensorInfo(2, swTD, DataType::Float32), weightData);

    tensorFullyConnectedDescriptor.m_BiasEnabled = true;
    // Todo: check whether bias enabled
    armnn::IConnectableLayer* fullyConnectedLayer = nullptr;
    if (tensorFullyConnectedDescriptor.m_BiasEnabled)
    {
        // BIAS VALUE
        vector<float> biasData(outputSize);

        GetDataFromBlob(layerParam, biasData, 1);

        const unsigned int sbTD[1] = { outputSize };

        ConstTensor biases(TensorInfo(1, sbTD, DataType::Float32), biasData);

        fullyConnectedLayer = m_Network->AddFullyConnectedLayer(tensorFullyConnectedDescriptor, weights, biases,
            layerParam.name().c_str());
    }
    else
    {
        fullyConnectedLayer = m_Network->AddFullyConnectedLayer(tensorFullyConnectedDescriptor, weights,
            layerParam.name().c_str());
    }

    TensorInfo outputInfo({ inputInfo.GetShape()[0], outputSize }, DataType::Float32);
    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).Connect(fullyConnectedLayer->GetInputSlot(0));
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), fullyConnectedLayer->GetOutputSlot(0));
}

void CaffeParser::ParseSoftmaxLayer(const LayerParameter& layerParam)
{
    ValidateNumInputsOutputs(layerParam, 1, 1);

    SoftmaxParameter param = layerParam.softmax_param();

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo();

    // Ignored Caffe Parameters
    //      axis
    //      Engine

    armnn::SoftmaxDescriptor softmaxDescriptor;
    armnn::IConnectableLayer* const softmaxLayer = m_Network->AddSoftmaxLayer(
        softmaxDescriptor,
        layerParam.name().c_str());
    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).Connect(softmaxLayer->GetInputSlot(0));
    softmaxLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), softmaxLayer->GetOutputSlot(0));
}

void CaffeParser::ParseEltwiseLayer(const LayerParameter& layerParam)
{
    ValidateNumInputsOutputs(layerParam, 2, 1);

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo();

    // Ignored Caffe Parameters
    //      coeff

    EltwiseParameter_EltwiseOp operation = EltwiseParameter_EltwiseOp_SUM; // default to sum as per caffe

    if (layerParam.has_eltwise_param() && layerParam.eltwise_param().has_operation())
    {
        operation = layerParam.eltwise_param().operation();
    }

    armnn::IConnectableLayer* newLayer = nullptr;
    switch (operation)
    {
        case EltwiseParameter_EltwiseOp_SUM:
        {
            newLayer = m_Network->AddAdditionLayer(layerParam.name().c_str());
            break;
        }
        case EltwiseParameter_EltwiseOp_PROD:
        {
            newLayer = m_Network->AddMultiplicationLayer(layerParam.name().c_str());
            break;
        }
        default:
        {
            throw ParseException("Unsupported operation in Eltwise layer");
        }
    }

    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).Connect(newLayer->GetInputSlot(0));
    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(1)).Connect(newLayer->GetInputSlot(1));
    newLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), newLayer->GetOutputSlot(0));
}

void CaffeParser::ParseConcatLayer(const LayerParameter& layerParam)
{
    unsigned int numInputs = static_cast<unsigned int>(layerParam.bottom_size());
    // we assume concat happens along the channel dimension, which is 1 in (0, 1, 2, 3)
    unsigned int concatDim = 1;
    unsigned int numOfDims = 4;

    OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numInputs), numOfDims);// we only consider 4-D tensor here
    std::vector<unsigned int>mergeDimSizes(numOfDims, 0u);

    unsigned int mergeDim = 0;
    for (unsigned int viewIndex = 0; viewIndex < numInputs; ++viewIndex)
    {
        const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(
            layerParam.bottom(boost::numeric_cast<int>(viewIndex))).GetTensorInfo();
        
        // Check whether the dimensions of the input tensors are actually 4
        if (inputInfo.GetNumDimensions()!=4)
        {
            throw ParseException("The number of dimensions for input tensors of the concatenation op should be 4.");
        }

        mergeDimSizes[0] = inputInfo.GetShape()[0];
        mergeDimSizes[1] = inputInfo.GetShape()[1];
        mergeDimSizes[2] = inputInfo.GetShape()[2];
        mergeDimSizes[3] = inputInfo.GetShape()[3];

        for (unsigned int j = 0; j < concatDim; ++j)
        {
            concatDescriptor.SetViewOriginCoord(viewIndex, j, 0);
        }

        concatDescriptor.SetViewOriginCoord(viewIndex, concatDim, mergeDim);
        mergeDim += mergeDimSizes[concatDim];

        for (unsigned int j = concatDim+1; j < numOfDims; ++j)
        {
            concatDescriptor.SetViewOriginCoord(viewIndex, j, 0);
        }
    }
    mergeDimSizes[concatDim] = mergeDim;

    armnn::IConnectableLayer *concatlayer = m_Network->AddMergerLayer(concatDescriptor, layerParam.name().c_str());
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        armnn::IOutputSlot& outputSlot = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(boost::numeric_cast<int>(i)));
        outputSlot.Connect(concatlayer->GetInputSlot(i));
    }

    concatlayer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(numOfDims, mergeDimSizes.data(), DataType::Float32));
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), concatlayer->GetOutputSlot(0));
}

void CaffeParser::ParseBatchNormLayer(const LayerParameter& layerParam)
{
    ValidateNumInputsOutputs(layerParam, 1, 1);

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo();

    string name = layerParam.name();

    BatchNormParameter param = layerParam.batch_norm_param();
    // If use_global_stats is not explicitly set in the model, assume it to be true (its default value
    // when the network is in the testing phase).
    if (param.has_use_global_stats())
    {
        if (!param.use_global_stats())
        {
            throw ParseException(boost::str(boost::format("Error parsing Batch Norm layer '%1%': "
                "Parameter 'use_global_stats' is set to false, which is unsupported (value used for training).")
                % name));
        }
    }

    BatchNormalizationDescriptor desc;
    desc.m_Eps = param.eps();

    unsigned int channels = inputInfo.GetShape()[1];
    unsigned int shape[]  = {channels};

    vector<float> meanData(channels);
    GetDataFromBlob(layerParam, meanData, 0);

    vector<float> varianceData(channels);
    GetDataFromBlob(layerParam, varianceData, 1);

    // identity scale operation
    vector<float> betaData(channels, 0.0f);
    vector<float> gammaData(channels, 1.0f);

    ConstTensor mean(TensorInfo(1, shape, armnn::DataType::Float32), meanData);
    ConstTensor variance(TensorInfo(1, shape, armnn::DataType::Float32), varianceData);
    ConstTensor beta(TensorInfo(1, shape, armnn::DataType::Float32), betaData);
    ConstTensor gamma(TensorInfo(1, shape, armnn::DataType::Float32), gammaData);

    armnn::IConnectableLayer* const batchNormLayer = m_Network->AddBatchNormalizationLayer(desc,
        mean, variance, beta, gamma, name.c_str());
    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).Connect(batchNormLayer->GetInputSlot(0));
    batchNormLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), batchNormLayer->GetOutputSlot(0));
}

void CaffeParser::ParseScaleLayer(const LayerParameter& layerParam)
{
    // current unoptimal solution: add a batchnormalization layer with 0 mean and 1 variance
    ValidateNumInputsOutputs(layerParam, 1, 1);

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo();

    string name = layerParam.name();

    ScaleParameter param = layerParam.scale_param();
    if (param.axis() != 1)
    {
        // Would have to use something other than BatchNormalizationLayer in this case
        throw ParseException("Loading Scale Layer: Only axis 1 supported currently");
    }

    unsigned int     channels = inputInfo.GetShape()[1];
    unsigned int     shape[]  = {channels};

    BatchNormalizationDescriptor desc;
    desc.m_Eps = 0.0f; // don't need epsilon if variance is 1
    vector<float> meanData(channels, 0.0f);
    vector<float> varianceData(channels, 1.0f);
    vector<float> betaData(channels, 0.0f);
    vector<float> gammaData(channels);

    GetDataFromBlob(layerParam, gammaData, 0);

    if(param.has_bias_term())
    {
        GetDataFromBlob(layerParam, betaData, 1);
    }

    ConstTensor mean(TensorInfo(1, shape, armnn::DataType::Float32), meanData);
    ConstTensor variance(TensorInfo(1, shape, armnn::DataType::Float32), varianceData);
    ConstTensor beta(TensorInfo(1, shape, armnn::DataType::Float32), betaData);
    ConstTensor gamma(TensorInfo(1, shape, armnn::DataType::Float32), gammaData);

    armnn::IConnectableLayer* const batchNormLayer = m_Network->AddBatchNormalizationLayer(desc,
        mean, variance, beta, gamma, name.c_str());
    GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).Connect(batchNormLayer->GetInputSlot(0));
    batchNormLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), batchNormLayer->GetOutputSlot(0));
}

void CaffeParser::ParseSplitLayer(const caffe::LayerParameter& layerParam)
{
    // Used in caffe to duplicate memory - not necessary in armnn
    if (layerParam.bottom_size() != 1)
    {
        throw ParseException("Split layer '" + layerParam.name() + "' should have exactly 1 bottom");
    }
    armnn::IOutputSlot& outputSlot = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0));
    for (int i = 0; i < layerParam.top_size(); i++)
    {
        SetArmnnOutputSlotForCaffeTop(layerParam.top(i), outputSlot);
    }
}

void CaffeParser::ParseDropoutLayer(const caffe::LayerParameter& layerParam)
{
    // Ignored for inference so patch the single input to its single output
    if (layerParam.bottom_size() != 1 || layerParam.top_size() != 1)
    {
        throw ParseException("Dropout layer '" + layerParam.name() + "' should have exactly 1 bottom and 1 top");
    }
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)));
}

void CaffeParser::TrackInputBinding(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, layer->GetName(), m_NetworkInputsBindingInfo);
}

void CaffeParser::TrackOutputBinding(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, layer->GetName(), m_NetworkOutputsBindingInfo);
}

void CaffeParser::TrackBindingPoint(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo,
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
        throw ParseException(boost::str(
            boost::format("Id %1% used by more than one %2% layer") % id % bindingPointDesc));
    }
}

armnn::IOutputSlot& CaffeParser::GetArmnnOutputSlotForCaffeTop(const std::string& caffeTopName) const
{
    auto it = m_ArmnnOutputSlotForCaffeTop.find(caffeTopName);
    if (it != m_ArmnnOutputSlotForCaffeTop.end())
    {
        return *it->second;
    }
    else
    {
        throw ParseException(boost::str(boost::format(
            "Could not find armnn output slot for Caffe top '%1%'") % caffeTopName));
    }
}

void CaffeParser::SetArmnnOutputSlotForCaffeTop(const std::string& caffeTopName, armnn::IOutputSlot& armnnOutputSlot)
{
    auto it = m_ArmnnOutputSlotForCaffeTop.find(caffeTopName);
    if (it == m_ArmnnOutputSlotForCaffeTop.end())
    {
        m_ArmnnOutputSlotForCaffeTop[caffeTopName] = &armnnOutputSlot;
    }
    else
    {
        throw ParseException("Attempting to add duplicate entry for Caffe top '" + caffeTopName + "'");
    }
}

void CaffeParser::ResolveInPlaceLayers(caffe::NetParameter& netParameter)
{
    // Find layers with the same top
    std::map<std::string, std::vector<caffe::LayerParameter*>> layersByTop;
    for (int layerIdx = 0; layerIdx < netParameter.layer_size(); ++layerIdx)
    {
        caffe::LayerParameter& layer = *netParameter.mutable_layer(layerIdx);
        for (int i = 0; i < layer.top_size(); ++i)
        {
            layersByTop[layer.top(i)].push_back(&layer);
        }
    }

    // For each set of layers with the same top, resolve them to a linear chain rather than in-place layers.
    // Note that for 'regular' layers, there will be a single layer in each group and so this will be a no-op.
    for (auto layersWithSameTopIt : layersByTop)
    {
        const std::string& top = layersWithSameTopIt.first;
        const std::vector<caffe::LayerParameter*>& layersWithSameTop = layersWithSameTopIt.second;

        // Chain the layers together in the order that they are listed in the prototxt (hopefully this is correct).
        // Note that the last layer will not have its top modified so that other layers will continue to reference it.
        for (unsigned int layerIdx = 0; layerIdx < layersWithSameTop.size() - 1; ++layerIdx)
        {
            caffe::LayerParameter& layer1 = *layersWithSameTop[layerIdx];
            caffe::LayerParameter& layer2 = *layersWithSameTop[layerIdx+1];
            if (layer1.top_size() != 1)
            {
                throw ParseException("Node '" + layer1.name() + "' is an in-place layer but "
                    "doesn't have exactly one top.");
            }
            std::string newTop = layer1.name() + "_top";
            layer1.set_top(0, newTop);
            if (layer2.bottom_size() != 1 || layer2.bottom(0) != top)
            {
                throw ParseException("Node '" + layer2.name() + "' is an in-place layer but "
                    " doesn't have exactly one bottom, or it doesn't match its top.");
            }
            layer2.set_bottom(0, newTop);
        }
    }
}

void CaffeParser::LoadNetParam(NetParameter& netParameter)
{
    // caffe models sometimes have an implicit input layer.
    // in that case, add an explicit one
    if (netParameter.input_size() > 0)
    {
        LayerParameter* newLayer = netParameter.add_layer();

        newLayer->set_type("Input");
        newLayer->set_name(netParameter.input(0));
        newLayer->add_top(netParameter.input(0));

        InputParameter* inputParam = newLayer->mutable_input_param();
        BlobShape* shape = inputParam->add_shape();

        int dim_size = netParameter.input_dim_size();
        for (int i = 0; i < dim_size; ++i)
        {
            shape->add_dim(netParameter.input_dim(i));
        }
    }

    // Replace in-place layers with regular ones to make the rest of the parsing easier.
    ResolveInPlaceLayers(netParameter);

    // Create a lookup of Caffe layers by name
    for (int i = 0; i < netParameter.layer_size(); ++i)
    {
        const caffe::LayerParameter& layer = netParameter.layer(i);
        for (int i = 0; i < layer.top_size(); ++i)
        {
            m_CaffeLayersByTopName[layer.top(i)] = &layer;
        }
    }

    // Find the output layers the user requested
    std::vector<const caffe::LayerParameter*> targetLayers;
    for (const std::string& requestedOutputName : m_RequestedOutputs)
    {
        auto nodeIt = m_CaffeLayersByTopName.find(requestedOutputName);
        if (nodeIt == m_CaffeLayersByTopName.end())
        {
            throw ParseException("Couldn't find requested output layer '" + requestedOutputName + "' in graph");
        }
        targetLayers.push_back(nodeIt->second);
    }

    // Sort them into a linear ordering such that all inputs of a node are before the node itself
    std::vector<const caffe::LayerParameter*> sortedNodes;
    if (!armnnUtils::GraphTopologicalSort<const caffe::LayerParameter*>(
        targetLayers,
        [this](const caffe::LayerParameter* node)
        {
            return GetInputs(*node);
        },
        sortedNodes))
    {
        throw ParseException("Cycle detected in graph");
    }

    // Parse each node in order, knowing that all inputs of a node will be processed before the node itself
    for (const caffe::LayerParameter* current : sortedNodes)
    {
        auto it = ms_CaffeLayerNameToParsingFunctions.find(current->type());
        if (it == ms_CaffeLayerNameToParsingFunctions.end())
        {
            throw ParseException("Unsupported layer type '" + current->type() + "'");
        }
        auto func = it->second;
        (this->*func)(*current);
    }

    // Add ArmNN output layers connected to each requested output
    for (const std::string& requestedOutput : m_RequestedOutputs)
    {
        armnn::IOutputSlot& outputSlot = GetArmnnOutputSlotForCaffeTop(requestedOutput);

        const armnn::LayerBindingId outputId = boost::numeric_cast<armnn::LayerBindingId>(
            m_NetworkOutputsBindingInfo.size());
        armnn::IConnectableLayer* const outputLayer = m_Network->AddOutputLayer(outputId, requestedOutput.c_str());
        outputSlot.Connect(outputLayer->GetInputSlot(0));

        TrackOutputBinding(outputLayer, outputId, outputLayer->GetInputSlot(0).GetConnection()->GetTensorInfo());
    }
}

INetworkPtr CaffeParser::CreateNetworkFromTextFile(const char* graphFile,
    const std::map<std::string, armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    FILE* fd = fopen(graphFile, "r");

    if (fd == nullptr)
    {
        std::stringstream error;
        error << "Graph file " << graphFile << " failed to open";
        throw FileNotFoundException(error.str());
    }

    // Parse the file into a message
    NetParameter netParam;
    auto         input   = new google::protobuf::io::FileInputStream(fileno(fd));
    bool         success = google::protobuf::TextFormat::Parse(input, &netParam);
    delete input;
    fclose(fd);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse graph file";
        throw ParseException(error.str());
    }

    return CreateNetworkFromNetParameter(netParam, inputShapes, requestedOutputs);
}

INetworkPtr CaffeParser::CreateNetworkFromString(const char* protoText,
    const std::map<std::string, armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    // Parse the string into a message
    NetParameter netParam;
    bool         success = google::protobuf::TextFormat::ParseFromString(protoText, &netParam);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse graph string";
        throw ParseException(error.str());
    }

    return CreateNetworkFromNetParameter(netParam, inputShapes, requestedOutputs);
}

INetworkPtr CaffeParser::CreateNetworkFromBinaryFile(const char* graphFile,
    const std::map<std::string, armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    FILE* fd = fopen(graphFile, "rb");

    if (fd == nullptr)
    {
        std::stringstream error;
        error << "Graph file " << graphFile << " failed to open";
        throw FileNotFoundException(error.str());
    }

    // Parse the file into a message
    NetParameter netParam;

    FileInputStream  inStream(fileno(fd));
    CodedInputStream codedStream(&inStream);
    codedStream.SetTotalBytesLimit(INT_MAX, INT_MAX);
    bool success = netParam.ParseFromCodedStream(&codedStream);
    fclose(fd);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse protobuf file" << graphFile;
        throw ParseException(error.str());
    }

    return CreateNetworkFromNetParameter(netParam, inputShapes, requestedOutputs);
}

INetworkPtr CaffeParser::CreateNetworkFromNetParameter(NetParameter& netParam,
    const std::map<std::string, armnn::TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    m_NetworkInputsBindingInfo.clear();
    m_NetworkOutputsBindingInfo.clear();

    m_Network = INetwork::Create();

    m_InputShapes = inputShapes;
    if (requestedOutputs.size() == 0)
    {
        throw ParseException("requestedOutputs must have at least one entry");
    }
    m_RequestedOutputs = requestedOutputs;

    std::cout << "in caffeParser create network from net parameter." << std::endl;

    try
    {
        LoadNetParam(netParam);
    }
    catch (const ParseException& e)
    {
        Cleanup();
        throw e;
    }

    Cleanup();

    return move(m_Network);
}

void CaffeParser::Cleanup()
{
    // cleanup, in case we reuse this parser
    m_CaffeLayersByTopName.clear();
    m_InputShapes.clear();
    m_RequestedOutputs.clear();
    m_ArmnnOutputSlotForCaffeTop.clear();
}

void CaffeParser::ParseReorgLayer(const caffe::LayerParameter& layerParam)
{
    BOOST_ASSERT(layerParam.type() == "Reorg");
    ValidateNumInputsOutputs(layerParam, 1, 1);
    ReorgParameter reorgParameter = layerParam.reorg_param();
    BlobShape inputShape = TensorDescToBlobShape(GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo());

    unsigned int stride;
    if ( reorgParameter.has_stride() )
    {
        stride = reorgParameter.stride();
    }
    else
    {
        throw ParseException("Loading Reorg Layer: stride defined Illegally");
    }

    if ( (inputShape.dim(2) % stride) != 0 || (inputShape.dim(2) % stride) != 0 )
    {
        throw ParseException("Loading Reorg Layer: stride can not devide inputshape dim 2 or 3");
    }

    armnn::IOutputSlot& inputConnection = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0));

    BlobShape outputShape;
    outputShape.add_dim(0);
    outputShape.set_dim(0, inputShape.dim(0));
    outputShape.add_dim(1);
    outputShape.set_dim(1, stride*stride*inputShape.dim(1));
    outputShape.add_dim(2);
    outputShape.set_dim( 2, inputShape.dim(2)/ stride );
    outputShape.add_dim(3);
    outputShape.set_dim( 3, inputShape.dim(3)/ stride );

    ReorgDescriptor reorgDescriptor;
    reorgDescriptor.m_Stride = stride;
    //reorgDescriptor.m_StrideX = strideW;
    //reorgDescriptor.m_StrideY  = strideH;

    IConnectableLayer* reorgLayer = nullptr;
    reorgLayer = m_Network->AddReorgLayer(reorgDescriptor,layerParam.name().c_str());
    inputConnection.Connect( reorgLayer->GetInputSlot(0) );

    BOOST_ASSERT(reorgLayer);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), reorgLayer->GetOutputSlot(0));
}

void CaffeParser::ParseDetectionOutputLayer(const caffe::LayerParameter& layerParam)
{
    BOOST_ASSERT(layerParam.type() == "DetectionOutput");
    ValidateNumInputsOutputs(layerParam, 1, 1);
    DetectionOutputParameter detectionoutputParameter = layerParam.detection_output_param();
    BlobShape inputShape = TensorDescToBlobShape(GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0)).GetTensorInfo());

    unsigned int coords;
    if ( detectionoutputParameter.has_coords() )
        coords = detectionoutputParameter.coords();
    else
        throw ParseException("Loading DetectionOutput Layer: coords defined Illegally");

    unsigned int classes;
    if ( detectionoutputParameter.has_num_classes() )
        classes = detectionoutputParameter.num_classes();
    else
        throw ParseException("Loading DetectionOutput Layer: classes defined Illegally");

    unsigned int side;
    if( detectionoutputParameter.has_side() )
        side = detectionoutputParameter.side();
    else
        throw ParseException("Loading DetectionOutput Layer: side defined Illegally");

    unsigned int numBox;
    if( detectionoutputParameter.has_num_box() )
        numBox = detectionoutputParameter.num_box();
    else
        throw ParseException("Loading DetectionOutput Layer: numbox defined Illegally");

    float confidenceThreshold;
    if( detectionoutputParameter.has_confidence_threshold() )
        confidenceThreshold = detectionoutputParameter.confidence_threshold();
    else
        throw ParseException("Loading DetectionOutput Layer: confidenceThreshold defined Illegally");

    float nmsThreshold;
    if( detectionoutputParameter.has_nms_threshold() )
        nmsThreshold = detectionoutputParameter.nms_threshold();
    else
        throw ParseException("Loading DetectionOutput Layer: nmsThreshold defined Illegally");

    std::vector<float> anchors;
    int size;
    size = detectionoutputParameter.anchor_size();
    if( size > 1 )
    {
        anchors.resize(static_cast<long unsigned int>(size));
        for(int i = 0; i < size ; ++i )
        {
            anchors.emplace_back(detectionoutputParameter.anchor(i));
        }
    }
    else
        throw ParseException("Loading DetectionOutput Layer: anchors defined Illegally");


    armnn::IOutputSlot& inputConnection = GetArmnnOutputSlotForCaffeTop(layerParam.bottom(0));

    BlobShape outputShape;
    outputShape.add_dim(0);
    outputShape.set_dim(0, inputShape.dim(0));
    outputShape.add_dim(1);
    outputShape.set_dim(1, side * side);
    outputShape.add_dim(2);
    outputShape.set_dim( 2, (classes + coords + 1)*numBox );

    DetectionOutputDescriptor detectionoutputDescriptor;
    // uint32 param
    detectionoutputDescriptor.m_Classes = classes;
    detectionoutputDescriptor.m_Side = side;
    detectionoutputDescriptor.m_NumBox = numBox;
    // float pram
    detectionoutputDescriptor.m_ConfidenceThreshold = confidenceThreshold;
    detectionoutputDescriptor.m_NmsThreshold = nmsThreshold;
    // vector param
    detectionoutputDescriptor.m_Biases = std::move(anchors);

    IConnectableLayer* detectionoutputLayer = nullptr;
    detectionoutputLayer = m_Network->AddDetectionOutputLayer(detectionoutputDescriptor,layerParam.name().c_str());
    inputConnection.Connect( detectionoutputLayer->GetInputSlot(0) );

    BOOST_ASSERT(detectionoutputLayer);
    SetArmnnOutputSlotForCaffeTop(layerParam.top(0), detectionoutputLayer->GetOutputSlot(0));
}

}


