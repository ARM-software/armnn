//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DepthwiseConvolution2dLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <string>

using namespace armnnUtils;

namespace armnn
{

DepthwiseConvolution2dLayer::DepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& param,
                                                         const char* name)
    : LayerWithParameters(1, 1, LayerType::DepthwiseConvolution2d, param, name)
{
}

void DepthwiseConvolution2dLayer::SerializeLayerParameters(ParameterStringifyFunction& fn) const
{
    const std::vector<TensorShape>& inputShapes =
    {
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        m_Weight->GetTensorInfo().GetShape()
    };
    const TensorShape filterShape = inputShapes[1];
    DataLayoutIndexed dataLayoutIndex(m_Param.m_DataLayout);
    unsigned int inputChannels = filterShape[1];
    unsigned int filterWidth = filterShape[3];
    unsigned int filterHeight = filterShape[2];
    unsigned int depthMultiplier = filterShape[0];

    fn("FilterWidth",std::to_string(filterWidth));
    fn("FilterHeight",std::to_string(filterHeight));
    fn("DepthMultiplier",std::to_string(depthMultiplier));
    fn("InputChannels",std::to_string(inputChannels));

    LayerWithParameters<DepthwiseConvolution2dDescriptor>::SerializeLayerParameters(fn);
}

std::unique_ptr<IWorkload> DepthwiseConvolution2dLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    // on this level constant data should not be released..
    ARMNN_ASSERT_MSG(m_Weight != nullptr, "DepthwiseConvolution2dLayer: Weights data should not be null.");

    DepthwiseConvolution2dQueueDescriptor descriptor;

    descriptor.m_Weight = m_Weight.get();

    if (m_Param.m_BiasEnabled)
    {
        ARMNN_ASSERT_MSG(m_Bias != nullptr, "DepthwiseConvolution2dLayer: Bias data should not be null.");
        descriptor.m_Bias = m_Bias.get();
    }

    SetAdditionalInfo(descriptor);

    return factory.CreateDepthwiseConvolution2d(descriptor, PrepInfoAndDesc(descriptor));
}

DepthwiseConvolution2dLayer* DepthwiseConvolution2dLayer::Clone(Graph& graph) const
{
    auto layer      = CloneBase<DepthwiseConvolution2dLayer>(graph, m_Param, GetName());
    layer->m_Weight = m_Weight ? std::make_unique<ScopedCpuTensorHandle>(*m_Weight) : nullptr;

    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? std::make_unique<ScopedCpuTensorHandle>(*m_Bias) : nullptr;
    }

    return std::move(layer);
}

std::vector<TensorShape>
DepthwiseConvolution2dLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);
    const TensorShape& inputShape  = inputShapes[0];
    const TensorShape& filterShape = inputShapes[1];

    ARMNN_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Convolutions will always have 4D input.");

    DataLayoutIndexed dataLayoutIndex(m_Param.m_DataLayout);

    unsigned int inputBatchSize = inputShape[0];
    unsigned int inputHeight    = inputShape[dataLayoutIndex.GetHeightIndex()];
    unsigned int inputWidth     = inputShape[dataLayoutIndex.GetWidthIndex()];
    unsigned int inputChannels  = inputShape[dataLayoutIndex.GetChannelsIndex()];

    // Expected filter shape: [ M, I, H, W ] - This shape does NOT depend on the data layout
    // Namely: [ depth multiplier, input channels, filter height, filter width ]
    // Output channels = input channels * depthMultiplier
    unsigned int depthMultiplier = filterShape[0];

    unsigned int filterHeight = filterShape[2];
    unsigned int dilatedFilterHeight = filterHeight + (m_Param.m_DilationY - 1) * (filterHeight - 1);
    unsigned int readHeight   = (inputHeight + m_Param.m_PadTop + m_Param.m_PadBottom) - dilatedFilterHeight;
    unsigned int outputHeight = 1 + (readHeight / m_Param.m_StrideY);

    unsigned int filterWidth = filterShape[3];
    unsigned int dilatedFilterWidth = filterWidth + (m_Param.m_DilationX - 1) * (filterWidth - 1);
    unsigned int readWidth   = (inputWidth + m_Param.m_PadLeft + m_Param.m_PadRight) - dilatedFilterWidth;
    unsigned int outputWidth = 1 + (readWidth / m_Param.m_StrideX);

    unsigned int outputChannels  = inputChannels * depthMultiplier;
    unsigned int outputBatchSize = inputBatchSize;

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NHWC ?
                              TensorShape{ outputBatchSize, outputHeight, outputWidth, outputChannels } :
                              TensorShape{ outputBatchSize, outputChannels, outputHeight, outputWidth };

    return std::vector<TensorShape>{ tensorShape };
}

void DepthwiseConvolution2dLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    // on this level constant data should not be released..
    ARMNN_ASSERT_MSG(m_Weight != nullptr, "DepthwiseConvolution2dLayer: Weights data should not be null.");

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        m_Weight->GetTensorInfo().GetShape()
     });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "DepthwiseConvolution2dLayer");
}

Layer::ConstantTensors DepthwiseConvolution2dLayer::GetConstantTensorsByRef()
{
    return {m_Weight, m_Bias};
}

void DepthwiseConvolution2dLayer::Accept(ILayerVisitor& visitor) const
{
    ConstTensor weightsTensor(m_Weight->GetTensorInfo(), m_Weight->Map(true));
    Optional<ConstTensor> optionalBiasTensor = EmptyOptional();

    if (GetParameters().m_BiasEnabled)
    {
        ConstTensor biasTensor(m_Bias->GetTensorInfo(), m_Bias->Map(true));
        optionalBiasTensor = Optional<ConstTensor>(biasTensor);
    }

    visitor.VisitDepthwiseConvolution2dLayer(this, GetParameters(), weightsTensor, optionalBiasTensor, GetName());
}

} // namespace armnn
