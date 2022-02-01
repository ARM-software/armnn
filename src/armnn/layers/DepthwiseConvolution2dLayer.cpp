//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DepthwiseConvolution2dLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

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

    return factory.CreateWorkload(LayerType::DepthwiseConvolution2d, descriptor, PrepInfoAndDesc(descriptor));
}

DepthwiseConvolution2dLayer* DepthwiseConvolution2dLayer::Clone(Graph& graph) const
{
    auto layer      = CloneBase<DepthwiseConvolution2dLayer>(graph, m_Param, GetName());
    layer->m_Weight = m_Weight ? m_Weight : nullptr;

    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? m_Bias : nullptr;
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

    ARMNN_ASSERT( m_Param.m_StrideX > 0);
    ARMNN_ASSERT( m_Param.m_StrideY > 0);

    DataLayoutIndexed dataLayoutIndex(m_Param.m_DataLayout);

    unsigned int inputBatchSize = inputShape[0];
    unsigned int inputHeight    = inputShape[dataLayoutIndex.GetHeightIndex()];
    unsigned int inputWidth     = inputShape[dataLayoutIndex.GetWidthIndex()];

    // Expected filter shape: [ 1, H, W, O ] - This shape does NOT depend on the data layout
    // Namely: [ 1, filter height, filter width, output channels ]

    unsigned int filterHeight = filterShape[1];
    unsigned int dilatedFilterHeight = filterHeight + (m_Param.m_DilationY - 1) * (filterHeight - 1);
    unsigned int readHeight   = (inputHeight + m_Param.m_PadTop + m_Param.m_PadBottom) - dilatedFilterHeight;
    unsigned int outputHeight = 1 + (readHeight / m_Param.m_StrideY);

    unsigned int filterWidth = filterShape[2];
    unsigned int dilatedFilterWidth = filterWidth + (m_Param.m_DilationX - 1) * (filterWidth - 1);
    unsigned int readWidth   = (inputWidth + m_Param.m_PadLeft + m_Param.m_PadRight) - dilatedFilterWidth;
    unsigned int outputWidth = 1 + (readWidth / m_Param.m_StrideX);

    unsigned int outputChannels  = filterShape[3];
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
    // For API stability DO NOT ALTER order and add new members to the end of vector
    return {m_Weight, m_Bias};
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void DepthwiseConvolution2dLayer::Accept(ILayerVisitor& visitor) const
{
    ManagedConstTensorHandle managedWeight(m_Weight);
    ConstTensor weightsTensor(managedWeight.GetTensorInfo(), managedWeight.Map());
    Optional<ConstTensor> optionalBiasTensor = EmptyOptional();

    ManagedConstTensorHandle managedBias(m_Bias);
    if (GetParameters().m_BiasEnabled)
    {
        ConstTensor biasTensor(managedBias.GetTensorInfo(), managedBias.Map());
        optionalBiasTensor = Optional<ConstTensor>(biasTensor);
    }

    visitor.VisitDepthwiseConvolution2dLayer(this, GetParameters(), weightsTensor, optionalBiasTensor, GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

void DepthwiseConvolution2dLayer::ExecuteStrategy(IStrategy& strategy) const
{
    ManagedConstTensorHandle managedWeight(m_Weight);
    std::vector<armnn::ConstTensor> constTensors { { managedWeight.GetTensorInfo(), managedWeight.Map() } };

    ManagedConstTensorHandle managedBias(m_Bias);
    if (GetParameters().m_BiasEnabled)
    {
        constTensors.emplace_back(ConstTensor(managedBias.GetTensorInfo(), managedBias.Map(true)));
    }

    strategy.ExecuteStrategy(this, GetParameters(), constTensors, GetName());
}

} // namespace armnn
