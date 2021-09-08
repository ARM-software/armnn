//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Convolution3dLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <backendsCommon/TensorHandle.hpp>

using namespace armnnUtils;

namespace armnn
{

Convolution3dLayer::Convolution3dLayer(const Convolution3dDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Convolution3d, param, name)
{
}

void Convolution3dLayer::SerializeLayerParameters(ParameterStringifyFunction& fn) const
{
    const std::vector<TensorShape>& inputShapes =
    {
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        m_Weight->GetTensorInfo().GetShape()
    };

    // Conv3d Filter Layout: [D,H,W,I,O]
    const TensorShape filterShape = inputShapes[1];
    DataLayoutIndexed dataLayoutIndex(m_Param.m_DataLayout);
    unsigned int filterDepth = filterShape[0];
    unsigned int filterHeight = filterShape[1];
    unsigned int filterWidth = filterShape[2];
    unsigned int inChannels = filterShape[3];
    unsigned int outChannels = filterShape[4];

    fn("FilterDepth",std::to_string(filterDepth));
    fn("FilterHeight",std::to_string(filterHeight));
    fn("FilterWidth",std::to_string(filterWidth));
    fn("InputChannels",std::to_string(inChannels));
    fn("OutputChannels",std::to_string(outChannels));

    LayerWithParameters<Convolution3dDescriptor>::SerializeLayerParameters(fn);
}

std::unique_ptr<IWorkload> Convolution3dLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    // At this level constant data should not be released.
    ARMNN_ASSERT_MSG(m_Weight != nullptr, "Convolution3dLayer: Weights data should not be null.");

    Convolution3dQueueDescriptor descriptor;
    descriptor.m_Weight = m_Weight.get();

    if (m_Param.m_BiasEnabled)
    {
        ARMNN_ASSERT_MSG(m_Bias != nullptr, "Convolution3dLayer: Bias data should not be null.");
        descriptor.m_Bias = m_Bias.get();
    }

    SetAdditionalInfo(descriptor);

    return factory.CreateConvolution3d(descriptor, PrepInfoAndDesc(descriptor));
}

Convolution3dLayer* Convolution3dLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<Convolution3dLayer>(graph, m_Param, GetName());

    layer->m_Weight = m_Weight ? m_Weight : nullptr;

    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? m_Bias : nullptr;
    }

    return std::move(layer);
}

std::vector<TensorShape> Convolution3dLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);
    const TensorShape& inputShape = inputShapes[0];
    const TensorShape& filterShape = inputShapes[1];

    ARMNN_ASSERT_MSG(inputShape.GetNumDimensions() == 5, "Convolutions will always have 5D input.");

    ARMNN_ASSERT( m_Param.m_StrideX > 0);
    ARMNN_ASSERT( m_Param.m_StrideY > 0);
    ARMNN_ASSERT( m_Param.m_StrideZ > 0);

    DataLayoutIndexed dataLayoutIndex(m_Param.m_DataLayout);

    unsigned int inWidth = inputShape[dataLayoutIndex.GetWidthIndex()];
    unsigned int inHeight = inputShape[dataLayoutIndex.GetHeightIndex()];
    unsigned int inDepth = inputShape[dataLayoutIndex.GetDepthIndex()];
    unsigned int inBatchSize = inputShape[0];

    // Conv3d Filter Layout: [D,H,W,I,O]
    unsigned int filterDepth = filterShape[0];
    unsigned int dilatedFilterDepth = filterDepth + (m_Param.m_DilationZ - 1) * (filterDepth - 1);
    unsigned int readDepth = (inDepth + m_Param.m_PadFront + m_Param.m_PadBack) - dilatedFilterDepth;
    unsigned int outDepth = 1 + (readDepth / m_Param.m_StrideZ);

    unsigned int filterHeight = filterShape[1];
    unsigned int dilatedFilterHeight = filterHeight + (m_Param.m_DilationY - 1) * (filterHeight - 1);
    unsigned int readHeight = (inHeight + m_Param.m_PadTop + m_Param.m_PadBottom) - dilatedFilterHeight;
    unsigned int outHeight = 1 + (readHeight / m_Param.m_StrideY);

    unsigned int filterWidth = filterShape[2];
    unsigned int dilatedFilterWidth = filterWidth + (m_Param.m_DilationX - 1) * (filterWidth - 1);
    unsigned int readWidth = (inWidth + m_Param.m_PadLeft + m_Param.m_PadRight) - dilatedFilterWidth;
    unsigned int outWidth = 1 + (readWidth / m_Param.m_StrideX);

    unsigned int outChannels = filterShape[4];
    unsigned int outBatchSize = inBatchSize;

    TensorShape tensorShape = TensorShape( { outBatchSize, outDepth, outHeight, outWidth, outChannels } );

    return std::vector<TensorShape>({ tensorShape });
}

void Convolution3dLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    // check if we m_Weight data is not nullptr
    ARMNN_ASSERT_MSG(m_Weight != nullptr, "Convolution3dLayer: Weights data should not be null.");

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        m_Weight->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "Convolution3dLayer");
}

Layer::ConstantTensors Convolution3dLayer::GetConstantTensorsByRef()
{
    return {m_Weight, m_Bias};
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void Convolution3dLayer::Accept(ILayerVisitor& visitor) const
{
    IgnoreUnused(visitor);
    throw armnn::Exception("Convolution3dLayer: VisitConvolution3dLayer is not implemented");
}
ARMNN_NO_DEPRECATE_WARN_END

void Convolution3dLayer::ExecuteStrategy(IStrategy& strategy) const
{
    ManagedConstTensorHandle managedWeight(m_Weight);
    std::vector<armnn::ConstTensor> constTensors { { managedWeight.GetTensorInfo(), managedWeight.Map() } };

    ManagedConstTensorHandle managedBias(m_Bias);
    if (GetParameters().m_BiasEnabled)
    {
        constTensors.emplace_back(ConstTensor(managedBias.GetTensorInfo(), managedBias.Map()));
    }

    strategy.ExecuteStrategy(this, GetParameters(), constTensors, GetName());
}

} // namespace armnn
