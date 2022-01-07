//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Convolution3dLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/TensorHandle.hpp>

using namespace armnnUtils;

namespace armnn
{

Convolution3dLayer::Convolution3dLayer(const Convolution3dDescriptor& param, const char* name)
    : LayerWithParameters(param.GetNumInputs(), 1, LayerType::Convolution3d, param, name)
{
}

void Convolution3dLayer::SerializeLayerParameters(ParameterStringifyFunction& fn) const
{
    const std::vector<TensorShape>& inputShapes =
    {
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape(),
    };

    // Conv3d Filter Layout: [D,H,W,I,O]
    const TensorShape filterShape = inputShapes[1];
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
    Convolution3dQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Convolution3d, descriptor, PrepInfoAndDesc(descriptor));
}

Convolution3dLayer* Convolution3dLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<Convolution3dLayer>(graph, m_Param, GetName());
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

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NDHWC ?
            TensorShape( { outBatchSize, outDepth, outHeight, outWidth, outChannels } ) :
            TensorShape( { outBatchSize, outChannels, outDepth, outHeight, outWidth });

    return std::vector<TensorShape>({ tensorShape });
}

void Convolution3dLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(m_Param.GetNumInputs(), CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    ARMNN_ASSERT_MSG(GetInputSlot(1).GetConnection(),
                     "Convolution3dLayer: Weights should be connected to input slot 1.");

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "Convolution3dLayer");
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
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
