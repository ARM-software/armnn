//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
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
    : LayerWithParameters(param.GetNumInputs(), 1, LayerType::DepthwiseConvolution2d, param, name)
{
}

void DepthwiseConvolution2dLayer::SerializeLayerParameters(ParameterStringifyFunction& fn) const
{
    const std::vector<TensorShape>& inputShapes =
    {
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()
    };
    const TensorShape filterShape = inputShapes[1];
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
    DepthwiseConvolution2dQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::DepthwiseConvolution2d, descriptor, PrepInfoAndDesc(descriptor));
}

DepthwiseConvolution2dLayer* DepthwiseConvolution2dLayer::Clone(Graph& graph) const
{
    auto layer      = CloneBase<DepthwiseConvolution2dLayer>(graph, m_Param, GetName());
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
    VerifyLayerConnections(m_Param.GetNumInputs(), CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    ARMNN_ASSERT_MSG(GetInputSlot(1).GetConnection(),
                     "DepthwiseConvolution2dLayer: Weights data should not be null.");

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()
    });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "DepthwiseConvolution2dLayer");
}

Layer::ImmutableConstantTensors DepthwiseConvolution2dLayer::GetConstantTensorsByRef() const
{
    Layer::ImmutableConstantTensors tensors = GetConnectedConstantAsInputTensors();
    return tensors;
}

void DepthwiseConvolution2dLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
