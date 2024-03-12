//
// Copyright © 2017-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Convolution2dLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <string>

using namespace armnnUtils;

namespace armnn
{

Convolution2dLayer::Convolution2dLayer(const Convolution2dDescriptor& param, const char* name)
    : LayerWithParameters(param.GetNumInputs(), 1, LayerType::Convolution2d, param, name)
{

}

void Convolution2dLayer::SerializeLayerParameters(ParameterStringifyFunction& fn) const
{
    //using DescriptorType = Parameters;
    const std::vector<TensorShape>& inputShapes =
    {
        GetInputSlot(0).GetTensorInfo().GetShape(),
        GetInputSlot(1).GetTensorInfo().GetShape()
    };
    const TensorShape filterShape = inputShapes[1];
    DataLayoutIndexed dataLayoutIndex(m_Param.m_DataLayout);
    unsigned int filterWidth = filterShape[dataLayoutIndex.GetWidthIndex()];
    unsigned int filterHeight = filterShape[dataLayoutIndex.GetHeightIndex()];
    unsigned int outChannels = filterShape[0];

    fn("OutputChannels",std::to_string(outChannels));
    fn("FilterWidth",std::to_string(filterWidth));
    fn("FilterHeight",std::to_string(filterHeight));
    LayerWithParameters<Convolution2dDescriptor>::SerializeLayerParameters(fn);
}

std::unique_ptr<IWorkload> Convolution2dLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Convolution2dLayer_CreateWorkload");
    Convolution2dQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Convolution2d, descriptor, PrepInfoAndDesc(descriptor));
}

Convolution2dLayer* Convolution2dLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<Convolution2dLayer>(graph, m_Param, GetName());
    return std::move(layer);
}

std::vector<TensorShape> Convolution2dLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    if (inputShapes.size() != 2)
    {
        throw armnn::Exception("inputShapes' size is \"" + std::to_string(inputShapes.size()) +
                               "\" - should be \"2\".");
    }

    const TensorShape& inputShape = inputShapes[0];
    const TensorShape filterShape = inputShapes[1];

    // If we support multiple batch dimensions in the future, then this assert will need to change.
    if (inputShape.GetNumDimensions() != 4)
    {
        throw armnn::Exception("Convolutions will always have 4D input.");
    }

    if (m_Param.m_StrideX == 0)
    {
        throw armnn::Exception("m_StrideX cannot be 0.");
    }

    if (m_Param.m_StrideY == 0)
    {
        throw armnn::Exception("m_StrideY cannot be 0.");
    }

    DataLayoutIndexed dataLayoutIndex(m_Param.m_DataLayout);

    unsigned int inWidth = inputShape[dataLayoutIndex.GetWidthIndex()];
    unsigned int inHeight = inputShape[dataLayoutIndex.GetHeightIndex()];
    unsigned int inBatchSize = inputShape[0];

    unsigned int filterWidth = filterShape[dataLayoutIndex.GetWidthIndex()];
    unsigned int dilatedFilterWidth = filterWidth + (m_Param.m_DilationX - 1) * (filterWidth - 1);
    unsigned int readWidth = (inWidth + m_Param.m_PadLeft + m_Param.m_PadRight) - dilatedFilterWidth;
    unsigned int outWidth = 1 + (readWidth / m_Param.m_StrideX);

    unsigned int filterHeight = filterShape[dataLayoutIndex.GetHeightIndex()];
    unsigned int dilatedFilterHeight = filterHeight + (m_Param.m_DilationY - 1) * (filterHeight - 1);
    unsigned int readHeight = (inHeight + m_Param.m_PadTop + m_Param.m_PadBottom) - dilatedFilterHeight;
    unsigned int outHeight = 1 + (readHeight / m_Param.m_StrideY);

    unsigned int outChannels = filterShape[0];
    unsigned int outBatchSize = inBatchSize;

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NHWC ?
        TensorShape( { outBatchSize, outHeight, outWidth, outChannels } ) :
        TensorShape( { outBatchSize, outChannels, outHeight, outWidth });

    return std::vector<TensorShape>({ tensorShape });
}

void Convolution2dLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(m_Param.GetNumInputs(), CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    if (!GetInputSlot(1).GetConnection())
    {
        throw armnn::NullPointerException("Convolution2dLayer: Weights should be connected to input slot 1.");
    }

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
             GetInputSlot(0).GetTensorInfo().GetShape(),
             GetInputSlot(1).GetTensorInfo().GetShape() });

    if (inferredShapes.size() != 1)
    {
        throw armnn::Exception("inferredShapes has "
                               + std::to_string(inferredShapes.size()) +
                               " elements - should only have 1.");
    }

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "Convolution2dLayer");
}

Layer::ImmutableConstantTensors Convolution2dLayer::GetConstantTensorsByRef() const
{
    Layer::ImmutableConstantTensors tensors = GetConnectedConstantAsInputTensors();
    return tensors;
}

void Convolution2dLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
