//
// Copyright © 2017-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PadLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <cstring>

namespace armnn
{

PadLayer::PadLayer(const armnn::PadDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Pad, param, name)
{}

std::unique_ptr<IWorkload> PadLayer::CreateWorkload(const armnn::IWorkloadFactory& factory) const
{
    PadQueueDescriptor descriptor;
    descriptor.m_Parameters.m_PadList = m_Param.m_PadList;
    descriptor.m_Parameters.m_PaddingMode = m_Param.m_PaddingMode;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Pad, descriptor, PrepInfoAndDesc(descriptor));
}

PadLayer* PadLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<PadLayer>(graph, m_Param, GetName());

    layer->m_Param.m_PadList = m_Param.m_PadList;
    layer->m_Param.m_PaddingMode = m_Param.m_PaddingMode;

    return std::move(layer);
}

std::vector<TensorShape> PadLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    if (inputShapes.size() != 1)
    {
        throw armnn::Exception("inputShapes' size is \"" + std::to_string(inputShapes.size()) +
                               "\" - should be \"1\".");
    }

    const TensorShape& inputShape = inputShapes[0];

    unsigned int rank = inputShape.GetNumDimensions();

    if (m_Param.m_PadList.size() != rank)
    {
        throw armnn::Exception("Mismatch in size of mPadList and rank (\""
                               + std::to_string(m_Param.m_PadList.size()) +
                               "\" vs "
                               + std::to_string(rank) + ")");
    }

    if (rank == 0)
    {
        throw armnn::Exception("rank must not equal 0.");
    }

    std::vector<unsigned int> outputDimensionSizes(rank);
    for (unsigned int i = 0; i < rank; ++i)
    {
        outputDimensionSizes[i] = inputShape[i] + m_Param.m_PadList[i].first + m_Param.m_PadList[i].second;
    }

    TensorShape tensorShape = TensorShape( rank, outputDimensionSizes.data());
    return std::vector<TensorShape>({ tensorShape });
}

void PadLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetTensorInfo().GetShape() });

    if (inferredShapes.size() != 1)
    {
        throw armnn::LayerValidationException("inferredShapes has "
                                              + std::to_string(inferredShapes.size()) +
                                              " elements - should only have 1.");
    }

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "PadLayer");
}

void PadLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
