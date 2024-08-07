//
// Copyright © 2019-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SliceLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

SliceLayer::SliceLayer(const SliceDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Slice, param, name)
{
}

std::unique_ptr<IWorkload> SliceLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    SliceQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Slice, descriptor, PrepInfoAndDesc(descriptor));
}

SliceLayer* SliceLayer::Clone(Graph& graph) const
{
    return CloneBase<SliceLayer>(graph, m_Param, GetName());
}

void SliceLayer::ValidateTensorShapesFromInputs()
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

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "SliceLayer");
}

std::vector<TensorShape> SliceLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    IgnoreUnused(inputShapes);

    if (inputShapes.size() != 1)
    {
        throw armnn::Exception("inputShapes' size is \"" + std::to_string(inputShapes.size()) +
                               "\" - should be \"1\".");
    }

    TensorShape outputShape(armnn::numeric_cast<unsigned int>(m_Param.m_Size.size()), m_Param.m_Size.data());

    return std::vector<TensorShape>({ outputShape });
}

void SliceLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
