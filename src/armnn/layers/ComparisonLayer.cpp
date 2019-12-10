//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ComparisonLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <algorithm>

namespace armnn
{

ComparisonLayer::ComparisonLayer(const ComparisonDescriptor& param, const char* name)
    : LayerWithParameters(2, 1, LayerType::Comparison, param, name)
{
}

std::unique_ptr<IWorkload> ComparisonLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ComparisonQueueDescriptor descriptor;
    return factory.CreateComparison(descriptor, PrepInfoAndDesc(descriptor));
}

ComparisonLayer* ComparisonLayer::Clone(Graph& graph) const
{
    return CloneBase<ComparisonLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ComparisonLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 2);
    const TensorShape& input0 = inputShapes[0];
    const TensorShape& input1 = inputShapes[1];

    BOOST_ASSERT(input0.GetNumDimensions() == input1.GetNumDimensions());
    unsigned int numDims = input0.GetNumDimensions();

    std::vector<unsigned int> dims(numDims);
    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0[i];
        unsigned int dim1 = input1[i];

        BOOST_ASSERT_MSG(dim0 == dim1 || dim0 == 1 || dim1 == 1,
                         "Dimensions should either match or one should be of size 1.");

        dims[i] = std::max(dim0, dim1);
    }

    return std::vector<TensorShape>({ TensorShape(numDims, dims.data()) });
}

void ComparisonLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()
    });
    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ComparisonLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void ComparisonLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitComparisonLayer(this, GetParameters(), GetName());
}

} // namespace armnn
