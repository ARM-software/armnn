//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "DivisionLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

DivisionLayer::DivisionLayer(const char* name)
        : Layer(2, 1, LayerType::Division, name)
{
}

std::unique_ptr<IWorkload> DivisionLayer::CreateWorkload(const Graph& graph,
                                                         const IWorkloadFactory& factory) const
{
    DivisionQueueDescriptor descriptor;

    return factory.CreateDivision(descriptor, PrepInfoAndDesc(descriptor, graph));
}

DivisionLayer* DivisionLayer::Clone(Graph& graph) const
{
    return CloneBase<DivisionLayer>(graph, GetName());
}

std::vector<TensorShape> DivisionLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 2);
    auto& input0 = inputShapes[0];
    auto& input1 = inputShapes[1];

    // Get the max of the inputs.
    BOOST_ASSERT(input0.GetNumDimensions() == input1.GetNumDimensions());
    unsigned int numDims = input0.GetNumDimensions();
    std::vector<unsigned int> dims(numDims);

    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0[i];
        unsigned int dim1 = input1[i];

        // Validates inputs are broadcast compatible.
#if !NDEBUG
        if (dim0 != dim1)
        {
            BOOST_ASSERT_MSG(dim0 == 1 || dim1 == 1, "Dimensions should either match or one should be of size 1.");
        }
#endif

        dims[i] = std::max(dim0, dim1);
    }

    return std::vector<TensorShape>({ TensorShape(numDims, dims.data()) });
}

void DivisionLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({
                                                    GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
                                                    GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()
                                            });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
            "DivisionLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
            GetOutputSlot(0).GetTensorInfo().GetShape(),
            inferredShapes[0]);
}

} // namespace armnn
