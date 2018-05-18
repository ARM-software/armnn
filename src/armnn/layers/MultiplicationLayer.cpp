//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "MultiplicationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

MultiplicationLayer::MultiplicationLayer(const char* name)
    : Layer(2, 1, LayerType::Multiplication, name)
{
}

std::unique_ptr<IWorkload> MultiplicationLayer::CreateWorkload(const Graph&            graph,
                                                               const IWorkloadFactory& factory) const
{
    MultiplicationQueueDescriptor descriptor;

    return factory.CreateMultiplication(descriptor, PrepInfoAndDesc(descriptor, graph));
}

MultiplicationLayer* MultiplicationLayer::Clone(Graph& graph) const
{
    return CloneBase<MultiplicationLayer>(graph, GetName());
}

void MultiplicationLayer::ValidateTensorShapesFromInputs()
{
    auto& input0 = GetInputSlot(0).GetConnection()->GetTensorInfo();
    auto& input1 = GetInputSlot(1).GetConnection()->GetTensorInfo();

    // Get the max of the inputs
    BOOST_ASSERT(input0.GetNumDimensions() == input1.GetNumDimensions());
    unsigned int numDims = input0.GetNumDimensions();
    std::vector<unsigned int> dims(numDims);

    // validate inputs are broadcast compatible
#if !NDEBUG
    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0.GetShape()[i];
        unsigned int dim1 = input1.GetShape()[i];
        if (dim0 != dim1)
        {
            BOOST_ASSERT_MSG(dim0 == 1 || dim1 == 1, "Dimensions should either match or one should be of size 1.");
        }
    }
#endif

    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0.GetShape()[i];
        unsigned int dim1 = input1.GetShape()[i];
        dims[i] = std::max(dim0, dim1);
    }

    TensorShape outShape(numDims, dims.data());
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MultiplicationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
