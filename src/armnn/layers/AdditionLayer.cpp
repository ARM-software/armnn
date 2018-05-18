//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "AdditionLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

AdditionLayer::AdditionLayer(const char* name)
    : Layer(2, 1, LayerType::Addition, name)
{
}

std::unique_ptr<IWorkload> AdditionLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    AdditionQueueDescriptor descriptor;
    return factory.CreateAddition(descriptor, PrepInfoAndDesc(descriptor, graph));
}

AdditionLayer* AdditionLayer::Clone(Graph& graph) const
{
    return CloneBase<AdditionLayer>(graph, GetName());
}

void AdditionLayer::ValidateTensorShapesFromInputs()
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
        "AdditionLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
