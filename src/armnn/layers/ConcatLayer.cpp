//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ConcatLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <queue>

namespace armnn
{

ConcatLayer::ConcatLayer(const OriginsDescriptor& param, const char* name)
    : LayerWithParameters(param.GetNumViews(), 1, LayerType::Concat, param, name)
{
}

std::unique_ptr<IWorkload> ConcatLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    ConcatQueueDescriptor descriptor;

    // Copies the view origins to the descriptor.
    descriptor.m_ViewOrigins.reserve(m_Param.GetNumViews());
    for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
    {
        descriptor.m_ViewOrigins.emplace_back(
            std::vector<unsigned int>(m_Param.GetViewOrigin(i), m_Param.GetViewOrigin(i) + m_Param.GetNumDimensions()));
    }

    return factory.CreateConcat(descriptor, PrepInfoAndDesc(descriptor, graph));
}

template<typename FactoryType>
void ConcatLayer::CreateTensors(const FactoryType& factory)
{
    //If sub tensors are supported then the concat
    //just needs to make sure that the outputs of the prev layer
    //are made subtensors of the output of the concat layer.
    m_OutputHandlers[0].CreateTensorHandles(factory);

    if (factory.SupportsSubTensors())
    {
        ITensorHandleFactory::FactoryId factoryId = GetOutputSlot(0).GetTensorHandleFactoryId();

        std::queue<ConcatLayer*> m_ConcatLayers;

        m_ConcatLayers.push(this);
        while (!m_ConcatLayers.empty())
        {
            ConcatLayer* currentLayer = m_ConcatLayers.front();
            ITensorHandle* parentTensor = currentLayer->GetOutputHandler(0).GetData();
            const TensorInfo& parentInfo = currentLayer->GetOutputHandler(0).GetTensorInfo();
            m_ConcatLayers.pop();

            const unsigned int numInputSlots = currentLayer->GetNumInputSlots();

            // First go through all the input slots and verify that we can sub-tensor all the inputs.
            std::vector<std::unique_ptr<ITensorHandle>> subTensors(0);
            subTensors.reserve(numInputSlots);
            for (unsigned int i = 0; i < numInputSlots; ++i)
            {
                OutputSlot* slot = currentLayer->GetInputSlot(i).GetConnectedOutputSlot();
                const TensorInfo& info = slot->GetTensorInfo();

                auto CreateSubTensor = [&]()
                {
                    // Make sure:
                    // 1) quantization parameters are in the same space
                    // 2) the same TensorHandleFactory is used for input and Concat layer output
                    // 3) the input does not come from a Constant layer or input layer
                    // 4) the input is only read by this concat layer
                    if (slot &&
                        parentInfo.IsTypeSpaceMatch(info) && //(1)
                        factoryId == slot->GetTensorHandleFactoryId() && //(2)
                        slot->GetOwningLayer().GetType() != LayerType::Constant && //(3)
                        slot->GetOwningLayer().GetType() != LayerType::Input && //(3)
                        slot->GetNumConnections() == 1) //(4)
                    {
                        return factory.CreateSubTensorHandle(*parentTensor,
                                                             info.GetShape(),
                                                             currentLayer->m_Param.GetViewOrigin(i));
                    }
                    return std::unique_ptr<ITensorHandle>();
                };

                auto subTensor = CreateSubTensor();
                if (!subTensor)
                {
                    break; //Failed to create a valid sub-tensor, so stop trying with the rest of the inputs.
                }
                else
                {
                    subTensors.push_back(std::move(subTensor)); // store the valid sub-tensor.
                }
            }

            // Ensure that ALL inputs can be substituted with valid sub-tensors
            if (subTensors.size() < numInputSlots)
            {
                continue; // Don't optimize this Concat layer with sub-tensors
            }

            // Substitute input tensors with sub-tensors by replacing the output tensors on the connected layers.
            unsigned int i=0;
            for (auto& subTensor : subTensors)
            {
                OutputSlot* slot = currentLayer->GetInputSlot(i).GetConnectedOutputSlot();
                OutputHandler& outputHandler = slot->GetOutputHandler();

                BOOST_ASSERT_MSG(subTensor, "ConcatLayer: Expected a valid sub-tensor for substitution.");
                outputHandler.SetData(std::move(subTensor));

                Layer& inputLayer = slot->GetOwningLayer();
                if (inputLayer.GetType() == LayerType::Concat)
                {
                    // Continue with the substitution if the connected inputs are also concat layers
                    m_ConcatLayers.push(boost::polymorphic_downcast<ConcatLayer*>(&inputLayer));
                }
                ++i;
            }
        }
    }
}

void ConcatLayer::CreateTensorHandles(const TensorHandleFactoryRegistry& registry,
                                      const IWorkloadFactory& workloadFactory)
{
    OutputSlot& slot = GetOutputSlot(0);
    ITensorHandleFactory::FactoryId factoryId = slot.GetTensorHandleFactoryId();

    if (factoryId == ITensorHandleFactory::LegacyFactoryId)
    {
        CreateTensors(workloadFactory);
    }
    else
    {
        ITensorHandleFactory* handleFactory = registry.GetFactory(factoryId);
        BOOST_ASSERT(handleFactory);
        CreateTensors(*handleFactory);
    }
}

ConcatLayer* ConcatLayer::Clone(Graph& graph) const
{
    return CloneBase<ConcatLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ConcatLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == m_Param.GetNumViews());

    unsigned int numDims = m_Param.GetNumDimensions();
    for (unsigned int i=0; i< inputShapes.size(); i++)
    {
        auto& inputShape = inputShapes[i];

        ConditionalThrowIfNotEqual<LayerValidationException>(
            "ConcatLayer: Num Dimensions must match all inputs.",
            numDims,
            inputShape.GetNumDimensions());
    }

    // Finds the bounding box (extents) of all the views.
    std::vector<unsigned int> extentMin(numDims);
    std::vector<unsigned int> extentMax(numDims);
    for (unsigned int i = 0; i < inputShapes.size(); i++)
    {
        const uint32_t* origin = m_Param.GetViewOrigin(i);
        const armnn::TensorShape& shape = inputShapes[i];
        for (unsigned int d = 0; d < numDims; d++)
        {
            extentMin[d] = std::min(extentMin[d], origin[d]);
            extentMax[d] = std::max(extentMax[d], origin[d] + shape[d]);
        }
    }

    // Checks that the bounding box starts at the origin.
    if (!std::all_of(extentMin.begin(), extentMin.end(), [](unsigned int s) { return s == 0; }))
    {
        throw LayerValidationException("ConcatLayer: there is no view that starts at the origin");
    }

    // Checks that there are no overlaps of views (this would lead to undefined output at those locations).
    // Checks each pair of views against each other
    // (and doesn't bother to check against self, or check the same pair both ways round).
    for (unsigned int a = 0; a < inputShapes.size(); a++)
    {
        const uint32_t* aOrigin = m_Param.GetViewOrigin(a);
        const armnn::TensorShape& aShape = inputShapes[a];
        for (unsigned int b = 0; b < a; b++)
        {
            const uint32_t* bOrigin = m_Param.GetViewOrigin(b);
            const armnn::TensorShape& bShape = inputShapes[b];

            bool allAxesOverlap = true;
            for (unsigned int d = 0; d < numDims && allAxesOverlap; d++)
            {
                unsigned int a1 = aOrigin[d];
                unsigned int a2 = aOrigin[d] + aShape[d];

                unsigned int b1 = bOrigin[d];
                unsigned int b2 = bOrigin[d] + bShape[d];

                if (a2 <= b1 || b2 <= a1)
                {
                    allAxesOverlap = false;
                }
            }
            if (allAxesOverlap)
            {
                throw LayerValidationException("ConcatLayer: Some views overlap.");
            }
        }
    }

    // Checks that there are no "holes", i.e. regions of the output which is not covered by a view.
    // Because we already checked that there are no overlaps, this can be done simply by checking that
    // the total 'volume' of the views is the same as the output.
    unsigned int totalViewsVolume = 0;
    for (unsigned int i = 0; i < inputShapes.size(); i++)
    {
        totalViewsVolume += inputShapes[i].GetNumElements();
    }
    unsigned int outputVolume = 1;
    for (unsigned int d = 0; d < numDims; d++)
    {
        outputVolume *= (extentMax[d] - extentMin[d]);
    }

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ConcatLayer: there are some gaps between views",
        totalViewsVolume,
        outputVolume);

    return std::vector<TensorShape>({ TensorShape({numDims, extentMax.data()}) });
}

void ConcatLayer::ValidateTensorShapesFromInputs()
{
    // Validates Concat layer.
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ConcatLayer: Num Inputs must match num views.",
        m_Param.GetNumViews(),
        GetNumInputSlots());

    VerifyLayerConnections(m_Param.GetNumViews(), CHECK_LOCATION());

    std::vector<TensorShape> inputShapes;
    for (unsigned int i = 0; i < GetNumInputSlots(); ++i)
    {
        inputShapes.push_back(GetInputSlot(i).GetConnection()->GetTensorInfo().GetShape());
    }

    auto inferredShapes = InferOutputShapes(inputShapes);

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ConcatLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void ConcatLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitConcatLayer(this, GetParameters(), GetName());
}

} // namespace armnn armnn
