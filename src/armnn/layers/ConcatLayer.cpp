//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ConcatLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <queue>

namespace armnn
{

ConcatLayer::ConcatLayer(const OriginsDescriptor& param, const char* name)
    : LayerWithParameters(param.GetNumViews(), 1, LayerType::Concat, param, name)
{
}

std::unique_ptr<IWorkload> ConcatLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ConcatQueueDescriptor descriptor;

    // Copies the view origins to the descriptor.
    descriptor.m_ViewOrigins.reserve(m_Param.GetNumViews());
    for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
    {
        descriptor.m_ViewOrigins.emplace_back(
            std::vector<unsigned int>(m_Param.GetViewOrigin(i), m_Param.GetViewOrigin(i) + m_Param.GetNumDimensions()));
    }
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Concat, descriptor, PrepInfoAndDesc(descriptor));
}

template<typename FactoryType>
void ConcatLayer::CreateTensors(const TensorHandleFactoryRegistry& registry,
                                const FactoryType& factory,
                                bool isMemoryManaged)
{
    //If sub tensors are supported then the concat
    //just needs to make sure that the outputs of the prev layer
    //are made subtensors of the output of the concat layer.
    m_OutputHandlers[0].CreateTensorHandles(factory, isMemoryManaged);

    if (factory.SupportsSubTensors())
    {
        // check if concat is along the x or y (2 innermost dimensions)
        uint32_t concatAxis = m_Param.GetConcatAxis();
        auto numberOfDimensions = m_Param.GetNumDimensions();
        bool isConcatOnXorY = m_Param.GetNumDimensions() >= 3
                                && ((concatAxis == numberOfDimensions - 1) || (concatAxis == numberOfDimensions - 2));

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

            // if concat along x or y (2 innermost dimensions) and the previous layers do not require padding
            bool canUseSubTensorOnXorY = true;
            bool isTensorHandleFactory = std::is_same<armnn::ITensorHandleFactory, FactoryType>::value;
            if (isTensorHandleFactory)
            {
                for (unsigned int i = 0; i < numInputSlots; ++i)
                {
                    OutputSlot* slot = currentLayer->GetInputSlot(i).GetConnectedOutputSlot();
                    ITensorHandleFactory* handleFactory  = registry.GetFactory(factoryId);
                    std::vector<Capability> capabilities =
                        handleFactory->GetCapabilities(&(slot->GetOwningLayer()),
                                                       currentLayer,
                                                       CapabilityClass::PaddingRequired);
                    if (isConcatOnXorY)
                    {
                        canUseSubTensorOnXorY = false;
                        if (capabilities.empty())
                        {
                            canUseSubTensorOnXorY = true;
                        }
                    }

                    // Splitter layer outputs are subtensors on the inputs whereas concat inputs are subtensors on
                    // the output. If the parent is a Splitter layer we cannot use subtensors.
                    if ((PolymorphicDowncast<const Layer*>(&(slot->GetOwningLayer())))->GetType() == LayerType::Splitter
                        && (PolymorphicDowncast<const Layer*>(currentLayer))->GetType() == LayerType::Concat)
                    {
                        canUseSubTensorOnXorY = false;
                    }

                    if (!canUseSubTensorOnXorY)
                    {
                        break;
                    }
                }
            }

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
                    // 5) if concat along x or y (2 innermost dimensions) and the previous layers do not require padding
                    if (slot &&
                        parentInfo.IsTypeSpaceMatch(info) && //(1)
                        factoryId == slot->GetTensorHandleFactoryId() && //(2)
                        slot->GetOwningLayer().GetType() != LayerType::Constant && //(3)
                        slot->GetOwningLayer().GetType() != LayerType::Input && //(3)
                        slot->GetNumConnections() == 1 &&
                        canUseSubTensorOnXorY) //(5)
                    {
                        ARMNN_NO_DEPRECATE_WARN_BEGIN
                        return factory.CreateSubTensorHandle(*parentTensor,
                                                             info.GetShape(),
                                                             currentLayer->m_Param.GetViewOrigin(i));
                        ARMNN_NO_DEPRECATE_WARN_END
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

                ARMNN_ASSERT_MSG(subTensor, "ConcatLayer: Expected a valid sub-tensor for substitution.");
                outputHandler.SetData(std::move(subTensor));

                Layer& inputLayer = slot->GetOwningLayer();
                if (inputLayer.GetType() == LayerType::Concat)
                {
                    // Continue with the substitution if the connected inputs are also concat layers
                    m_ConcatLayers.push(PolymorphicDowncast<ConcatLayer*>(&inputLayer));
                }
                ++i;
            }
        }
    }
}

void ConcatLayer::CreateTensorHandles(const TensorHandleFactoryRegistry& registry,
                                      const IWorkloadFactory& workloadFactory,
                                      const bool isMemoryManaged)
{
    OutputSlot& slot = GetOutputSlot(0);
    ITensorHandleFactory::FactoryId factoryId = slot.GetTensorHandleFactoryId();

    if (factoryId == ITensorHandleFactory::LegacyFactoryId)
    {
        CreateTensors(registry, workloadFactory, isMemoryManaged);
    }
    else
    {
        ITensorHandleFactory* handleFactory = registry.GetFactory(factoryId);
        ARMNN_ASSERT(handleFactory);
        CreateTensors(registry, *handleFactory, isMemoryManaged);
    }
}

ConcatLayer* ConcatLayer::Clone(Graph& graph) const
{
    return CloneBase<ConcatLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ConcatLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == m_Param.GetNumViews());

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

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inputShapes;
    for (unsigned int i = 0; i < GetNumInputSlots(); ++i)
    {
        inputShapes.push_back(GetInputSlot(i).GetConnection()->GetTensorInfo().GetShape());
    }

    auto inferredShapes = InferOutputShapes(inputShapes);

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ConcatLayer");
}

void ConcatLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn armnn
