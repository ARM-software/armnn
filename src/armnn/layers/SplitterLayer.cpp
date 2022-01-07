//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "SplitterLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

SplitterLayer::SplitterLayer(const ViewsDescriptor& param, const char* name)
    : LayerWithParameters(1, param.GetNumViews(), LayerType::Splitter, param, name)
{
}

std::unique_ptr<IWorkload> SplitterLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    SplitterQueueDescriptor descriptor;

    // Copies the window origins to the descriptor.
    for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
    {
        descriptor.m_ViewOrigins.emplace_back(
            std::vector<unsigned int>(m_Param.GetViewOrigin(i), m_Param.GetViewOrigin(i) + m_Param.GetNumDimensions()));
    }

    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Splitter, descriptor, PrepInfoAndDesc(descriptor));
}

template<typename FactoryType>
void SplitterLayer::CreateTensors(const TensorHandleFactoryRegistry& registry,
                                  const FactoryType& factory,
                                  bool isMemoryManaged)
{
    //If sub tensors are supported than all the "splitter" need to do is to
    //set the outputs to be appropriate sub tensors of the input.
    bool useSubTensors = factory.SupportsSubTensors();

    if (useSubTensors)
    {
        // Get outputHandler of previous layer
        const OutputHandler& outputHandler = GetInputSlots()[0].GetConnectedOutputSlot()->GetOutputHandler();
        const OutputSlot* slot = GetInputSlots()[0].GetConnectedOutputSlot();

        const TensorInfo& parentInfo = outputHandler.GetTensorInfo();

        ITensorHandle* inputData = outputHandler.GetData();

        std::vector<std::unique_ptr<ITensorHandle>> subTensors;

        // check if split is along the x or y (2 innermost dimensions)
        auto numberOfDimensions = m_Param.GetNumDimensions();

        // Compute split axis within class as aclCommon function causes header issues when included
        auto ComputeSplitAxis = [&](const armnn::SplitterDescriptor& desc, const TensorShape& input)
        {
            unsigned int numSplit = desc.GetNumViews();
            unsigned int numDimensions = desc.GetNumDimensions();
            std::set<unsigned int> splitAxis;

            for (unsigned int i = 0; i < numSplit; ++i)
            {
                for (unsigned int dimIdx = 0; dimIdx < numDimensions; ++dimIdx)
                {
                    if (desc.GetViewSizes(i)[dimIdx] != input[dimIdx])
                    {
                        splitAxis.insert(dimIdx);
                    }
                }
            }
            return splitAxis;
        };

        std::set<unsigned int> axis = ComputeSplitAxis(m_Param, parentInfo.GetShape());
        std::set<unsigned int>::iterator axisIt = axis.begin();

        bool isOnXorY = m_Param.GetNumDimensions() >= 3 &&
                            ((*axisIt == numberOfDimensions - 1) ||
                                (*axisIt == numberOfDimensions - 2));

        //Creates the outputs as subtensors of the input.
        for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
        {
            const TensorInfo& info = m_OutputHandlers[i].GetTensorInfo();

            OutputSlot& outSlot = GetOutputSlot(i);
            ITensorHandleFactory::FactoryId factoryId = outSlot.GetTensorHandleFactoryId();

            const unsigned int numOutputSlots = GetNumOutputSlots();

            // if split along x or y (2 innermost dimensions) and the next layers do not require padding
            bool canUseSubTensorOnXorY = true;
            bool isTensorHandleFactory = std::is_same<armnn::ITensorHandleFactory, FactoryType>::value;
            if (isTensorHandleFactory)
            {
                for (unsigned int it = 0; it < numOutputSlots; ++it)
                {
                    InputSlot* inputSlot = GetOutputSlot(it).GetConnection(0);
                    ITensorHandleFactory* handleFactory  = registry.GetFactory(factoryId);
                    std::vector<Capability> capabilities =
                        handleFactory->GetCapabilities(&(inputSlot->GetOwningLayer()),
                                                       this,
                                                       CapabilityClass::PaddingRequired);
                    if (isOnXorY)
                    {
                        canUseSubTensorOnXorY = false;
                        if (capabilities.empty())
                        {
                            canUseSubTensorOnXorY = true;
                        }
                    }

                    if (!canUseSubTensorOnXorY)
                    {
                        break;
                    }
                }
            }

            auto CreateSubTensor = [&]()
            {
                // Make sure:
                // 1) quantization parameters are in the same space
                // 2) the same TensorHandleFactory is used for input and split layer output
                // 3) the output does not go to a Constant layer or input layer
                // 4) if split along x or y (2 innermost dimensions) and the next layers do not require padding
                if (parentInfo.IsTypeSpaceMatch(info) && //(1)
                    factoryId == slot->GetTensorHandleFactoryId() && //(2)
                    GetOutputSlot(i).GetConnection(0)->GetOwningLayer().GetType() != LayerType::Constant && //(3)
                    GetOutputSlot(i).GetConnection(0)->GetOwningLayer().GetType() != LayerType::Input && //(3)
                    canUseSubTensorOnXorY) //(4)
                {
                    ARMNN_NO_DEPRECATE_WARN_BEGIN
                    return factory.CreateSubTensorHandle(*inputData,
                                                         info.GetShape(),
                                                         this->m_Param.GetViewOrigin(i));
                    ARMNN_NO_DEPRECATE_WARN_END
                }
                return std::unique_ptr<ITensorHandle>();
            };

            auto subTensor = CreateSubTensor();
            if (!subTensor)
            {
                useSubTensors = false;
                break; //Failed to create a valid sub-tensor, so stop trying with the rest of the views.
            }
            subTensors.push_back(std::move(subTensor));
        }

        if (useSubTensors)
        {
            unsigned int i = 0;
            for (auto& subTensor : subTensors)
            {
                m_OutputHandlers[i].SetData(std::move(subTensor));
                ++i;
            }
        }
    }

    if (!useSubTensors)
    {
        for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
        {
            m_OutputHandlers[i].CreateTensorHandles(factory, isMemoryManaged);
        }
    }
}

void SplitterLayer::CreateTensorHandles(const TensorHandleFactoryRegistry& registry,
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

SplitterLayer* SplitterLayer::Clone(Graph& graph) const
{
    return CloneBase<SplitterLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> SplitterLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    IgnoreUnused(inputShapes);
    ARMNN_ASSERT(inputShapes.size() ==  m_Param.GetNumViews());
    std::vector<TensorShape> outShapes;
    //Output shapes must match View shapes.
    for (unsigned int viewIdx = 0; viewIdx < m_Param.GetNumViews(); viewIdx++)
    {
        const uint32_t* sizes = m_Param.GetViewSizes(viewIdx);
        outShapes.push_back(TensorShape(m_Param.GetNumDimensions(), sizes));
    }
    return outShapes;
}

void SplitterLayer::ValidateTensorShapesFromInputs()
{
    std::for_each(BeginOutputSlots(), EndOutputSlots(), [&](OutputSlot& outputSlot)
    {
        VerifyShapeInferenceType(outputSlot.GetTensorInfo().GetShape(), m_ShapeInferenceMethod);
    });

    std::vector<TensorShape> views;
    for (unsigned int viewIdx = 0; viewIdx < m_Param.GetNumViews(); viewIdx++)
    {
        const uint32_t* sizes = m_Param.GetViewSizes(viewIdx);
        views.push_back(TensorShape(m_Param.GetNumDimensions(), sizes));
    }

    auto inferredShapes = InferOutputShapes(views);

    ARMNN_ASSERT(inferredShapes.size() == m_Param.GetNumViews());

    for (unsigned int viewIdx = 0; viewIdx < m_Param.GetNumViews(); viewIdx++)
    {
        ValidateAndCopyShape(GetOutputSlot(viewIdx).GetTensorInfo().GetShape(),
                             inferredShapes[viewIdx],
                             m_ShapeInferenceMethod,
                             "SplitterLayer",
                             viewIdx);
    }
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void SplitterLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitSplitterLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
