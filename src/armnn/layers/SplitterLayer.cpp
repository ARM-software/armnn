//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "SplitterLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

SplitterLayer::SplitterLayer(const ViewsDescriptor& param, const char* name)
    : LayerWithParameters(1, param.GetNumViews(), LayerType::Splitter, param, name)
{
}

std::unique_ptr<IWorkload> SplitterLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    SplitterQueueDescriptor descriptor;

    // Copies the window origins to the descriptor.
    for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
    {
        descriptor.m_ViewOrigins.emplace_back(
            std::vector<unsigned int>(m_Param.GetViewOrigin(i), m_Param.GetViewOrigin(i) + m_Param.GetNumDimensions()));
    }

    return factory.CreateSplitter(descriptor, PrepInfoAndDesc(descriptor, graph));
}

template<typename FactoryType>
void SplitterLayer::CreateTensors(const FactoryType& factory)
{
    //If sub tensors are supported than all the "splitter" need to do is to
    //set the outputs to be appropriate sub tensors of the input.
    bool useSubTensors = factory.SupportsSubTensors();

    if (useSubTensors)
    {
        const OutputSlot* slot = GetInputSlots()[0].GetConnectedOutputSlot();
        const OutputHandler& outputHandler = GetInputSlots()[0].GetConnectedOutputSlot()->GetOutputHandler();

        const TensorInfo& parentInfo = outputHandler.GetTensorInfo();

        ITensorHandle* inputData = outputHandler.GetData();

        std::vector<std::unique_ptr<ITensorHandle>> subTensors;

        //Creates the outputs as subtensors of the input.
        for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
        {
            const TensorInfo& info = m_OutputHandlers[i].GetTensorInfo();

            OutputSlot& outSlot = GetOutputSlot(i);
            ITensorHandleFactory::FactoryId factoryId = outSlot.GetTensorHandleFactoryId();
            auto CreateSubTensor = [&]()
            {
                // Make sure quantization parameters are in the same space
                if (parentInfo.IsTypeSpaceMatch(info) &&
                    factoryId == slot->GetTensorHandleFactoryId())
                {
                    return factory.CreateSubTensorHandle(*inputData,
                                                         info.GetShape(),
                                                         this->m_Param.GetViewOrigin(i));
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
            m_OutputHandlers[i].CreateTensorHandles(factory);
        }
    }
}

void SplitterLayer::CreateTensorHandles(const TensorHandleFactoryRegistry& registry,
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

SplitterLayer* SplitterLayer::Clone(Graph& graph) const
{
    return CloneBase<SplitterLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> SplitterLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() ==  m_Param.GetNumViews());
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
    std::vector<TensorShape> views;
    for (unsigned int viewIdx = 0; viewIdx < m_Param.GetNumViews(); viewIdx++)
    {
        const uint32_t* sizes = m_Param.GetViewSizes(viewIdx);
        views.push_back(TensorShape(m_Param.GetNumDimensions(), sizes));
    }

    auto inferredShapes = InferOutputShapes(views);

    BOOST_ASSERT(inferredShapes.size() == m_Param.GetNumViews());

    for (unsigned int viewIdx = 0; viewIdx < m_Param.GetNumViews(); viewIdx++)
    {
        ConditionalThrowIfNotEqual<LayerValidationException>(
            "SplitterLayer: View sizes must match output tensor shapes.",
            GetOutputSlot(viewIdx).GetTensorInfo().GetShape(),
            inferredShapes[viewIdx]);
    }
}

void SplitterLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitSplitterLayer(this, GetParameters(), GetName());
}

} // namespace armnn
