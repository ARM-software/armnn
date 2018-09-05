//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "SplitterLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

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

void SplitterLayer::CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory)
{
    //If sub tensors are supported than all the "splitter" need to do is to
    //set the outputs to be appropriate sub tensors of the input.
    if (factory.SupportsSubTensors())
    {
        const OutputHandler& outputHandler = GetInputSlots()[0].GetConnectedOutputSlot()->GetOutputHandler();

        ITensorHandle* inputData = outputHandler.GetData();
        //Creates the outputs as subtensors of the input.
        for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
        {
            m_OutputHandlers[i].SetData(factory.CreateSubTensorHandle(*inputData,
                                                                      m_OutputHandlers[i].GetTensorInfo().GetShape(),
                                                                      m_Param.GetViewOrigin(i)));
        }
    }
    else
    {
        for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
        {
            m_OutputHandlers[i].CreateTensorHandles(factory);
        }
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

} // namespace armnn
