//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Graph.hpp"
#include "SubGraph.hpp"
#include "LayersFwd.hpp"

#include <armnn/Utils.hpp>
#include <armnn/TypesUtils.hpp>

#include <boost/polymorphic_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>

#include <unordered_map>
#include <DotSerializer.hpp>
#include <sstream>

namespace armnn
{

Graph::Graph(const Graph& other)
:   m_LayersInOrder(other.m_LayersInOrder)
{
    std::unordered_map<const Layer*, Layer*> otherToClonedMap;

    for (auto&& otherLayer : other.m_Layers)
    {
        Layer* const layer = otherLayer->Clone(*this);
        otherToClonedMap.emplace(otherLayer, layer);
    }

    // Copies slot connections.
    for (auto&& otherLayer : other.m_Layers)
    {
        Layer* const thisLayer = otherToClonedMap[otherLayer];

        auto outputSlot = thisLayer->BeginOutputSlots();
        for (auto&& otherOutputSlot : otherLayer->GetOutputSlots())
        {
            for (auto&& otherInputSlot : otherOutputSlot.GetConnections())
            {
                const Layer& otherTgtLayer = otherInputSlot->GetOwningLayer();
                Layer* const thisTgtLayer = otherToClonedMap[&otherTgtLayer];

                InputSlot& inputSlot = thisTgtLayer->GetInputSlot(otherInputSlot->GetSlotIndex());
                outputSlot->Connect(inputSlot);
            }
            outputSlot->SetTensorInfo(otherOutputSlot.GetTensorInfo());
            ++outputSlot;
        }
    }
}

Status Graph::Print() const
{
    if (m_Layers.empty())
    {
        BOOST_LOG_TRIVIAL(info) << "\n Graph is empty.\n";
        return Status::Success;
    }
    BOOST_LOG_TRIVIAL(info) << "\n";
    BOOST_LOG_TRIVIAL(info) << "Walking Pattern: \n";

    for (auto&& it : TopologicalSort())
    {
        BOOST_LOG_TRIVIAL(info) << it->GetName() << ":" << GetLayerTypeAsCString(it->GetType())
                                << ":" << it->GetBackendId().Get();
    }
    BOOST_LOG_TRIVIAL(info) << "\n\n";

    return Status::Success;
}

Status Graph::SerializeToDot(std::ostream& stream)
{
    {
        DotGraph graph(stream, "Optimized");

        {
            // Default node attributes:
            DotDefaults nodes(stream, "node");
            nodes.GetAttributeSet()
                .AddAttribute("shape", "record");
        }

        {
            // Default edge attributes:
            DotDefaults edges(stream, "edge");
            edges.GetAttributeSet()
                .AddAttribute("fontsize", 8)
                .AddAttribute("fontcolor", "blue")
                .AddAttribute("fontname", "arial-bold");
        }

        // First declares the nodes.
        for (auto&& layer : m_Layers)
        {
            DotNode node(stream, layer->GetGuid(), GetLayerTypeAsCString(layer->GetType()));
            // Extracts the layer parameters.
            ParameterStringifyFunction extractParams = [&node](const std::string & name, const std::string & value){
                node.GetContents().AddContent(name + " : " + value);
            };
            layer->SerializeLayerParameters(extractParams);
        }

        // Second declares the edges.
        for (auto&& layer : m_Layers)
        {
            LayerGuid toId = layer->GetGuid();

            for (unsigned int i=0;i<layer->GetNumInputSlots(); i++)
            {
                OutputSlot* outputSlot = static_cast<OutputSlot*>(layer->GetInputSlot(i).GetConnection());
                LayerGuid fromId = outputSlot->GetOwningLayer().GetGuid();
                DotEdge edge(stream, fromId, toId);

                // Now print the tensor shape on the edge.
                {
                    // Constructs the label attribute with HTML markup.
                    std::stringstream ss;
                    ss << "< " << outputSlot->GetTensorInfo().GetShape() << " >";
                    edge.GetAttributeSet().AddAttribute("label", ss);
                }
            }
        }
    }

    if (stream.bad())
    {
        return Status::Failure;
    }
    return Status::Success;
}

Status Graph::AllocateDynamicBuffers()
{
    // Layers must be sorted in topological order
    BOOST_ASSERT(m_LayersInOrder);

    std::unordered_set<const ITensorHandle*> preallocatedTensors;
    std::unordered_map<const ITensorHandle*, unsigned int> handleReferenceCounts;

    // Finds the first TensorHandle ancestor of a SubTensorHandle. If the ITensorHandle provided
    // is a TensorHandle, the function just returns it
    auto TraceSubTensorHandleAncestry = [](ITensorHandle* const subTensorHandle)
    {
        ITensorHandle* ancestor = subTensorHandle;
        while (ancestor && ancestor->GetParent())
        {
            ancestor = ancestor->GetParent();
        }
        return ancestor;
    };

    // Checks whether a TensorHandle has been pre-allocated
    auto IsPreallocated = [&](ITensorHandle* const tensorHandle)
    {
        return tensorHandle && preallocatedTensors.find(tensorHandle) != preallocatedTensors.end();
    };

    // Constant tensor handles need to last from the beginning of execution till the end,
    // therefore we pre-allocate them upfront
    for (auto&& layer : m_Layers)
    {
        if (layer->GetType() == LayerType::Constant)
        {
            for (auto&& slot = layer->BeginOutputSlots(); slot != layer->EndOutputSlots(); ++slot)
            {
                ITensorHandle *tensorHandle = TraceSubTensorHandleAncestry(slot->GetOutputHandler().GetData());

                if (tensorHandle && !IsPreallocated(tensorHandle))
                {
                    tensorHandle->Allocate();
                    preallocatedTensors.insert(tensorHandle);
                }
            }
        }
    }

    // Iterate over the network in topological order
    for (auto&& layer : m_Layers)
    {
        // Count the amount of times each output slot references a certain buffer (ITensorHandle).
        // The first time we encounter a new tensor handle, we start managing its lifetime.
        for (auto&& slot = layer->BeginOutputSlots(); slot != layer->EndOutputSlots(); ++slot)
        {
            ITensorHandle *tensorHandle = TraceSubTensorHandleAncestry(slot->GetOutputHandler().GetData());

            if (tensorHandle && !IsPreallocated(tensorHandle))
            {
                unsigned int numConnections = slot->GetNumConnections();
                if (handleReferenceCounts.find(tensorHandle) == handleReferenceCounts.end())
                {
                    handleReferenceCounts[tensorHandle] = numConnections;
                    tensorHandle->Manage();
                }
                else
                {
                    handleReferenceCounts[tensorHandle] += numConnections;
                }
            }
        }

        // Loop through the input slots in the same layer and decrement the reference counter associated
        // to each tensor handle we encounter. Once it reaches zero, we end the lifetime of the tensor handle
        for (auto&& slot = layer->BeginInputSlots(); slot != layer->EndInputSlots(); ++slot)
        {
            ITensorHandle *tensorHandle = TraceSubTensorHandleAncestry(
                slot->GetConnectedOutputSlot()->GetOutputHandler().GetData());

            if (tensorHandle && !IsPreallocated(tensorHandle))
            {
                --handleReferenceCounts[tensorHandle];

                if (handleReferenceCounts[tensorHandle] == 0u)
                {
                    // Stop managing lifetime of tensor handle
                    tensorHandle->Allocate();
                    handleReferenceCounts.erase(tensorHandle);
                }
            }
        }
    }

    return Status::Success;
}

const Graph& Graph::TopologicalSort() const
{
    if (!m_LayersInOrder)
    {
        // Resets layer order.
        for (auto&& it : m_Layers)
        {
            it->ResetPriority();
        }

        auto compareLayerPriority = [](const LayerList::value_type& layerA, const LayerList::value_type& layerB)
            {
                return layerA->GetPriority() < layerB->GetPriority();
            };

        m_Layers.sort(compareLayerPriority);

        m_LayersInOrder = true;
    }

    return *this;
}

void Graph::AddCopyLayers()
{
    // Returns true if the given layer could potentially need an intermediate copy layer (depending on its
    // connections to other layers). At the time of writing, copy layers will be inserted in the following situations:
    // CPU -> CL (and viceversa)
    // CPU -> Neon (and viceversa)
    auto MayNeedCopyLayer = [](const Layer& layer)
        {
            // All layers should have been associated with a valid compute device at this point.
            BOOST_ASSERT(layer.GetBackendId() != Compute::Undefined);
            // Does not need another copy layer if a copy layer is already present.
            return layer.GetType() != LayerType::MemCopy;
        };

    for (auto&& srcLayer : m_Layers)
    {
        if (MayNeedCopyLayer(*srcLayer))
        {
            unsigned int srcOutputIndex = 0;
            for (auto&& srcOutput : srcLayer->GetOutputSlots())
            {
                std::vector<InputSlot*> connectionCopy = srcOutput.GetConnections();
                for (auto&& dstInput : connectionCopy)
                {
                    Layer& dstLayer = dstInput->GetOwningLayer();
                    if (MayNeedCopyLayer(dstLayer) && (dstLayer.GetBackendId() != srcLayer->GetBackendId()))
                    {
                        // A copy layer is needed in between the source and destination layers.
                        // Record the operation rather than attempting to modify the graph as we go.
                        // (invalidating iterators)
                        const std::string copyLayerName = boost::str(boost::format("[ %1% (%2%) -> %3% (%4%) ]")
                                                                     % srcLayer->GetName()
                                                                     % srcOutputIndex
                                                                     % dstLayer.GetName()
                                                                     % dstInput->GetSlotIndex());

                        MemCopyLayer* const copyLayer = InsertNewLayer<MemCopyLayer>(*dstInput, copyLayerName.c_str());
                        copyLayer->SetBackendId(dstLayer.GetBackendId());
                    }
                }
                ++srcOutputIndex;
            }
        }
    }
}

void Graph::SubstituteSubGraph(std::unique_ptr<SubGraph> subGraph, IConnectableLayer* substituteLayer)
{
    BOOST_ASSERT(subGraph != nullptr);
    BOOST_ASSERT(substituteLayer != nullptr);

    ReplaceSubGraphConnections(*subGraph, substituteLayer);
    EraseSubGraphLayers(*subGraph);
}

void Graph::SubstituteSubGraph(std::unique_ptr<SubGraph> subGraph, const SubGraph& substituteSubGraph)
{
    BOOST_ASSERT(subGraph);

    ReplaceSubGraphConnections(*subGraph, substituteSubGraph);
    EraseSubGraphLayers(*subGraph);
}

void Graph::ReplaceSubGraphConnections(const SubGraph& subGraph, IConnectableLayer* substituteLayer)
{
    BOOST_ASSERT(substituteLayer != nullptr);

    // Create a new sub-graph with only the given layer, using
    // the given sub-graph as a reference of which parent graph to use
    SubGraph substituteSubGraph(subGraph, substituteLayer);
    ReplaceSubGraphConnections(subGraph, substituteSubGraph);
}

void Graph::ReplaceSubGraphConnections(const SubGraph& subGraph, const SubGraph& substituteSubGraph)
{
    BOOST_ASSERT_MSG(!substituteSubGraph.GetLayers().empty(), "New sub-graph used for substitution must not be empty");

    const SubGraph::Layers& substituteSubGraphLayers = substituteSubGraph.GetLayers();
    std::for_each(substituteSubGraphLayers.begin(), substituteSubGraphLayers.end(), [&](Layer* layer)
    {
        BOOST_ASSERT_MSG(std::find(m_Layers.begin(), m_Layers.end(), layer) != m_Layers.end(),
                         "Substitute layer is not a member of graph");
    });

    const SubGraph::InputSlots& subGraphInputSlots = subGraph.GetInputSlots();
    const SubGraph::OutputSlots& subGraphOutputSlots = subGraph.GetOutputSlots();

    unsigned int subGraphNumInputSlots = boost::numeric_cast<unsigned int>(subGraphInputSlots.size());
    unsigned int subGraphNumOutputSlots = boost::numeric_cast<unsigned int>(subGraphOutputSlots.size());

    const SubGraph::InputSlots& substituteSubGraphInputSlots = substituteSubGraph.GetInputSlots();
    const SubGraph::OutputSlots& substituteSubGraphOutputSlots = substituteSubGraph.GetOutputSlots();

    BOOST_ASSERT(subGraphNumInputSlots == substituteSubGraphInputSlots.size());
    BOOST_ASSERT(subGraphNumOutputSlots == substituteSubGraphOutputSlots.size());

    // Disconnect the sub-graph and replace it with the substitute sub-graph

    // Step 1: process input slots
    for (unsigned int inputSlotIdx = 0; inputSlotIdx < subGraphNumInputSlots; ++inputSlotIdx)
    {
        InputSlot* subGraphInputSlot = subGraphInputSlots.at(inputSlotIdx);
        BOOST_ASSERT(subGraphInputSlot);

        IOutputSlot* connectedOutputSlot = subGraphInputSlot->GetConnection();
        BOOST_ASSERT(connectedOutputSlot);
        connectedOutputSlot->Disconnect(*subGraphInputSlot);

        IInputSlot* substituteInputSlot = substituteSubGraphInputSlots.at(inputSlotIdx);
        BOOST_ASSERT(substituteInputSlot);
        connectedOutputSlot->Connect(*substituteInputSlot);
    }

    // Step 2: process output slots
    for(unsigned int outputSlotIdx = 0; outputSlotIdx < subGraphNumOutputSlots; ++outputSlotIdx)
    {
        OutputSlot* subGraphOutputSlot = subGraphOutputSlots.at(outputSlotIdx);
        BOOST_ASSERT(subGraphOutputSlot);

        OutputSlot* substituteOutputSlot = substituteSubGraphOutputSlots.at(outputSlotIdx);
        BOOST_ASSERT(substituteOutputSlot);
        subGraphOutputSlot->MoveAllConnections(*substituteOutputSlot);
    }
}

void Graph::EraseSubGraphLayers(const SubGraph &subGraph)
{
    for (auto layer : subGraph.GetLayers())
    {
        EraseLayer(layer);
    }
}

void Graph::InferTensorInfos()
{
    for (auto&& layer : TopologicalSort())
    {
        for (auto&& input : layer->GetInputSlots())
        {
            boost::ignore_unused(input);
            BOOST_ASSERT_MSG(input.GetConnectedOutputSlot()->IsTensorInfoSet(),
                             "All inputs must have the TensorInfo set at this point.");
        }
        layer->ValidateTensorShapesFromInputs();
    }
}

} // namespace armnn
