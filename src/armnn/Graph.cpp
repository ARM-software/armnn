//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Graph.hpp>
#include <LayersFwd.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/SubgraphView.hpp>

#include <armnn/BackendId.hpp>
#include <armnn/Logging.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <fmt/format.h>

#include <unordered_map>
#include <DotSerializer.hpp>
#include <sstream>

namespace armnn
{

Graph::Graph(const Graph& other)
:   m_LayersInOrder(other.m_LayersInOrder)
,   m_Profiler(other.m_Profiler)
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
        ARMNN_LOG(info) << "\n Graph is empty.\n";
        return Status::Success;
    }
    ARMNN_LOG(info) << "\n";
    ARMNN_LOG(info) << "Walking Pattern: \n";

    for (auto&& it : TopologicalSort())
    {
        auto numInputSlots = it->GetNumInputSlots();
        auto numOutputSlots = it->GetNumOutputSlots();

        ARMNN_LOG(info) << it->GetName() << ":" << GetLayerTypeAsCString(it->GetType())
                                << ":" << it->GetBackendId().Get()
                                << " has " << numInputSlots << " input slots"
                                << " and " << numOutputSlots << " output slots.";

        for (auto i : it->GetInputSlots())
        {
            std::ostringstream message;
            auto inputTensorShape = i.GetConnectedOutputSlot()->GetTensorInfo().GetShape();
            unsigned int numDims = inputTensorShape.GetNumDimensions();

            message << "The input slot has shape [ ";
            for (unsigned int dim=0; dim < numDims; dim++)
            {
                message << inputTensorShape[dim] << ",";
            }
            message << " ]";
            ARMNN_LOG(info) << message.str();
        }

        for (unsigned int i = 0; i < it->GetNumOutputSlots(); i++)
        {
            const armnn::Layer *layer = it;
            std::ostringstream message;
            auto outputTensorShape = layer->GetOutputSlots()[i].GetTensorInfo().GetShape();
            unsigned int numDims = outputTensorShape.GetNumDimensions();

            message << "The output slot has shape [ ";
            for (unsigned int dim=0; dim < numDims; dim++)
            {
                message << outputTensorShape[dim] << ",";
            }
            message << " ]";
            ARMNN_LOG(info) << message.str();
        }
        ARMNN_LOG(info) << "\n";
    }
    ARMNN_LOG(info) << "\n\n";

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
    ARMNN_ASSERT(m_LayersInOrder);
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "LoadNetwork_AllocateDynamicBuffers");

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
                    if (handleReferenceCounts[tensorHandle] == 0u)
                    {
                          // if nobody consumes this tensor we call Allocate()
                          tensorHandle->Allocate();
                    }
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

void Graph::AddCompatibilityLayers(std::map<BackendId, std::unique_ptr<IBackendInternal>>& backends,
                                   TensorHandleFactoryRegistry& registry)
{
    // Returns true if the given layer could potentially need an intermediate copy/import layer (depending on its
    // connections to other layers).
    auto MayNeedCompatibilityLayer = [](const Layer& layer)
    {
        // All layers should have been associated with a valid compute device at this point.
        ARMNN_ASSERT(layer.GetBackendId() != Compute::Undefined);
        // Does not need another compatibility layer if a copy or import layer is already present.
        return layer.GetType() != LayerType::MemCopy &&
               layer.GetType() != LayerType::MemImport;
    };

    auto IsCompatibilityStrategy = [](EdgeStrategy strategy)
    {
        return strategy == EdgeStrategy::CopyToTarget ||
               strategy == EdgeStrategy::ExportToTarget;
    };

    ForEachLayer([this, &backends, &registry, MayNeedCompatibilityLayer, IsCompatibilityStrategy](Layer* srcLayer)
    {
        ARMNN_ASSERT(srcLayer);

        if (!MayNeedCompatibilityLayer(*srcLayer))
        {
            // The current layer does not need copy layers, move to the next one
            return;
        }

        const std::vector<OutputSlot>& srcOutputSlots = srcLayer->GetOutputSlots();
        for (unsigned int srcOutputIndex = 0; srcOutputIndex < srcOutputSlots.size(); srcOutputIndex++)
        {
            OutputSlot& srcOutputSlot = srcLayer->GetOutputSlot(srcOutputIndex);
            const std::vector<InputSlot*> srcConnections = srcOutputSlot.GetConnections();
            const std::vector<EdgeStrategy> srcEdgeStrategies = srcOutputSlot.GetEdgeStrategies();
            for (unsigned int srcConnectionIndex = 0; srcConnectionIndex < srcConnections.size(); srcConnectionIndex++)
            {
                InputSlot* dstInputSlot = srcConnections[srcConnectionIndex];
                ARMNN_ASSERT(dstInputSlot);

                EdgeStrategy strategy = srcEdgeStrategies[srcConnectionIndex];
                ARMNN_ASSERT_MSG(strategy != EdgeStrategy::Undefined,
                                 "Undefined memory strategy found while adding copy layers for compatibility");

                const Layer& dstLayer = dstInputSlot->GetOwningLayer();
                if (MayNeedCompatibilityLayer(dstLayer) &&
                    IsCompatibilityStrategy(strategy))
                {
                    // A copy layer is needed in between the source and destination layers.
                    // Record the operation rather than attempting to modify the graph as we go.
                    // (invalidating iterators)
                    const std::string compLayerName = fmt::format("[ {} ({}) -> {} ({}) ]",
                                                                  srcLayer->GetName(),
                                                                  srcOutputIndex,
                                                                  dstLayer.GetName(),
                                                                  dstInputSlot->GetSlotIndex());
                    Layer* compLayer = nullptr;
                    if (strategy == EdgeStrategy::CopyToTarget)
                    {
                        compLayer = InsertNewLayer<MemCopyLayer>(*dstInputSlot, compLayerName.c_str());
                    }
                    else
                    {
                        ARMNN_ASSERT_MSG(strategy == EdgeStrategy::ExportToTarget, "Invalid edge strategy found.");
                        compLayer = InsertNewLayer<MemImportLayer>(*dstInputSlot, compLayerName.c_str());
                    }

                    compLayer->SetBackendId(dstLayer.GetBackendId());

                    OutputSlot& compOutputSlot = compLayer->GetOutputSlot(0);
                    auto backendIt = backends.find(dstLayer.GetBackendId());
                    if (backendIt != backends.end() &&
                        backendIt->second &&
                        backendIt->second->SupportsTensorAllocatorAPI())
                    {
                        auto backend = backendIt->second.get();
                        auto tensorHandleFactoryIds = backend->GetHandleFactoryPreferences();
                        bool found = false;

                        for (auto preference : tensorHandleFactoryIds)
                        {
                            auto factory = registry.GetFactory(preference);
                            if (factory)
                            {
                                auto srcPref = srcOutputSlot.GetTensorHandleFactoryId();
                                auto srcFactory = registry.GetFactory(srcPref);

                                if (srcFactory)
                                {
                                    bool canExportImport =
                                        (factory->GetImportFlags() & srcFactory->GetExportFlags()) != 0;

                                    if (factory->SupportsMapUnmap() || canExportImport)
                                    {
                                        compOutputSlot.SetTensorHandleFactory(preference);
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }

                        if (!found)
                        {
                            compOutputSlot.SetTensorHandleFactory(ITensorHandleFactory::LegacyFactoryId);
                        }
                    }
                    else
                    {
                        compOutputSlot.SetTensorHandleFactory(ITensorHandleFactory::LegacyFactoryId);
                    }

                    // The output strategy of a compatibility layer is always DirectCompatibility.
                    compOutputSlot.SetEdgeStrategy(0, EdgeStrategy::DirectCompatibility);

                    // Recalculate the connection index on the previous layer as we have just inserted into it.
                    const std::vector<InputSlot*>& newSourceConnections = srcOutputSlot.GetConnections();
                    auto newSrcConnectionIndex = std::distance(newSourceConnections.begin(),
                                                               std::find(newSourceConnections.begin(),
                                                                         newSourceConnections.end(),
                                                                         &compLayer->GetInputSlot(0)));

                    // The input strategy of a compatibility layer is always DirectCompatibilty.
                    srcOutputSlot.SetEdgeStrategy(armnn::numeric_cast<unsigned int>(newSrcConnectionIndex),
                                                    EdgeStrategy::DirectCompatibility);
                }
            }
        }
    });
}

void Graph::SubstituteSubgraph(SubgraphView& subgraph, IConnectableLayer* substituteLayer)
{
    ARMNN_ASSERT(substituteLayer != nullptr);

    // Create a new sub-graph with only the given layer, using
    // the given sub-graph as a reference of which parent graph to use
    SubgraphView substituteSubgraph(substituteLayer);

    SubstituteSubgraph(subgraph, substituteSubgraph);
}

void Graph::SubstituteSubgraph(SubgraphView& subgraph, const SubgraphView& substituteSubgraph)
{
    // Look through each layer in the new subgraph and add any that are not already a member of this graph
    substituteSubgraph.ForEachIConnectableLayer([this](IConnectableLayer* iConnectableLayer)
    {
        if (std::find(std::begin(m_Layers),
                      std::end(m_Layers),
                      iConnectableLayer) == std::end(m_Layers))
        {
            auto layer = PolymorphicDowncast<Layer*>(iConnectableLayer);
            layer->Reparent(*this, m_Layers.end());
            m_LayersInOrder = false;
        }
    });

    ReplaceSubgraphConnections(subgraph, substituteSubgraph);
    EraseSubgraphLayers(subgraph);
    TopologicalSort();
}

void Graph::ReplaceSubgraphConnections(const SubgraphView& subgraph, const SubgraphView& substituteSubgraph)
{
    ARMNN_ASSERT_MSG(!substituteSubgraph.GetIConnectableLayers().empty(),
                     "New sub-graph used for substitution must not be empty");

    const SubgraphView::IConnectableLayers& substituteSubgraphLayers = substituteSubgraph.GetIConnectableLayers();
    std::for_each(substituteSubgraphLayers.begin(), substituteSubgraphLayers.end(), [&](IConnectableLayer* layer)
    {
        IgnoreUnused(layer);
        layer = PolymorphicDowncast<Layer*>(layer);
        ARMNN_ASSERT_MSG(std::find(m_Layers.begin(), m_Layers.end(), layer) != m_Layers.end(),
                         "Substitute layer is not a member of graph");
    });

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraph.GetIOutputSlots();

    unsigned int subgraphNumInputSlots = armnn::numeric_cast<unsigned int>(subgraphInputSlots.size());
    unsigned int subgraphNumOutputSlots = armnn::numeric_cast<unsigned int>(subgraphOutputSlots.size());

    const SubgraphView::IInputSlots& substituteSubgraphInputSlots = substituteSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& substituteSubgraphOutputSlots = substituteSubgraph.GetIOutputSlots();

    ARMNN_ASSERT(subgraphNumInputSlots == substituteSubgraphInputSlots.size());
    ARMNN_ASSERT(subgraphNumOutputSlots == substituteSubgraphOutputSlots.size());

    // Disconnect the sub-graph and replace it with the substitute sub-graph

    // Step 1: process input slots
    for (unsigned int inputSlotIdx = 0; inputSlotIdx < subgraphNumInputSlots; ++inputSlotIdx)
    {
        IInputSlot* subgraphInputSlot = subgraphInputSlots.at(inputSlotIdx);
        ARMNN_ASSERT(subgraphInputSlot);

        IOutputSlot* connectedOutputSlot = subgraphInputSlot->GetConnection();
        ARMNN_ASSERT(connectedOutputSlot);
        connectedOutputSlot->Disconnect(*subgraphInputSlot);

        IInputSlot* substituteInputSlot = substituteSubgraphInputSlots.at(inputSlotIdx);
        ARMNN_ASSERT(substituteInputSlot);
        connectedOutputSlot->Connect(*substituteInputSlot);
    }

    // Step 2: process output slots
    for(unsigned int outputSlotIdx = 0; outputSlotIdx < subgraphNumOutputSlots; ++outputSlotIdx)
    {
        auto subgraphOutputSlot =
                PolymorphicDowncast<OutputSlot*>(subgraphOutputSlots.at(outputSlotIdx));
        ARMNN_ASSERT(subgraphOutputSlot);

        auto substituteOutputSlot =
                PolymorphicDowncast<OutputSlot*>(substituteSubgraphOutputSlots.at(outputSlotIdx));
        ARMNN_ASSERT(substituteOutputSlot);

        subgraphOutputSlot->MoveAllConnections(*substituteOutputSlot);
    }
}

void Graph::EraseSubgraphLayers(SubgraphView &subgraph)
{

    for (auto iConnectableLayer : subgraph.GetIConnectableLayers())
    {
        auto layer = PolymorphicDowncast<Layer*>(iConnectableLayer);
        EraseLayer(layer);
    }
    subgraph.Clear();
}

/// For each ConstantLayer in Graph, ensures TensorInfo is set on all output slots.
/// LayerValidationException thrown if no TensorInfo is set.
///
/// @throws LayerValidationException
void Graph::VerifyConstantLayerSetTensorInfo() const
{
    for (auto&& layer : TopologicalSort())
    {
        if (layer->GetType() == armnn::LayerType::Constant)
        {
            for (auto&& output: layer->GetOutputSlots())
            {
                if (!output.IsTensorInfoSet())
                {
                    std::ostringstream message;
                    message << "Output slot TensorInfo not set on "
                            << GetLayerTypeAsCString(layer->GetType())
                            << " layer \""
                            << layer->GetName()
                            << "\"";
                    throw LayerValidationException(message.str());
                }
            }
        }
    }
}

void Graph::InferTensorInfos()
{
    for (auto&& layer : TopologicalSort())
    {
        for (auto&& input : layer->GetInputSlots())
        {
            const IOutputSlot* source = input.GetConnectedOutputSlot();
            if (source == NULL)
            {
                // Throws exception due to a layer input not being connected to an output slot.
                // Verifies input slot weights and bias are set for FullyConnected layers.
                ConstructErrorMessageForUnconnectedInputs(layer, input.GetSlotIndex());
            }

            if (!source->IsTensorInfoSet())
            {
                std::ostringstream message;
                message << "Output slot TensorInfo not set on "
                        << GetLayerTypeAsCString(layer->GetType())
                        << " layer "
                        << std::quoted(layer->GetName());
                throw LayerValidationException(message.str());
            }
        }

        if (layer->m_ShapeInferenceMethod == ShapeInferenceMethod::ValidateOnly)
        {
            layer->ValidateTensorShapesFromInputs();
        }
    }
}

/// Throws exception due to a layer input not being connected to an output slot.
/// Verifies weights and bias are set for layers on input slots 1
/// and 2 respectively. Method checks if bias is enabled before ensuring it is set.
///
/// @param layer constant pointer to a Layer object
/// @param slotIndex input slot index of layer
/// @throws LayerValidationException
void Graph::ConstructErrorMessageForUnconnectedInputs(Layer* const layer,
                                                      unsigned int slotIndex)
{
    std::ostringstream message;
    bool noWeightsAndBias = false;

    if ((layer->GetType() == armnn::LayerType::FullyConnected ||
         layer->GetType() == armnn::LayerType::Convolution3d) && slotIndex > 0)
    {
        // If weights are not set and is bias enabled, also check if bias is set
        if (slotIndex == 1 && layer->GetNumInputSlots() == 3)
        {
            const IOutputSlot* biasSource = layer->GetInputSlot(2).GetConnectedOutputSlot();
            if (biasSource == NULL)
            {
                message << layer->GetName() << " layer weights and bias not set: ";
                noWeightsAndBias = true;
            }
        }

        // Only weights or bias are not set
        if (!noWeightsAndBias)
        {
            if (slotIndex == 1)
            {
                message << layer->GetName() << " layer weights not set: ";
            }
            else
            {
                message << layer->GetName() << " layer bias not set: ";
            }
        }
    }

    std::string slotString = noWeightsAndBias ? "1 & 2" : std::to_string(slotIndex);
    message << "Input slot(s) "
            << slotString
            << " not connected to an output slot on "
            << GetLayerTypeAsCString(layer->GetType())
            << " layer "
            << std::quoted(layer->GetName());
    throw LayerValidationException(message.str());
}

const std::shared_ptr<IProfiler>& Graph::GetProfiler() const
{
    return m_Profiler;
}

} // namespace armnn
