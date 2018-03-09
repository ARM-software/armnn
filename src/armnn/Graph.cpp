//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Graph.hpp"
#include "Layers.hpp"

#include <armnn/Utils.hpp>
#include <armnn/TypesUtils.hpp>

#include <boost/polymorphic_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>

#include <unordered_map>

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

    // Copy slot connections
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
                                << ":" << GetComputeDeviceAsCString(it->GetComputeDevice());
    }
    BOOST_LOG_TRIVIAL(info) << "\n\n";

    return Status::Success;
}

Status Graph::AllocateDynamicBuffers()
{
    for (auto&& layer : m_Layers)
    {
        for (auto slot = layer->BeginOutputSlots(); slot != layer->EndOutputSlots(); ++slot)
        {
            slot->GetOutputHandler().AllocateTensors();
        }
    }
    return Status::Success;
}

const Graph& Graph::TopologicalSort() const
{
    if (!m_LayersInOrder)
    {
        //Reset layer order
        for (auto&& it : m_Layers)
        {
            it->ResetPriority();
        }

        auto compareLayerPriority = [](const LayersList::value_type& layerA, const LayersList::value_type& layerB)
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
            // All layers should have been associated with a valid compute device at this point
            BOOST_ASSERT(layer.GetComputeDevice() != Compute::Undefined);
            // Do not need another copy layer if copy layer is already present
            return layer.GetType() != LayerType::MemCopy;
        };

    for (auto&& srcLayer : m_Layers)
    {
        if (MayNeedCopyLayer(*srcLayer))
        {
            unsigned int srcOutputIndex = 0;
            for (auto&& srcOutput : srcLayer->GetOutputSlots())
            {
                for (auto&& dstInput : srcOutput.GetConnections())
                {
                    Layer& dstLayer = dstInput->GetOwningLayer();

                    if (MayNeedCopyLayer(dstLayer) && (dstLayer.GetComputeDevice() != srcLayer->GetComputeDevice()))
                    {
                        // A copy layer is needed in between the source and destination layers
                        // Record the operation rather than attempting to modify the graph as we go
                        // (invalidating iterators)
                        const std::string copyLayerName = boost::str(boost::format("[ %1% (%2%) -> %3% (%4%) ]")
                                                                     % srcLayer->GetName()
                                                                     % srcOutputIndex
                                                                     % dstLayer.GetName()
                                                                     % dstInput->GetSlotIndex());

                        MemCopyLayer* const copyLayer = InsertNewLayer<MemCopyLayer>(*dstInput, copyLayerName.c_str());
                        copyLayer->SetComputeDevice(dstLayer.GetComputeDevice());
                    }
                }
                ++srcOutputIndex;
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
            boost::ignore_unused(input);
            BOOST_ASSERT_MSG(input.GetConnectedOutputSlot()->IsTensorInfoSet(),
                             "All inputs must have the TensorInfo set at this point.");
        }
        layer->ValidateTensorShapesFromInputs();
    }
}

} // namespace armnn
