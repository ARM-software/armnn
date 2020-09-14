//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubgraphView.hpp"
#include "Graph.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <utility>

namespace armnn
{

namespace
{

template <class C>
void AssertIfNullsOrDuplicates(const C& container, const std::string& errorMessage)
{
    using T = typename C::value_type;
    std::unordered_set<T> duplicateSet;
    std::for_each(container.begin(), container.end(), [&duplicateSet, &errorMessage](const T& i)
    {
        // Ignore unused for release builds
        IgnoreUnused(errorMessage);

        // Check if the item is valid
        ARMNN_ASSERT_MSG(i, errorMessage.c_str());

        // Check if a duplicate has been found
        ARMNN_ASSERT_MSG(duplicateSet.find(i) == duplicateSet.end(), errorMessage.c_str());

        duplicateSet.insert(i);
    });
}

} // anonymous namespace

SubgraphView::SubgraphView(Graph& graph)
    : m_InputSlots{}
    , m_OutputSlots{}
    , m_Layers(graph.begin(), graph.end())
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers)
    : m_InputSlots{inputs}
    , m_OutputSlots{outputs}
    , m_Layers{layers}
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(const SubgraphView& subgraph)
    : m_InputSlots(subgraph.m_InputSlots.begin(), subgraph.m_InputSlots.end())
    , m_OutputSlots(subgraph.m_OutputSlots.begin(), subgraph.m_OutputSlots.end())
    , m_Layers(subgraph.m_Layers.begin(), subgraph.m_Layers.end())
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(SubgraphView&& subgraph)
    : m_InputSlots(std::move(subgraph.m_InputSlots))
    , m_OutputSlots(std::move(subgraph.m_OutputSlots))
    , m_Layers(std::move(subgraph.m_Layers))
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(IConnectableLayer* layer)
    : m_InputSlots{}
    , m_OutputSlots{}
    , m_Layers{PolymorphicDowncast<Layer*>(layer)}
{
    unsigned int numInputSlots = layer->GetNumInputSlots();
    m_InputSlots.resize(numInputSlots);
    for (unsigned int i = 0; i < numInputSlots; i++)
    {
        m_InputSlots.at(i) = PolymorphicDowncast<InputSlot*>(&(layer->GetInputSlot(i)));
    }

    unsigned int numOutputSlots = layer->GetNumOutputSlots();
    m_OutputSlots.resize(numOutputSlots);
    for (unsigned int i = 0; i < numOutputSlots; i++)
    {
        m_OutputSlots.at(i) = PolymorphicDowncast<OutputSlot*>(&(layer->GetOutputSlot(i)));
    }

    CheckSubgraph();
}

SubgraphView& SubgraphView::operator=(SubgraphView&& other)
{
    m_InputSlots = std::move(other.m_InputSlots);
    m_OutputSlots = std::move(other.m_OutputSlots);
    m_Layers = std::move(other.m_Layers);

    CheckSubgraph();

    return *this;
}

void SubgraphView::CheckSubgraph()
{
    // Check for invalid or duplicate input slots
    AssertIfNullsOrDuplicates(m_InputSlots, "Sub-graphs cannot contain null or duplicate input slots");

    // Check for invalid or duplicate output slots
    AssertIfNullsOrDuplicates(m_OutputSlots, "Sub-graphs cannot contain null or duplicate output slots");

    // Check for invalid or duplicate layers
    AssertIfNullsOrDuplicates(m_Layers, "Sub-graphs cannot contain null or duplicate layers");
}

const SubgraphView::InputSlots& SubgraphView::GetInputSlots() const
{
    return m_InputSlots;
}

const SubgraphView::OutputSlots& SubgraphView::GetOutputSlots() const
{
    return m_OutputSlots;
}

const InputSlot* SubgraphView::GetInputSlot(unsigned int index) const
{
    return m_InputSlots.at(index);
}

InputSlot* SubgraphView::GetInputSlot(unsigned int index)
{
    return m_InputSlots.at(index);
}

const OutputSlot* SubgraphView::GetOutputSlot(unsigned int index) const
{
    return m_OutputSlots.at(index);
}

OutputSlot* SubgraphView::GetOutputSlot(unsigned int index)
{
    return m_OutputSlots.at(index);
}

unsigned int SubgraphView::GetNumInputSlots() const
{
    return armnn::numeric_cast<unsigned int>(m_InputSlots.size());
}

unsigned int SubgraphView::GetNumOutputSlots() const
{
    return armnn::numeric_cast<unsigned int>(m_OutputSlots.size());
}

const SubgraphView::Layers& SubgraphView::GetLayers() const
{
    return m_Layers;
}

SubgraphView::Iterator SubgraphView::begin()
{
    return m_Layers.begin();
}

SubgraphView::Iterator SubgraphView::end()
{
    return m_Layers.end();
}

SubgraphView::ConstIterator SubgraphView::begin() const
{
    return m_Layers.begin();
}

SubgraphView::ConstIterator SubgraphView::end() const
{
    return m_Layers.end();
}

SubgraphView::ConstIterator SubgraphView::cbegin() const
{
    return begin();
}

SubgraphView::ConstIterator SubgraphView::cend() const
{
    return end();
}

void SubgraphView::Clear()
{
    m_InputSlots.clear();
    m_OutputSlots.clear();
    m_Layers.clear();
}

} // namespace armnn
