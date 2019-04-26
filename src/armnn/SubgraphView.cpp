//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubgraphView.hpp"
#include "Graph.hpp"

#include <boost/numeric/conversion/cast.hpp>

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
        boost::ignore_unused(errorMessage);

        // Check if the item is valid
        BOOST_ASSERT_MSG(i, errorMessage.c_str());

        // Check if a duplicate has been found
        BOOST_ASSERT_MSG(duplicateSet.find(i) == duplicateSet.end(), errorMessage.c_str());

        duplicateSet.insert(i);
    });
}

} // anonymous namespace

SubgraphView::SubgraphView(Graph& graph)
    : m_InputSlots{}
    , m_OutputSlots{}
    , m_Layers(graph.begin(), graph.end())
    , m_ParentGraph(&graph)
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(Graph* parentGraph, InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers)
    : m_InputSlots{inputs}
    , m_OutputSlots{outputs}
    , m_Layers{layers}
    , m_ParentGraph(parentGraph)
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(const SubgraphView& referenceSubgraph,
                           InputSlots&& inputs,
                           OutputSlots&& outputs,
                           Layers&& layers)
    : m_InputSlots{inputs}
    , m_OutputSlots{outputs}
    , m_Layers{layers}
    , m_ParentGraph(referenceSubgraph.m_ParentGraph)
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(const SubgraphView& subgraph)
    : m_InputSlots(subgraph.m_InputSlots.begin(), subgraph.m_InputSlots.end())
    , m_OutputSlots(subgraph.m_OutputSlots.begin(), subgraph.m_OutputSlots.end())
    , m_Layers(subgraph.m_Layers.begin(), subgraph.m_Layers.end())
    , m_ParentGraph(subgraph.m_ParentGraph)
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(SubgraphView&& subgraph)
    : m_InputSlots(std::move(subgraph.m_InputSlots))
    , m_OutputSlots(std::move(subgraph.m_OutputSlots))
    , m_Layers(std::move(subgraph.m_Layers))
    , m_ParentGraph(std::exchange(subgraph.m_ParentGraph, nullptr))
{
    CheckSubgraph();
}

SubgraphView::SubgraphView(const SubgraphView& referenceSubgraph, IConnectableLayer* layer)
    : m_InputSlots{}
    , m_OutputSlots{}
    , m_Layers{boost::polymorphic_downcast<Layer*>(layer)}
    , m_ParentGraph(referenceSubgraph.m_ParentGraph)
{
    unsigned int numInputSlots = layer->GetNumInputSlots();
    m_InputSlots.resize(numInputSlots);
    for (unsigned int i = 0; i < numInputSlots; i++)
    {
        m_InputSlots.at(i) = boost::polymorphic_downcast<InputSlot*>(&(layer->GetInputSlot(i)));
    }

    unsigned int numOutputSlots = layer->GetNumOutputSlots();
    m_OutputSlots.resize(numOutputSlots);
    for (unsigned int i = 0; i < numOutputSlots; i++)
    {
        m_OutputSlots.at(i) = boost::polymorphic_downcast<OutputSlot*>(&(layer->GetOutputSlot(i)));
    }

    CheckSubgraph();
}

void SubgraphView::CheckSubgraph()
{
    // Check that the sub-graph has a valid parent graph
    BOOST_ASSERT_MSG(m_ParentGraph, "Sub-graphs must have a parent graph");

    // Check for invalid or duplicate input slots
    AssertIfNullsOrDuplicates(m_InputSlots, "Sub-graphs cannot contain null or duplicate input slots");

    // Check for invalid or duplicate output slots
    AssertIfNullsOrDuplicates(m_OutputSlots, "Sub-graphs cannot contain null or duplicate output slots");

    // Check for invalid or duplicate layers
    AssertIfNullsOrDuplicates(m_Layers, "Sub-graphs cannot contain null or duplicate layers");

    // Check that all the layers of the sub-graph belong to the parent graph
    std::for_each(m_Layers.begin(), m_Layers.end(), [&](const Layer* l)
    {
        BOOST_ASSERT_MSG(std::find(m_ParentGraph->begin(), m_ParentGraph->end(), l) != m_ParentGraph->end(),
                         "Sub-graph layer is not a member of the parent graph");
    });
}

void SubgraphView::Update(Graph &graph)
{
    m_InputSlots.clear();
    m_OutputSlots.clear();
    m_Layers.assign(graph.begin(), graph.end());
    m_ParentGraph = &graph;

    CheckSubgraph();
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
    return  m_InputSlots.at(index);
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
    return boost::numeric_cast<unsigned int>(m_InputSlots.size());
}

unsigned int SubgraphView::GetNumOutputSlots() const
{
    return boost::numeric_cast<unsigned int>(m_OutputSlots.size());
}

const SubgraphView::Layers & SubgraphView::GetLayers() const
{
    return m_Layers;
}

SubgraphView::Layers::iterator SubgraphView::begin()
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

} // namespace armnn
