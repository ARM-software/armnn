//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Layer.hpp"
#include "SubGraph.hpp"

#include <boost/numeric/conversion/cast.hpp>

namespace armnn
{

SubGraph::SubGraph()
{
}

SubGraph::SubGraph(InputSlots && inputs,
                   OutputSlots && outputs,
                   Layers && layers)
: m_InputSlots{inputs}
, m_OutputSlots{outputs}
, m_Layers{layers}
{
}

const SubGraph::InputSlots & SubGraph::GetInputSlots() const
{
    return m_InputSlots;
}

const SubGraph::OutputSlots & SubGraph::GetOutputSlots() const
{
    return m_OutputSlots;
}

const InputSlot* SubGraph::GetInputSlot(unsigned int index) const
{
    return m_InputSlots.at(index);
}

InputSlot* SubGraph::GetInputSlot(unsigned int index)
{
    return  m_InputSlots.at(index);
}

const OutputSlot* SubGraph::GetOutputSlot(unsigned int index) const
{
    return m_OutputSlots.at(index);
}

OutputSlot* SubGraph::GetOutputSlot(unsigned int index)
{
    return m_OutputSlots.at(index);
}

unsigned int SubGraph::GetNumInputSlots() const
{
    return boost::numeric_cast<unsigned int>(m_InputSlots.size());
}

unsigned int SubGraph::GetNumOutputSlots() const
{
    return boost::numeric_cast<unsigned int>(m_OutputSlots.size());
}

const SubGraph::Layers & SubGraph::GetLayers() const
{
    return m_Layers;
}

SubGraph::Layers::iterator SubGraph::begin()
{
    return m_Layers.begin();
}

SubGraph::Layers::iterator SubGraph::end()
{
    return m_Layers.end();
}

SubGraph::Layers::const_iterator SubGraph::begin() const
{
    return m_Layers.begin();
}

SubGraph::Layers::const_iterator SubGraph::end() const
{
    return m_Layers.end();
}

SubGraph::Layers::const_iterator SubGraph::cbegin() const
{
    return begin();
}

SubGraph::Layers::const_iterator SubGraph::cend() const
{
    return end();
}

} // namespace armnn
