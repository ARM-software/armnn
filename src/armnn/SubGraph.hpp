//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Layer.hpp"

#include <vector>
#include <list>

namespace armnn
{

///
/// The SubGraph class represents a subgraph of a Graph.
/// The data it holds, points to data held by layers of the Graph, so the
/// the contents of the SubGraph becomes invalid when the Layers are destroyed
/// or changed.
///
class SubGraph final
{
public:
    using InputSlots = std::vector<InputSlot *>;
    using OutputSlots = std::vector<OutputSlot *>;
    using Layers = std::list<Layer*>;

    SubGraph();
    SubGraph(InputSlots && inputs,
             OutputSlots && outputs,
             Layers && layers);

    const InputSlots & GetInputSlots() const;
    const OutputSlots & GetOutputSlots() const;
    const Layers & GetLayers() const;

    const InputSlot* GetInputSlot(unsigned int index) const;
    InputSlot* GetInputSlot(unsigned int index);

    const OutputSlot* GetOutputSlot(unsigned int index) const;
    OutputSlot* GetOutputSlot(unsigned int index);

    unsigned int GetNumInputSlots() const;
    unsigned int GetNumOutputSlots() const;

    Layers::iterator begin();
    Layers::iterator end();

    Layers::const_iterator begin() const;
    Layers::const_iterator end() const;

    Layers::const_iterator cbegin() const;
    Layers::const_iterator cend() const;

private:
    InputSlots m_InputSlots;
    OutputSlots m_OutputSlots;
    Layers m_Layers;
};

} // namespace armnn
