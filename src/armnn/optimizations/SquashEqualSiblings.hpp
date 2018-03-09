//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "Optimization.hpp"

namespace armnn
{
namespace optimizations
{

template <typename Comparable>
class SquashEqualSiblingsImpl
{
public:
    /// Run for every connection between a base Layer (any) and a child ComparableLayer.
    /// For all siblings of the child layer that compare equal to it, bypasses and removes
    /// them. I.e., moves the connections in the outputs of the siblings to the outputs of
    /// the child layer, so the siblings are left unconnected (and later removed).
    void Run(Graph& graph, InputSlot& connection) const
    {
        auto& child = connection.GetOwningLayer();

        if (!child.IsOutputUnconnected())
        {
            OutputSlot& baseOutput = *connection.GetConnectedOutputSlot();
            auto& comparableChild = *boost::polymorphic_downcast<Comparable*>(&child);

            for (auto&& it : baseOutput.GetConnections())
            {
                Layer& sibling = it->GetOwningLayer();
                if ((&sibling != &child) && comparableChild.IsEqual(sibling))
                {
                    // Bypass sibling. It will be removed as it's left unconnected.
                    auto siblingOut = sibling.BeginOutputSlots();
                    for (auto childOut = child.BeginOutputSlots(); childOut != child.EndOutputSlots(); ++childOut)
                    {
                        siblingOut->MoveAllConnections(*childOut);
                        ++siblingOut;
                    }
                }
            }
        }
    }

protected:
    SquashEqualSiblingsImpl() = default;
    ~SquashEqualSiblingsImpl() = default;
};

using SquashEqualPermuteSiblings = OptimizeForConnection<Layer, PermuteLayer, SquashEqualSiblingsImpl<PermuteLayer>>;
using SquashEqualReshapeSiblings = OptimizeForConnection<Layer, ReshapeLayer, SquashEqualSiblingsImpl<ReshapeLayer>>;

} // namespace optimizations
} // namespace armnn
