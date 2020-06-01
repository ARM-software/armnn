//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

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
        IgnoreUnused(graph);
        auto& child = connection.GetOwningLayer();

        if (!child.IsOutputUnconnected())
        {
            OutputSlot& baseOutput = *connection.GetConnectedOutputSlot();

            if (baseOutput.GetNumConnections() > 1)
            {
                auto& comparableChild = *PolymorphicDowncast<Comparable*>(&child);

                Layer* lowestPriorityChild = &child;
                for (auto&& it : baseOutput.GetConnections())
                {
                    Layer* sibling = &it->GetOwningLayer();
                    if ((sibling != lowestPriorityChild) && comparableChild.IsEqual(*sibling))
                    {
                        if (sibling->GetPriority() < lowestPriorityChild->GetPriority())
                        {
                            std::swap(sibling, lowestPriorityChild);
                        }
                        // Bypasses sibling. It will be removed as it's left unconnected.
                        auto siblingOut = sibling->BeginOutputSlots();
                        for (auto lowestPriorityChildOut = lowestPriorityChild->BeginOutputSlots();
                             lowestPriorityChildOut != lowestPriorityChild->EndOutputSlots(); ++lowestPriorityChildOut)
                        {
                            siblingOut->MoveAllConnections(*lowestPriorityChildOut);
                            ++siblingOut;
                        }
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
using SquashEqualTransposeSiblings = OptimizeForConnection<Layer, TransposeLayer,
    SquashEqualSiblingsImpl<TransposeLayer>>;
using SquashEqualReshapeSiblings = OptimizeForConnection<Layer, ReshapeLayer, SquashEqualSiblingsImpl<ReshapeLayer>>;

} // namespace optimizations
} // namespace armnn
