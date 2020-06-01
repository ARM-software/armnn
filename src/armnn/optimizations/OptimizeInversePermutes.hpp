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

template <typename PermuteType>
class OptimizeInversePermutesImpl
{
public:
    /// Run for every connection between a base PermuteLayer and a child PermuteLayer.
    /// Bypasses both layers for that connection if one is the inverse of the other.
    void Run(Graph& graph, InputSlot& connection) const
    {
        IgnoreUnused(graph);
        Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
        auto child = PolymorphicDowncast<PermuteType*>(&connection.GetOwningLayer());

        if (child->IsInverse(*PolymorphicDowncast<PermuteType*>(&base)))
        {
            // Bypass both layers. Child will be removed as it's left unconnected.
            // Base layer will be removed if left unconnected.
            child->GetOutputSlot().MoveAllConnections(*base.GetInputSlot(0).GetConnectedOutputSlot());
        }
    }

protected:
    OptimizeInversePermutesImpl() = default;
    ~OptimizeInversePermutesImpl() = default;
};

using OptimizeInversePermutes = OptimizeForConnection<PermuteLayer, PermuteLayer,
    OptimizeInversePermutesImpl<PermuteLayer>>;
using OptimizeInverseTransposes = OptimizeForConnection<TransposeLayer, TransposeLayer,
    OptimizeInversePermutesImpl<TransposeLayer>>;

} // namespace optimizations
} // namespace armnn
