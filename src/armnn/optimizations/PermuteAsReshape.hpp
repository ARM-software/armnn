//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"

namespace armnn
{
namespace optimizations
{

class PermuteAsReshapeImpl
{
public:
    /// Run for every PermuteLayer. Replaces it with a ReshapeLayer if they are equivalent.
    void Run(Graph& graph, PermuteLayer& permute) const
    {
        if (IsReshape(permute))
        {
            const TensorInfo& outInfo = permute.GetOutputHandler().GetTensorInfo();

            const std::string name = std::string("as_reshape-") + permute.GetName();
            const ReshapeDescriptor descriptor{outInfo.GetShape()};
            // Inserts NewLayer so layers don't need to be re-sorted.
            auto reshape = graph.InsertNewLayer<ReshapeLayer>(permute.GetInputSlot(0), descriptor, name.c_str());

            // Bypass permute. It will be deleted since it's left unconnected.
            permute.GetOutputSlot().MoveAllConnections(reshape->GetOutputSlot());
        }
    }

protected:
    PermuteAsReshapeImpl() = default;
    ~PermuteAsReshapeImpl() = default;

private:
    static bool IsReshape(const PermuteLayer& layer)
    {
        const TensorShape& outShape = layer.GetOutputHandler().GetTensorInfo().GetShape();
        const PermutationVector& permutation = layer.GetPermutation();

        const unsigned int numDimensions = permutation.GetSize();

        unsigned int lastGtOne = 0;
        while ((lastGtOne < numDimensions) && (outShape[(permutation[lastGtOne])] == 1U))
        {
            ++lastGtOne;
        }

        bool isReshape = true;
        for (unsigned int i = lastGtOne + 1U; isReshape && (i < numDimensions); ++i)
        {
            if (outShape[permutation[i]] > 1U)
            {
                isReshape = permutation[lastGtOne] < permutation[i];
                lastGtOne = i;
            }
        }

        return isReshape;
    }
};

using PermuteAsReshape = OptimizeForType<PermuteLayer, PermuteAsReshapeImpl>;

} // namespace optimizations
} // namespace armnn
