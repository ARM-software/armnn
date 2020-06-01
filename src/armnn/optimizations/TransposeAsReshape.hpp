//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"

namespace armnn
{
namespace optimizations
{

class TransposeAsReshapeImpl
{
public:
    /// Run for every TransposeLayer. Replaces it with a ReshapeLayer if they are equivalent.
    void Run(Graph& graph, TransposeLayer& transpose) const
    {
        if (IsReshape(transpose))
        {
            const TensorInfo& outInfo = transpose.GetOutputHandler().GetTensorInfo();

            const std::string name = std::string("as_reshape-") + transpose.GetName();
            const ReshapeDescriptor descriptor{outInfo.GetShape()};
            // Inserts NewLayer so layers don't need to be re-sorted.
            auto reshape = graph.InsertNewLayer<ReshapeLayer>(transpose.GetInputSlot(0), descriptor, name.c_str());

            // Bypass transpose. It will be deleted since it's left unconnected.
            transpose.GetOutputSlot().MoveAllConnections(reshape->GetOutputSlot());
        }
    }

protected:
    TransposeAsReshapeImpl() = default;
    ~TransposeAsReshapeImpl() = default;

private:
    static bool IsReshape(const TransposeLayer& layer)
    {
        const TensorShape& outShape = layer.GetOutputHandler().GetTensorInfo().GetShape();
        const PermutationVector& permutation = layer.GetPermutation();

        const unsigned int numDimensions = permutation.GetSize();
        std::map<unsigned int, unsigned int> permuteMappings;
        for (unsigned int i = 0; i < permutation.GetSize(); ++i)
        {
            permuteMappings[permutation[i]] = i;
        }

        std::vector<unsigned int> permuteVector;
        for (unsigned int i = 0; i < permutation.GetSize(); ++i)
        {
            permuteVector.push_back(permuteMappings.at(i));
        }

        unsigned int lastGtOne = 0;
        while ((lastGtOne < numDimensions) && (outShape[(permuteVector[lastGtOne])] == 1U))
        {
            ++lastGtOne;
        }

        bool isReshape = true;
        for (unsigned int i = lastGtOne + 1U; isReshape && (i < numDimensions); ++i)
        {
            if (outShape[permuteVector[i]] > 1U)
            {
                isReshape = permuteVector[lastGtOne] < permuteVector[i];
                lastGtOne = i;
            }
        }

        return isReshape;
    }
};

using TransposeAsReshape = OptimizeForType<TransposeLayer, TransposeAsReshapeImpl>;

} // namespace optimizations
} // namespace armnn
