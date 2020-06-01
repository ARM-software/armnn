//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnUtils/Transpose.hpp>

namespace armnn
{
namespace optimizations
{
class MoveTransposeUpImpl
{
public:
    /// Run for every connection between a base Layer (any) and a child TransposeLayer. If the type
    /// of the base layer allows it, it moves the permutation to the inputs of the base layer.
    /// I.e., adds equivalent permutations before the inputs of the base layer and moves the
    /// connections in the output of the child transpose layer to the output of the base layer.
    void Run(Graph& graph, InputSlot& connection) const
    {
        OutputSlot& baseOutput = *connection.GetConnectedOutputSlot();

        if (baseOutput.GetNumConnections() == 1U)
        {
            Layer& base = baseOutput.GetOwningLayer();

            if (CanMoveTransposeToInputs(base))
            {
                auto transpose = PolymorphicDowncast<TransposeLayer*>(&connection.GetOwningLayer());
                const PermutationVector& perm = transpose->GetPermutation();

                // Inserts an equivalent transpose before every input of the base layer.
                for (auto baseInput = base.BeginInputSlots(); baseInput != base.EndInputSlots(); ++baseInput)
                {
                    // Inserts a new transpose layer.
                    const std::string name = std::string("moved_up-") + transpose->GetName();
                    TransposeLayer& permLayer = *graph.InsertNewLayer<TransposeLayer>(*baseInput, perm, name.c_str());

                    // Sets output tensor info for the new layer.
                    OutputSlot& parentOutput = *permLayer.GetInputSlot(0).GetConnectedOutputSlot();
                    const TensorInfo permOutInfo = armnnUtils::TransposeTensorShape(parentOutput.GetTensorInfo(), perm);
                    permLayer.GetOutputHandler().SetTensorInfo(permOutInfo);
                }

                // Bypasses transpose. It will be removed as it's left unconnected.
                transpose->GetOutputSlot().MoveAllConnections(base.GetOutputSlot());
            }
        }
    }

protected:
    MoveTransposeUpImpl() = default;
    ~MoveTransposeUpImpl() = default;

private:
    static bool CanMoveTransposeToInputs(const Layer& base)
    {
        switch (base.GetType())
        {
            case LayerType::Activation:
            case LayerType::Addition:
            case LayerType::FakeQuantization:
            case LayerType::Floor:
            case LayerType::MemCopy:
            case LayerType::Multiplication:
                return true;
            default:
                return false;
        }
    }
};

using MoveTransposeUp = OptimizeForConnection<Layer, TransposeLayer, MoveTransposeUpImpl>;

} // namespace optimizations
} // namespace armnn
