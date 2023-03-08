//
// Copyright © 2017-2018,2020,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnUtils/Permute.hpp>

namespace armnn
{
namespace optimizations
{
class MovePermuteUpImpl
{
public:
    /// Run for every connection between a base Layer (any) and a child PermuteLayer. If the type
    /// of the base layer allows it, it moves the permutation to the inputs of the base layer.
    /// I.e., adds equivalent permutations before the inputs of the base layer and moves the
    /// connections in the output of the child permute layer to the output of the base layer.
    void Run(Graph& graph, InputSlot& connection) const
    {
        OutputSlot& baseOutput = *connection.GetConnectedOutputSlot();

        if (baseOutput.GetNumConnections() == 1U)
        {
            Layer& base = baseOutput.GetOwningLayer();

            if (CanMovePermuteToInputs(base))
            {
                auto permute = PolymorphicDowncast<PermuteLayer*>(&connection.GetOwningLayer());
                const PermutationVector& perm = permute->GetPermutation();

                // Inserts an equivalent permute before every input of the base layer.
                for (auto baseInput = base.BeginInputSlots(); baseInput != base.EndInputSlots(); ++baseInput)
                {
                    // Inserts a new permute layer.
                    const std::string name = std::string("moved_up-") + permute->GetName();
                    PermuteLayer& permLayer = *graph.InsertNewLayer<PermuteLayer>(*baseInput, perm, name.c_str());

                    // Sets output tensor info for the new layer.
                    OutputSlot& parentOutput = *permLayer.GetInputSlot(0).GetConnectedOutputSlot();
                    const TensorInfo permOutInfo = armnnUtils::Permuted(parentOutput.GetTensorInfo(), perm);
                    permLayer.GetOutputHandler().SetTensorInfo(permOutInfo);
                }

                // Bypasses permute. It will be removed as it's left unconnected.
                permute->GetOutputSlot().MoveAllConnections(base.GetOutputSlot());
            }
        }
    }

protected:
    MovePermuteUpImpl() = default;
    ~MovePermuteUpImpl() = default;

private:
    static bool CanMovePermuteToInputs(const Layer& base)
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
            case LayerType::ElementwiseBinary:
            {
                auto descriptor = PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&base.GetParameters());
                return (descriptor->m_Operation == BinaryOperation::Add ||
                        descriptor->m_Operation == BinaryOperation::Mul);
            }
            default:
                return false;
        }
    }
};

using MovePermuteUp = OptimizeForConnection<Layer, PermuteLayer, MovePermuteUpImpl>;

} // namespace optimizations
} // namespace armnn
