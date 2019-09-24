//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"

namespace armnn
{
namespace optimizations
{

/// Replaces Permute leading into BatchToSpace with a DepthToSpace
/// in the case where the Permute swaps the batch and channels dimensions
/// such that the replacement is valid.
class PermuteAndBatchToSpaceAsDepthToSpaceImpl
{
public:
    void Run(Graph& graph, InputSlot& connection) const;
};

using PermuteAndBatchToSpaceAsDepthToSpace =
    OptimizeForConnection<PermuteLayer, BatchToSpaceNdLayer, PermuteAndBatchToSpaceAsDepthToSpaceImpl>;

}    // namespace optimizations
}    // namespace armnn
