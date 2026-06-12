//
// Copyright © 2026 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendOptions.hpp>

namespace armnn
{

class Graph;

void ApplySme2ShapePolicy(const Graph& graph, bool reduceFp32ToFp16, ModelOptions& modelOptions);

} // namespace armnn
