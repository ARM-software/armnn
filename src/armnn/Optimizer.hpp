//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <vector>

namespace armnn
{

class Graph;
class Optimization;

class Optimizer
{
public:

    static void Optimize(Graph& graph);

private:
    ~Optimizer() = default;

    Optimizer();

    std::vector<Optimization*> m_Optimizations;
};

} // namespace armnn
