//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <vector>
#include <memory>
#include "optimizations/All.hpp"

namespace armnn
{

class Optimizer
{
public:
    using OptimizationPtr = std::unique_ptr<Optimization>;
    using Optimizations = std::vector<OptimizationPtr>;

    static void Pass(Graph& graph, const Optimizations& optimizations);

private:
    ~Optimizer() = default;

    Optimizer();
};


template<typename T>
void Append(Optimizer::Optimizations& optimizations, T&& optimization)
{
    optimizations.emplace_back(new T(optimization));
};

template<typename Front, typename... Others>
void Append(Optimizer::Optimizations& optimizations, Front&& front, Others&&... others)
{
    Append<Front>(optimizations, std::forward<Front>(front));
    Append<Others...>(optimizations, std::forward<Others>(others)...);
};

template<typename... Args>
Optimizer::Optimizations MakeOptimizations(Args&&... args)
{
    Optimizer::Optimizations optimizations;

    Append(optimizations, std::forward<Args>(args)...);

    return optimizations;
}

} // namespace armnn
