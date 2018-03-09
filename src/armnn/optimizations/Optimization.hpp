//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "Graph.hpp"
#include "LayersFwd.hpp"

namespace armnn
{

class Optimization
{
public:
    virtual void Run(Graph& graph, Graph::Iterator& pos) const = 0;
protected:
    ~Optimization() = default;
};

// Wrappers
// The implementation of the following wrappers make use of the CRTP C++ idiom
// (curiously recurring template pattern).
// For details, see https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

/// Wrapper Optimization base class that calls Wrapped::Run for every layer of type BaseType.
/// - Wrapped class mustn't remove the base layer.
/// - Base layer is removed if left unconnected after applying the wrapped optimization.
template <typename BaseType, typename Wrapped>
class OptimizeForTypeImpl : public armnn::Optimization, public Wrapped
{
public:
    using Wrapped::Wrapped;

    void Run(Graph& graph, Graph::Iterator& pos) const override
    {
        Layer* const base = *pos;

        if (base->GetType() == LayerEnumOf<BaseType>())
        {
            Wrapped::Run(graph, *boost::polymorphic_downcast<BaseType*>(base));
        }
    }

protected:
    ~OptimizeForTypeImpl() = default;
};

/// Specialization that calls Wrapped::Run for any layer type
template <typename Wrapped>
class OptimizeForTypeImpl<Layer, Wrapped> : public armnn::Optimization, public Wrapped
{
public:
    using Wrapped::Wrapped;

    void Run(Graph& graph, Graph::Iterator& pos) const override
    {
        Wrapped::Run(graph, **pos);
    }

protected:
    ~OptimizeForTypeImpl() = default;
};

template <typename BaseType, typename Wrapped>
class OptimizeForType final : public OptimizeForTypeImpl<BaseType, Wrapped>
{
public:
    using OptimizeForTypeImpl<BaseType, Wrapped>::OptimizeForTypeImpl;
};

/// Wrapper Optimization class that calls Wrapped::Run for every connection BaseType -> ChildType.
/// - Wrapped class mustn't remove the base layer.
/// - Wrapped class mustn't affect existing connections in the same output. It might add new ones.
/// - Base and children layers are removed if left unconnected after applying the wrapped optimization.
template <typename BaseType, typename ChildType, typename Wrapped>
class OptimizeForConnectionImpl : public Wrapped
{
public:
    using Wrapped::Wrapped;

    void Run(Graph& graph, BaseType& base) const
    {
        for (auto output = base.BeginOutputSlots(); output != base.EndOutputSlots(); ++output)
        {
            for (auto&& childInput : output->GetConnections())
            {
                if (childInput->GetOwningLayer().GetType() == LayerEnumOf<ChildType>())
                {
                    Wrapped::Run(graph, *childInput);
                }
            }

            // Remove unconnected children
            for (unsigned int i = 0; i < output->GetNumConnections();)
            {
                Layer* child = &output->GetConnection(i)->GetOwningLayer();

                if (child->IsOutputUnconnected())
                {
                    graph.EraseLayer(child);
                }
                else
                {
                    ++i;
                }
            }
        }
    }

protected:
    ~OptimizeForConnectionImpl() = default;
};

template <typename BaseType, typename ChildType, typename Wrapped>
class OptimizeForConnection final
    : public OptimizeForTypeImpl<BaseType, OptimizeForConnectionImpl<BaseType, ChildType, Wrapped>>
{
public:
    using OptimizeForTypeImpl<BaseType, OptimizeForConnectionImpl<BaseType, ChildType, Wrapped>>::OptimizeForTypeImpl;
};

} // namespace armnn
