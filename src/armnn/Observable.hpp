//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "IGraphObservable.hpp"
#include "Graph.hpp"

namespace armnn
{

template <typename ObservedType>
class GraphObservable : public IGraphObservable
{
public:
    using Iterator = typename std::list<ObservedType>::const_iterator;

    GraphObservable(Graph& subject, GraphEvent notifyOnEvent)
    : m_Subject(&subject)
    {
        m_NotifyOnEvent = notifyOnEvent;
        m_Subject->AttachObservable(this, m_NotifyOnEvent);
    };

    void Clear() { m_ObservedObjects.clear(); };

    Iterator begin() { return m_ObservedObjects.begin(); }

    Iterator end() { return m_ObservedObjects.end(); }

protected:
    ~GraphObservable()
    {
        if (m_Subject)
        {
            m_Subject->DetachObservable(this, m_NotifyOnEvent);
        }
    }

    GraphEvent m_NotifyOnEvent;
    Graph* m_Subject;
    std::list<ObservedType> m_ObservedObjects;
};

class AddedLayerObservable : public GraphObservable<Layer*>
{
public:
    explicit AddedLayerObservable(Graph& subject)
    : GraphObservable<Layer*>(subject, GraphEvent::LayerAdded)
    {};

    void Update(Layer* graphLayer) override;
};

class ErasedLayerNamesObservable : public GraphObservable<std::string>
{
public:
    explicit ErasedLayerNamesObservable(Graph& subject)
    : GraphObservable<std::string>(subject, GraphEvent::LayerErased)
    {};

    void Update(Layer* graphLayer) override;
};

} //namespace armnn

