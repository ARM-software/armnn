//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TestLayerVisitor.hpp"
#include <doctest/doctest.h>

namespace armnn
{

void CheckLayerBindingId(LayerBindingId visitorId, LayerBindingId id)
{
    CHECK_EQ(visitorId, id);
}

// Concrete TestLayerVisitor subclasses for layers taking LayerBindingId argument with overridden VisitLayer methods
class TestInputLayerVisitor : public TestLayerVisitor
{
private:
    LayerBindingId visitorId;

public:
    explicit TestInputLayerVisitor(LayerBindingId id, const char* name = nullptr)
        : TestLayerVisitor(name)
        , visitorId(id)
    {};

    void VisitInputLayer(const IConnectableLayer* layer,
                         LayerBindingId id,
                         const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckLayerBindingId(visitorId, id);
        CheckLayerName(name);
    };
};

class TestOutputLayerVisitor : public TestLayerVisitor
{
private:
    LayerBindingId visitorId;

public:
    explicit TestOutputLayerVisitor(LayerBindingId id, const char* name = nullptr)
        : TestLayerVisitor(name)
        , visitorId(id)
    {};

    void VisitOutputLayer(const IConnectableLayer* layer,
                          LayerBindingId id,
                          const char* name = nullptr) override
    {
        CheckLayerPointer(layer);
        CheckLayerBindingId(visitorId, id);
        CheckLayerName(name);
    };
};

} //namespace armnn
