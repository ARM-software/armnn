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

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Input:
            {
                CheckLayerPointer(layer);
                CheckLayerBindingId(visitorId, id);
                CheckLayerName(name);
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }
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

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id);
        switch (layer->GetType())
        {
            case armnn::LayerType::Output:
            {
                CheckLayerPointer(layer);
                CheckLayerBindingId(visitorId, id);
                CheckLayerName(name);
                break;
            }
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }
};

} //namespace armnn
