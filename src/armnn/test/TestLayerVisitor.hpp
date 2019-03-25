//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/LayerVisitorBase.hpp>
#include <armnn/Descriptors.hpp>

namespace armnn
{
// Abstract base class with do nothing implementations for all layer visit methods
class TestLayerVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
{
protected:
    virtual ~TestLayerVisitor() {}

    void CheckLayerName(const char* name);

    void CheckLayerPointer(const IConnectableLayer* layer);

    void CheckConstTensors(const ConstTensor& expected, const ConstTensor& actual);

    void CheckOptionalConstTensors(const Optional<ConstTensor>& expected, const Optional<ConstTensor>& actual);

private:
    const char* m_LayerName;

public:
    explicit TestLayerVisitor(const char* name) : m_LayerName(name)
    {
        if (name == nullptr)
        {
            m_LayerName = "";
        }
    }
};

} //namespace armnn
