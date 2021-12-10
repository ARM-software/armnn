//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/StrategyBase.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/backends/TensorHandle.hpp>

namespace armnn
{
// Abstract base class with do nothing implementations for all layers
class TestLayerVisitor : public StrategyBase<NoThrowStrategy>
{
protected:
    virtual ~TestLayerVisitor() {}

    void CheckLayerName(const char* name);

    void CheckLayerPointer(const IConnectableLayer* layer);

    void CheckConstTensors(const ConstTensor& expected,
                           const ConstTensor& actual);
    void CheckConstTensors(const ConstTensor& expected,
                           const ConstTensorHandle& actual);

    void CheckConstTensorPtrs(const std::string& name,
                              const ConstTensor* expected,
                              const ConstTensor* actual);
    void CheckConstTensorPtrs(const std::string& name,
                              const ConstTensor* expected,
                              const std::shared_ptr<ConstTensorHandle> actual);

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
