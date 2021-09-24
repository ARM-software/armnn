//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once


#include <armnn/INetwork.hpp>
#include <armnn/IStrategy.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

struct ThrowingStrategy
{
    void Apply(const std::string& errorMessage = "") { throw UnimplementedException(errorMessage); };
};

struct NoThrowStrategy
{
    void Apply(const std::string&) {};
};

/// Strategy base class with empty implementations.
template <typename DefaultStrategy>
class StrategyBase : public IStrategy
{
protected:
    virtual ~StrategyBase() {};

public:
    virtual void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                                 const armnn::BaseDescriptor& descriptor,
                                 const std::vector<armnn::ConstTensor>& constants,
                                 const char* name,
                                 const armnn::LayerBindingId id = 0) override
    {
        armnn::IgnoreUnused(descriptor, constants, id, name);
        switch (layer->GetType())
        {
            default:
            {
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType()));
            }
        }
    }

protected:
    DefaultStrategy m_DefaultStrategy;

};


} // namespace armnn
