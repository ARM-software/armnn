//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/BackendOptions.hpp>
#include <armnn/IRuntime.hpp>
#include <memory>

namespace armnn
{

class IBackendContext
{
protected:
    IBackendContext(const IRuntime::CreationOptions&) {}

public:
    /// Before and after Load network events
    virtual bool BeforeLoadNetwork(NetworkId networkId) = 0;
    virtual bool AfterLoadNetwork(NetworkId networkId) = 0;

    /// Before and after Unload network events
    virtual bool BeforeUnloadNetwork(NetworkId networkId) = 0;
    virtual bool AfterUnloadNetwork(NetworkId networkId) = 0;

    virtual ~IBackendContext() {}
};

using IBackendContextUniquePtr = std::unique_ptr<IBackendContext>;

class IBackendModelContext
{
public:
    virtual ~IBackendModelContext() {}
};

} // namespace armnn