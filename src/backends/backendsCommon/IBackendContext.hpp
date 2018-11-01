//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/IRuntime.hpp>
#include <memory>

namespace armnn
{

class IBackendContext
{
public:
    virtual ~IBackendContext() {}

protected:
    IBackendContext(const IRuntime::CreationOptions& options) {}

private:
    IBackendContext() = delete;
};

using IBackendContextUniquePtr = std::unique_ptr<IBackendContext>;

} // namespace armnn
