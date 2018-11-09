//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/IRuntime.hpp>
#include <vector>

namespace armnn
{
class IWorkloadFactory;
class IBackendContext;
class Optimization;

class IBackendInternal : public IBackend
{
protected:
    // Creation must be done through a specific
    // backend interface.
    IBackendInternal() = default;

public:
    // Allow backends created by the factory function
    // to be destroyed through IBackendInternal.
    ~IBackendInternal() override = default;

    using IWorkloadFactoryPtr = std::unique_ptr<IWorkloadFactory>;
    using IBackendContextPtr = std::unique_ptr<IBackendContext>;
    using OptimizationPtr = std::unique_ptr<Optimization>;
    using Optimizations = std::vector<OptimizationPtr>;

    virtual IWorkloadFactoryPtr CreateWorkloadFactory() const = 0;
    virtual IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const = 0;
    virtual Optimizations GetOptimizations() const = 0;
};

using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;

} // namespace armnn
