//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>

namespace armnn
{
class IWorkloadFactory;

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
    virtual IWorkloadFactoryPtr CreateWorkloadFactory() const = 0;
};

using IBackendInternalUniquePtr = std::unique_ptr<IBackendInternal>;

} // namespace armnn
