//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

class IBackendInternal : public IBackend
{
protected:
    IBackendInternal() {}
    virtual ~IBackendInternal() {}

public:
    virtual std::unique_ptr<IWorkloadFactory> CreateWorkloadFactory() const = 0;
};

} // namespace armnn
