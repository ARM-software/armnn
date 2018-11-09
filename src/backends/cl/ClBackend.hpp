//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/IBackendInternal.hpp>

namespace armnn
{

class ClBackend : public IBackendInternal
{
public:
    ClBackend()  = default;
    ~ClBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory() const override;
    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;
    IBackendInternal::Optimizations GetOptimizations() const override;
};

} // namespace armnn