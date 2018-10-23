//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backends/IBackendInternal.hpp>

namespace armnn
{

class ClBackend : public IBackendInternal
{
public:
    ClBackend()  = default;
    ~ClBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IWorkloadFactoryPtr CreateWorkloadFactory() const override;
};

} // namespace armnn