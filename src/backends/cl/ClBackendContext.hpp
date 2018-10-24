//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backends/IBackendContext.hpp>

namespace armnn
{

class ClBackendContext : public IBackendContext
{
public:
    ClBackendContext(const IRuntime::CreationOptions& options);
    ~ClBackendContext() override;

    struct ContextControlWrapper;
private:
    std::shared_ptr<ContextControlWrapper> m_ContextControl;
};

} // namespace armnn