//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/IBackendContext.hpp>
#include <unordered_set>
#include <mutex>

namespace armnn
{

class ClBackendContext : public IBackendContext
{
public:
    ClBackendContext(const IRuntime::CreationOptions& options);

    bool BeforeLoadNetwork(NetworkId networkId) override;
    bool AfterLoadNetwork(NetworkId networkId) override;

    bool BeforeUnloadNetwork(NetworkId networkId) override;
    bool AfterUnloadNetwork(NetworkId networkId) override;

    ~ClBackendContext() override;

private:
    std::mutex m_Mutex;
    struct ClContextControlWrapper;
    std::unique_ptr<ClContextControlWrapper> m_ClContextControlWrapper;

    std::unordered_set<NetworkId> m_NetworkIds;

};

} // namespace armnn