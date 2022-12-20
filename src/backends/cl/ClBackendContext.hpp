//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendContext.hpp>
#include <unordered_set>
#include <mutex>

#include <arm_compute/runtime/CL/CLTuner.h>
#include <arm_compute/runtime/CL/CLGEMMHeuristicsHandle.h>

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

    bool AfterEnqueueWorkload(NetworkId networkId) override;

    ~ClBackendContext() override;

private:
    std::mutex m_Mutex;
    struct ClContextControlWrapper;
    std::unique_ptr<ClContextControlWrapper> m_ClContextControlWrapper;

    std::unordered_set<NetworkId> m_NetworkIds;

    std::unique_ptr<arm_compute::CLTuner> m_Tuner;
    std::string m_TuningFile;

protected:
    arm_compute::CLGEMMHeuristicsHandle m_MLGOTuner;
    std::string m_MLGOTuningFile;
};

} // namespace armnn