//
// Copyright © 2017, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <aclCommon/ArmComputeTuningUtils.hpp>

namespace armnn
{

// ARM Compute OpenCL context control.
class ClContextControl
{
public:

    ClContextControl(arm_compute::CLTuner* = nullptr,
                     arm_compute::CLGEMMHeuristicsHandle* = nullptr,
                     bool profilingEnabled = false);

    virtual ~ClContextControl();

    void LoadOpenClRuntime();

    // Users should call this (after freeing all of the cl::Context objects they use)
    // to release the cached memory used by the compute library.
    void UnloadOpenClRuntime();

    // Clear the CL cache, without losing the tuned parameter settings.
    void ClearClCache();

private:

    void DoLoadOpenClRuntime(bool updateTunedParameters);

    arm_compute::CLTuner* m_Tuner;
    arm_compute::CLGEMMHeuristicsHandle* m_HeuristicsHandle;

    bool m_ProfilingEnabled;
};

} // namespace armnn
