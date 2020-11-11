//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>

#include <set>
#include <string>
#include <vector>

namespace armnnDelegate
{

class DelegateOptions
{
public:
    DelegateOptions(armnn::Compute computeDevice, const std::vector<armnn::BackendOptions>& backendOptions = {});

    DelegateOptions(const std::vector<armnn::BackendId>& backends,
                    const std::vector<armnn::BackendOptions>& backendOptions = {});

    const std::vector<armnn::BackendId>& GetBackends() const { return m_Backends; }

    void SetBackends(const std::vector<armnn::BackendId>& backends) { m_Backends = backends; }

    const std::vector<armnn::BackendOptions>& GetBackendOptions() const { return m_BackendOptions; }

private:
    /// Which backend to run Delegate on.
    /// Examples of possible values are: CpuRef, CpuAcc, GpuAcc.
    /// CpuRef as default.
    std::vector<armnn::BackendId> m_Backends = { armnn::Compute::CpuRef };

    /// Pass backend specific options to Delegate
    ///
    /// For example, tuning can be enabled on GpuAcc like below
    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// m_BackendOptions.emplace_back(
    ///     BackendOptions{"GpuAcc",
    ///       {
    ///         {"TuningLevel", 2},
    ///         {"TuningFile", filename}
    ///       }
    ///     });
    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// The following backend options are available:
    /// GpuAcc:
    ///   "TuningLevel" : int [0..3] (0=UseOnly(default) | 1=RapidTuning | 2=NormalTuning | 3=ExhaustiveTuning)
    ///   "TuningFile" : string [filenameString]
    ///   "KernelProfilingEnabled" : bool [true | false]
    std::vector<armnn::BackendOptions> m_BackendOptions;
};

} // namespace armnnDelegate
