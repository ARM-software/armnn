//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Logging.hpp>
#include <armnn/Optional.hpp>

#include <set>
#include <string>
#include <vector>

namespace armnnDelegate
{

class DelegateOptions
{
public:
    DelegateOptions(armnn::Compute computeDevice,
                    const std::vector<armnn::BackendOptions>& backendOptions = {},
                    armnn::Optional<armnn::LogSeverity> logSeverityLevel = armnn::EmptyOptional());

    DelegateOptions(const std::vector<armnn::BackendId>& backends,
                    const std::vector<armnn::BackendOptions>& backendOptions = {},
                    armnn::Optional<armnn::LogSeverity> logSeverityLevel = armnn::EmptyOptional());

    const std::vector<armnn::BackendId>& GetBackends() const { return m_Backends; }

    void SetBackends(const std::vector<armnn::BackendId>& backends) { m_Backends = backends; }

    const std::vector<armnn::BackendOptions>& GetBackendOptions() const { return m_BackendOptions; }

    /// Appends a backend option to the list of backend options
    void AddBackendOption(const armnn::BackendOptions& option) { m_BackendOptions.push_back(option); }

    /// Sets the severity level for logging within ArmNN that will be used on creation of the delegate
    void SetLoggingSeverity(const armnn::LogSeverity& level) { m_LoggingSeverity = level; }
    void SetLoggingSeverity(const std::string& level) { m_LoggingSeverity = armnn::StringToLogLevel(level); }

    /// Returns the severity level for logging within ArmNN
    armnn::LogSeverity GetLoggingSeverity() { return m_LoggingSeverity.value(); }

    bool IsLoggingEnabled() { return m_LoggingSeverity.has_value(); }

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

    /// Severity level for logging within ArmNN that will be used on creation of the delegate
    armnn::Optional<armnn::LogSeverity> m_LoggingSeverity;
};

} // namespace armnnDelegate
