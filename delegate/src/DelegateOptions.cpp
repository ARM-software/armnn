//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <DelegateOptions.hpp>

namespace armnnDelegate
{

DelegateOptions::DelegateOptions(armnn::Compute computeDevice,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
    : m_Backends({computeDevice}), m_BackendOptions(backendOptions), m_LoggingSeverity(logSeverityLevel)
{
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
    : m_Backends(backends), m_BackendOptions(backendOptions), m_LoggingSeverity(logSeverityLevel)
{
}

} // namespace armnnDelegate
