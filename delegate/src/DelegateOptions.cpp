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
    : m_Backends({computeDevice}), m_RuntimeOptions(), m_LoggingSeverity(logSeverityLevel)
{
    m_RuntimeOptions.m_BackendOptions = backendOptions;
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const armnn::Optional<armnn::LogSeverity> logSeverityLevel)
    : m_Backends(backends), m_RuntimeOptions(), m_LoggingSeverity(logSeverityLevel)
{
    m_RuntimeOptions.m_BackendOptions = backendOptions;
}

DelegateOptions::DelegateOptions(armnn::Compute computeDevice,
                                 const armnn::OptimizerOptions& optimizerOptions,
                                 const armnn::Optional<armnn::LogSeverity>& logSeverityLevel,
                                 const armnn::Optional<armnn::DebugCallbackFunction>& func)
    : m_Backends({computeDevice}),
      m_RuntimeOptions(),
      m_OptimizerOptions(optimizerOptions),
      m_LoggingSeverity(logSeverityLevel),
      m_DebugCallbackFunc(func)
{
}

DelegateOptions::DelegateOptions(const std::vector<armnn::BackendId>& backends,
                                 const armnn::OptimizerOptions& optimizerOptions,
                                 const armnn::Optional<armnn::LogSeverity>& logSeverityLevel,
                                 const armnn::Optional<armnn::DebugCallbackFunction>& func)
    : m_Backends(backends),
      m_RuntimeOptions(),
      m_OptimizerOptions(optimizerOptions),
      m_LoggingSeverity(logSeverityLevel),
      m_DebugCallbackFunc(func)
{
}

} // namespace armnnDelegate
