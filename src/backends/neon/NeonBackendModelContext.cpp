//
// Copyright © 2026 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackendModelContext.hpp"

#include <arm_compute/core/CPP/CPPTypes.h>

namespace
{

bool ParseBool(const armnn::BackendOptions::Var& value, bool defaultValue)
{
    if (value.IsBool())
    {
        return value.AsBool();
    }
    return defaultValue;
}

unsigned int ParseUnsignedInt(const armnn::BackendOptions::Var& value, unsigned int defaultValue)
{
    if (value.IsUnsignedInt())
    {
        return value.AsUnsignedInt();
    }
    return defaultValue;
}

} // namespace anonymous

namespace armnn
{

NeonBackendModelContext::NeonBackendModelContext(const ModelOptions& modelOptions)
    : m_IsFastMathEnabled(false), m_NumberOfThreads(0), m_IsSveEnabled(true), m_IsSmeEnabled(true)
{
   if (!modelOptions.empty())
   {
       ParseOptions(modelOptions, "CpuAcc", [&](std::string name, const BackendOptions::Var& value)
       {
           if (name == "FastMathEnabled")
           {
               m_IsFastMathEnabled = ParseBool(value, m_IsFastMathEnabled);
           }
           if (name == "NumberOfThreads")
           {
               m_NumberOfThreads = ParseUnsignedInt(value, m_NumberOfThreads);
           }
           if (name == "SmeEnabled")
           {
               m_IsSmeEnabled = ParseBool(value, m_IsSmeEnabled);
           }
           if (name == "SveEnabled")
           {
               m_IsSveEnabled = ParseBool(value, m_IsSveEnabled);
           }
       });
   }

   ApplyAclIsaPolicy();
}

bool NeonBackendModelContext::IsFastMathEnabled() const
{
    return m_IsFastMathEnabled;
}

unsigned int NeonBackendModelContext::GetNumberOfThreads() const
{
    return m_NumberOfThreads;
}

void NeonBackendModelContext::ApplyAclIsaPolicy() const
{
    arm_compute::CPUInfo::get().set_sve_allowed(m_IsSveEnabled);
    arm_compute::CPUInfo::get().set_sme_allowed(m_IsSmeEnabled);
}

} // namespace armnn
