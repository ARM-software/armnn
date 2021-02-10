//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackendModelContext.hpp"

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
    : m_IsFastMathEnabled(false), m_NumberOfThreads(0)
{
   if (!modelOptions.empty())
   {
       ParseOptions(modelOptions, "CpuAcc", [&](std::string name, const BackendOptions::Var& value)
       {
           if (name == "FastMathEnabled")
           {
               m_IsFastMathEnabled |= ParseBool(value, false);
           }
           if (name == "NumberOfThreads")
           {
               m_NumberOfThreads |= ParseUnsignedInt(value, 0);
           }
       });
   }
}

bool NeonBackendModelContext::IsFastMathEnabled() const
{
    return m_IsFastMathEnabled;
}

unsigned int NeonBackendModelContext::GetNumberOfThreads() const
{
    return m_NumberOfThreads;
}

} // namespace armnn