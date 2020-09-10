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

} // namespace anonymous

namespace armnn
{

NeonBackendModelContext::NeonBackendModelContext(const ModelOptions& modelOptions)
    : m_IsFastMathEnabled(false)
{
   if (!modelOptions.empty())
   {
       ParseOptions(modelOptions, "CpuAcc", [&](std::string name, const BackendOptions::Var& value)
       {
           if (name == "FastMathEnabled")
           {
               m_IsFastMathEnabled |= ParseBool(value, false);
           }
       });
   }
}

bool NeonBackendModelContext::IsFastMathEnabled() const
{
    return m_IsFastMathEnabled;
}

} // namespace armnn