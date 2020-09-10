//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackendModelContext.hpp"

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

ClBackendModelContext::ClBackendModelContext(const ModelOptions& modelOptions)
    : m_IsFastMathEnabled(false)
{
   if (!modelOptions.empty())
   {
       ParseOptions(modelOptions, "GpuAcc", [&](std::string name, const BackendOptions::Var& value)
       {
           if (name == "FastMathEnabled")
           {
               m_IsFastMathEnabled |= ParseBool(value, false);
           }
       });
   }
}

bool ClBackendModelContext::IsFastMathEnabled() const
{
    return m_IsFastMathEnabled;
}

} // namespace armnn