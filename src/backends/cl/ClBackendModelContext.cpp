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

std::string ParseFile(const armnn::BackendOptions::Var& value, std::string defaultValue)
{
    if (value.IsString())
    {
        return value.AsString();
    }
    return defaultValue;
}

} // namespace anonymous

namespace armnn
{

ClBackendModelContext::ClBackendModelContext(const ModelOptions& modelOptions)
    : m_CachedNetworkFilePath(""), m_IsFastMathEnabled(false), m_SaveCachedNetwork(false), m_CachedFileDescriptor(-1)
{
   if (!modelOptions.empty())
   {
       ParseOptions(modelOptions, "GpuAcc", [&](std::string name, const BackendOptions::Var& value)
       {
           if (name == "FastMathEnabled")
           {
               m_IsFastMathEnabled |= ParseBool(value, false);
           }
           if (name == "SaveCachedNetwork")
           {
               m_SaveCachedNetwork |= ParseBool(value, false);
           }
           if (name == "CachedNetworkFilePath")
           {
               m_CachedNetworkFilePath = ParseFile(value, "");
           }
           if (name == "CachedFileDescriptor")
           {
               m_CachedFileDescriptor = armnn::ParseIntBackendOption(value, -1);
           }
       });
   }
}

std::string ClBackendModelContext::GetCachedNetworkFilePath() const
{
    return m_CachedNetworkFilePath;
}

bool ClBackendModelContext::IsFastMathEnabled() const
{
    return m_IsFastMathEnabled;
}

bool ClBackendModelContext::SaveCachedNetwork() const
{
    return m_SaveCachedNetwork;
}

int ClBackendModelContext::GetCachedFileDescriptor() const
{
    return m_CachedFileDescriptor;
}

} // namespace armnn