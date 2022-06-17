//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stdio.h>
#include <string>
#include <iostream>
#include <sys/system_properties.h>
#include <log/log.h>

namespace {
template<typename T>
struct ConvStringTo;

template<>
struct ConvStringTo<float>
{
    static float Func(std::string s) { return std::stof(s); }
};

template<>
struct ConvStringTo<int>
{
    static int Func(std::string s) { return std::stoi(s); }
};

template<>
struct ConvStringTo<bool>
{
    static bool Func(std::string s) { return !!std::stoi(s); }
};

template<typename T>
void GetCapabilitiesProperties([[maybe_unused]]void* cookie,
                               [[maybe_unused]]const char *name,
                               [[maybe_unused]]const char *value,
                               [[maybe_unused]]uint32_t serial)
{
    T &prop = *reinterpret_cast<T*>(cookie);
    prop = ConvStringTo<T>::Func(std::string(value));
}

template<typename T>
T ParseSystemProperty(const char* name, T defaultValue)
{
    try
    {
        const prop_info *pInfo = __system_property_find(name);
        if (!pInfo)
        {
            ALOGW("ArmnnDriver::ParseSystemProperty(): Could not find property [%s].", name);
        } else
        {
            T property;
            __system_property_read_callback(pInfo, &GetCapabilitiesProperties<T>, &property);
            std::stringstream messageBuilder;
            messageBuilder << "ArmnnDriver::ParseSystemProperty(): Setting [" << name << "]=[" << property << "].";
            ALOGD("%s", messageBuilder.str().c_str());
            return property;
        }
    }
    catch(const std::invalid_argument& e)
    {
        ALOGD("ArmnnDriver::ParseSystemProperty(): Property [%s] has invalid data type.", name);
    }
    catch(const std::out_of_range& e)
    {
        ALOGD("ArmnnDriver::ParseSystemProperty(): Property [%s] out of range for the data type.", name);
    }
    catch (...)
    {
        ALOGD("ArmnnDriver::ParseSystemProperty(): Unexpected exception reading system "
            "property [%s].", name);
    }

    std::stringstream messageBuilder;
    messageBuilder << "ArmnnDriver::ParseSystemProperty(): Falling back to default value [" << defaultValue << "]";
    ALOGD("%s", messageBuilder.str().c_str());
    return defaultValue;
}
} //namespace
