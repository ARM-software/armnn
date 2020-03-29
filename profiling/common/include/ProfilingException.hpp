//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <stdexcept>
#include <string>

namespace armnnProfiling
{

/// General Exception class for Profiling code
class ProfilingException : public std::exception
{
public:
    explicit ProfilingException(const std::string& message) : m_Message(message) {};

    /// @return - Error message of ProfilingException
    virtual const char* what() const noexcept override
    {
        return m_Message.c_str();
    }

private:
    std::string m_Message;
};

} // namespace armnnProfiling
