//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

namespace arm
{

namespace pipe
{

struct Location
{
    const char* m_Function;
    const char* m_File;
    unsigned int m_Line;

    Location(const char* func,
             const char* file,
             unsigned int line)
    : m_Function{func}
    , m_File{file}
    , m_Line{line}
    {
    }

    std::string AsString() const
    {
        std::stringstream ss;
        ss << " at function " << m_Function
           << " [" << m_File << ':' << m_Line << "]";
        return ss.str();
    }

    std::string FileLine() const
    {
        std::stringstream ss;
        ss << " [" << m_File << ':' << m_Line << "]";
        return ss.str();
    }
};

/// General Exception class for Profiling code
class ProfilingException : public std::exception
{
public:
    explicit ProfilingException(const std::string& message) : m_Message(message) {};

    explicit ProfilingException(const std::string& message,
                                const Location& location) : m_Message(message + location.AsString()) {};

    /// @return - Error message of ProfilingException
    virtual const char *what() const noexcept override
    {
         return m_Message.c_str();
    }

private:
    std::string m_Message;
};

class TimeoutException : public ProfilingException
{
public:
    using ProfilingException::ProfilingException;
};

class InvalidArgumentException : public ProfilingException
{
public:
    using ProfilingException::ProfilingException;
};

} // namespace pipe
} // namespace arm

#define LOCATION() arm::pipe::Location(__func__, __FILE__, __LINE__)
