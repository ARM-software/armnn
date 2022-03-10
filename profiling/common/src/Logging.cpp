//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/Logging.hpp>
#include <common/include/IgnoreUnused.hpp>
#include <common/include/Assert.hpp>

#if defined(_MSC_VER)
#include <common/include/WindowsWrapper.hpp>
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include <iostream>

namespace arm
{

namespace pipe
{

template<>
SimpleLogger<LogSeverity::Debug>& SimpleLogger<LogSeverity::Debug>::Get()
{
    static SimpleLogger<LogSeverity::Debug> logger;
    return logger;
}

template<>
SimpleLogger<LogSeverity::Trace>& SimpleLogger<LogSeverity::Trace>::Get()
{
    static SimpleLogger<LogSeverity::Trace> logger;
    return logger;
}

template<>
SimpleLogger<LogSeverity::Info>& SimpleLogger<LogSeverity::Info>::Get()
{
    static SimpleLogger<LogSeverity::Info> logger;
    return logger;
}

template<>
SimpleLogger<LogSeverity::Warning>& SimpleLogger<LogSeverity::Warning>::Get()
{
    static SimpleLogger<LogSeverity::Warning> logger;
    return logger;
}

template<>
SimpleLogger<LogSeverity::Error>& SimpleLogger<LogSeverity::Error>::Get()
{
    static SimpleLogger<LogSeverity::Error> logger;
    return logger;
}

template<>
SimpleLogger<LogSeverity::Fatal>& SimpleLogger<LogSeverity::Fatal>::Get()
{
    static SimpleLogger<LogSeverity::Fatal> logger;
    return logger;
}

void SetLogFilter(LogSeverity level)
{
    SimpleLogger<LogSeverity::Trace>::Get().Enable(false);
    SimpleLogger<LogSeverity::Debug>::Get().Enable(false);
    SimpleLogger<LogSeverity::Info>::Get().Enable(false);
    SimpleLogger<LogSeverity::Warning>::Get().Enable(false);
    SimpleLogger<LogSeverity::Error>::Get().Enable(false);
    SimpleLogger<LogSeverity::Fatal>::Get().Enable(false);
    switch (level)
    {
        case LogSeverity::Trace:
            SimpleLogger<LogSeverity::Trace>::Get().Enable(true);
            ARM_PIPE_FALLTHROUGH;
        case LogSeverity::Debug:
            SimpleLogger<LogSeverity::Debug>::Get().Enable(true);
            ARM_PIPE_FALLTHROUGH;
        case LogSeverity::Info:
            SimpleLogger<LogSeverity::Info>::Get().Enable(true);
            ARM_PIPE_FALLTHROUGH;
        case LogSeverity::Warning:
            SimpleLogger<LogSeverity::Warning>::Get().Enable(true);
            ARM_PIPE_FALLTHROUGH;
        case LogSeverity::Error:
            SimpleLogger<LogSeverity::Error>::Get().Enable(true);
            ARM_PIPE_FALLTHROUGH;
        case LogSeverity::Fatal:
            SimpleLogger<LogSeverity::Fatal>::Get().Enable(true);
            break;
        default:
            ARM_PIPE_ASSERT(false);
    }
}

class StandardOutputColourSink : public LogSink
{
public:
    StandardOutputColourSink(LogSeverity level = LogSeverity::Info)
    : m_Level(level)
    {
    }

    void Consume(const std::string& s) override
    {
        std::cout << GetColour(m_Level) << s << ResetColour() << std::endl;
    }

private:
    std::string ResetColour()
    {
        return "\033[0m";
    }

    std::string GetColour(LogSeverity level)
    {
        switch(level)
        {
            case LogSeverity::Trace:
                return "\033[35m";
            case LogSeverity::Debug:
                return "\033[32m";
            case LogSeverity::Info:
                return "\033[0m";
            case LogSeverity::Warning:
                return "\033[33m";
            case LogSeverity::Error:
                return "\033[31m";
            case LogSeverity::Fatal:
                return "\033[41;30m";

            default:
                return "\033[0m";
        }
    }
    LogSeverity m_Level;
};

class DebugOutputSink : public LogSink
{
public:
    void Consume(const std::string& s) override
    {
        IgnoreUnused(s);
#if defined(_MSC_VER)
        OutputDebugString(s.c_str());
        OutputDebugString("\n");
#elif defined(__ANDROID__)
        __android_log_write(ANDROID_LOG_DEBUG, "armnn", s.c_str());
#else
        IgnoreUnused(s);
#endif
    }
};

template<LogSeverity Level>
inline void SetLoggingSinks(bool standardOut, bool debugOut, bool coloured)
{
    SimpleLogger<Level>::Get().RemoveAllSinks();

    if (standardOut)
    {
        if (coloured)
        {
            SimpleLogger<Level>::Get().AddSink(
                std::make_shared<StandardOutputColourSink>(Level));
        } else
        {
            SimpleLogger<Level>::Get().AddSink(
                std::make_shared<StandardOutputSink>());
        }
    }

    if (debugOut)
    {
        SimpleLogger<Level>::Get().AddSink(
            std::make_shared<DebugOutputSink>());
    }
}

void SetAllLoggingSinks(bool standardOut, bool debugOut, bool coloured)
{
    SetLoggingSinks<LogSeverity::Trace>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Debug>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Info>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Warning>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Error>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Fatal>(standardOut, debugOut, coloured);
}

void ConfigureLogging(bool printToStandardOutput, bool printToDebugOutput, LogSeverity severity)
{
    SetAllLoggingSinks(printToStandardOutput, printToDebugOutput, false);
    SetLogFilter(severity);
}

} // namespace pipe

} // namespace armnn
