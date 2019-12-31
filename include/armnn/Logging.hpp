//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>

#include "Utils.hpp"


#if defined(_MSC_VER)
#include <Windows.h>
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include <boost/assert.hpp>
#include <boost/core/ignore_unused.hpp>

namespace armnn
{

inline std::string LevelToString(LogSeverity level)
{
    switch(level)
    {
        case LogSeverity::Trace:
            return "Trace";
        case LogSeverity::Debug:
            return "Debug";
        case LogSeverity::Info:
            return "Info";
        case LogSeverity::Warning:
            return "Warning";
        case LogSeverity::Error:
            return "Error";
        case LogSeverity::Fatal:
            return "Fatal";
        default:
            return "Log";
    }
}

class LogSink
{
public:
    virtual ~LogSink(){};

    virtual void Consume(const std::string&) = 0;
private:

};

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

class StandardOutputSink : public LogSink
{
public:
    void Consume(const std::string& s) override
    {
        std::cout << s << std::endl;
    }
};

class DebugOutputSink : public LogSink
{
public:
    void Consume(const std::string& s) override
    {
        boost::ignore_unused(s);
#if defined(_MSC_VER)
        OutputDebugString(s.c_str());
        OutputDebugString("\n");
#elif defined(__ANDROID__)
        __android_log_write(ANDROID_LOG_DEBUG, "armnn", s.c_str());
#else
        boost::ignore_unused(s);
#endif
    }
};

struct ScopedRecord
{
    ScopedRecord(const std::vector<std::shared_ptr<LogSink>>& sinks, LogSeverity level, bool enabled)
    : m_LogSinks(sinks)
    , m_Enabled(enabled)
    {
        if (enabled)
        {
            m_Os << LevelToString(level) << ": ";
        }
    }

    ~ScopedRecord()
    {
        if (m_Enabled)
        {
            for (auto sink : m_LogSinks)
            {
                if (sink)
                {
                    sink->Consume(m_Os.str());
                }
            }
        }
    }

    ScopedRecord(const ScopedRecord&) = delete;
    ScopedRecord& operator=(const ScopedRecord&) = delete;
    ScopedRecord(ScopedRecord&& other) = default;
    ScopedRecord& operator=(ScopedRecord&&) = default;

    template<typename Streamable>
    ScopedRecord& operator<<(const Streamable& s)
    {
        if (m_Enabled)
        {
            m_Os << s;
        }
        return (*this);
    }

private:
    const std::vector<std::shared_ptr<LogSink>>& m_LogSinks;
    std::ostringstream m_Os;
    bool m_Enabled;
};

template<LogSeverity Level>
class SimpleLogger
{
public:
    SimpleLogger()
        : m_Sinks{std::make_shared<StandardOutputSink>()}
        , m_Enable(true)
    {
    }

    static SimpleLogger& Get()
    {
        static SimpleLogger<Level> logger;
        return logger;
    }

    void Enable(bool enable = true)
    {
        m_Enable = enable;
    }

    ScopedRecord StartNewRecord()
    {
        ScopedRecord record(m_Sinks, Level, m_Enable);
        return record;
    }

    void RemoveAllSinks()
    {
        m_Sinks.clear();
    }

    void AddSink(std::shared_ptr<LogSink> sink)
    {
        m_Sinks.push_back(sink);
    }
private:
    std::vector<std::shared_ptr<LogSink>> m_Sinks;
    bool m_Enable;
};

inline void SetLogFilter(LogSeverity level)
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
            ARMNN_FALLTHROUGH;
        case LogSeverity::Debug:
            SimpleLogger<LogSeverity::Debug>::Get().Enable(true);
            ARMNN_FALLTHROUGH;
        case LogSeverity::Info:
            SimpleLogger<LogSeverity::Info>::Get().Enable(true);
            ARMNN_FALLTHROUGH;
        case LogSeverity::Warning:
            SimpleLogger<LogSeverity::Warning>::Get().Enable(true);
            ARMNN_FALLTHROUGH;
        case LogSeverity::Error:
            SimpleLogger<LogSeverity::Error>::Get().Enable(true);
            ARMNN_FALLTHROUGH;
        case LogSeverity::Fatal:
            SimpleLogger<LogSeverity::Fatal>::Get().Enable(true);
            break;
        default:
            BOOST_ASSERT(false);
    }
}

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

inline void SetAllLoggingSinks(bool standardOut, bool debugOut, bool coloured)
{
    SetLoggingSinks<LogSeverity::Trace>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Debug>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Info>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Warning>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Error>(standardOut, debugOut, coloured);
    SetLoggingSinks<LogSeverity::Fatal>(standardOut, debugOut, coloured);
}

enum class BoostLogSeverityMapping
{
    trace,
    debug,
    info,
    warning,
    error,
    fatal
};

constexpr LogSeverity ConvertLogSeverity(BoostLogSeverityMapping severity)
{
    return static_cast<LogSeverity>(severity);
}


#define ARMNN_LOG(severity) \
    armnn::SimpleLogger<ConvertLogSeverity(armnn::BoostLogSeverityMapping::severity)>::Get().StartNewRecord()

} //namespace armnn
