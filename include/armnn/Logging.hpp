//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Utils.hpp>
#include <iostream>
#include <algorithm>

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

inline LogSeverity StringToLogLevel(std::string level)
{
    // Transfer to lower case
    std::transform(level.begin(), level.end(), level.begin(),
                   [](unsigned char c){ return std::tolower(c); }
    );

    if (level == "trace")
    {
        return LogSeverity::Trace;
    }
    else if (level == "debug")
    {
        return LogSeverity::Debug;
    }
    else if (level == "info")
    {
        return LogSeverity::Info;
    }
    else if (level == "warning")
    {
        return LogSeverity::Warning;
    }
    else if (level == "error")
    {
        return LogSeverity::Error;
    }
    else if (level == "fatal")
    {
        return LogSeverity::Fatal;
    }
    else
    {
        throw armnn::Exception("Unknown severity level for logging: '" + level +
                               "'. Valid options: trace, debug, info, warning, error, fatal");
    }
}

class LogSink
{
public:
    virtual ~LogSink(){};

    virtual void Consume(const std::string&) = 0;
private:

};

class StandardOutputSink : public LogSink
{
public:
    void Consume(const std::string& s) override
    {
        std::cout << s << std::endl;
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
    ScopedRecord& operator=(ScopedRecord&&) = delete;

    ScopedRecord(ScopedRecord&& other) = default;

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

void SetLogFilter(LogSeverity level);

void SetAllLoggingSinks(bool standardOut, bool debugOut, bool coloured);

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
