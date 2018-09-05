//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Logging.hpp"

#include <string>
#include <iostream>

#if defined(_MSC_VER)
#include <Windows.h>
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include <boost/make_shared.hpp>
#include <boost/log/core.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sinks/debug_output_backend.hpp>
#include <boost/log/sinks/basic_sink_backend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/utility/setup/console.hpp>

namespace armnnUtils
{

struct DebugOutputSink : boost::log::sinks::basic_formatted_sink_backend<char, boost::log::sinks::concurrent_feeding>
{
    void consume(boost::log::record_view const& rec, std::string const& formatted_message)
    {
#if defined(_MSC_VER)
        OutputDebugString(formatted_message.c_str());
        OutputDebugString("\n");
#endif
#if defined(__ANDROID__)
        __android_log_write(ANDROID_LOG_DEBUG, "armnn", formatted_message.c_str());
#endif
    }
};

void ConfigureLogging(boost::log::core* core, bool printToStandardOutput, bool printToDebugOutput,
    armnn::LogSeverity severity)
{
    // Even if we remove all the sinks, Boost will fallback to the 'default sink' and still print stuff to
    // stdout, so we have to explicitly disable logging in this case.
    core->set_logging_enabled(printToStandardOutput || printToDebugOutput);

    // Sets up severity filter.
    boost::log::trivial::severity_level boostSeverity;
    switch (severity)
    {
    case armnn::LogSeverity::Trace:
        boostSeverity = boost::log::trivial::trace;
        break;
    case armnn::LogSeverity::Debug:
        boostSeverity = boost::log::trivial::debug;
        break;
    case armnn::LogSeverity::Info:
        boostSeverity = boost::log::trivial::info;
        break;
    case armnn::LogSeverity::Warning:
        boostSeverity = boost::log::trivial::warning;
        break;
    case armnn::LogSeverity::Error:
        boostSeverity = boost::log::trivial::error;
        break;
    case armnn::LogSeverity::Fatal:
        boostSeverity = boost::log::trivial::fatal;
        break;
    default:
        BOOST_ASSERT_MSG(false, "Invalid severity");
    }
    core->set_filter(boost::log::trivial::severity >= boostSeverity);

    core->remove_all_sinks();
    if (printToStandardOutput)
    {
        typedef boost::log::sinks::basic_text_ostream_backend<char> backend_t;
        boost::shared_ptr<backend_t>                                backend = boost::make_shared<backend_t>();

        boost::shared_ptr<std::basic_ostream<char>> stream(&std::cout, boost::null_deleter());
        backend->add_stream(stream);

        typedef boost::log::sinks::synchronous_sink<backend_t> sink_t;
        boost::shared_ptr<sink_t>                              standardOutputSink = boost::make_shared<sink_t>(backend);

        core->add_sink(standardOutputSink);
    }
    if (printToDebugOutput)
    {
        typedef boost::log::sinks::synchronous_sink<DebugOutputSink> sink_t;
        boost::shared_ptr<sink_t>                                    debugOutputSink(new sink_t());
        core->add_sink(debugOutputSink);
    }
}

}
