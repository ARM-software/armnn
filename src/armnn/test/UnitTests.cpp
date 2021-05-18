//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#define BOOST_TEST_MODULE UnitTests
#include <boost/test/unit_test.hpp>

#include "UnitTests.hpp"
#include <armnn/Logging.hpp>
#include <armnn/utility/NumericCast.hpp>

struct ConfigureLoggingFixture
{
    ConfigureLoggingFixture()
    {
        ConfigureLoggingTest();
    }
};

BOOST_GLOBAL_FIXTURE(ConfigureLoggingFixture);

BOOST_AUTO_TEST_SUITE(LoggerSuite)

BOOST_AUTO_TEST_CASE(LoggerTest)
{
    std::stringstream ss;

    {
        struct StreamRedirector
        {
        public:
            StreamRedirector(std::ostream& stream, std::streambuf* newStreamBuffer)
                : m_Stream(stream)
                , m_BackupBuffer(m_Stream.rdbuf(newStreamBuffer))
            {}
            ~StreamRedirector() { m_Stream.rdbuf(m_BackupBuffer); }

        private:
            std::ostream& m_Stream;
            std::streambuf* m_BackupBuffer;
        };


        StreamRedirector redirect(std::cout, ss.rdbuf());

        using namespace armnn;
        SetLogFilter(LogSeverity::Trace);
        SetAllLoggingSinks(true, false, false);


        ARMNN_LOG(trace) << "My trace message; " << -2;
        ARMNN_LOG(debug) << "My debug message; " << -1;
        ARMNN_LOG(info) << "My info message; " << 0;
        ARMNN_LOG(warning) << "My warning message; "  << 1;
        ARMNN_LOG(error) << "My error message; " << 2;
        ARMNN_LOG(fatal) << "My fatal message; "  << 3;

        SetLogFilter(LogSeverity::Fatal);

    }

    BOOST_CHECK(ss.str().find("Trace: My trace message; -2") != std::string::npos);
    BOOST_CHECK(ss.str().find("Debug: My debug message; -1") != std::string::npos);
    BOOST_CHECK(ss.str().find("Info: My info message; 0") != std::string::npos);
    BOOST_CHECK(ss.str().find("Warning: My warning message; 1") != std::string::npos);
    BOOST_CHECK(ss.str().find("Error: My error message; 2") != std::string::npos);
    BOOST_CHECK(ss.str().find("Fatal: My fatal message; 3") != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
