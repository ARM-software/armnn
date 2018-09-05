//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#define BOOST_TEST_MODULE UnitTests
#include <boost/test/unit_test.hpp>

#include "UnitTests.hpp"

struct ConfigureLoggingFixture
{
    ConfigureLoggingFixture()
    {
        ConfigureLoggingTest();
    }
};

BOOST_GLOBAL_FIXTURE(ConfigureLoggingFixture);

// On Windows, duplicate the boost test logging output to the Visual Studio output window using OutputDebugString.
#if defined(_MSC_VER)

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/tee.hpp>
#include <iostream>
#include <Windows.h>

using namespace boost::iostreams;
using namespace std;

struct DebugOutputSink : boost::iostreams::sink
{
    std::streamsize write(const char* s, std::streamsize n)
    {
        // The given string is not null-terminated, so we need to copy it.
        std::string s2(s, boost::numeric_cast<size_t>(n));
        OutputDebugString(s2.c_str());
        return n;
    }
};

class SetupDebugOutput
{
public:
    SetupDebugOutput()
    {
        // Sends the output to both cout (as standard) and the debug output.
        m_OutputStream.push(tee(std::cout));
        m_OutputStream.push(m_DebugOutputSink);

        boost::unit_test::unit_test_log.set_stream(m_OutputStream);
    }
private:
    filtering_ostream m_OutputStream;
    DebugOutputSink m_DebugOutputSink;
};

BOOST_GLOBAL_FIXTURE(SetupDebugOutput);

#endif // defined(_MSC_VER)