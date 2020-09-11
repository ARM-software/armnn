//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../ProfilingConnectionDumpToFileDecorator.hpp"
#include <Filesystem.hpp>
#include <Runtime.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <fstream>
#include <sstream>

#include <boost/test/unit_test.hpp>

using namespace armnn::profiling;

namespace
{

const std::vector<char> g_Data       = { 'd', 'u', 'm', 'm', 'y' };
const uint32_t          g_DataLength = armnn::numeric_cast<uint32_t>(g_Data.size());
const unsigned char*    g_DataPtr    = reinterpret_cast<const unsigned char*>(g_Data.data());

class DummyProfilingConnection : public IProfilingConnection
{
public:
    DummyProfilingConnection()
        : m_Open(true)
        , m_PacketData(std::make_unique<unsigned char[]>(g_DataLength))
    {
        // populate packet data and construct packet
        std::memcpy(m_PacketData.get(), g_DataPtr, g_DataLength);
        m_Packet = std::make_unique<arm::pipe::Packet>(0u, g_DataLength, m_PacketData);
    }

    ~DummyProfilingConnection() = default;

    bool IsOpen() const override
    {
        return m_Open;
    }

    void Close() override
    {
        m_Open = false;
    }

    bool WritePacket(const unsigned char* buffer, uint32_t length) override
    {
        armnn::IgnoreUnused(buffer);
        armnn::IgnoreUnused(length);
        return true;
    }

    arm::pipe::Packet ReadPacket(uint32_t timeout) override
    {
        armnn::IgnoreUnused(timeout);
        return std::move(*m_Packet);
    }

private:
    bool m_Open;
    std::unique_ptr<unsigned char[]> m_PacketData;
    std::unique_ptr<arm::pipe::Packet> m_Packet;
};

std::vector<char> ReadDumpFile(const std::string& dumpFileName)
{
    std::ifstream input(dumpFileName, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input), {});
}

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(ProfilingConnectionDumpToFileDecoratorTests)

BOOST_AUTO_TEST_CASE(DumpIncomingInvalidFile)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_IncomingCaptureFile = "/";
    options.m_OutgoingCaptureFile =  "";
    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, false);
    BOOST_CHECK_THROW(decorator.ReadPacket(0), armnn::RuntimeException);
}

BOOST_AUTO_TEST_CASE(DumpIncomingInvalidFileIgnoreErrors)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_IncomingCaptureFile = "/";
    options.m_OutgoingCaptureFile =  "";
    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, true);
    BOOST_CHECK_NO_THROW(decorator.ReadPacket(0));
}

BOOST_AUTO_TEST_CASE(DumpIncomingValidFile)
{
    fs::path fileName = armnnUtils::Filesystem::NamedTempFile("Armnn-DumpIncomingValidFileTest-TempFile");

    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_IncomingCaptureFile = fileName.string();
    options.m_OutgoingCaptureFile =  "";

    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, false);

    // NOTE: unique_ptr is needed here because operator=() is deleted for Packet
    std::unique_ptr<arm::pipe::Packet> packet;
    BOOST_CHECK_NO_THROW(packet = std::make_unique<arm::pipe::Packet>(decorator.ReadPacket(0)));

    decorator.Close();

    std::vector<char> data = ReadDumpFile(options.m_IncomingCaptureFile);
    const char* packetData = reinterpret_cast<const char*>(packet->GetData());

    // check if the data read back from the dump file matches the original
    constexpr unsigned int bytesToSkip = 2u * sizeof(uint32_t); // skip header and packet length
    int diff = std::strncmp(data.data() + bytesToSkip, packetData, g_DataLength);
    BOOST_CHECK(diff == 0);
    fs::remove(fileName);
}

BOOST_AUTO_TEST_CASE(DumpOutgoingInvalidFile)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_IncomingCaptureFile = "";
    options.m_OutgoingCaptureFile = "/";
    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, false);
    BOOST_CHECK_THROW(decorator.WritePacket(g_DataPtr, g_DataLength), armnn::RuntimeException);
}

BOOST_AUTO_TEST_CASE(DumpOutgoingInvalidFileIgnoreErrors)
{
    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_IncomingCaptureFile = "";
    options.m_OutgoingCaptureFile = "/";

    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, true);
    BOOST_CHECK_NO_THROW(decorator.WritePacket(g_DataPtr, g_DataLength));

    bool success = decorator.WritePacket(g_DataPtr, g_DataLength);
    BOOST_CHECK(!success);
}

BOOST_AUTO_TEST_CASE(DumpOutgoingValidFile)
{
    fs::path fileName = armnnUtils::Filesystem::NamedTempFile("Armnn-DumpOutgoingValidFileTest-TempFile");

    armnn::Runtime::CreationOptions::ExternalProfilingOptions options;
    options.m_IncomingCaptureFile = "";
    options.m_OutgoingCaptureFile = fileName.string();

    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, false);

    bool success = false;
    BOOST_CHECK_NO_THROW(success = decorator.WritePacket(g_DataPtr, g_DataLength));
    BOOST_CHECK(success);

    decorator.Close();

    std::vector<char> data = ReadDumpFile(options.m_OutgoingCaptureFile);

    // check if the data read back from the dump file matches the original
    int diff = std::strncmp(data.data(), g_Data.data(), g_DataLength);
    BOOST_CHECK(diff == 0);
    fs::remove(fileName);
}

BOOST_AUTO_TEST_SUITE_END()
