//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <client/src/ProfilingConnectionDumpToFileDecorator.hpp>

#include <Runtime.hpp>

#include <armnnUtils/Filesystem.hpp>

#include <common/include/IgnoreUnused.hpp>
#include <common/include/NumericCast.hpp>


#include <fstream>
#include <sstream>

#include <doctest/doctest.h>

using namespace arm::pipe;

namespace
{

const std::vector<char> g_Data       = { 'd', 'u', 'm', 'm', 'y' };
const uint32_t          g_DataLength = arm::pipe::numeric_cast<uint32_t>(g_Data.size());
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
        m_Packet = std::make_unique<Packet>(0u, g_DataLength, m_PacketData);
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
        arm::pipe::IgnoreUnused(buffer);
        arm::pipe::IgnoreUnused(length);
        return true;
    }

    Packet ReadPacket(uint32_t timeout) override
    {
        arm::pipe::IgnoreUnused(timeout);
        return std::move(*m_Packet);
    }

private:
    bool m_Open;
    std::unique_ptr<unsigned char[]> m_PacketData;
    std::unique_ptr<Packet> m_Packet;
};

std::vector<char> ReadDumpFile(const std::string& dumpFileName)
{
    std::ifstream input(dumpFileName, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input), {});
}

} // anonymous namespace

TEST_SUITE("ProfilingConnectionDumpToFileDecoratorTests")
{
TEST_CASE("DumpIncomingInvalidFile")
{
    ProfilingOptions options;
    options.m_IncomingCaptureFile = "/";
    options.m_OutgoingCaptureFile =  "";
    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, false);
    CHECK_THROWS_AS(decorator.ReadPacket(0), arm::pipe::ProfilingException);
}

TEST_CASE("DumpIncomingInvalidFileIgnoreErrors")
{
    ProfilingOptions options;
    options.m_IncomingCaptureFile = "/";
    options.m_OutgoingCaptureFile =  "";
    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, true);
    CHECK_NOTHROW(decorator.ReadPacket(0));
}

TEST_CASE("DumpIncomingValidFile")
{
    fs::path fileName = armnnUtils::Filesystem::NamedTempFile("Armnn-DumpIncomingValidFileTest-TempFile");

    ProfilingOptions options;
    options.m_IncomingCaptureFile = fileName.string();
    options.m_OutgoingCaptureFile =  "";

    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, false);

    // NOTE: unique_ptr is needed here because operator=() is deleted for Packet
    std::unique_ptr<Packet> packet;
    CHECK_NOTHROW(packet = std::make_unique<Packet>(decorator.ReadPacket(0)));

    decorator.Close();

    std::vector<char> data = ReadDumpFile(options.m_IncomingCaptureFile);
    const char* packetData = reinterpret_cast<const char*>(packet->GetData());

    // check if the data read back from the dump file matches the original
    constexpr unsigned int bytesToSkip = 2u * sizeof(uint32_t); // skip header and packet length
    int diff = std::strncmp(data.data() + bytesToSkip, packetData, g_DataLength);
    CHECK(diff == 0);
    fs::remove(fileName);
}

TEST_CASE("DumpOutgoingInvalidFile")
{
    ProfilingOptions options;
    options.m_IncomingCaptureFile = "";
    options.m_OutgoingCaptureFile = "/";
    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, false);
    CHECK_THROWS_AS(decorator.WritePacket(g_DataPtr, g_DataLength), arm::pipe::ProfilingException);
}

TEST_CASE("DumpOutgoingInvalidFileIgnoreErrors")
{
    ProfilingOptions options;
    options.m_IncomingCaptureFile = "";
    options.m_OutgoingCaptureFile = "/";

    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, true);
    CHECK_NOTHROW(decorator.WritePacket(g_DataPtr, g_DataLength));

    bool success = decorator.WritePacket(g_DataPtr, g_DataLength);
    CHECK(!success);
}

TEST_CASE("DumpOutgoingValidFile")
{
    fs::path fileName = armnnUtils::Filesystem::NamedTempFile("Armnn-DumpOutgoingValidFileTest-TempFile");

    ProfilingOptions options;
    options.m_IncomingCaptureFile = "";
    options.m_OutgoingCaptureFile = fileName.string();

    ProfilingConnectionDumpToFileDecorator decorator(std::make_unique<DummyProfilingConnection>(), options, false);

    bool success = false;
    CHECK_NOTHROW(success = decorator.WritePacket(g_DataPtr, g_DataLength));
    CHECK(success);

    decorator.Close();

    std::vector<char> data = ReadDumpFile(options.m_OutgoingCaptureFile);

    // check if the data read back from the dump file matches the original
    int diff = std::strncmp(data.data(), g_Data.data(), g_DataLength);
    CHECK(diff == 0);
    fs::remove(fileName);
}

}
