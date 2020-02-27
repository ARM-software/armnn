//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PacketVersionResolver.hpp"
#include "CommandFileParser.hpp"
#include "CommandLineProcessor.hpp"
#include "DirectoryCaptureCommandHandler.hpp"
#include "GatordMockService.hpp"
#include "PeriodicCounterCaptureCommandHandler.hpp"
#include "PeriodicCounterSelectionResponseHandler.hpp"
#include <TimelineDecoder.hpp>
#include <TimelineDirectoryCaptureCommandHandler.hpp>
#include <TimelineCaptureCommandHandler.hpp>

#include <iostream>
#include <string>
#include <NetworkSockets.hpp>
#include <signal.h>

using namespace armnn;
using namespace gatordmock;

// Used to capture ctrl-c so we can close any remaining sockets before exit
static volatile bool run = true;
void exit_capture(int signum)
{
    IgnoreUnused(signum);
    run = false;
}

bool CreateMockService(armnnUtils::Sockets::Socket clientConnection, std::string commandFile, bool isEchoEnabled)
{
    profiling::PacketVersionResolver packetVersionResolver;
    // Create the Command Handler Registry
    profiling::CommandHandlerRegistry registry;

    timelinedecoder::TimelineDecoder timelineDecoder;
    timelineDecoder.SetDefaultCallbacks();

    // This functor will receive back the selection response packet.
    PeriodicCounterSelectionResponseHandler periodicCounterSelectionResponseHandler(
            0, 4, packetVersionResolver.ResolvePacketVersion(0, 4).GetEncodedValue());
    // This functor will receive the counter data.
    PeriodicCounterCaptureCommandHandler counterCaptureCommandHandler(
            3, 0, packetVersionResolver.ResolvePacketVersion(3, 0).GetEncodedValue());

    profiling::DirectoryCaptureCommandHandler directoryCaptureCommandHandler(
            0, 2, packetVersionResolver.ResolvePacketVersion(0, 2).GetEncodedValue(), false);

    timelinedecoder::TimelineCaptureCommandHandler timelineCaptureCommandHandler(
            1, 1, packetVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), timelineDecoder);

    timelinedecoder::TimelineDirectoryCaptureCommandHandler timelineDirectoryCaptureCommandHandler(
            1, 0, packetVersionResolver.ResolvePacketVersion(1, 0).GetEncodedValue(),
            timelineCaptureCommandHandler, false);

    // Register different derived functors
    registry.RegisterFunctor(&periodicCounterSelectionResponseHandler);
    registry.RegisterFunctor(&counterCaptureCommandHandler);
    registry.RegisterFunctor(&directoryCaptureCommandHandler);
    registry.RegisterFunctor(&timelineDirectoryCaptureCommandHandler);
    registry.RegisterFunctor(&timelineCaptureCommandHandler);

    GatordMockService mockService(clientConnection, registry, isEchoEnabled);

    // Send receive the strweam metadata and send connection ack.
    if (!mockService.WaitForStreamMetaData())
    {
        return EXIT_FAILURE;
    }
    mockService.SendConnectionAck();

    // Prepare to receive data.
    mockService.LaunchReceivingThread();

    // Process the SET and WAIT command from the file.
    CommandFileParser commandLineParser;
    commandLineParser.ParseFile(commandFile, mockService);

    // Once we've finished processing the file wait for the receiving thread to close.
    mockService.WaitForReceivingThread();

    if(isEchoEnabled)
    {
        timelineDecoder.print();
    }

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    // We need to capture ctrl-c so we can close any remaining sockets before exit
    signal(SIGINT, exit_capture);

    // Process command line arguments
    CommandLineProcessor cmdLine;
    if (!cmdLine.ProcessCommandLine(argc, argv))
    {
        return EXIT_FAILURE;
    }

    std::vector<std::thread> threads;
    std::string commandFile = cmdLine.GetCommandFile();

    armnnUtils::Sockets::Initialize();
    armnnUtils::Sockets::Socket listeningSocket = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);

    if (!GatordMockService::OpenListeningSocket(listeningSocket, cmdLine.GetUdsNamespace(), 10))
    {
        return EXIT_FAILURE;
    }
    std::cout << "Bound to UDS namespace: \\0" << cmdLine.GetUdsNamespace() << std::endl;

    // make the socket non-blocking so we can exit the loop
    armnnUtils::Sockets::SetNonBlocking(listeningSocket);
    while (run)
    {
        armnnUtils::Sockets::Socket clientConnection =
                armnnUtils::Sockets::Accept(listeningSocket, nullptr, nullptr, SOCK_CLOEXEC);

        if (clientConnection > 0)
        {
            threads.emplace_back(
                    std::thread(CreateMockService, clientConnection, commandFile, cmdLine.IsEchoEnabled()));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100u));
    }

    armnnUtils::Sockets::Close(listeningSocket);
    std::for_each(threads.begin(), threads.end(), [](std::thread& t){t.join();});
}