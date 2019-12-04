//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../../../src/profiling/PacketVersionResolver.hpp"
#include "../../../src/profiling/PeriodicCounterSelectionCommandHandler.hpp"
#include "CommandFileParser.hpp"
#include "CommandLineProcessor.hpp"
#include "DirectoryCaptureCommandHandler.hpp"
#include "GatordMockService.hpp"
#include "PeriodicCounterCaptureCommandHandler.hpp"
#include "PeriodicCounterSelectionResponseHandler.hpp"

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    // Process command line arguments
    armnn::gatordmock::CommandLineProcessor cmdLine;
    if (!cmdLine.ProcessCommandLine(argc, argv))
    {
        return EXIT_FAILURE;
    }

    armnn::profiling::PacketVersionResolver packetVersionResolver;
    // Create the Command Handler Registry
    armnn::profiling::CommandHandlerRegistry registry;

    // This functor will receive back the selection response packet.
    armnn::gatordmock::PeriodicCounterSelectionResponseHandler periodicCounterSelectionResponseHandler(
        0, 4, packetVersionResolver.ResolvePacketVersion(0, 4).GetEncodedValue());
    // This functor will receive the counter data.
    armnn::gatordmock::PeriodicCounterCaptureCommandHandler counterCaptureCommandHandler(
        3, 0, packetVersionResolver.ResolvePacketVersion(3, 0).GetEncodedValue());

    armnn::profiling::DirectoryCaptureCommandHandler directoryCaptureCommandHandler(
        0, 2, packetVersionResolver.ResolvePacketVersion(0, 2).GetEncodedValue(), false);

    // Register different derived functors
    registry.RegisterFunctor(&periodicCounterSelectionResponseHandler);
    registry.RegisterFunctor(&counterCaptureCommandHandler);
    registry.RegisterFunctor(&directoryCaptureCommandHandler);

    armnn::gatordmock::GatordMockService mockService(registry, cmdLine.IsEchoEnabled());

    if (!mockService.OpenListeningSocket(cmdLine.GetUdsNamespace()))
    {
        return EXIT_FAILURE;
    }
    std::cout << "Bound to UDS namespace: \\0" << cmdLine.GetUdsNamespace() << std::endl;

    // Wait for a single connection.
    if (-1 == mockService.BlockForOneClient())
    {
        return EXIT_FAILURE;
    }
    std::cout << "Client connection established." << std::endl;

    // Send receive the strweam metadata and send connection ack.
    if (!mockService.WaitForStreamMetaData())
    {
        return EXIT_FAILURE;
    }
    mockService.SendConnectionAck();

    // Prepare to receive data.
    mockService.LaunchReceivingThread();

    // Process the SET and WAIT command from the file.
    armnn::gatordmock::CommandFileParser commandLineParser;
    commandLineParser.ParseFile(cmdLine.GetCommandFile(), mockService);

    // Once we've finished processing the file wait for the receiving thread to close.
    mockService.WaitForReceivingThread();

    return EXIT_SUCCESS;
}
