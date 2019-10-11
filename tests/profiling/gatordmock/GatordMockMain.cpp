//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandFileParser.hpp"
#include "CommandLineProcessor.hpp"
#include "GatordMockService.hpp"
#include "MockUtils.hpp"
#include "PeriodicCounterCaptureCommandHandler.hpp"

#include <string>

int main(int argc, char *argv[])
{
    // Process command line arguments
    armnn::gatordmock::CommandLineProcessor cmdLine;
    if (!cmdLine.ProcessCommandLine(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // Initialise functors and register into the CommandHandlerRegistry
    uint32_t version = 1;

    // Create headers
    uint32_t counterCaptureCommandHeader = armnn::gatordmock::ConstructHeader(1,0,0);

    // Create the Command Handler Registry
    armnn::profiling::CommandHandlerRegistry registry;

    // Update with derived functors
    armnn::gatordmock::PeriodicCounterCaptureCommandHandler counterCaptureCommandHandler(counterCaptureCommandHeader,
                                                                                         version,
                                                                                         cmdLine.IsEchoEnabled());

    // Register different derived functors
    registry.RegisterFunctor(&counterCaptureCommandHandler);

    armnn::gatordmock::GatordMockService mockService(registry, cmdLine.IsEchoEnabled());

    if (!mockService.OpenListeningSocket(cmdLine.GetUdsNamespace()))
    {
        return EXIT_FAILURE;
    }

    // Wait for a single connection.
    if (-1 == mockService.BlockForOneClient())
    {
        return EXIT_FAILURE;
    }

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
