//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandFileParser.hpp"
#include "CommandLineProcessor.hpp"
#include "GatordMockService.hpp"

#include <server/include/basePipeServer/ConnectionHandler.hpp>

#include <string>
#include <signal.h>

using namespace armnn;
using namespace gatordmock;

// Used to capture ctrl-c so we can close any remaining sockets before exit
static volatile bool run = true;
void exit_capture(int signum)
{
    arm::pipe::IgnoreUnused(signum);
    run = false;
}

bool CreateMockService(std::unique_ptr<arm::pipe::BasePipeServer> basePipeServer,
                       std::string commandFile,
                       bool isEchoEnabled)
{
    GatordMockService mockService(std::move(basePipeServer), isEchoEnabled);

    // Send receive the stream metadata and send connection ack.
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

    // make the socket non-blocking so we can exit the loop
    arm::pipe::ConnectionHandler connectionHandler(cmdLine.GetUdsNamespace(), true);

    while (run)
    {
        auto basePipeServer = connectionHandler.GetNewBasePipeServer(cmdLine.IsEchoEnabled());

        if (basePipeServer != nullptr)
        {
            threads.emplace_back(
                    std::thread(CreateMockService, std::move(basePipeServer), commandFile, cmdLine.IsEchoEnabled()));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100u));
    }

    std::for_each(threads.begin(), threads.end(), [](std::thread& t){t.join();});
}