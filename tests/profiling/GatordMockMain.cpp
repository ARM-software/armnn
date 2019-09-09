//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"
#include "GatordMockService.hpp"

int main(int argc, char *argv[])
{
    armnn::gatordmock::CommandLineProcessor cmdline;
    if (!cmdline.ProcessCommandLine(argc, argv))
    {
        return EXIT_FAILURE;
    }
    armnn::gatordmock::GatordMockService mockService;
    if (!mockService.OpenListeningSocket(cmdline.GetUdsNamespace()))
    {
        return EXIT_FAILURE;
    }
    int clientFd = mockService.BlockForOneClient();
    if (-1 == clientFd)
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
