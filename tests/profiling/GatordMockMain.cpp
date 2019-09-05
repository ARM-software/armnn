//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"

#include <cstdlib>

int main(int argc, char *argv[])
{
    armnn::gatordmock::CommandLineProcessor cmdline;
    if (!cmdline.ProcessCommandLine(argc, argv))
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
