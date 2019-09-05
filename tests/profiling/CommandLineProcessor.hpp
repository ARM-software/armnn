//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

namespace armnn
{

namespace gatordmock
{

// Parses the command line to extract:
//

class CommandLineProcessor
{
public:
    bool ProcessCommandLine(int argc, char *argv[]);

};

} // namespace gatordmock

} // namespace armnn
