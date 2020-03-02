//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandFileParser.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

namespace armnn
{

namespace gatordmock
{

void CommandFileParser::ParseFile(std::string CommandFile, GatordMockService& mockService)
{
    std::ifstream infile(CommandFile);
    std::string line;

    std::cout << "Parsing command file: " << CommandFile << std::endl;

    while (mockService.ReceiveThreadRunning() && std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<std::string> tokens;

        std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
                  std::back_inserter(tokens));
        if (tokens.size() > 0)
        {
            std::string command = tokens[0];
            if (command == "LIST")
            {
                // Expected format for the SET command
                //
                //      LIST
                //

                mockService.SendRequestCounterDir();
            }
            if (command == "SET")
            {
                // Expected format for the SET command
                //
                //      SET 500000 1 2 5 10
                //
                // This breaks down to:
                // SET          command
                // 500000       polling period in micro seconds
                // 1 2 5 10     counter list

                if (tokens.size() > 2) // minimum of 3 tokens.
                {
                    uint32_t period = static_cast<uint32_t>(std::stoul(tokens[1]));

                    std::vector<uint16_t> counters;

                    std::transform(tokens.begin() + 2, tokens.end(), std::back_inserter(counters),
                                   [](const std::string& str)
                                       { return static_cast<uint16_t>(std::stoul(str)); });

                    mockService.SendPeriodicCounterSelectionList(period, counters);
                }
                else
                {
                    std::cerr << "Invalid SET command. Format is: SET <polling period> <id list>" << std::endl;
                }
            }
            else if (command == "WAIT")
            {
                // Expected format for the SET command
                //
                //      WAIT 11000000
                //
                // This breaks down to:
                // WAIT         command
                // 11000000     timeout period in micro seconds
                if (tokens.size() > 1) // minimum of 2 tokens.
                {
                    uint32_t timeout = static_cast<uint32_t>(std::stoul(tokens[1]));
                    mockService.WaitCommand(timeout);
                }
                else
                {
                    std::cerr << "Invalid WAIT command. Format is: WAIT <interval>" << std::endl;
                }
            }
        }
    }
}

}    // namespace gatordmock

}    // namespace armnn
