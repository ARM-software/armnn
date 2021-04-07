//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CmdArgsParser.hpp"
#include <iostream>
/*
 * Checks that a particular option was specified by the user
 */
bool CheckOptionSpecified(const std::map<std::string, std::string>& options, const std::string& option)
{
    auto it = options.find(option);
    return it!=options.end();
}

/*
 * Retrieves the user provided option
 */
std::string GetSpecifiedOption(const std::map<std::string, std::string>& options, const std::string& option)
{
    if (CheckOptionSpecified(options, option)){
        return options.at(option);
    }
    else
    {
        throw std::invalid_argument("Required option: " + option + " not defined.");
    }
}

/*
 * Parses all the command line options provided by the user and stores in a map.
 */
int ParseOptions(std::map<std::string, std::string>& options, std::map<std::string, std::string>& acceptedOptions,
                 char *argv[], int argc)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string currentOption = std::string(argv[i]);
        auto it = acceptedOptions.find(currentOption);
        if (it != acceptedOptions.end())
        {
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
            {
                std::string value = argv[++i];
                options.insert({it->first, value});
            }
            else if (std::string(argv[i]) == "HELP")
            {
                std::cout << "Available options" << std::endl;
                for (auto & acceptedOption : acceptedOptions)
                {
                    std::cout << acceptedOption.first << " : " << acceptedOption.second << std::endl;
                }
                return 2;
            }
            else
            {
                std::cerr << std::string(argv[i]) << " option requires one argument." << std::endl;
                return 1;
            }
        }
        else
        {
            std::cerr << "Unrecognised option: " << std::string(argv[i]) << std::endl;
            return 1;
        }
    }
    return 0;
}
