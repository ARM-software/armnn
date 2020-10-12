//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cxxopts/cxxopts.hpp>

/**
 * Ensure all mandatory command-line parameters have been passed to cxxopts.
 * @param result returned from the cxxopts parse(argc, argv) call
 * @param required vector of strings listing the mandatory parameters to be input from the command-line
 * @return boolean value - true if all required parameters satisfied, false otherwise
 * */
inline bool CheckRequiredOptions(const cxxopts::ParseResult& result, const std::vector<std::string>& required)
{
    for(const std::string& str : required)
    {
        if(result.count(str) == 0)
        {
            std::cerr << "--" << str << " parameter is mandatory" << std::endl;
            return false;
        }
    }
    return true;
}
