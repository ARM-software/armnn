//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <string>
#include <map>

/*
 * Checks that a particular option was specified by the user
 */
bool CheckOptionSpecified(const std::map<std::string, std::string>& options, const std::string& option);


/*
 * Retrieves the user provided option
 */
std::string GetSpecifiedOption(const std::map<std::string, std::string>& options, const std::string& option);


/*
 * Parses all the command line options provided by the user and stores in a map.
 */
int ParseOptions(std::map<std::string, std::string>& options, std::map<std::string, std::string>& acceptedOptions,
                 char *argv[], int argc);