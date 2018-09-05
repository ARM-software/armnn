//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <vector>
#include <string>

namespace armnnUtils
{

struct CsvRow
{
    std::vector<std::string> values;
};

class CsvReader
{
public:
    static std::vector<CsvRow> ParseFile(const std::string& csvFile);

    static std::vector<CsvRow> ParseVector(const std::vector<std::string>& csvVector);
};
} // namespace armnnUtils
