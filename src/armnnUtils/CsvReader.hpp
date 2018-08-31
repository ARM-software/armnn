//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
