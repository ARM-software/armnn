//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CsvReader.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>

#include <fstream>
#include <string>
#include <vector>

using Tokenizer = boost::tokenizer<boost::escaped_list_separator<char>>;

namespace armnnUtils
{

CsvRow ParseLine(const std::string& csvLine)
{
    Tokenizer tokenizer(csvLine);
    CsvRow entry;

    for (const auto &token : tokenizer)
    {
        entry.values.push_back(boost::trim_copy(token));
    }
    return entry;
}

std::vector<CsvRow> CsvReader::ParseFile(const std::string& csvFile)
{
    std::vector<CsvRow> result;

    std::ifstream in(csvFile.c_str());
    if (!in.is_open())
        return result;

    std::string line;
    while (getline(in, line))
    {
        if(!line.empty())
        {
            CsvRow entry = ParseLine(line);
            result.push_back(entry);
        }
    }
    return result;
}

std::vector<CsvRow> CsvReader::ParseVector(const std::vector<std::string>& csvVector)
{
    std::vector<CsvRow> result;

    for (auto const& line: csvVector)
    {
        CsvRow entry = ParseLine(line);
        result.push_back(entry);
    }
    return result;
}
} // namespace armnnUtils