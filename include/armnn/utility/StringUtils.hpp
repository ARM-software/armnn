//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <armnn/Exceptions.hpp>

namespace armnn
{

namespace stringUtils
{

/// Function to take a string and a list of delimiters and split the string into tokens based on those delimiters
/// This assumes that tokens are also to be split by newlines
/// Enabling tokenCompression merges adjacent delimiters together, preventing empty tokens
inline std::vector<std::string> StringTokenizer(const std::string& str,
                                                const char* delimiters,
                                                bool tokenCompression = true)
{
    std::stringstream stringStream(str);
    std::string line;
    std::vector<std::string> tokenVector;
    while (std::getline(stringStream, line))
    {
        std::size_t prev = 0;
        std::size_t pos;
        while ((pos = line.find_first_of(delimiters, prev)) != std::string::npos)
        {
            // Ignore adjacent tokens
            if (pos > prev)
            {
                tokenVector.push_back(line.substr(prev, pos - prev));
            }
            // Unless token compression is disabled
            else if (!tokenCompression)
            {
                tokenVector.push_back(line.substr(prev, pos - prev));
            }
            prev = pos + 1;
        }
        if (prev < line.length())
        {
            tokenVector.push_back(line.substr(prev, std::string::npos));
        }
    }
    return tokenVector;
}

// Set of 3 utility functions for trimming std::strings
// Default char set for common whitespace characters

///
/// Trim from the start of a string
///
inline std::string& StringStartTrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

///
/// Trim for the end of a string
///
inline std::string& StringEndTrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

///
/// Trim from both the start and the end of a string
///
inline std::string& StringTrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    return StringStartTrim(StringEndTrim(str, chars), chars);
}

///
/// Trim from both the start and the end of a string, returns a trimmed copy of the string
///
inline std::string StringTrimCopy(const std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    std::string strCopy = str;
    return StringStartTrim(StringEndTrim(strCopy, chars), chars);
}

/// Takes a vector of strings and concatenates them together into one long std::string with an optional
/// seperator between each.
inline std::string StringConcat(const std::vector<std::string>& strings, std::string seperator = "")
{
    std::stringstream ss;
    for (auto string : strings)
    {
        ss << string << seperator;
    }
    return ss.str();
}

///
/// Iterates over a given str and replaces all instance of substring oldStr with newStr
///
inline void StringReplaceAll(std::string& str,
                             const std::string& oldStr,
                             const std::string& newStr)
{
    std::string::size_type pos = 0u;
    while ((pos = str.find(oldStr, pos)) != std::string::npos)
    {
        str.replace(pos, oldStr.length(), newStr);
        pos += newStr.length();
    }
}

///
/// Converts a string to bool.
/// Accepts "true", "false" (case-insensitive) and numbers, 1 (true) or 0 (false).
///
/// \param s               String to convert to bool
/// \param throw_on_error  Bool variable to suppress error if conversion failed (Will return false in that case)
/// \return bool value
///
inline bool StringToBool(const std::string& s, bool throw_on_error = true)
{
    // in case of failure to convert returns false
    auto result = false;

    // isstringstream fails if parsing didn't work
    std::istringstream is(s);

    // try integer conversion first. For the case s is a number
    is >> result;

    if (is.fail())
    {
        // transform to lower case to make case-insensitive
        std::string s_lower = s;
        std::transform(s_lower.begin(),
                       s_lower.end(),
                       s_lower.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        is.str(s_lower);
        // try boolean -> s="false" or "true"
        is.clear();
        is >> std::boolalpha >> result;
    }

    if (is.fail() && throw_on_error)
    {
        throw armnn::InvalidArgumentException(s + " is not convertable to bool");
    }

    return result;
}

} // namespace stringUtils

} // namespace armnn