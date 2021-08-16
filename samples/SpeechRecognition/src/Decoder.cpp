//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Decoder.hpp"

namespace asr 
{

Decoder::Decoder(std::map<int, std::string>& labels) :
            m_labels(labels) {}

std::string Decoder::FilterCharacters(std::vector<char>& unfiltered) 
{
    std::string filtered;

    for (int i = 0; i < unfiltered.size(); ++i) 
    {
        if (unfiltered.at(i) == '$') 
        {
            continue;
        } 
        else if (i + 1 < unfiltered.size() && unfiltered.at(i) == unfiltered.at(i + 1)) 
        {
            continue;
        } 
        else 
        {
            filtered += unfiltered.at(i);
        }
    }
    return filtered;
}
} // namespace asr

