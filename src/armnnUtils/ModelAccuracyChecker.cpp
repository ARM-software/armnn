//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ModelAccuracyChecker.hpp"

#include <armnn/Logging.hpp>

#include <map>
#include <vector>

namespace armnnUtils
{

armnnUtils::ModelAccuracyChecker::ModelAccuracyChecker(const std::map<std::string, std::string>& validationLabels,
                                                       const std::vector<LabelCategoryNames>& modelOutputLabels)
    : m_GroundTruthLabelSet(validationLabels)
    , m_ModelOutputLabels(modelOutputLabels)
{}

float ModelAccuracyChecker::GetAccuracy(unsigned int k)
{
    if (k > 10)
    {
        ARMNN_LOG(warning) << "Accuracy Tool only supports a maximum of Top 10 Accuracy. "
                              "Printing Top 10 Accuracy result!";
        k = 10;
    }
    unsigned int total = 0;
    for (unsigned int i = k; i > 0; --i)
    {
        total += m_TopK[i];
    }
    return static_cast<float>(total * 100) / static_cast<float>(m_ImagesProcessed);
}

// Split a string into tokens by a delimiter
std::vector<std::string>
    SplitBy(const std::string& originalString, const std::string& delimiter, bool includeEmptyToken)
{
    std::vector<std::string> tokens;
    size_t cur  = 0;
    size_t next = 0;
    while ((next = originalString.find(delimiter, cur)) != std::string::npos)
    {
        // Skip empty tokens, unless explicitly stated to include them.
        if (next - cur > 0 || includeEmptyToken)
        {
            tokens.push_back(originalString.substr(cur, next - cur));
        }
        cur = next + delimiter.size();
    }
    // Get the remaining token
    // Skip empty tokens, unless explicitly stated to include them.
    if (originalString.size() - cur > 0 || includeEmptyToken)
    {
        tokens.push_back(originalString.substr(cur, originalString.size() - cur));
    }
    return tokens;
}

// Remove any preceding and trailing character specified in the characterSet.
std::string Strip(const std::string& originalString, const std::string& characterSet)
{
    ARMNN_ASSERT(!characterSet.empty());
    const std::size_t firstFound = originalString.find_first_not_of(characterSet);
    const std::size_t lastFound  = originalString.find_last_not_of(characterSet);
    // Return empty if the originalString is empty or the originalString contains only to-be-striped characters
    if (firstFound == std::string::npos || lastFound == std::string::npos)
    {
        return "";
    }
    return originalString.substr(firstFound, lastFound + 1 - firstFound);
}
}    // namespace armnnUtils