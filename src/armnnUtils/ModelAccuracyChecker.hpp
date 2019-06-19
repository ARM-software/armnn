//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <string>
#include <map>
#include <vector>
#include <boost/variant/apply_visitor.hpp>
#include <iostream>
#include <armnn/Types.hpp>
#include <functional>
#include <algorithm>

namespace armnnUtils
{

using namespace armnn;

class ModelAccuracyChecker
{
public:
    ModelAccuracyChecker(const std::map<std::string, int>& validationLabelSet);

    float GetAccuracy(unsigned int k);

    template<typename TContainer>
    void AddImageResult(const std::string& imageName, std::vector<TContainer> outputTensor)
    {
        // Increment the total number of images processed
        ++m_ImagesProcessed;

        std::map<int, float> confidenceMap;
        auto & output = outputTensor[0];

        // Create a map of all predictions
        boost::apply_visitor([&](auto && value)
                             {
                                 int index = 0;
                                 for (const auto & o : value)
                                 {
                                     if (o > 0)
                                     {
                                         confidenceMap.insert(std::pair<int, float>(index, static_cast<float>(o)));
                                     }
                                     ++index;
                                 }
                             },
                             output);

        // Create a comparator for sorting the map in order of highest probability
        typedef std::function<bool(std::pair<int, float>, std::pair<int, float>)> Comparator;

        Comparator compFunctor =
            [](std::pair<int, float> element1, std::pair<int, float> element2)
            {
                return element1.second > element2.second;
            };

        // Do the sorting and store in an ordered set
        std::set<std::pair<int, float>, Comparator> setOfPredictions(
            confidenceMap.begin(), confidenceMap.end(), compFunctor);

        std::string trimmedName = GetTrimmedImageName(imageName);
        int value = m_GroundTruthLabelSet.find(trimmedName)->second;

        unsigned int index = 1;
        for (std::pair<int, float> element : setOfPredictions)
        {
            if (index >= m_TopK.size())
            {
                break;
            }
            if (element.first == value)
            {
                ++m_TopK[index];
                break;
            }
            ++index;
        }
    }

    std::string GetTrimmedImageName(const std::string& imageName) const
    {
        std::string trimmedName;
        size_t lastindex = imageName.find_last_of(".");
        if(lastindex != std::string::npos)
        {
            trimmedName = imageName.substr(0, lastindex);
        } else
        {
            trimmedName = imageName;
        }
        return trimmedName;
    }

private:
    const std::map<std::string, int> m_GroundTruthLabelSet;
    std::vector<unsigned int> m_TopK = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned int m_ImagesProcessed = 0;
};
} //namespace armnnUtils

