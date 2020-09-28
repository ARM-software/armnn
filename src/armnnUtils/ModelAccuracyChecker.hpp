//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <algorithm>
#include <armnn/Types.hpp>
#include <armnn/utility/Assert.hpp>
#include <mapbox/variant.hpp>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace armnnUtils
{

using namespace armnn;

// Category names associated with a label
using LabelCategoryNames = std::vector<std::string>;

/** Split a string into tokens by a delimiter
 *
 * @param[in] originalString    Original string to be split
 * @param[in] delimiter         Delimiter used to split \p originalString
 * @param[in] includeEmptyToekn If true, include empty tokens in the result
 * @return A vector of tokens split from \p originalString by \delimiter
 */
std::vector<std::string>
    SplitBy(const std::string& originalString, const std::string& delimiter = " ", bool includeEmptyToken = false);

/** Remove any preceding and trailing character specified in the characterSet.
 *
 * @param[in] originalString    Original string to be stripped
 * @param[in] characterSet      Set of characters to be stripped from \p originalString
 * @return A string stripped of all characters specified in \p characterSet from \p originalString
 */
std::string Strip(const std::string& originalString, const std::string& characterSet = " ");

class ModelAccuracyChecker
{
public:
    /** Constructor for a model top k accuracy checker
     *
     * @param[in] validationLabelSet Mapping from names of images to be validated, to category names of their
                                     corresponding ground-truth labels.
     * @param[in] modelOutputLabels  Mapping from output nodes to the category names of their corresponding labels
                                     Note that an output node can have multiple category names.
     */
    ModelAccuracyChecker(const std::map<std::string, std::string>& validationLabelSet,
                         const std::vector<LabelCategoryNames>& modelOutputLabels);

    /** Get Top K accuracy
     *
     * @param[in] k The number of top predictions to use for validating the ground-truth label. For example, if \p k is
                    3, then a prediction is considered correct as long as the ground-truth appears in the top 3
                    predictions.
     * @return  The accuracy, according to the top \p k th predictions.
     */
    float GetAccuracy(unsigned int k);

    /** Record the prediction result of an image
     *
     * @param[in] imageName     Name of the image.
     * @param[in] outputTensor  Output tensor of the network running \p imageName.
     */
    template <typename TContainer>
    void AddImageResult(const std::string& imageName, std::vector<TContainer> outputTensor)
    {
        // Increment the total number of images processed
        ++m_ImagesProcessed;

        std::map<int, float> confidenceMap;
        auto& output = outputTensor[0];

        // Create a map of all predictions
        mapbox::util::apply_visitor([&confidenceMap](auto && value)
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

        const std::string correctLabel = m_GroundTruthLabelSet.at(imageName);

        unsigned int index = 1;
        for (std::pair<int, float> element : setOfPredictions)
        {
            if (index >= m_TopK.size())
            {
                break;
            }
            // Check if the ground truth label value is included in the topi prediction.
            // Note that a prediction can have multiple prediction labels.
            const LabelCategoryNames predictionLabels = m_ModelOutputLabels[static_cast<size_t>(element.first)];
            if (std::find(predictionLabels.begin(), predictionLabels.end(), correctLabel) != predictionLabels.end())
            {
                ++m_TopK[index];
                break;
            }
            ++index;
        }
    }

private:
    const std::map<std::string, std::string> m_GroundTruthLabelSet;
    const std::vector<LabelCategoryNames> m_ModelOutputLabels;
    std::vector<unsigned int> m_TopK = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    unsigned int m_ImagesProcessed   = 0;
};
} //namespace armnnUtils

