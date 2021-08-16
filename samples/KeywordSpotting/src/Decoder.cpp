//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Decoder.hpp"

std::pair<int, float> kws::Decoder::decodeOutput(std::vector<int8_t>& modelOutput) 
{

    std::vector<float> dequantisedOutput;
    //Normalise vector values into new vector
    for (auto& value : modelOutput) 
    {
        float normalisedModelOutput = this->quantisationScale * (static_cast<float >(value) -
                                                                 static_cast<float >(this->quantisationOffset));
        dequantisedOutput.push_back(normalisedModelOutput);
    }

    //Get largest value in modelOutput
    const std::vector<float>::iterator& maxElementIterator = std::max_element(dequantisedOutput.begin(),
                                                                              dequantisedOutput.end());
    //Find the labelMapIndex of the largest value which corresponds to a key in a label map
    int labelMapIndex = static_cast<int>(std::distance(dequantisedOutput.begin(), maxElementIterator));

    //Round to two DP
    float maxModelOutputProbability = std::roundf((*maxElementIterator) * 100) / 100;

    return std::make_pair(labelMapIndex, maxModelOutputProbability);

}




