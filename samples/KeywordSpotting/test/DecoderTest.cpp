//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <catch.hpp>
#include <map>
#include "Decoder.hpp"


TEST_CASE("Test KWS decoder")
{
//    Actual output probability: [0.0, 0.06, 0.02, 0.03, 0.0, 0.0, 0.05, 0.0, 0.83, 0.0, 0.1, 0.0]
//    int8 quantised Model output [1, 4, 2, 3, 1, 1, 3, 1, 43, 1, 6, 1]
//    Reconstructed  dequantised probability [0.0, 0.06, 0.02, 0.04, 0.0, 0.0, 0.04, 0.0, 0.84, 0.0, 0.1, 0.0]

    int quantisationOffset = 1;
    float quantisationScale = 0.02;

    std::vector<int8_t> modelOutput = {1, 4, 2, 3, 1, 1, 3, 1, 43, 1, 6, 1};

    kws::Decoder decoder(quantisationOffset,quantisationScale);

    std::pair<int,float> result =  decoder.decodeOutput(modelOutput);


    CHECK(result == std::pair<int,float>(8,0.84));
}
