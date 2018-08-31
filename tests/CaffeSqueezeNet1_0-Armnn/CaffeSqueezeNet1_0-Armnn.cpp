//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../CaffePreprocessor.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    return armnn::test::ClassifierInferenceTestMain<CaffePreprocessor, armnnCaffeParser::ICaffeParser>(
        argc, argv, "squeezenet.caffemodel", true,
        "data", "output", { 0 },
        [](const char* dataDir) { return CaffePreprocessor(dataDir); });
}
