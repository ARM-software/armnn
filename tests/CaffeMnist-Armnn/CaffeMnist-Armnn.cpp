//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../MnistDatabase.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    return armnn::test::ClassifierInferenceTestMain<MnistDatabase, armnnCaffeParser::ICaffeParser>(
        argc, argv, "lenet_iter_9000.caffemodel", true, "data", "prob",
        { 0, 1, 5, 8, 9 },
        [](const char* dataDir) { return MnistDatabase(dataDir); });
}
