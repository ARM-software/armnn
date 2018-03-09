//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../Cifar10Database.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    return armnn::test::ClassifierInferenceTestMain<Cifar10Database, armnnCaffeParser::ICaffeParser>(
        argc, argv, "cifar10_full_iter_60000.caffemodel", true, "data", "prob",
        { 0, 1, 2, 4, 7 },
        [](const char* dataDir) { return Cifar10Database(dataDir); });
}
