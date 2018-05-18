//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../Cifar10Database.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    int retVal = EXIT_FAILURE;
    try
    {
        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<Cifar10Database, armnnCaffeParser::ICaffeParser>(
                    argc, argv, "cifar10_full_iter_60000.caffemodel", true, "data", "prob",
                    { 0, 1, 2, 4, 7 },
                    [](const char* dataDir) { return Cifar10Database(dataDir); });
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: CaffeCifar10AcrossChannels-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
