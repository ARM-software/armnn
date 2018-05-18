//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../ImageNetDatabase.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    int retVal = EXIT_FAILURE;
    try
    {
        // Coverity fix: The following code may throw an exception of type std::length_error.
        std::vector<ImageSet> imageSet =
        {
            {"shark.jpg", 3694}
        };

        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<ImageNetDatabase, armnnCaffeParser::ICaffeParser>(
                    argc, argv, "Inception-BN-batchsize1.caffemodel", true,
                    "data", "softmax", { 0 },
                    [&imageSet](const char* dataDir) { return ImageNetDatabase(dataDir, 224, 224, imageSet); });
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: CaffeInception_BN-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
