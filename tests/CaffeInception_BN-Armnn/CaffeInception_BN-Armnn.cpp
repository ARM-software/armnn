//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../ImageNetDatabase.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    std::vector<ImageSet> imageSet =
    {
        {"shark.jpg", 3694}
    };
    return armnn::test::ClassifierInferenceTestMain<ImageNetDatabase, armnnCaffeParser::ICaffeParser>(
        argc, argv, "Inception-BN-batchsize1.caffemodel", true,
        "data", "softmax", { 0 },
        [&imageSet](const char* dataDir) { return ImageNetDatabase(dataDir, 224, 224, imageSet); });
}
