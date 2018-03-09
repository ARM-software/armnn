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
        {"ILSVRC2012_val_00000018.JPEG",  21 },
        {"shark.jpg", 2}
    };

    armnn::TensorShape inputTensorShape({ 1, 3, 224, 224 });
    return armnn::test::ClassifierInferenceTestMain<ImageNetDatabase, armnnCaffeParser::ICaffeParser>(
        argc, argv, "ResNet_50_ilsvrc15_model.caffemodel", true,
        "data", "prob", { 0, 1 },
        [&imageSet](const char* dataDir) { return ImageNetDatabase(dataDir, 224, 224, imageSet); },
        &inputTensorShape);
}
