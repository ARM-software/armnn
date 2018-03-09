//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../ImageNetDatabase.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    armnn::TensorShape inputTensorShape({ 1, 3, 224, 224 });
    return armnn::test::ClassifierInferenceTestMain<ImageNetDatabase, armnnCaffeParser::ICaffeParser>(
        argc, argv, "VGG_CNN_S.caffemodel", true,
        "input", "prob", { 0 },
        [](const char* dataDir) { return ImageNetDatabase(dataDir, 224, 224); },
        &inputTensorShape);
}
