//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../MobileNetDatabase.hpp"
#include "armnnTfParser/ITfParser.hpp"

int main(int argc, char* argv[])
{
    std::vector<ImageSet> imageSet =
    {
        { "Dog.jpg", 208 },
        { "Cat.jpg", 283 },
        { "shark.jpg", 3 },
    };
    armnn::TensorShape inputTensorShape({ 1, 299, 299, 3 });
    return armnn::test::ClassifierInferenceTestMain<MobileNetDatabase, armnnTfParser::ITfParser>(
        argc, argv, "inception_v3_2016_08_28_frozen_transformed.pb", true,
        "input", "InceptionV3/Predictions/Reshape_1", { 0, 1, 2, },
        [&imageSet](const char* dataDir) { return MobileNetDatabase(dataDir, 299, 299, imageSet); },
        &inputTensorShape);
}
