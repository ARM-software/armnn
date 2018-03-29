//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../Cifar10Database.hpp"
#include "armnnTfParser/ITfParser.hpp"

int main(int argc, char* argv[])
{
    armnn::TensorShape inputTensorShape({ 1, 32, 32, 3 });
    return armnn::test::ClassifierInferenceTestMain<Cifar10Database, armnnTfParser::ITfParser>(
        argc, argv, "cifar10_tf.prototxt", false,
        "data", "prob", { 0, 1, 2, 4, 7 },
        [](const char* dataDir) { return Cifar10Database(dataDir, true); },
        &inputTensorShape);
}
