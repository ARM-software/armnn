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
        {"Dog.jpg", 209},
        // top five predictions in tensorflow:
        // -----------------------------------
        // 209:Labrador retriever 0.949995
        // 160:Rhodesian ridgeback 0.0270182
        // 208:golden retriever 0.0192866
        // 853:tennis ball 0.000470382
        // 239:Greater Swiss Mountain dog 0.000464451
        {"Cat.jpg", 283},
        // top five predictions in tensorflow:
        // -----------------------------------
        // 283:tiger cat 0.579016
        // 286:Egyptian cat 0.319676
        // 282:tabby, tabby cat 0.0873346
        // 288:lynx, catamount 0.011163
        // 289:leopard, Panthera pardus 0.000856755
        {"shark.jpg", 3},
        // top five predictions in tensorflow:
        // -----------------------------------
        // 3:great white shark, white shark, ... 0.996926
        // 4:tiger shark, Galeocerdo cuvieri 0.00270528
        // 149:killer whale, killer, orca, ... 0.000121848
        // 395:sturgeon 7.78977e-05
        // 5:hammerhead, hammerhead shark 6.44127e-055
    };

    armnn::TensorShape inputTensorShape({ 1, 224, 224, 3  });
    return armnn::test::ClassifierInferenceTestMain<MobileNetDatabase, armnnTfParser::ITfParser>(
        argc, argv, "mobilenet_v1_1.0_224_fp32.pb", true, "input", "output", { 0, 1, 2 },
        [&imageSet](const char* dataDir) {
            return MobileNetDatabase(
                dataDir,
                224,
                224,
                imageSet);
        },
        &inputTensorShape);
}
