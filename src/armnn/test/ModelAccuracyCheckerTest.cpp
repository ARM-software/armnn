//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ModelAccuracyChecker.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <string>
#include <boost/log/core/core.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

using namespace armnnUtils;

struct TestHelper {
    const std::map<std::string, int> GetValidationLabelSet()
    {
        std::map<std::string, int> validationLabelSet;
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000001", 2));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000002", 9));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000003", 1));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000004", 6));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000005", 5));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000006", 0));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000007", 8));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000008", 4));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000009", 3));
        validationLabelSet.insert( std::make_pair("ILSVRC2012_val_00000009", 7));

        return validationLabelSet;
    }
};

BOOST_AUTO_TEST_SUITE(ModelAccuracyCheckerTest)

using TContainer = boost::variant<std::vector<float>, std::vector<int>, std::vector<unsigned char>>;

BOOST_FIXTURE_TEST_CASE(TestFloat32OutputTensorAccuracy, TestHelper)
{
    ModelAccuracyChecker checker(GetValidationLabelSet());

    // Add image 1 and check accuracy
    std::vector<float> inferenceOutputVector1 = {0.05f, 0.10f, 0.70f, 0.15f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    TContainer inference1Container(inferenceOutputVector1);
    std::vector<TContainer> outputTensor1;
    outputTensor1.push_back(inference1Container);

    std::string imageName = "ILSVRC2012_val_00000001.JPEG";
    checker.AddImageResult<TContainer>(imageName, outputTensor1);

    // Top 1 Accuracy
    float totalAccuracy = checker.GetAccuracy(1);
    BOOST_CHECK(totalAccuracy == 100.0f);

    // Add image 2 and check accuracy
    std::vector<float> inferenceOutputVector2 = {0.10f, 0.0f, 0.0f, 0.0f, 0.05f, 0.70f, 0.0f, 0.0f, 0.0f, 0.15f};
    TContainer inference2Container(inferenceOutputVector2);
    std::vector<TContainer> outputTensor2;
    outputTensor2.push_back(inference2Container);

    imageName = "ILSVRC2012_val_00000002.JPEG";
    checker.AddImageResult<TContainer>(imageName, outputTensor2);

    // Top 1 Accuracy
    totalAccuracy = checker.GetAccuracy(1);
    BOOST_CHECK(totalAccuracy == 50.0f);

    // Top 2 Accuracy
    totalAccuracy = checker.GetAccuracy(2);
    BOOST_CHECK(totalAccuracy == 100.0f);

    // Add image 3 and check accuracy
    std::vector<float> inferenceOutputVector3 = {0.0f, 0.10f, 0.0f, 0.0f, 0.05f, 0.70f, 0.0f, 0.0f, 0.0f, 0.15f};
    TContainer inference3Container(inferenceOutputVector3);
    std::vector<TContainer> outputTensor3;
    outputTensor3.push_back(inference3Container);

    imageName = "ILSVRC2012_val_00000003.JPEG";
    checker.AddImageResult<TContainer>(imageName, outputTensor3);

    // Top 1 Accuracy
    totalAccuracy = checker.GetAccuracy(1);
    BOOST_CHECK(totalAccuracy == 33.3333321f);

    // Top 2 Accuracy
    totalAccuracy = checker.GetAccuracy(2);
    BOOST_CHECK(totalAccuracy == 66.6666641f);

    // Top 3 Accuracy
    totalAccuracy = checker.GetAccuracy(3);
    BOOST_CHECK(totalAccuracy == 100.0f);
}

BOOST_AUTO_TEST_SUITE_END()
