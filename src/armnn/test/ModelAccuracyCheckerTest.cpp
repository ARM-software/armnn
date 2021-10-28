//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ModelAccuracyChecker.hpp"
#include <armnnUtils/TContainer.hpp>

#include <doctest/doctest.h>

#include <iostream>
#include <string>

using namespace armnnUtils;

namespace {
struct TestHelper
{
    const std::map<std::string, std::string> GetValidationLabelSet()
    {
        std::map<std::string, std::string> validationLabelSet;
        validationLabelSet.insert(std::make_pair("val_01.JPEG", "goldfinch"));
        validationLabelSet.insert(std::make_pair("val_02.JPEG", "magpie"));
        validationLabelSet.insert(std::make_pair("val_03.JPEG", "brambling"));
        validationLabelSet.insert(std::make_pair("val_04.JPEG", "robin"));
        validationLabelSet.insert(std::make_pair("val_05.JPEG", "indigo bird"));
        validationLabelSet.insert(std::make_pair("val_06.JPEG", "ostrich"));
        validationLabelSet.insert(std::make_pair("val_07.JPEG", "jay"));
        validationLabelSet.insert(std::make_pair("val_08.JPEG", "snowbird"));
        validationLabelSet.insert(std::make_pair("val_09.JPEG", "house finch"));
        validationLabelSet.insert(std::make_pair("val_09.JPEG", "bulbul"));

        return validationLabelSet;
    }
    const std::vector<armnnUtils::LabelCategoryNames> GetModelOutputLabels()
    {
        const std::vector<armnnUtils::LabelCategoryNames> modelOutputLabels =
        {
            {"ostrich", "Struthio camelus"},
            {"brambling", "Fringilla montifringilla"},
            {"goldfinch", "Carduelis carduelis"},
            {"house finch", "linnet", "Carpodacus mexicanus"},
            {"junco", "snowbird"},
            {"indigo bunting", "indigo finch", "indigo bird", "Passerina cyanea"},
            {"robin", "American robin", "Turdus migratorius"},
            {"bulbul"},
            {"jay"},
            {"magpie"}
        };
        return modelOutputLabels;
    }
};
}

TEST_SUITE("ModelAccuracyCheckerTest")
{

TEST_CASE_FIXTURE(TestHelper, "TestFloat32OutputTensorAccuracy")
{
    ModelAccuracyChecker checker(GetValidationLabelSet(), GetModelOutputLabels());

    // Add image 1 and check accuracy
    std::vector<float> inferenceOutputVector1 = {0.05f, 0.10f, 0.70f, 0.15f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    armnnUtils::TContainer inference1Container(inferenceOutputVector1);
    std::vector<armnnUtils::TContainer> outputTensor1;
    outputTensor1.push_back(inference1Container);

    std::string imageName = "val_01.JPEG";
    checker.AddImageResult<armnnUtils::TContainer>(imageName, outputTensor1);

    // Top 1 Accuracy
    float totalAccuracy = checker.GetAccuracy(1);
    CHECK(totalAccuracy == 100.0f);

    // Add image 2 and check accuracy
    std::vector<float> inferenceOutputVector2 = {0.10f, 0.0f, 0.0f, 0.0f, 0.05f, 0.70f, 0.0f, 0.0f, 0.0f, 0.15f};
    armnnUtils::TContainer inference2Container(inferenceOutputVector2);
    std::vector<armnnUtils::TContainer> outputTensor2;
    outputTensor2.push_back(inference2Container);

    imageName = "val_02.JPEG";
    checker.AddImageResult<armnnUtils::TContainer>(imageName, outputTensor2);

    // Top 1 Accuracy
    totalAccuracy = checker.GetAccuracy(1);
    CHECK(totalAccuracy == 50.0f);

    // Top 2 Accuracy
    totalAccuracy = checker.GetAccuracy(2);
    CHECK(totalAccuracy == 100.0f);

    // Add image 3 and check accuracy
    std::vector<float> inferenceOutputVector3 = {0.0f, 0.10f, 0.0f, 0.0f, 0.05f, 0.70f, 0.0f, 0.0f, 0.0f, 0.15f};
    armnnUtils::TContainer inference3Container(inferenceOutputVector3);
    std::vector<armnnUtils::TContainer> outputTensor3;
    outputTensor3.push_back(inference3Container);

    imageName = "val_03.JPEG";
    checker.AddImageResult<armnnUtils::TContainer>(imageName, outputTensor3);

    // Top 1 Accuracy
    totalAccuracy = checker.GetAccuracy(1);
    CHECK(totalAccuracy == 33.3333321f);

    // Top 2 Accuracy
    totalAccuracy = checker.GetAccuracy(2);
    CHECK(totalAccuracy == 66.6666641f);

    // Top 3 Accuracy
    totalAccuracy = checker.GetAccuracy(3);
    CHECK(totalAccuracy == 100.0f);
}

}
