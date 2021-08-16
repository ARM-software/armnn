//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <limits>

#include "AudioCapture.hpp"

TEST_CASE("Test capture of audio file")
{
    std::string testResources = TEST_RESOURCE_DIR;
    REQUIRE(testResources != "");
    std::string file =  testResources + "/" + "myVoiceIsMyPassportVerifyMe04.wav";
    audio::AudioCapture capture;
    std::vector<float> audioData = capture.LoadAudioFile(file);
    capture.InitSlidingWindow(audioData.data(), audioData.size(), 47712, 16000);

    std::vector<float> firstAudioBlock = capture.Next();
    float actual1 = firstAudioBlock.at(0);
    float actual2 = firstAudioBlock.at(47000);
    CHECK(std::to_string(actual1) == "0.000352");
    CHECK(std::to_string(actual2) == "-0.056441");
    CHECK(firstAudioBlock.size() == 47712);

    CHECK(capture.HasNext() == true);

    std::vector<float> secondAudioBlock = capture.Next();
    float actual3 = secondAudioBlock.at(0);
    float actual4 = secondAudioBlock.at(47000);
    CHECK(std::to_string(actual3) == "0.102077");
    CHECK(std::to_string(actual4) == "0.000194");
    CHECK(capture.HasNext() == true);

    std::vector<float> thirdAudioBlock = capture.Next();
    float actual5 = thirdAudioBlock.at(0);
    float actual6 = thirdAudioBlock.at(33500);
    float actual7 = thirdAudioBlock.at(33600);
    CHECK(std::to_string(actual5) == "-0.076416");
    CHECK(std::to_string(actual6) == "-0.000275");
    CHECK(std::to_string(actual7) == "0.000000");
    CHECK(capture.HasNext() == false);
}

TEST_CASE("Test sliding window of audio capture")
{
    std::string testResources = TEST_RESOURCE_DIR;
    REQUIRE(testResources != "");
    std::string file =  testResources + "/" + "myVoiceIsMyPassportVerifyMe04.wav";
    audio::AudioCapture capture;
    std::vector<float> audioData = capture.LoadAudioFile(file);
    capture.InitSlidingWindow(audioData.data(), audioData.size(), 47712, 16000);
    capture.Next();
    capture.Next();

    CHECK(capture.HasNext() == true);
    capture.Next();
    CHECK(capture.HasNext() == false);
}
