//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <opencv2/opencv.hpp>

#include "IFrameReader.hpp"
#include "CvVideoFrameReader.hpp"

SCENARIO("Read frames from video file using CV frame reader", "[framereader]") {

    GIVEN("a valid video file") {

        std::string testResources = TEST_RESOURCE_DIR;
        REQUIRE(testResources != "");
        std::string file =  testResources + "/" + "Megamind.avi";
        WHEN("Frame reader is initialised") {

            common::CvVideoFrameReader reader;
            THEN("no exception is thrown") {
                reader.Init(file);

                AND_WHEN("when source parameters are read") {

                    auto fps = reader.GetSourceFps();
                    auto height = reader.GetSourceHeight();
                    auto width = reader.GetSourceWidth();
                    auto encoding = reader.GetSourceEncoding();
                    auto framesCount = reader.GetFrameCount();

                    THEN("they are aligned with video file") {

                        REQUIRE(height == 528);
                        REQUIRE(width == 720);
                        REQUIRE(encoding == "XVID");
                        REQUIRE(fps == 23.976);
                        REQUIRE(framesCount == 270);
                    }

                }

                AND_WHEN("frame is read") {
                    auto framePtr = reader.ReadFrame();

                    THEN("it is not a NULL pointer") {
                        REQUIRE(framePtr != nullptr);
                    }

                    AND_THEN("it is not empty") {
                        REQUIRE(!framePtr->empty());
                        REQUIRE(!reader.IsExhausted(framePtr));
                    }
                }

                AND_WHEN("all frames were read from the file") {

                    for (int i = 0; i < 270; i++) {
                        auto framePtr = reader.ReadFrame();
                    }

                    THEN("last + 1 frame is empty") {
                        auto framePtr = reader.ReadFrame();

                        REQUIRE(framePtr->empty());
                        REQUIRE(reader.IsExhausted(framePtr));
                    }

                }

                AND_WHEN("frames are read from the file, pointers point to the different objects") {

                    auto framePtr = reader.ReadFrame();

                    cv::Mat *frame = framePtr.get();

                    for (int i = 0; i < 30; i++) {
                        REQUIRE(frame != reader.ReadFrame().get());
                    }

                }
            }
        }
    }

    GIVEN("an invalid video file") {

        std::string file = "nosuchfile.avi";

        WHEN("Frame reader is initialised") {

            common::CvVideoFrameReader reader;

            THEN("exception is thrown") {
                REQUIRE_THROWS(reader.Init(file));
            }
        }

    }
}