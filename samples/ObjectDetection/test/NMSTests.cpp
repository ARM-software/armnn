//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <catch.hpp>

#include "NonMaxSuppression.hpp"

TEST_CASE("Non_Max_Suppression_1")
{
    // Box with iou exactly 0.5.
    od::DetectedObject detectedObject1;
    detectedObject1.SetLabel("2");
    detectedObject1.SetScore(171);
    detectedObject1.SetBoundingBox({0, 0, 150, 150});

    // Strongest detection.
    od::DetectedObject detectedObject2;
    detectedObject2.SetLabel("2");
    detectedObject2.SetScore(230);
    detectedObject2.SetBoundingBox({0, 75, 150, 75});

    // Weaker detection with same coordinates of strongest.
    od::DetectedObject detectedObject3;
    detectedObject3.SetLabel("2");
    detectedObject3.SetScore(20);
    detectedObject3.SetBoundingBox({0, 75, 150, 75});

    // Detection not overlapping strongest.
    od::DetectedObject detectedObject4;
    detectedObject4.SetLabel("2");
    detectedObject4.SetScore(222);
    detectedObject4.SetBoundingBox({0, 0, 50, 50});

    // Small detection inside strongest.
    od::DetectedObject detectedObject5;
    detectedObject5.SetLabel("2");
    detectedObject5.SetScore(201);
    detectedObject5.SetBoundingBox({100, 100, 20, 20});

    // Box with iou exactly 0.5 but different label.
    od::DetectedObject detectedObject6;
    detectedObject6.SetLabel("1");
    detectedObject6.SetScore(75);
    detectedObject6.SetBoundingBox({0, 0, 150, 150});

    od::DetectedObjects expectedResults {detectedObject1,
        detectedObject2,
        detectedObject3,
        detectedObject4,
        detectedObject5,
        detectedObject6};

    auto sorted = od::NonMaxSuppression(expectedResults, 0.49);

    // 1st and 3rd detection should be suppressed.
    REQUIRE(sorted.size() == 4);

    // Final detects should be ordered strongest to weakest.
    REQUIRE(sorted[0] == 1);
    REQUIRE(sorted[1] == 3);
    REQUIRE(sorted[2] == 4);
    REQUIRE(sorted[3] == 5);
}

TEST_CASE("Non_Max_Suppression_2")
{
    // Real box examples.
    od::DetectedObject detectedObject1;
    detectedObject1.SetLabel("2");
    detectedObject1.SetScore(220);
    detectedObject1.SetBoundingBox({430, 158, 68, 68});

    od::DetectedObject detectedObject2;
    detectedObject2.SetLabel("2");
    detectedObject2.SetScore(171);
    detectedObject2.SetBoundingBox({438, 158, 68, 68});

    od::DetectedObjects expectedResults {detectedObject1,
        detectedObject2};

    auto sorted = od::NonMaxSuppression(expectedResults, 0.5);

    // 2nd detect should be suppressed.
    REQUIRE(sorted.size() == 1);

    // First detect should be strongest and kept.
    REQUIRE(sorted[0] == 0);
}
