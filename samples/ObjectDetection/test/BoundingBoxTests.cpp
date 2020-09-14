//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <catch.hpp>
#include "BoundingBox.hpp"

namespace
{
    static constexpr unsigned int s_X = 100u;
    static constexpr unsigned int s_Y = 200u;
    static constexpr unsigned int s_W = 300u;
    static constexpr unsigned int s_H = 400u;
} // anonymous namespace

TEST_CASE("BoundingBoxTest_Default")
{
    od::BoundingBox boundingBox;

    REQUIRE(boundingBox.GetX()      == 0u);
    REQUIRE(boundingBox.GetY()      == 0u);
    REQUIRE(boundingBox.GetWidth()  == 0u);
    REQUIRE(boundingBox.GetHeight() == 0u);
}

TEST_CASE("BoundingBoxTest_Custom")
{
    od::BoundingBox boundingBox(s_X, s_Y, s_W, s_H);

    REQUIRE(boundingBox.GetX()      == s_X);
    REQUIRE(boundingBox.GetY()      == s_Y);
    REQUIRE(boundingBox.GetWidth()  == s_W);
    REQUIRE(boundingBox.GetHeight() == s_H);
}

TEST_CASE("BoundingBoxTest_Setters")
{
    od::BoundingBox boundingBox;

    boundingBox.SetX(s_X);
    boundingBox.SetY(s_Y);
    boundingBox.SetWidth(s_W);
    boundingBox.SetHeight(s_H);

    REQUIRE(boundingBox.GetX()      == s_X);
    REQUIRE(boundingBox.GetY()      == s_Y);
    REQUIRE(boundingBox.GetWidth()  == s_W);
    REQUIRE(boundingBox.GetHeight() == s_H);
}

static inline bool AreBoxesEqual(od::BoundingBox& b1, od::BoundingBox& b2)
{
    return (b1.GetX() == b2.GetX() && b1.GetY() == b2.GetY() &&
        b1.GetWidth() == b2.GetWidth() && b1.GetHeight() == b2.GetHeight());
}

TEST_CASE("BoundingBoxTest_GetValidBoundingBox")
{
    od::BoundingBox boxIn { 0,  0, 10, 20 };
    od::BoundingBox boxOut;

    WHEN("Limiting box is completely within the input box")
    {
        od::BoundingBox boxLmt{ 1,  1,  9, 18 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxLmt,boxOut));
    }

    WHEN("Limiting box cuts off the top and left")
    {
        od::BoundingBox boxLmt{ 1,  1, 10, 20 };
        od::BoundingBox boxExp{ 1,  1,  9, 19 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }

    WHEN("Limiting box cuts off the bottom")
    {
        od::BoundingBox boxLmt{ 0,  0, 10, 19 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxLmt, boxOut));
    }

    WHEN("Limiting box cuts off the right")
    {
        od::BoundingBox boxLmt{ 0,  0,  9, 20 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxLmt, boxOut));
    }

    WHEN("Limiting box cuts off the bottom and right")
    {
        od::BoundingBox boxLmt{ 0,  0,  9, 19 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxLmt, boxOut));
    }

    WHEN("Limiting box cuts off the bottom and left")
    {
        od::BoundingBox boxLmt{ 1,  0, 10, 19 };
        od::BoundingBox boxExp{ 1,  0,  9, 19 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }

    WHEN("Limiting box does not impose any limit")
    {
        od::BoundingBox boxLmt{ 0,  0, 10, 20 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxIn, boxOut));
    }

    WHEN("Limiting box zeros out the width")
    {
        od::BoundingBox boxLmt{ 0,  0,  0, 20 };
        od::BoundingBox boxExp{ 0,  0,  0,  0 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }

    WHEN("Limiting box zeros out the height")
    {
        od::BoundingBox boxLmt{ 0,  0, 10,  0 };
        od::BoundingBox boxExp{ 0,  0,  0,  0 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }

    WHEN("Limiting box with negative starts - top and left with 1 sq pixel cut-off")
    {
        od::BoundingBox boxLmt{ -1, -1, 10, 20 };
        od::BoundingBox boxExp{  0,  0,  9, 19 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }

    WHEN("Limiting box with negative starts - top and left with full overlap")
    {
        od::BoundingBox boxLmt{ -1, -1, 11, 21 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxIn, boxOut));
    }

    WHEN("Limiting box with zero overlap")
    {
        od::BoundingBox boxLmt{-10,-20, 10, 20 };
        od::BoundingBox boxExp{  0,  0,  0,  0 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }

    WHEN("Limiting box with one square pixel overlap")
    {
        od::BoundingBox boxLmt{-9,-19, 10, 20 };
        od::BoundingBox boxExp{ 0,  0,  1,  1 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }

    WHEN("Limiting box with unrealistically high values in positive quadrant")
    {
        od::BoundingBox boxLmt{INT32_MAX, INT32_MAX, UINT32_MAX, UINT32_MAX };
        od::BoundingBox boxExp{ 0,  0,  0,  0 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }

    /* This should actually return a valid bounding box, currently not handled. */
    WHEN("Limiting box with unrealistic values spanning 32 bit space")
    {
        od::BoundingBox boxLmt{-(INT32_MAX), -(INT32_MAX), UINT32_MAX, UINT32_MAX};
        od::BoundingBox boxExp{ 0,  0,  0,  0 };
        GetValidBoundingBox(boxIn, boxOut, boxLmt);
        REQUIRE(AreBoxesEqual(boxExp, boxOut));
    }
}