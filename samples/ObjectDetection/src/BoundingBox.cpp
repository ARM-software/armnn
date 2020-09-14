//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BoundingBox.hpp"
#include <algorithm>
namespace od
{

BoundingBox::BoundingBox() :
        BoundingBox(0, 0, 0u, 0u) {}

BoundingBox::BoundingBox(
        int x,
        int y,
        unsigned int width,
        unsigned int height) :
        m_X(x),
        m_Y(y),
        m_Width(width),
        m_Height(height) {}

BoundingBox::BoundingBox(const BoundingBox& other) :
        m_X(other.m_X),
        m_Y(other.m_Y),
        m_Width(other.m_Width),
        m_Height(other.m_Height) {}

int BoundingBox::GetX() const {
    return m_X;
}

int BoundingBox::GetY() const {
    return m_Y;
}

unsigned int BoundingBox::GetWidth() const {
    return m_Width;
}

unsigned int BoundingBox::GetHeight() const {
    return m_Height;
}

void BoundingBox::SetX(int x) {
    m_X = x;
}

void BoundingBox::SetY(int y) {
    m_Y = y;
}

void BoundingBox::SetWidth(unsigned int width) {
    m_Width = width;
}

void BoundingBox::SetHeight(unsigned int height) {
    m_Height = height;
}

BoundingBox& BoundingBox::operator=(const BoundingBox& other) {
    m_X = other.m_X;
    m_Y = other.m_Y;

    m_Width = other.m_Width;
    m_Height = other.m_Height;

    return *this;
}

/* Helper function to get a "valid" bounding box */
void GetValidBoundingBox(const BoundingBox& boxIn, BoundingBox& boxOut,
                         const BoundingBox& boxLimits) {
    boxOut.SetX(std::max(boxIn.GetX(), boxLimits.GetX()));
    boxOut.SetY(std::max(boxIn.GetY(), boxLimits.GetY()));

    /* If we have changed x and/or y, we compensate by reducing the height and/or width */
    int boxOutWidth = static_cast<int>(boxIn.GetWidth()) -
                      std::max(0, (boxOut.GetX() - boxIn.GetX()));
    int boxOutHeight = static_cast<int>(boxIn.GetHeight()) -
                       std::max(0, (boxOut.GetY() - boxIn.GetY()));

    /* This suggests that there was no overlap on x or/and y axis */
    if (boxOutHeight <= 0 || boxOutWidth <= 0)
    {
        boxOut = BoundingBox{0, 0, 0, 0};
        return;
    }

    const int limitBoxRightX = boxLimits.GetX() + static_cast<int>(boxLimits.GetWidth());
    const int limitBoxRightY = boxLimits.GetY() + static_cast<int>(boxLimits.GetHeight());
    const int boxRightX = boxOut.GetX() + boxOutWidth;
    const int boxRightY = boxOut.GetY() + boxOutHeight;

    if (boxRightX > limitBoxRightX)
    {
        boxOutWidth -= (boxRightX - limitBoxRightX);
    }

    if (boxRightY > limitBoxRightY)
    {
        boxOutHeight -= (boxRightY - limitBoxRightY);
    }

    /* This suggests value has rolled over because of very high numbers, not handled for now */
    if (boxOutHeight <= 0 || boxOutWidth <= 0)
    {
        boxOut = BoundingBox{0, 0, 0, 0};
        return;
    }

    boxOut.SetHeight(static_cast<unsigned int>(boxOutHeight));
    boxOut.SetWidth(static_cast<unsigned int>(boxOutWidth));
}
}// namespace od