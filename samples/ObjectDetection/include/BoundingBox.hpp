//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace od
{
/**
* @brief Class used to store and receive bounding box location and size information
*
*/
class BoundingBox
{
public:
    /**
    * @brief Default constructor
    */
    BoundingBox();

    /**
    * @brief Constructor with parameters to configure the bounding box dimensions
    * @param[in]  x                       int value representing the x coordinate.
    * @param[in]  y                       int value representing the y coordinate.
    * @param[in]  width                   unsigned int value representing the width value.
    * @param[in]  height                  unsigned int value representing the height value.
    */
    BoundingBox(int x, int y, unsigned int width, unsigned int height);

    /**
    * @brief Constructor with a BoundingBox type parameter to copy from.
    * @param[in]  other                   Bounding box to copy.
    */
    BoundingBox(const BoundingBox& other);

    ~BoundingBox() = default;

    /**
    * @brief Function to retrieve the X coordinate.
    */
    int GetX() const;

    /**
    * @brief Function to retrieve the Y coordinate.
    */
    int GetY() const;

    /**
    * @brief Function to retrieve the width.
    */
    unsigned int GetWidth() const;

    /**
    * @brief Function to retrieve the height.
    */
    unsigned int GetHeight() const;

    /**
    * @brief Function to set the X coordinate.
    * @param[in]  x                      int value representing x coordinate
    */
    void SetX(int x);

    /**
    * @brief Function to set the Y coordinate.
    * @param[in]  y                      int value representing y coordinate
    */
    void SetY(int y);

    /**
    * @brief Function to set the width of the BoundingBox.
    * @param[in]  width                  int value representing the width
    */
    void SetWidth(unsigned int width);

    /**
    * @brief Function to set the height of the BoundingBox.
    * @param[in]  height                 int value representing the height
    */
    void SetHeight(unsigned int height);

    /**
    * @brief Function to check equality with another BoundingBox
    * @param[in]  other                  BoundingBox to compare with
    */
    BoundingBox& operator=(const BoundingBox& other);

private:
    int m_X;
    int m_Y;
    unsigned int m_Width;
    unsigned int m_Height;
};

/*
 * @brief: Get a bounding box within the limits of another bounding box
 *
 * @param[in]   boxIn       Input bounding box
 * @param[out]  boxOut      Output bounding box
 * @param[in]   boxLimits   Bounding box defining the limits which the output
 *                          needs to conform to.
 * @return      none
 */
void GetValidBoundingBox(const BoundingBox& boxIn, BoundingBox& boxOut,
    const BoundingBox& boxLimits);

}// namespace od