//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <vector>
#include <cmath>
#include <cstdint>
#include <numeric>

class MathUtils
{

public:

    /**
     * @brief       Computes the FFT for the input vector
     * @param[in]   input       Floating point vector of input elements
     * @param[out]  fftOutput   Output buffer to be populated by computed
     *                          FFTs
     * @return      none
     */
    static void FftF32(std::vector<float>& input,
                       std::vector<float>& fftOutput);


    /**
     * @brief       Computes the dot product of two 1D floating point
     *              vectors.
     *              result = sum(srcA[0]*srcB[0] + srcA[1]*srcB[1] + ..)
     * @param[in]   srcPtrA     pointer to the first element of first
     *                          array
     * @param[in]   srcPtrB     pointer to the first element of second
     *                          array
     * @param[in]   srcLen      Number of elements in the array/vector
     * @return      dot product
     */
    static float DotProductF32(const float* srcPtrA, float* srcPtrB,
                               int srcLen);

    /**
     * @brief       Computes the squared magnitude of floating point
     *              complex number array.
     * @param[in]   ptrSrc      pointer to the first element of input
     *                          array
     * @param[in]   srcLen      Number of elements in the array/vector
     * @param[out]  ptrDst      Output buffer to be populated
     * @param[in]   dstLen      output buffer len (for sanity check only)
     * @return      true if successful, false otherwise
     */
    static bool ComplexMagnitudeSquaredF32(const float* ptrSrc,
                                           int srcLen,
                                           float* ptrDst,
                                           int dstLen);

    /**
         * @brief       Computes the natural logarithms of input floating point
         *              vector
         * @param[in]   input   Floating point input vector
         * @param[out]  output  Pre-allocated buffer to be populated with
         *                      natural log values of each input element
         * @return      none
         */
    static void VecLogarithmF32(std::vector <float>& input,
                                std::vector <float>& output);

    /**
         * @brief       Gets the mean of a floating point array of elements
         * @param[in]   ptrSrc  pointer to the first element
         * @param[in]   srcLen  Number of elements in the array/vector
         * @return      average value
         */
    static float MeanF32(const float* ptrSrc, uint32_t srcLen);

    /**
     * @brief       Gets the standard deviation of a floating point array
     *              of elements
     * @param[in]   ptrSrc  pointer to the first element
     * @param[in]   srcLen  Number of elements in the array/vector
     * @param[in]   mean    pre-computed mean value
     * @return      standard deviation value
     */
    static float StdDevF32(const float* ptrSrc, uint32_t srcLen,
                           float mean);
};
