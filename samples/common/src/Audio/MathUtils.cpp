//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MathUtils.hpp"
#include <vector>
#include <cmath>
#include <cstdio>

void MathUtils::FftF32(std::vector<float>& input,
                       std::vector<float>& fftOutput)
{
    const int inputLength = input.size();

    for (int k = 0; k <= inputLength / 2; k++)
    {
        float sumReal = 0, sumImag = 0;

        for (int t = 0; t < inputLength; t++)
        {
            float angle = 2 * M_PI * t * k / inputLength;
            sumReal += input[t] * cosf(angle);
            sumImag += -input[t] * sinf(angle);
        }

        /* Arrange output to [real0, realN/2, real1, im1, real2, im2, ...] */
        if (k == 0)
        {
            fftOutput[0] = sumReal;
        }
        else if (k == inputLength / 2)
        {
            fftOutput[1] = sumReal;
        }
        else
        {
            fftOutput[k*2] = sumReal;
            fftOutput[k*2 + 1] = sumImag;
        };
    }
}

float MathUtils::DotProductF32(const float* srcPtrA, float* srcPtrB,
                               const int srcLen)
{
    float output = 0.f;

    for (int i = 0; i < srcLen; ++i)
    {
        output += *srcPtrA++ * *srcPtrB++;
    }
    return output;
}

bool MathUtils::ComplexMagnitudeSquaredF32(const float* ptrSrc,
                                           int srcLen,
                                           float* ptrDst,
                                           int dstLen)
{
    if (dstLen < srcLen/2)
    {
        printf("dstLen must be greater than srcLen/2");
        return false;
    }

    for (int j = 0; j < dstLen; ++j)
    {
        const float real = *ptrSrc++;
        const float im = *ptrSrc++;
        *ptrDst++ = real*real + im*im;
    }
    return true;
}

void MathUtils::VecLogarithmF32(std::vector <float>& input,
                                std::vector <float>& output)
{
    for (auto in = input.begin(), out = output.begin();
         in != input.end(); ++in, ++out)
    {
        *out = logf(*in);
    }
}

float MathUtils::MeanF32(const float* ptrSrc, const uint32_t srcLen)
{
    if (!srcLen)
    {
        return 0.f;
    }

    float acc = std::accumulate(ptrSrc, ptrSrc + srcLen, 0.0);
    return acc/srcLen;
}

float MathUtils::StdDevF32(const float* ptrSrc, uint32_t srcLen, float mean)
{
    if (!srcLen)
    {
        return 0.f;
    }
    auto VarianceFunction = [mean, srcLen](float acc, const float value) {
        return acc + (((value - mean) * (value - mean))/ srcLen);
    };

    float acc = std::accumulate(ptrSrc, ptrSrc + srcLen, 0.0,
                                VarianceFunction);
    return sqrtf(acc);
}

