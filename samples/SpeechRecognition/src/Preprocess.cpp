//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <numeric>
#include <math.h>
#include <string.h>

#include "MathUtils.hpp"
#include "Preprocess.hpp"

Preprocess::Preprocess(
        const uint32_t  windowLen,
        const uint32_t  windowStride,
        const MFCC mfccInst):
        _m_mfcc(mfccInst),
        _m_mfccBuf(mfccInst._m_params.m_numMfccFeatures, mfccInst._m_params.m_numMfccVectors),
        _m_delta1Buf(mfccInst._m_params.m_numMfccFeatures, mfccInst._m_params.m_numMfccVectors),
        _m_delta2Buf(mfccInst._m_params.m_numMfccFeatures, mfccInst._m_params.m_numMfccVectors),
        _m_windowLen(windowLen),
        _m_windowStride(windowStride)
{
    if (mfccInst._m_params.m_numMfccFeatures > 0 && windowLen > 0)
    {
        this->_m_mfcc.Init();
    }
}

Preprocess::~Preprocess()
{
}

bool Preprocess::Invoke( const float*  audioData, const uint32_t  audioDataLen, std::vector<int8_t>& output,
        int quantOffset, float quantScale)
{
    this->_m_window = SlidingWindow<const float>(
            audioData, audioDataLen,
            this->_m_windowLen, this->_m_windowStride);

    uint32_t mfccBufIdx = 0;

    // Init buffers with 0
    std::fill(_m_mfccBuf.begin(), _m_mfccBuf.end(), 0.f);
    std::fill(_m_delta1Buf.begin(), _m_delta1Buf.end(), 0.f);
    std::fill(_m_delta2Buf.begin(), _m_delta2Buf.end(), 0.f);

    /* While we can slide over the window */
    while (this->_m_window.HasNext())
    {
        const float*  mfccWindow = this->_m_window.Next();
        auto mfccAudioData = std::vector<float>(
                mfccWindow,
                mfccWindow + this->_m_windowLen);

        auto mfcc = this->_m_mfcc.MfccCompute(mfccAudioData);
        for (size_t i = 0; i < this->_m_mfccBuf.size(0); ++i)
        {
            this->_m_mfccBuf(i, mfccBufIdx) = mfcc[i];
        }
        ++mfccBufIdx;
    }

    /* Pad MFCC if needed by repeating last feature vector */
    while (mfccBufIdx != this->_m_mfcc._m_params.m_numMfccVectors)
    {
        memcpy(&this->_m_mfccBuf(0, mfccBufIdx),
               &this->_m_mfccBuf(0, mfccBufIdx-1), sizeof(float)*this->_m_mfcc._m_params.m_numMfccFeatures);
        ++mfccBufIdx;
    }

    /* Compute first and second order deltas from MFCCs */
    this->_ComputeDeltas(this->_m_mfccBuf,
                         this->_m_delta1Buf,
                         this->_m_delta2Buf);

    /* Normalise */
    this->_Normalise();

    return this->_Quantise<int8_t>(output.data(), quantOffset, quantScale);
}

bool Preprocess::_ComputeDeltas(Array2d<float>& mfcc,
                                Array2d<float>& delta1,
                                Array2d<float>& delta2)
{
    const std::vector <float> delta1Coeffs =
            {6.66666667e-02,  5.00000000e-02,  3.33333333e-02,
             1.66666667e-02, -3.46944695e-18, -1.66666667e-02,
             -3.33333333e-02, -5.00000000e-02, -6.66666667e-02};

    const std::vector <float> delta2Coeffs =
            {0.06060606,      0.01515152,     -0.01731602,
             -0.03679654,     -0.04329004,     -0.03679654,
             -0.01731602,      0.01515152,      0.06060606};

    if (delta1.size(0) == 0 || delta2.size(0) != delta1.size(0) ||
        mfcc.size(0) == 0 || mfcc.size(1) == 0)
    {
        return false;
    }

    /* Get the middle index; coeff vec len should always be odd */
    const size_t coeffLen = delta1Coeffs.size();
    const size_t fMidIdx = (coeffLen - 1)/2;
    const size_t numFeatures = mfcc.size(0);
    const size_t numFeatVectors = mfcc.size(1);

    /* iterate through features in MFCC vector*/
    for (size_t i = 0; i < numFeatures; ++i)
    {
        /* for each feature, iterate through time (t) samples representing feature evolution and
        * calculate d/dt and d^2/dt^2, using 1d convolution with differential kernels.
        * Convolution padding = valid, result size is `time length - kernel length + 1`.
        * The result is padded with 0 from both sides to match the size of initial time samples data.
        *
        * For the small filter, conv1d implementation as a simple loop is efficient enough.
        * Filters of a greater size would need CMSIS-DSP functions to be used, like arm_fir_f32.
        */

        for (size_t j = fMidIdx; j < numFeatVectors - fMidIdx; ++j)
        {
            float d1 = 0;
            float d2 = 0;
            const size_t mfccStIdx = j - fMidIdx;

            for (size_t k = 0, m = coeffLen - 1; k < coeffLen; ++k, --m)
            {

                d1 +=  mfcc(i,mfccStIdx + k) * delta1Coeffs[m];
                d2 +=  mfcc(i,mfccStIdx + k) * delta2Coeffs[m];
            }

            delta1(i,j) = d1;
            delta2(i,j) = d2;
        }
    }

    return true;
}

float Preprocess::_GetMean(Array2d<float>& vec)
{
    return MathUtils::MeanF32(vec.begin(), vec.totalSize());
}

float Preprocess::_GetStdDev(Array2d<float>& vec, const float mean)
{
    return MathUtils::StdDevF32(vec.begin(), vec.totalSize(), mean);
}

void Preprocess::_NormaliseVec(Array2d<float>& vec)
{
    auto mean = Preprocess::_GetMean(vec);
    auto stddev = Preprocess::_GetStdDev(vec, mean);

    if (stddev == 0)
    {
        std::fill(vec.begin(), vec.end(), 0);
    }
    else
    {
        const float stddevInv = 1.f/stddev;
        const float normalisedMean = mean/stddev;

        auto NormalisingFunction = [=](float &value) {
            value = value * stddevInv - normalisedMean;
        };
        std::for_each(vec.begin(), vec.end(), NormalisingFunction);
    }
}

void Preprocess::_Normalise()
{
    Preprocess::_NormaliseVec(this->_m_mfccBuf);
    Preprocess::_NormaliseVec(this->_m_delta1Buf);
    Preprocess::_NormaliseVec(this->_m_delta2Buf);
}

float Preprocess::_GetQuantElem(
        const float     elem,
        const float     quantScale,
        const int       quantOffset,
        const float     minVal,
        const float     maxVal)
{
    float val = std::round((elem/quantScale) + quantOffset);
    float maxim = std::max<float>(val, minVal);
    float returnVal = std::min<float>(std::max<float>(val, minVal), maxVal);
    return returnVal;
}