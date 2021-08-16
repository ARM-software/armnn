//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MathUtils.hpp"
#include <cstring>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <memory>
#include "Wav2LetterPreprocessor.hpp"
#include "Wav2LetterMFCC.hpp"

float Wav2LetterPreprocessor::GetMean(Array2d<float>& vec)
{
    return MathUtils::MeanF32(vec.begin(), vec.totalSize());
}

float Wav2LetterPreprocessor::GetStdDev(Array2d<float>& vec, const float mean)
{
    return MathUtils::StdDevF32(vec.begin(), vec.totalSize(), mean);
}

void Wav2LetterPreprocessor::NormaliseVec(Array2d<float>& vec)
{
    auto mean = Wav2LetterPreprocessor::GetMean(vec);
    auto stddev = Wav2LetterPreprocessor::GetStdDev(vec, mean);

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

void Wav2LetterPreprocessor::Normalise()
{
    Wav2LetterPreprocessor::NormaliseVec(this->m_mfccBuf);
    Wav2LetterPreprocessor::NormaliseVec(this->m_delta1Buf);
    Wav2LetterPreprocessor::NormaliseVec(this->m_delta2Buf);
}

float Wav2LetterPreprocessor::GetQuantElem(
        const float     elem,
        const float     quantScale,
        const int       quantOffset,
        const float     minVal,
        const float     maxVal)
{
    float val = std::round((elem/quantScale) + quantOffset);
    float returnVal = std::min<float>(std::max<float>(val, minVal), maxVal);
    return returnVal;
}

bool Wav2LetterPreprocessor::Invoke(const float*  audioData, const uint32_t  audioDataLen, std::vector<int8_t>& output,
                                     int quantOffset, float quantScale)
{
    this->m_window = SlidingWindow<const float>(
            audioData, audioDataLen,
            this->m_windowLen, this->m_windowStride);

    uint32_t mfccBufIdx = 0;

    // Init buffers with 0
    std::fill(m_mfccBuf.begin(), m_mfccBuf.end(), 0.f);
    std::fill(m_delta1Buf.begin(), m_delta1Buf.end(), 0.f);
    std::fill(m_delta2Buf.begin(), m_delta2Buf.end(), 0.f);

    // While we can slide over the window 
    while (this->m_window.HasNext())
    {
        const float* mfccWindow = this->m_window.Next();
        auto mfccAudioData = std::vector<float>(
                mfccWindow,
                mfccWindow + this->m_windowLen);

        auto mfcc = this->m_mfcc->MfccCompute(mfccAudioData);
        for (size_t i = 0; i < this->m_mfccBuf.size(0); ++i)
        {
            this->m_mfccBuf(i, mfccBufIdx) = mfcc[i];
        }
        ++mfccBufIdx;
    }

    // Pad MFCC if needed by repeating last feature vector 
    while (mfccBufIdx != this->m_mfcc->m_params.m_numMfccVectors)
    {
        memcpy(&this->m_mfccBuf(0, mfccBufIdx),
               &this->m_mfccBuf(0, mfccBufIdx - 1), sizeof(float) * this->m_mfcc->m_params.m_numMfccFeatures);
        ++mfccBufIdx;
    }

    // Compute first and second order deltas from MFCCs 
    Wav2LetterPreprocessor::ComputeDeltas(this->m_mfccBuf,
                        this->m_delta1Buf,
                        this->m_delta2Buf);

    // Normalise 
    this->Normalise();

    return this->Quantise<int8_t>(output.data(), quantOffset, quantScale);
}

bool Wav2LetterPreprocessor::ComputeDeltas(Array2d<float>& mfcc,
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

    // Get the middle index; coeff vec len should always be odd 
    const size_t coeffLen = delta1Coeffs.size();
    const size_t fMidIdx = (coeffLen - 1)/2;
    const size_t numFeatures = mfcc.size(0);
    const size_t numFeatVectors = mfcc.size(1);

    // iterate through features in MFCC vector
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

Wav2LetterPreprocessor::Wav2LetterPreprocessor(const uint32_t  windowLen,
                                               const uint32_t  windowStride,
                                               std::unique_ptr<Wav2LetterMFCC> mfccInst):
    m_mfcc(std::move(mfccInst)),
    m_mfccBuf(m_mfcc->m_params.m_numMfccFeatures, m_mfcc->m_params.m_numMfccVectors),
    m_delta1Buf(m_mfcc->m_params.m_numMfccFeatures, m_mfcc->m_params.m_numMfccVectors),
    m_delta2Buf(m_mfcc->m_params.m_numMfccFeatures, m_mfcc->m_params.m_numMfccVectors),
    m_windowLen(windowLen),
    m_windowStride(windowStride) 
{
    if (m_mfcc->m_params.m_numMfccFeatures > 0 && windowLen > 0) 
    {
        this->m_mfcc->Init();
    }
    std::fill(m_mfccBuf.begin(), m_mfccBuf.end(), 0.f);
}