//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <cstdio>
#include <float.h>

#include "MFCC.hpp"
#include "MathUtils.hpp"


MfccParams::MfccParams(
        const float samplingFreq,
        const int numFbankBins,
        const float melLoFreq,
        const float melHiFreq,
        const int numMfccFeats,
        const int frameLen,
        const bool useHtkMethod,
        const int numMfccVectors):
        m_samplingFreq(samplingFreq),
        m_numFbankBins(numFbankBins),
        m_melLoFreq(melLoFreq),
        m_melHiFreq(melHiFreq),
        m_numMfccFeatures(numMfccFeats),
        m_frameLen(frameLen),
        m_numMfccVectors(numMfccVectors),

        /* Smallest power of 2 >= frame length. */
        m_frameLenPadded(pow(2, ceil((log(frameLen)/log(2))))),
        m_useHtkMethod(useHtkMethod)
{}

std::string MfccParams::Str()
{
    char strC[1024];
    snprintf(strC, sizeof(strC) - 1, "\n   \
            \n\t Sampling frequency:         %f\
            \n\t Number of filter banks:     %u\
            \n\t Mel frequency limit (low):  %f\
            \n\t Mel frequency limit (high): %f\
            \n\t Number of MFCC features:    %u\
            \n\t Frame length:               %u\
            \n\t Padded frame length:        %u\
            \n\t Using HTK for Mel scale:    %s\n",
             this->m_samplingFreq, this->m_numFbankBins, this->m_melLoFreq,
             this->m_melHiFreq, this->m_numMfccFeatures, this->m_frameLen,
             this->m_frameLenPadded, this->m_useHtkMethod ? "yes" : "no");
    return std::string{strC};
}

MFCC::MFCC(const MfccParams& params):
        _m_params(params),
        _m_filterBankInitialised(false)
{
    this->_m_buffer = std::vector<float>(
            this->_m_params.m_frameLenPadded, 0.0);
    this->_m_frame = std::vector<float>(
            this->_m_params.m_frameLenPadded, 0.0);
    this->_m_melEnergies = std::vector<float>(
            this->_m_params.m_numFbankBins, 0.0);

    this->_m_windowFunc = std::vector<float>(this->_m_params.m_frameLen);
    const float multiplier = 2 * M_PI / this->_m_params.m_frameLen;

    /* Create window function. */
    for (size_t i = 0; i < this->_m_params.m_frameLen; i++)
    {
        this->_m_windowFunc[i] = (0.5 - (0.5 * cos(static_cast<float>(i) * multiplier)));
    }
}

void MFCC::Init()
{
    this->_InitMelFilterBank();
}

float MFCC::MelScale(const float freq, const bool useHTKMethod)
{
    if (useHTKMethod)
    {
        return 1127.0f * logf (1.0f + freq / 700.0f);
    }
    else
    {
        /* Slaney formula for mel scale. */
        float mel = freq / freqStep;

        if (freq >= minLogHz)
        {
            mel = minLogMel + logf(freq / minLogHz) / logStep;
        }
        return mel;
    }
}

float MFCC::InverseMelScale(const float melFreq, const bool useHTKMethod)
{
    if (useHTKMethod)
    {
        return 700.0f * (expf (melFreq / 1127.0f) - 1.0f);
    }
    else
    {
        /* Slaney formula for mel scale. */
        float freq = freqStep * melFreq;

        if (melFreq >= minLogMel)
        {
            freq = minLogHz * expf(logStep * (melFreq - minLogMel));
        }
        return freq;
    }
}


bool MFCC::ApplyMelFilterBank(
        std::vector<float>&                 fftVec,
        std::vector<std::vector<float>>&    melFilterBank,
        std::vector<int32_t>&               filterBankFilterFirst,
        std::vector<int32_t>&               filterBankFilterLast,
        std::vector<float>&                 melEnergies)
{
    const size_t numBanks = melEnergies.size();

    if (numBanks != filterBankFilterFirst.size() ||
        numBanks != filterBankFilterLast.size())
    {
        printf("unexpected filter bank lengths\n");
        return false;
    }

    for (size_t bin = 0; bin < numBanks; ++bin)
    {
        auto filterBankIter = melFilterBank[bin].begin();
        float melEnergy = 1e-10; /* Avoid log of zero at later stages */
        const int32_t firstIndex = filterBankFilterFirst[bin];
        const int32_t lastIndex = filterBankFilterLast[bin];

        for (int32_t i = firstIndex; i <= lastIndex; ++i)
        {
            melEnergy += (*filterBankIter++ * fftVec[i]);
        }

        melEnergies[bin] = melEnergy;
    }

    return true;
}

void MFCC::ConvertToLogarithmicScale(std::vector<float>& melEnergies)
{
    float maxMelEnergy = -FLT_MAX;

    /* Container for natural logarithms of mel energies */
    std::vector <float> vecLogEnergies(melEnergies.size(), 0.f);

    /* Because we are taking natural logs, we need to multiply by log10(e).
     * Also, for wav2letter model, we scale our log10 values by 10 */
    constexpr float multiplier = 10.0 * /* default scalar */
                                 0.4342944819032518; /* log10f(std::exp(1.0))*/

    /* Take log of the whole vector */
    MathUtils::VecLogarithmF32(melEnergies, vecLogEnergies);

    /* Scale the log values and get the max */
    for (auto iterM = melEnergies.begin(), iterL = vecLogEnergies.begin();
         iterM != melEnergies.end(); ++iterM, ++iterL)
    {
        *iterM = *iterL * multiplier;

        /* Save the max mel energy. */
        if (*iterM > maxMelEnergy)
        {
            maxMelEnergy = *iterM;
        }
    }

    /* Clamp the mel energies */
    constexpr float maxDb = 80.0;
    const float clampLevelLowdB = maxMelEnergy - maxDb;
    for (auto iter = melEnergies.begin(); iter != melEnergies.end(); ++iter)
    {
        *iter = std::max(*iter, clampLevelLowdB);
    }
}

void MFCC::_ConvertToPowerSpectrum()
{
    const uint32_t halfDim = this->_m_params.m_frameLenPadded / 2;

    /* Handle this special case. */
    float firstEnergy = this->_m_buffer[0] * this->_m_buffer[0];
    float lastEnergy = this->_m_buffer[1] * this->_m_buffer[1];

    MathUtils::ComplexMagnitudeSquaredF32(
            this->_m_buffer.data(),
            this->_m_buffer.size(),
            this->_m_buffer.data(),
            this->_m_buffer.size()/2);

    this->_m_buffer[0] = firstEnergy;
    this->_m_buffer[halfDim] = lastEnergy;
}

std::vector<float> MFCC::CreateDCTMatrix(
        const int32_t inputLength,
        const int32_t coefficientCount)
{
    std::vector<float> dctMatix(inputLength * coefficientCount);

    /* Orthonormal normalization. */
    const float normalizerK0 = 2 * sqrt(1.0 / static_cast<float>(4*inputLength));
    const float normalizer = 2 * sqrt(1.0 / static_cast<float>(2*inputLength));

    const float angleIncr = M_PI/inputLength;
    float angle = angleIncr; /* we start using it at k = 1 loop */

    /* First row of DCT will use normalizer K0 */
    for (int32_t n = 0; n < inputLength; ++n)
    {
        dctMatix[n] = normalizerK0;
    }

    /* Second row (index = 1) onwards, we use standard normalizer */
    for (int32_t k = 1, m = inputLength; k < coefficientCount; ++k, m += inputLength)
    {
        for (int32_t n = 0; n < inputLength; ++n)
        {
            dctMatix[m+n] = normalizer *
                            cos((n + 0.5) * angle);
        }
        angle += angleIncr;
    }
    return dctMatix;
}

float MFCC::GetMelFilterBankNormaliser(
        const float&    leftMel,
        const float&    rightMel,
        const bool      useHTKMethod)
{
/* Slaney normalization for mel weights. */
    return (2.0f / (MFCC::InverseMelScale(rightMel, useHTKMethod) -
                    MFCC::InverseMelScale(leftMel, useHTKMethod)));
}

void MFCC::_InitMelFilterBank()
{
    if (!this->_IsMelFilterBankInited())
    {
        this->_m_melFilterBank = this->_CreateMelFilterBank();
        this->_m_dctMatrix = this->CreateDCTMatrix(
                this->_m_params.m_numFbankBins,
                this->_m_params.m_numMfccFeatures);
        this->_m_filterBankInitialised = true;
    }
}

bool MFCC::_IsMelFilterBankInited()
{
    return this->_m_filterBankInitialised;
}

void MFCC::_MfccComputePreFeature(const std::vector<float>& audioData)
{
    this->_InitMelFilterBank();

    /* TensorFlow way of normalizing .wav data to (-1, 1). */
    constexpr float normaliser = 1.0;
    for (size_t i = 0; i < this->_m_params.m_frameLen; i++)
    {
        this->_m_frame[i] = static_cast<float>(audioData[i]) * normaliser;
    }

    /* Apply window function to input frame. */
    for(size_t i = 0; i < this->_m_params.m_frameLen; i++)
    {
        this->_m_frame[i] *= this->_m_windowFunc[i];
    }

    /* Set remaining frame values to 0. */
    std::fill(this->_m_frame.begin() + this->_m_params.m_frameLen,this->_m_frame.end(), 0);

    /* Compute FFT. */
    MathUtils::FftF32(this->_m_frame, this->_m_buffer);

    /* Convert to power spectrum. */
    this->_ConvertToPowerSpectrum();

    /* Apply mel filterbanks. */
    if (!this->ApplyMelFilterBank(this->_m_buffer,
                                  this->_m_melFilterBank,
                                  this->_m_filterBankFilterFirst,
                                  this->_m_filterBankFilterLast,
                                  this->_m_melEnergies))
    {
        printf("Failed to apply MEL filter banks\n");
    }

    /* Convert to logarithmic scale */
    this->ConvertToLogarithmicScale(this->_m_melEnergies);
}

std::vector<float> MFCC::MfccCompute(const std::vector<float>& audioData)
{
    this->_MfccComputePreFeature(audioData);

    std::vector<float> mfccOut(this->_m_params.m_numMfccFeatures);

    float * ptrMel = this->_m_melEnergies.data();
    float * ptrDct = this->_m_dctMatrix.data();
    float * ptrMfcc = mfccOut.data();

    /* Take DCT. Uses matrix mul. */
    for (size_t i = 0, j = 0; i < mfccOut.size();
         ++i, j += this->_m_params.m_numFbankBins)
    {
        *ptrMfcc++ = MathUtils::DotProductF32(
                ptrDct + j,
                ptrMel,
                this->_m_params.m_numFbankBins);
    }

    return mfccOut;
}

std::vector<std::vector<float>> MFCC::_CreateMelFilterBank()
{
    size_t numFftBins = this->_m_params.m_frameLenPadded / 2;
    float fftBinWidth = static_cast<float>(this->_m_params.m_samplingFreq) / this->_m_params.m_frameLenPadded;

    float melLowFreq = MFCC::MelScale(this->_m_params.m_melLoFreq,
                                      this->_m_params.m_useHtkMethod);
    float melHighFreq = MFCC::MelScale(this->_m_params.m_melHiFreq,
                                       this->_m_params.m_useHtkMethod);
    float melFreqDelta = (melHighFreq - melLowFreq) / (this->_m_params.m_numFbankBins + 1);

    std::vector<float> thisBin = std::vector<float>(numFftBins);
    std::vector<std::vector<float>> melFilterBank(
            this->_m_params.m_numFbankBins);
    this->_m_filterBankFilterFirst =
            std::vector<int32_t>(this->_m_params.m_numFbankBins);
    this->_m_filterBankFilterLast =
            std::vector<int32_t>(this->_m_params.m_numFbankBins);

    for (size_t bin = 0; bin < this->_m_params.m_numFbankBins; bin++)
    {
        float leftMel = melLowFreq + bin * melFreqDelta;
        float centerMel = melLowFreq + (bin + 1) * melFreqDelta;
        float rightMel = melLowFreq + (bin + 2) * melFreqDelta;

        int32_t firstIndex = -1;
        int32_t lastIndex = -1;
        const float normaliser = this->GetMelFilterBankNormaliser(leftMel, rightMel, this->_m_params.m_useHtkMethod);

        for (size_t i = 0; i < numFftBins; i++)
        {
            float freq = (fftBinWidth * i); /* Center freq of this fft bin. */
            float mel = MFCC::MelScale(freq, this->_m_params.m_useHtkMethod);
            thisBin[i] = 0.0;

            if (mel > leftMel && mel < rightMel)
            {
                float weight;
                if (mel <= centerMel)
                {
                    weight = (mel - leftMel) / (centerMel - leftMel);
                }
                else
                {
                    weight = (rightMel - mel) / (rightMel - centerMel);
                }

                thisBin[i] = weight * normaliser;
                if (firstIndex == -1)
                {
                    firstIndex = i;
                }
                lastIndex = i;
            }
        }

        this->_m_filterBankFilterFirst[bin] = firstIndex;
        this->_m_filterBankFilterLast[bin] = lastIndex;

        /* Copy the part we care about. */
        for (int32_t i = firstIndex; i <= lastIndex; i++)
        {
            melFilterBank[bin].push_back(thisBin[i]);
        }
    }

    return melFilterBank;
}

