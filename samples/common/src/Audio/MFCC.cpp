//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MFCC.hpp"
#include "MathUtils.hpp"

#include <cfloat>
#include <cinttypes>
#include <cstring>

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
    m_params(params),
    m_filterBankInitialised(false)
{
    this->m_buffer = std::vector<float>(
            this->m_params.m_frameLenPadded, 0.0);
    this->m_frame = std::vector<float>(
            this->m_params.m_frameLenPadded, 0.0);
    this->m_melEnergies = std::vector<float>(
            this->m_params.m_numFbankBins, 0.0);

    this->m_windowFunc = std::vector<float>(this->m_params.m_frameLen);
    const auto multiplier = static_cast<float>(2 * M_PI / this->m_params.m_frameLen);

    /* Create window function. */
    for (size_t i = 0; i < this->m_params.m_frameLen; i++) 
    {
        this->m_windowFunc[i] = (0.5 - (0.5 * cosf(static_cast<float>(i) * multiplier)));
    }

}

void MFCC::Init()
{
    this->InitMelFilterBank();
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
        float mel = freq / ms_freqStep;

        if (freq >= ms_minLogHz) 
        {
            mel = ms_minLogMel + logf(freq / ms_minLogHz) / ms_logStep;
        }
        return mel;
    }
}

float MFCC::InverseMelScale(const float melFreq, const bool useHTKMethod)
{
    if (useHTKMethod) {
        return 700.0f * (expf (melFreq / 1127.0f) - 1.0f);
    } 
    else 
    {
        /* Slaney formula for mel scale. */
        float freq = ms_freqStep * melFreq;

        if (melFreq >= ms_minLogMel) 
        {
            freq = ms_minLogHz * expf(ms_logStep * (melFreq - ms_minLogMel));
        }
        return freq;
    }
}


bool MFCC::ApplyMelFilterBank(
        std::vector<float>&                 fftVec,
        std::vector<std::vector<float>>&    melFilterBank,
        std::vector<uint32_t>&              filterBankFilterFirst,
        std::vector<uint32_t>&              filterBankFilterLast,
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
        auto end = melFilterBank[bin].end();
        float melEnergy = FLT_MIN;  /* Avoid log of zero at later stages */
        const uint32_t firstIndex = filterBankFilterFirst[bin];
        const uint32_t lastIndex = std::min<uint32_t>(filterBankFilterLast[bin], fftVec.size() - 1);

        for (uint32_t i = firstIndex; i <= lastIndex && filterBankIter != end; i++) 
        {
            float energyRep = sqrt(fftVec[i]);
            melEnergy += (*filterBankIter++ * energyRep);
        }

        melEnergies[bin] = melEnergy;
    }

    return true;
}

void MFCC::ConvertToLogarithmicScale(std::vector<float>& melEnergies)
{
    for (float& melEnergy : melEnergies) 
    {
        melEnergy = logf(melEnergy);
    }
}

void MFCC::ConvertToPowerSpectrum()
{
    const uint32_t halfDim = this->m_buffer.size() / 2;

    /* Handle this special case. */
    float firstEnergy = this->m_buffer[0] * this->m_buffer[0];
    float lastEnergy = this->m_buffer[1] * this->m_buffer[1];

    MathUtils::ComplexMagnitudeSquaredF32(
            this->m_buffer.data(),
            this->m_buffer.size(),
            this->m_buffer.data(),
            this->m_buffer.size()/2);

    this->m_buffer[0] = firstEnergy;
    this->m_buffer[halfDim] = lastEnergy;
}

std::vector<float> MFCC::CreateDCTMatrix(
                            const int32_t inputLength,
                            const int32_t coefficientCount)
{
    std::vector<float> dctMatrix(inputLength * coefficientCount);

    const float normalizer = sqrtf(2.0f/inputLength);
    const float angleIncr = M_PI/inputLength;
    float angle = 0;

    for (int32_t k = 0, m = 0; k < coefficientCount; k++, m += inputLength) 
    {
        for (int32_t n = 0; n < inputLength; n++) 
        {
            dctMatrix[m + n] = normalizer * cosf((n + 0.5f) * angle);
        }
        angle += angleIncr;
    }

    return dctMatrix;
}

float MFCC::GetMelFilterBankNormaliser(
                const float&    leftMel,
                const float&    rightMel,
                const bool      useHTKMethod)
{
    /* By default, no normalisation => return 1 */
    return 1.f;
}

void MFCC::InitMelFilterBank()
{
    if (!this->IsMelFilterBankInited()) 
    {
        this->m_melFilterBank = this->CreateMelFilterBank();
        this->m_dctMatrix = this->CreateDCTMatrix(
                                this->m_params.m_numFbankBins,
                                this->m_params.m_numMfccFeatures);
        this->m_filterBankInitialised = true;
    }
}

bool MFCC::IsMelFilterBankInited() const
{
    return this->m_filterBankInitialised;
}

void MFCC::MfccComputePreFeature(const std::vector<float>& audioData)
{
    this->InitMelFilterBank();

    auto size = std::min(std::min(this->m_frame.size(), audioData.size()),
                         static_cast<size_t>(this->m_params.m_frameLen)) * sizeof(float);
    std::memcpy(this->m_frame.data(), audioData.data(), size);

    /* Apply window function to input frame. */
    for(size_t i = 0; i < this->m_params.m_frameLen; i++) 
    {
        this->m_frame[i] *= this->m_windowFunc[i];
    }

    /* Set remaining frame values to 0. */
    std::fill(this->m_frame.begin() + this->m_params.m_frameLen,this->m_frame.end(), 0);

    /* Compute FFT. */
    MathUtils::FftF32(this->m_frame, this->m_buffer);

    /* Convert to power spectrum. */
    this->ConvertToPowerSpectrum();

    /* Apply mel filterbanks. */
    if (!this->ApplyMelFilterBank(this->m_buffer,
                                  this->m_melFilterBank,
                                  this->m_filterBankFilterFirst,
                                  this->m_filterBankFilterLast,
                                  this->m_melEnergies)) 
    {
        printf("Failed to apply MEL filter banks\n");
    }

    /* Convert to logarithmic scale. */
    this->ConvertToLogarithmicScale(this->m_melEnergies);
}

std::vector<float> MFCC::MfccCompute(const std::vector<float>& audioData)
{
    this->MfccComputePreFeature(audioData);

    std::vector<float> mfccOut(this->m_params.m_numMfccFeatures);

    float * ptrMel = this->m_melEnergies.data();
    float * ptrDct = this->m_dctMatrix.data();
    float * ptrMfcc = mfccOut.data();

    /* Take DCT. Uses matrix mul. */
    for (size_t i = 0, j = 0; i < mfccOut.size();
                ++i, j += this->m_params.m_numFbankBins) 
    {
        *ptrMfcc++ = MathUtils::DotProductF32(
                ptrDct + j,
                ptrMel,
                this->m_params.m_numFbankBins);
    }
    return mfccOut;
}

std::vector<std::vector<float>> MFCC::CreateMelFilterBank()
{
    size_t numFftBins = this->m_params.m_frameLenPadded / 2;
    float fftBinWidth = static_cast<float>(this->m_params.m_samplingFreq) / this->m_params.m_frameLenPadded;

    float melLowFreq = MFCC::MelScale(this->m_params.m_melLoFreq,
                                      this->m_params.m_useHtkMethod);
    float melHighFreq = MFCC::MelScale(this->m_params.m_melHiFreq,
                                       this->m_params.m_useHtkMethod);
    float melFreqDelta = (melHighFreq - melLowFreq) / (this->m_params.m_numFbankBins + 1);

    std::vector<float> thisBin = std::vector<float>(numFftBins);
    std::vector<std::vector<float>> melFilterBank(
                                        this->m_params.m_numFbankBins);
    this->m_filterBankFilterFirst =
                    std::vector<uint32_t>(this->m_params.m_numFbankBins);
    this->m_filterBankFilterLast =
                    std::vector<uint32_t>(this->m_params.m_numFbankBins);

    for (size_t bin = 0; bin < this->m_params.m_numFbankBins; bin++) 
    {
        float leftMel = melLowFreq + bin * melFreqDelta;
        float centerMel = melLowFreq + (bin + 1) * melFreqDelta;
        float rightMel = melLowFreq + (bin + 2) * melFreqDelta;

        uint32_t firstIndex = 0;
        uint32_t lastIndex = 0;
        bool firstIndexFound = false;
        const float normaliser = this->GetMelFilterBankNormaliser(leftMel, rightMel, this->m_params.m_useHtkMethod);

        for (size_t i = 0; i < numFftBins; i++) 
        {
            float freq = (fftBinWidth * i);  /* Center freq of this fft bin. */
            float mel = MFCC::MelScale(freq, this->m_params.m_useHtkMethod);
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
                if (!firstIndexFound) 
                {
                    firstIndex = i;
                    firstIndexFound = true;
                }
                lastIndex = i;
            }
        }

        this->m_filterBankFilterFirst[bin] = firstIndex;
        this->m_filterBankFilterLast[bin] = lastIndex;

        /* Copy the part we care about. */
        for (uint32_t i = firstIndex; i <= lastIndex; i++) 
        {
            melFilterBank[bin].push_back(thisBin[i]);
        }
    }

    return melFilterBank;
}
