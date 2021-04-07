//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>

/* MFCC's consolidated parameters */
class MfccParams
{
public:
    float       m_samplingFreq;
    int         m_numFbankBins;
    float       m_melLoFreq;
    float       m_melHiFreq;
    int         m_numMfccFeatures;
    int         m_frameLen;
    int         m_frameLenPadded;
    bool        m_useHtkMethod;
    int         m_numMfccVectors;

    /** @brief  Constructor */
    MfccParams(const float samplingFreq, const int numFbankBins,
               const float melLoFreq, const float melHiFreq,
               const int numMfccFeats, const int frameLen,
               const bool useHtkMethod, const int numMfccVectors);

    /* Delete the default constructor */
    MfccParams()  = delete;

    /* Default destructor */
    ~MfccParams() = default;

    /** @brief  String representation of parameters */
    std::string Str();
};

/**
 * @brief   Class for MFCC feature extraction.
 *          Based on https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Source/MFCC/mfcc.cpp
 *          This class is designed to be generic and self-sufficient but
 *          certain calculation routines can be overridden to accommodate
 *          use-case specific requirements.
 */
class MFCC
{

public:

    /**
    * @brief        Extract MFCC  features for one single small frame of
    *               audio data e.g. 640 samples.
    * @param[in]    audioData - Vector of audio samples to calculate
    *               features for.
    * @return       Vector of extracted MFCC features.
    **/
    std::vector<float> MfccCompute(const std::vector<float>& audioData);

    MfccParams _m_params;

    /**
     * @brief       Constructor
     * @param[in]   params - MFCC parameters
    */
    MFCC(const MfccParams& params);

    /* Delete the default constructor */
    MFCC() = delete;

    /** @brief  Default destructor */
    ~MFCC() = default;

    /** @brief  Initialise */
    void Init();

    /**
     * @brief        Extract MFCC features and quantise for one single small
     *               frame of audio data e.g. 640 samples.
     * @param[in]    audioData - Vector of audio samples to calculate
     *               features for.
     * @param[in]    quantScale - quantisation scale.
     * @param[in]    quantOffset - quantisation offset
     * @return      Vector of extracted quantised MFCC features.
     **/
    template<typename T>
    std::vector<T> MfccComputeQuant(const std::vector<float>& audioData,
                                    const float quantScale,
                                    const int quantOffset)
    {
        this->_MfccComputePreFeature(audioData);
        float minVal = std::numeric_limits<T>::min();
        float maxVal = std::numeric_limits<T>::max();

        std::vector<T> mfccOut(this->_m_params.m_numMfccFeatures);
        const size_t numFbankBins = this->_m_params.m_numFbankBins;

        /* Take DCT. Uses matrix mul. */
        for (size_t i = 0, j = 0; i < mfccOut.size(); ++i, j += numFbankBins)
        {
            float sum = 0;
            for (size_t k = 0; k < numFbankBins; ++k)
            {
                sum += this->_m_dctMatrix[j + k] * this->_m_melEnergies[k];
            }
            /* Quantize to T. */
            sum = std::round((sum / quantScale) + quantOffset);
            mfccOut[i] = static_cast<T>(std::min<float>(std::max<float>(sum, minVal), maxVal));
        }

        return mfccOut;
    }

    /* Constants */
    static constexpr float logStep = 1.8562979903656 / 27.0;
    static constexpr float freqStep = 200.0 / 3;
    static constexpr float minLogHz = 1000.0;
    static constexpr float minLogMel = minLogHz / freqStep;

protected:
    /**
     * @brief       Project input frequency to Mel Scale.
     * @param[in]   freq - input frequency in floating point
     * @param[in]   useHTKmethod - bool to signal if HTK method is to be
     *              used for calculation
     * @return      Mel transformed frequency in floating point
     **/
    static float MelScale(const float    freq,
                          const bool     useHTKMethod = true);

    /**
     * @brief       Inverse Mel transform - convert MEL warped frequency
     *              back to normal frequency
     * @param[in]   freq - Mel frequency in floating point
     * @param[in]   useHTKmethod - bool to signal if HTK method is to be
     *              used for calculation
     * @return      Real world frequency in floating point
     **/
    static float InverseMelScale(const float melFreq,
                                 const bool  useHTKMethod = true);

    /**
     * @brief       Populates MEL energies after applying the MEL filter
     *              bank weights and adding them up to be placed into
     *              bins, according to the filter bank's first and last
     *              indices (pre-computed for each filter bank element
     *              by _CreateMelFilterBank function).
     * @param[in]   fftVec                  Vector populated with FFT magnitudes
     * @param[in]   melFilterBank           2D Vector with filter bank weights
     * @param[in]   filterBankFilterFirst   Vector containing the first indices of filter bank
     *                                      to be used for each bin.
     * @param[in]   filterBankFilterLast    Vector containing the last indices of filter bank
     *                                      to be used for each bin.
     * @param[out]  melEnergies             Pre-allocated vector of MEL energies to be
     *                                      populated.
     * @return      true if successful, false otherwise
     */
    virtual bool ApplyMelFilterBank(
            std::vector<float>&                 fftVec,
            std::vector<std::vector<float>>&    melFilterBank,
            std::vector<int32_t>&               filterBankFilterFirst,
            std::vector<int32_t>&               filterBankFilterLast,
            std::vector<float>&                 melEnergies);

    /**
     * @brief           Converts the Mel energies for logarithmic scale
     * @param[in/out]   melEnergies - 1D vector of Mel energies
     **/
    virtual void ConvertToLogarithmicScale(std::vector<float>& melEnergies);

    /**
     * @brief       Create a matrix used to calculate Discrete Cosine
     *              Transform.
     * @param[in]   inputLength - input length of the buffer on which
     *              DCT will be performed
     * @param[in]   coefficientCount - Total coefficients per input
     *              length
     * @return      1D vector with inputLength x coefficientCount elements
     *              populated with DCT coefficients.
     */
    virtual std::vector<float> CreateDCTMatrix(
            const int32_t inputLength,
            const int32_t coefficientCount);

    /**
     * @brief       Given the low and high Mel values, get the normaliser
     *              for weights to be applied when populating the filter
     *              bank.
     * @param[in]   leftMel - low Mel frequency value
     * @param[in]   rightMel - high Mel frequency value
     * @param[in]   useHTKMethod - bool to signal if HTK method is to be
     *              used for calculation
     */
    virtual float GetMelFilterBankNormaliser(
            const float&   leftMel,
            const float&   rightMel,
            const bool     useHTKMethod);

private:

    std::vector<float>              _m_frame;
    std::vector<float>              _m_buffer;
    std::vector<float>              _m_melEnergies;
    std::vector<float>              _m_windowFunc;
    std::vector<std::vector<float>> _m_melFilterBank;
    std::vector<float>              _m_dctMatrix;
    std::vector<int32_t>            _m_filterBankFilterFirst;
    std::vector<int32_t>            _m_filterBankFilterLast;
    bool                            _m_filterBankInitialised;

    /**
     * @brief       Initialises the filter banks and the DCT matrix **/
    void _InitMelFilterBank();

    /**
     * @brief       Signals whether the instance of MFCC has had its
     *              required buffers initialised
     * @return      True if initialised, false otherwise
     **/
    bool _IsMelFilterBankInited();

    /**
     * @brief       Create mel filter banks for MFCC calculation.
     * @return      2D vector of floats
     **/
    std::vector<std::vector<float>> _CreateMelFilterBank();

    /**
     * @brief       Computes and populates internal memeber buffers used
     *              in MFCC feature calculation
     * @param[in]   audioData - 1D vector of 16-bit audio data
     */
    void _MfccComputePreFeature(const std::vector<float>& audioData);

    /** @brief       Computes the magnitude from an interleaved complex array */
    void _ConvertToPowerSpectrum();

};

