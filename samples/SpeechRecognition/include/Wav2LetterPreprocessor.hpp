//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#ifndef SPEECH_RECOGNITION_EXAMPLE_WAV2LETTERPREPROCESSOR_HPP
#define SPEECH_RECOGNITION_EXAMPLE_WAV2LETTERPREPROCESSOR_HPP

#include <numeric>
#include "DataStructures.hpp"
#include "SlidingWindow.hpp"
#include "MFCC.hpp"
#include "Wav2LetterMFCC.hpp"
// Class to facilitate pre-processing calculation for Wav2Letter model for ASR 
using AudioWindow = SlidingWindow<const float>;

class Wav2LetterPreprocessor 
{
public:
    Wav2LetterPreprocessor(uint32_t windowLen, uint32_t windowStride,
                           std::unique_ptr<Wav2LetterMFCC> mfccInst);

    /**
     * @brief       Calculates the features required from audio data. This
     *              includes MFCC, first and second order deltas,
     *              normalisation and finally, quantisation. The tensor is
     *              populated with feature from a given window placed along
     *              in a single row.
     * @param[in]   audioData     pointer to the first element of audio data
     * @param[in]   audioDataLen  number of elements in the audio data
     * @param[in]   tensor        tensor to be populated
     * @return      true if successful, false in case of error.
     */
    bool Invoke(const float* audioData, uint32_t audioDataLen, std::vector<int8_t>& output, int quantOffset,
                float quantScale);

    std::unique_ptr<MFCC> m_mfcc;

    // Actual buffers to be populated 
    Array2d<float> m_mfccBuf;         // Contiguous buffer 1D: MFCC 
    Array2d<float> m_delta1Buf;       // Contiguous buffer 1D: Delta 1 
    Array2d<float> m_delta2Buf;       // Contiguous buffer 1D: Delta 2

    uint32_t m_windowLen;       // Window length for MFCC 
    uint32_t m_windowStride;    // Window stride len for MFCC 
    AudioWindow m_window;       // Sliding window 

protected:
    /**
     * @brief Computes the first and second order deltas for the
     *        MFCC buffers - they are assumed to be populated.
     *
     * @param[in]  mfcc   MFCC buffers
     * @param[out] delta1 result of the first diff computation
     * @param[out] delta2 result of the second diff computation
     *
     * @return true if successful, false otherwise
     */
    static bool ComputeDeltas(Array2d<float>& mfcc,
                              Array2d<float>& delta1,
                              Array2d<float>& delta2);

protected:

    /**
     * @brief      Given a 2D vector of floats, computes the mean
     * @param[in]   vec      vector of vector of floats
     * @return      mean value
     */
    static float GetMean(Array2d<float>& vec);

    /**
     * @brief       Given a 2D vector of floats, computes the stddev
     * @param[in]   vec   vector of vector of floats
     * @param[in]   mean     mean value of the vector passed in
     * @return      stddev value
     */
    static float GetStdDev(Array2d<float>& vec, float mean);

    /**
     * @brief           Given a 2D vector of floats, normalises it using
     *                  the mean and the stddev
     * @param[in/out]   vec      vector of vector of floats
     * @return
     */
    static void NormaliseVec(Array2d<float>& vec);

    /**
     * @brief       Normalises the MFCC and delta buffers
     * @return
     */
    void Normalise();

    /**
     * @brief       Given the quantisation and data type limits, computes
     *              the quantised values of a floating point input data.
     * @param[in]   elem            Element to be quantised
     * @param[in]   quantScale      Scale
     * @param[in]   quantOffset     Offset
     * @param[in]   minVal          Numerical limit - minimum
     * @param[in]   maxVal          Numerical limit - maximum
     * @return      floating point quantised value
     */
    static float GetQuantElem(
            float elem,
            float quantScale,
            int quantOffset,
            float minVal,
            float maxVal);

    /**
     * @brief       Quantises the MFCC and delta buffers, and places them
     *              in the output buffer. While doing so, it transposes
     *              the data. Reason: Buffers in this class are arranged
     *              for "time" axis to be row major. Primary reason for
     *              this being the convolution speed up (as we can use
     *              contiguous memory). The output, however, requires the
     *              time axis to be in column major arrangement.
     * @param[in]   outputBuf       pointer to the output buffer
     * @param[in]   outputBufSz     output buffer's size
     * @param[in]   quantScale      quantisation scale
     * @param[in]   quantOffset     quantisation offset
     */
    template<typename T>
    bool Quantise(T*outputBuf, int quantOffset, float quantScale) 
    {
        // Populate 
        T* outputBufMfcc = outputBuf;
        T* outputBufD1 = outputBuf + this->m_mfcc->m_params.m_numMfccFeatures;
        T* outputBufD2 = outputBufD1 + this->m_mfcc->m_params.m_numMfccFeatures;
        const uint32_t ptrIncr = this->m_mfcc->m_params.m_numMfccFeatures * 2; // (3 vectors - 1 vector) 

        const float minVal = std::numeric_limits<T>::min();
        const float maxVal = std::numeric_limits<T>::max();

        // We need to do a transpose while copying and concatenating the tensor
        for (uint32_t j = 0; j < this->m_mfcc->m_params.m_numMfccVectors; ++j) 
        {
            for (uint32_t i = 0; i < this->m_mfcc->m_params.m_numMfccFeatures; ++i) 
            {
                *outputBufMfcc++ = static_cast<T>(Wav2LetterPreprocessor::GetQuantElem(
                        this->m_mfccBuf(i, j), quantScale,
                        quantOffset, minVal, maxVal));
                *outputBufD1++ = static_cast<T>(Wav2LetterPreprocessor::GetQuantElem(
                        this->m_delta1Buf(i, j), quantScale,
                        quantOffset, minVal, maxVal));
                *outputBufD2++ = static_cast<T>(Wav2LetterPreprocessor::GetQuantElem(
                        this->m_delta2Buf(i, j), quantScale,
                        quantOffset, minVal, maxVal));
            }
            outputBufMfcc += ptrIncr;
            outputBufD1 += ptrIncr;
            outputBufD2 += ptrIncr;
        }
        return true;
    }
};

#endif //SPEECH_RECOGNITION_EXAMPLE_WAV2LETTERPREPROCESSOR_HPP
