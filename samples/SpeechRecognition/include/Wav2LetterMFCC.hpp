//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "MFCC.hpp"

/* Class to provide Wav2Letter specific MFCC calculation requirements. */
class Wav2LetterMFCC : public MFCC 
{

public:
    explicit Wav2LetterMFCC(const MfccParams& params)
        :  MFCC(params)
    {}

    Wav2LetterMFCC()  = delete;
    ~Wav2LetterMFCC() = default;

protected:

    /**
     * @brief       Overrides base class implementation of this function.
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
    bool ApplyMelFilterBank(
        std::vector<float>&                 fftVec,
        std::vector<std::vector<float>>&    melFilterBank,
        std::vector<uint32_t>&              filterBankFilterFirst,
        std::vector<uint32_t>&              filterBankFilterLast,
        std::vector<float>&                 melEnergies) override;

    /**
     * @brief           Override for the base class implementation convert mel
     *                  energies to logarithmic scale. The difference from
     *                  default behaviour is that the power is converted to dB
     *                  and subsequently clamped.
     * @param[in,out]   melEnergies   1D vector of Mel energies
     **/
    void ConvertToLogarithmicScale(std::vector<float>& melEnergies) override;

    /**
     * @brief       Create a matrix used to calculate Discrete Cosine
     *              Transform. Override for the base class' default
     *              implementation as the first and last elements
     *              use a different normaliser.
     * @param[in]   inputLength        input length of the buffer on which
     *                                 DCT will be performed
     * @param[in]   coefficientCount   Total coefficients per input length.
     * @return      1D vector with inputLength x coefficientCount elements
     *              populated with DCT coefficients.
     */
    std::vector<float> CreateDCTMatrix(int32_t inputLength,
                                       int32_t coefficientCount) override;

    /**
     * @brief       Given the low and high Mel values, get the normaliser
     *              for weights to be applied when populating the filter
     *              bank. Override for the base class implementation.
     * @param[in]   leftMel        Low Mel frequency value.
     * @param[in]   rightMel       High Mel frequency value.
     * @param[in]   useHTKMethod   bool to signal if HTK method is to be
     *                             used for calculation.
     * @return      Value to use for normalising.
     */
    float GetMelFilterBankNormaliser(const float&   leftMel,
                                     const float&   rightMel,
                                     bool     useHTKMethod) override;
};