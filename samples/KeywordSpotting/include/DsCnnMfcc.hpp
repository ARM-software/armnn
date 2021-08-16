//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "MFCC.hpp"

/* Class to provide DS-CNN specific MFCC calculation requirements. */
class DsCnnMFCC : public MFCC 
{

public:

    explicit DsCnnMFCC(MfccParams& params)
        :  MFCC(params)
    {}
    DsCnnMFCC()  = delete;
    ~DsCnnMFCC() = default;
};
