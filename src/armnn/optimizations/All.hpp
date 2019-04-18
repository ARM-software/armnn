//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ConvertConstants.hpp"
#include "OptimizeInversePermutes.hpp"
#include "PermuteAsReshape.hpp"
#include "OptimizeConsecutiveReshapes.hpp"
#include "SquashEqualSiblings.hpp"
#include "MovePermuteUp.hpp"
#include "OptimizeInverseConversions.hpp"
#include "ConvertFp32NetworkToFp16.hpp"
#include "AddDebug.hpp"
#include "FoldPadIntoConvolution2d.hpp"
