//
// Copyright © 2018-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "AddBroadcastReshapeLayer.hpp"
#include "AddDebug.hpp"
#include "ConvertConstants.hpp"
#include "ConvertConstDequantisationLayersToConstLayers.hpp"
#include "ConvertConstPermuteLayersToConstLayers.hpp"
#include "ConvertFp32NetworkToFp16.hpp"
#include "DeleteBroadcastTo.hpp"
#include "FoldPadIntoLayer2d.hpp"
#include "FuseBatchNorm.hpp"
#include "MaxMinIntoBoundedRelu.hpp"
#include "MovePermuteUp.hpp"
#include "MoveTransposeUp.hpp"
#include "OptimizeConsecutiveReshapes.hpp"
#include "OptimizeInverseConversions.hpp"
#include "OptimizeInversePermutes.hpp"
#include "PermuteAsReshape.hpp"
#include "PermuteAndBatchToSpaceAsDepthToSpace.hpp"
#include "PermuteDepthwiseConv2dWeights.hpp"
#include "SquashEqualSiblings.hpp"
#include "TransposeAsReshape.hpp"
