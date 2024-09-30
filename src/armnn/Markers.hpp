//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

/**
 * This empty macro has been inserted at the end of LoadedNetwork constructor. It marks the point after Arm NN has
 * created all workloads associated with an optimized network and the network is ready for execution.
 */
#define MARK_OPTIMIZED_NETWORK_LOADED()

/**
 * This empty macro has been inserted at LoadedNetwork::Execute. It marks the point just before Arm NN will:
 *  - perform any copy/import operation on subgraph inputs.
 *  - execute all the workloads in this subgraph.
 *  - perform any copy/export operation on subgraph outputs.
 */
#define MARK_INFERENCE_EXECUTION_BEGIN()

/**
 * This empty macro has been inserted at LoadedNetwork::Execute. It marks the point just after Arm NN has executed
 * all workloads in this subgraph and processed any copy/export operation on subgraph output.
 */
#define MARK_INFERENCE_EXECUTION_END()

/**
 * This empty macro has been inserted at LoadedNetwork::Execute. It marks the point just before a workload begins
 * execution.
 */
#define MARK_WORKLOAD_EXECUTION_BEGIN()

/**
* This empty macro has been inserted at LoadedNetwork::Execute. It marks the point just after a workload completes
* execution.
 */
#define MARK_WORKLOAD_EXECUTION_END()

} //namespace armnn
