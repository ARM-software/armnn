//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingStateMachine.hpp"

#include <armnn/Exceptions.hpp>

namespace armnn
{

namespace profiling
{

ProfilingState ProfilingStateMachine::GetCurrentState() const
{
    return m_State;
}

void ProfilingStateMachine::TransitionToState(ProfilingState newState)
{
     switch (newState)
     {
         case ProfilingState::Uninitialised:
         {
             ProfilingState expectedState = m_State.load(std::memory_order::memory_order_relaxed);
             do {
                 if (!IsOneOfStates(expectedState, ProfilingState::Uninitialised))
                 {
                     throw armnn::Exception(std::string("Cannot transition from state [")
                                            + GetProfilingStateName(expectedState)
                                            +"] to [" + GetProfilingStateName(newState) + "]");
                 }
             } while (!m_State.compare_exchange_strong(expectedState, newState,
                      std::memory_order::memory_order_relaxed));

             break;
         }
         case  ProfilingState::NotConnected:
         {
             ProfilingState expectedState = m_State.load(std::memory_order::memory_order_relaxed);
             do {
                 if (!IsOneOfStates(expectedState, ProfilingState::Uninitialised, ProfilingState::NotConnected,
                                    ProfilingState::Active))
                 {
                     throw armnn::Exception(std::string("Cannot transition from state [")
                                            + GetProfilingStateName(expectedState)
                                            +"] to [" + GetProfilingStateName(newState) + "]");
                 }
             } while (!m_State.compare_exchange_strong(expectedState, newState,
                      std::memory_order::memory_order_relaxed));

             break;
         }
         case ProfilingState::WaitingForAck:
         {
             ProfilingState expectedState = m_State.load(std::memory_order::memory_order_relaxed);
             do {
                 if (!IsOneOfStates(expectedState, ProfilingState::NotConnected, ProfilingState::WaitingForAck))
                 {
                     throw armnn::Exception(std::string("Cannot transition from state [")
                                            + GetProfilingStateName(expectedState)
                                            +"] to [" + GetProfilingStateName(newState) + "]");
                 }
             } while (!m_State.compare_exchange_strong(expectedState, newState,
                      std::memory_order::memory_order_relaxed));

             break;
         }
         case ProfilingState::Active:
         {
             ProfilingState expectedState = m_State.load(std::memory_order::memory_order_relaxed);
             do {
                 if (!IsOneOfStates(expectedState, ProfilingState::WaitingForAck, ProfilingState::Active))
                 {
                     throw armnn::Exception(std::string("Cannot transition from state [")
                                            + GetProfilingStateName(expectedState)
                                            +"] to [" + GetProfilingStateName(newState) + "]");
                 }
             } while (!m_State.compare_exchange_strong(expectedState, newState,
                      std::memory_order::memory_order_relaxed));

             break;
         }
         default:
             break;
     }
}

} //namespace profiling

} //namespace armnn