#!/bin/bash
#set -x
#
# Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#
# This script will run a TfLite model through ExecuteNetwork trying all available backends to measure
# both speed and accuracy. In addition, it will try some of the performance options that are available.
#
# Prerequisites: ExecuteNetwork must be built with:
# * CpuRef enabled (-DARMNNREF=1)
# * TfLite delegate enabled (-DBUILD_CLASSIC_DELEGATE=1)
# * TfLite parser enabled (-DBUILD_TF_LITE_PARSER=1)
# * Any backend you want to test against. E.g. -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1
# * The model must be fully supported by Arm NN.
#
# It can run on both native aarch64 linux and on Android via ADB (Android Debug Bridge)
#
# Usage:
# evaluate_network.sh -e <Path to ExecuteNetwork> -m <Tfite model to test> [-a]
#
# Sample usage:
# evaluate_network.sh -e ./build/release/armnn/test -m ./my_tflite_model.tflite
#

CMD=$( basename "$0" )

usage() {
  echo "Usage: $CMD -e <Path to ExecuteNetwork> -m <Test model> [-android]"
  echo "Options:        -e <Path to ExecuteNetwork>"
  echo "                -m <Test model>"
  echo "                -a Use ADB to run on a connected Android device."
  exit 1
}

# Errors if the previous command had a non-zero exit code.
function AssertZeroExitCode {
  EXITCODE=$?
  if [ $EXITCODE -ne 0 ]; then
    echo -e "Previous command exited with code $EXITCODE"
    exit 1
  fi
}

# Defult to Linux not Android.
USE_ADB=0

OPTION_COUNTER=0
while getopts "e:m:a" opt; do
  ((OPTION_COUNTER+=1))
  case "$opt" in
    h|\?) usage;;
    e) EXECUTE_NETWORK_PATH="$OPTARG";;
    m) MODEL="$OPTARG";;
    a) USE_ADB=1
  esac
done
shift $((OPTIND - 1))

# Both parameters are mandatory.
if [ -z "$EXECUTE_NETWORK_PATH" ] || [ -z "$MODEL" ]; then
    usage
    exit 1
fi

# Check that adb is available in the path.
if [ $USE_ADB -eq 1 ]; then
    ADB=$(which adb)
    if [ $? -eq 0 ]; then
        echo -e "Using adb from\t\t\t\t\t: $ADB"
    else
        echo "ADB was enabled but unable to locate it in the path."
        usage
        exit 1
    fi
fi

# Check the path to execute network will find the executable.
if [ $USE_ADB -eq 1 ]; then
    EXECUTE_NETWORK=$($ADB shell ls $EXECUTE_NETWORK_PATH/ExecuteNetwork 2> /dev/null)
    if [ $? -eq 0 ]; then
        echo -e "Using Execute Network from\t\t\t: $EXECUTE_NETWORK"
    else
        echo "Execute Network does not exist at \"$EXECUTE_NETWORK_PATH/ExecuteNetwork\""
        usage
        exit 1
    fi
    # We will assume the library files are in the same location as the executable.
    ADB+=" shell LD_LIBRARY_PATH=$EXECUTE_NETWORK_PATH:$EXECUTE_NETWORK_PATH/delegate"
else
    if [ -x "$EXECUTE_NETWORK_PATH/ExecuteNetwork" ]; then
        echo -e "Using Execute Network from\t\t\t: $EXECUTE_NETWORK_PATH/ExecuteNetwork"
        EXECUTE_NETWORK="$EXECUTE_NETWORK_PATH/ExecuteNetwork"
    else
        echo "Execute Network does not exist at \"$EXECUTE_NETWORK_PATH/ExecuteNetwork\""
        usage
        exit 1
    fi
fi


# Check that the model exists and has a supported extension.
MODEL_PATH=$($ADB ls $MODEL 2> /dev/null)
if [ $? -eq 0 ]; then
    if [[ ! $MODEL_PATH =~ (tflite)$ ]]; then
        echo "Only \".tflite\" files are supported."
        exit 1
    fi
else
    echo Model file: "\"$MODEL\" could not be found."
    usage
    exit 1
fi

# Find out the available backends. Unfortunaltey the list of backends spans multiple lines.
# This means we have to do this in several steps.
echo -n -e "Available backends on this executable\t\t:"
HELP_OUTOUT=`$ADB $EXECUTE_NETWORK --help 2> /dev/null`
BACKENDS=`echo $HELP_OUTOUT | sed  's/.*: \[//' | sed 's/\].*//' | sed 's/,//g'`
# Remove the leading space to make it look prettier.
BACKENDS="${BACKENDS:1}"
if [ -z "$BACKENDS" ]; then
    echo ""
    echo "Execute Network reported no available backends!"
    exit 1
else
    echo " $BACKENDS"
    # We really need the CpuRef to be in there.
    if [[ ! $BACKENDS =~ "CpuRef" ]]; then
        echo ""
        echo "Fatal: Please recompile ExecuteNetwork to include the CpuRef backend. (-DARMNNREF=1)"
        exit 1
    fi
fi

# On Android if GpuAcc is a valid backend then we need to find correct libOpenCL.so and libGLES_mali.so.
if [ $USE_ADB -eq 1 ]; then
    if [[ $BACKENDS =~ "GpuAcc" ]]; then
        echo -n -e "Looking for 64bit libOpenCL.so\t\t\t: "
        TMP_STRING=$($ADB find -name libOpenCL.so -exec file {} \\\; 2> /dev/null | grep "64-bit")
        # Only take the first path found.
        OPENCL_PATH=`echo $TMP_STRING | cut -d ' ' -f1`
        OPENCL_PATH="${OPENCL_PATH:1}"
        OPENCL_PATH="${OPENCL_PATH::-1}"
        # Reduce to parent directory.
        OPENCL_PATH="$(dirname $OPENCL_PATH)"
        echo $OPENCL_PATH

        echo -n -e "Looking for 64bit libGLES_mali.so\t\t: "
        TMP_STRING=$($ADB find -name libGLES_mali.so -exec file {} \\\; 2> /dev/null | grep "64-bit")
        # Only take the first path found.
        MALILIB_PATH=`echo $TMP_STRING | cut -d ' ' -f1`
        MALILIB_PATH="${MALILIB_PATH:1}"
        MALILIB_PATH="${MALILIB_PATH::-1}"
        # Reduce to parent directory.
        MALILIB_PATH="$(dirname $MALILIB_PATH)"
        echo $MALILIB_PATH
        # Add both paths to the LD_LIBRARY_PATH
        ADB+=":$OPENCL_PATH:$MALILIB_PATH"
    fi
fi

# This is where the real work starts.
# Model execution can take a long time. Trap ctrl-c and tell the user.
trap ctrl_c INT

function ctrl_c() {
        echo -e "Interrupted.\nNo patience eh? Try a smaller model."
        exit 1
}


# We need to check that the delegate is supported otherwise we can't run through the tf runtime.
echo -n -e "Is the delegate supported on this executable?\t:"
TFLITE_EXECUTION=`$ADB $EXECUTE_NETWORK -m $MODEL -T tflite -c CpuRef -N 2> /dev/null`
# Check for an error message about building with the delegate.
if [[ $TFLITE_EXECUTION =~ "Tensorflow-Lite delegate support" ]]; then
    echo ""
    echo "Fatal: Please recompile ExecuteNetwork with TfLite delegate support enabled. (-DBUILD_CLASSIC_DELEGATE=1)"
    exit 1
else
    echo " Yes"
fi


# Run through CpuRef to see if Arm NN supports the model.
echo -n -e "Is the model fully supported by Arm NN?\t\t:"
REF_EXECUTION=`$ADB $EXECUTE_NETWORK -m $MODEL -c CpuRef -N 2> /dev/null`
# If it failed look for the most common reason - an unsupported layer.
if [ $? -ne 0 ]; then
    if [[ $REF_EXECUTION =~ "is not supported on requested backend CpuRef" ]]; then
        echo -e " No - One or more layers are not supported by Arm NN"
    else
        echo -e " No - Execution using CpuRef backend failed."
    fi
    echo -e "The Reported problems were\t:"
    echo `echo "$REF_EXECUTION" | sed '/Warning\|ERROR\|Fatal/!d'`
    echo "To recreate this error try: \"$EXECUTE_NETWORK -m $MODEL -c CpuRef\" "
    exit 1
fi
echo " Yes"

# Extract the ABI version while we're at it.
VERSION=`echo "$REF_EXECUTION" | sed '/ArmNN/!d' | cut -d " " -f 3`
echo -e "Arm NN ABI version is\t\t\t\t: $VERSION"

# This function will execute the model and return a string representation of the results. This is the
# first time the model will be executed.
# Is done wth -c $BACKEND,CpuRef to allow the odd layer to be supported by an unaccelerated backend.
#
# Parameters:
# $1 Backend string like CpuRef.
# $2 Additional ExecuteNetwork parameters.
#
function RunAccuracyOnBackendWithParameters {
    BACKEND=$1
    ADDITIONAL_PARAM=$2
    # Run on BACKEND to check accuracy against TfLite runtime first. This will be a warning not a failure.
    ACCURACY_RUN=`$ADB $EXECUTE_NETWORK -m $MODEL -c $BACKEND $ADDITIONAL_PARAM -A -N 2> /dev/null`
    # Start by checking the return code.
    if [ $? -ne 0 ]; then
        # Maybe this backend isn't supported.
        if [[ $ACCURACY_RUN =~ "None of the preferred backends [$BACKEND ] are supported" ]]; then
            echo -e "\t\t***Is not supported***"
            return 1
        elif [[ $ACCURACY_RUN =~ "is not supported on requested backend" ]]; then
            # One or more layers require a fall back. Run again with CpuRef fall back.
            ACCURACY_RUN=`$ADB $EXECUTE_NETWORK -m $MODEL -c $BACKEND,CpuRef $ADDITIONAL_PARAM -A -N 2> /dev/null`
            REQUIRES_CPUREF="*"
        else
            # In the case of a general failure against this backend tell the user what we tried and then
            # ignore this backend.
            echo -e "\t***Execution failed. Ignoring this backend. Command was: \"$EXECUTE_NETWORK -m $MODEL -c $BACKEND -A -N\""
            return 1
        fi
    fi
    # Now check the RMS value. If it isn't 0 then mark this as questionable accuracy.
    ACCURACY_VALUE=`echo "$ACCURACY_RUN" | grep 'Byte level'`
    if [[ ! $ACCURACY_VALUE == *0 ]]; then
        ACCURACY=!`echo $ACCURACY_VALUE | sed 's/[a-zA-Z:]*//g'`
    else
        ACCURACY="OK"
    fi
    # Add on the * if we needed to add CpuRef.
    if [ -z $REQUIRES_CPUREF ]; then
        echo -e "$ACCURACY $REQUIRES_CPUREF\t\t"
    else
        echo -e "$ACCURACY\t\t"
    fi
}

# This function will execute the model and return a string representation of the results. The execution
# Is done wth -c $BACKEND,CpuRef to allow the odd layer to ot be supported by an accelerated backend.
#
# Parameters:
# $1 Backend string like CpuRef.
# $2 Additional ExecuteNetwork parameters.
#
function RunPerformanceOnBackendWithParameters {
    BACKEND=$1
    ADDITIONAL_PARAM=$2
    # Execute with 6 inferences. Mark the first as initial inference. Average the rest.
    SPEED_RUN=`$ADB $EXECUTE_NETWORK -m $MODEL -c $BACKEND,CpuRef -I 6 -N $ADDITIONAL_PARAM 2> /dev/null`

    # Extract the model load time
    MODEL_LOAD_TIME=`echo "$SPEED_RUN" | grep "Initialization time" | sed 's/[a-zA-Z:]*//g'`
    MODEL_LOAD_TIME=`echo ${MODEL_LOAD_TIME::-2}` # Remove the tailing space and full stop.
    # and the optimization time.
    OPTIMIZATION_TIME=`echo "$SPEED_RUN" | grep "Optimization time" | sed 's/[a-zA-Z:]*//g'`
    OPTIMIZATION_TIME=`echo ${OPTIMIZATION_TIME::-1}` # Remove the tailing space.

    # All 6 inference times.
    RAW_INFERENCE=`echo "$SPEED_RUN" | grep "Inference time"`
    # This will take "Info: Inference time: 0.03 ms Info:..." and transform to "0.03 0.01 0.01"
    INFERENCE_TIMES=`echo $RAW_INFERENCE | sed 's/[a-zA-Z:]*//g'`
    INITIAL_INFERENCE_TIME=`echo $INFERENCE_TIMES | cut -d ' ' -f 1`
    # Now remove the initial inference time as it will skew the average.
    INFERENCE_TIMES=`echo $INFERENCE_TIMES | sed 's/[^ ]* //'`
    # Use awk to sum and average the remaining 5 numbers.
    AVERAGE_INFERENCE_TIME=`echo $INFERENCE_TIMES | awk '{s+=$1}END{print s/NR}' RS=" "`

    # Result format is: MODEL LOAD | OPTIMIZATION | INITIAL INFERENCE | AVERAGE INFERENCE
    echo -e "$MODEL_LOAD_TIME\t\t$OPTIMIZATION_TIME\t\t\t$INITIAL_INFERENCE_TIME\t\t\t$AVERAGE_INFERENCE_TIME\t"
}


# Check execution in all available backends.
echo    "==================================================================================="
echo -e "BACKEND\t\tACCURACY\tMODEL LOAD(ms)\tOPTIMIZATION(ms)\tINITIAL INFERENCE(ms)\tAVERAGE INFERENCE(ms)"
for backend in $BACKENDS
do
    echo -n -e "$backend\t\t"
    RESULT=$(RunAccuracyOnBackendWithParameters $backend)
    echo -n -e "$RESULT"
    if [[ $RESULT =~ "*" ]]; then
        REQUIRED_CPU_REF=1
    fi
    # It's possible the backend wasn't supported.
    if [[ ! "$RESULT" =~ "not supported" ]]; then
        # It was, continue.
        RESULT=$(RunPerformanceOnBackendWithParameters $backend)
        echo -n -e "$RESULT"
        # Save some specific values for use later.
        if [ $backend == "CpuAcc" ]; then
            # In the case of CpuAcc we save the avrage inference time.
            CPUACC_AVERAGE_INFERENCE_TIME=`echo $RESULT | cut -d ' ' -f 4`
        fi
        if [ $backend == "GpuAcc" ]; then
            # In the case of GpuAcc we save the avrage inference time.
            GPUACC_AVERAGE_INFERENCE_TIME=`echo $RESULT | cut -d ' ' -f 4`
        fi
    else
        # Remove this backend from future tests.
        BACKENDS=`echo $BACKENDS | sed "s/$backend//"`
    fi
    echo
done
# Only print this if it was required.
if [ ! -z $REQUIRED_CPU_REF ]; then
    echo "* denotes this backend required fallback to CpuRef."
    echo
fi

# Now its time to look at backend specific parameters.

# This function first run the accuracy test and then the performance test. It uses the average from earlier
# to compare to.
function RunAccuracyAndPerformanceWithExtraParameter
{
    BACKEND=$1
    EXTRA_PARAM=$2
    AVERAGE_INFERENCE_TIME=$3
    echo -e "ACCURACY\tMODEL LOAD(ms)\tOPTIMIZATION(ms)\tINITIAL INFERENCE(ms)\tAVERAGE INFERENCE(ms)\t\tDELTA(ms)"
    RESULT=$(RunAccuracyOnBackendWithParameters $BACKEND,CpuRef $EXTRA_PARAM)
    echo -n "$RESULT"
    RESULT=$(RunPerformanceOnBackendWithParameters $BACKEND,CpuRef $EXTRA_PARAM)
    PARAM_AVERAGE_INFERENCE_TIME=`echo $RESULT | cut -d ' ' -f 4`
    # If adding the parameter was faster then incude by how much.
    if [[ "$PARAM_AVERAGE_INFERENCE_TIME" < "$AVERAGE_INFERENCE_TIME" ]]; then
        DELTA=`echo $AVERAGE_INFERENCE_TIME - $PARAM_AVERAGE_INFERENCE_TIME | bc`
        echo -e "$RESULT\t\t\t$DELTA  ($PARAM_AVERAGE_INFERENCE_TIME v $AVERAGE_INFERENCE_TIME)"
    else
        echo -e "$RESULT\t\t\t**No improvment**"
    fi
}


# Start with CpuAcc. Three knobs to twiddle, threads, fast-math and fp16.
if [[ $BACKENDS =~ "CpuAcc" ]]; then
    echo
    echo    "CpuAcc optimizations."
    echo    "============================"
    echo    "The value of \"number-of-threads\" parameter by default is decided on by the backend."
    echo    "Cycle through number-of-threads=1 -> 12 and see if any are faster than the default."
    echo
    for i in {1..12}
    do
        RESULT=$(RunPerformanceOnBackendWithParameters "CpuAcc,CpuRef" "--number-of-threads $i")
        AVERAGE_INFERENCE_TIME=`echo $RESULT | cut -d ' ' -f 4`
        # Print something out if the returned average is less than the previously saved average.
        if (( $(echo "$AVERAGE_INFERENCE_TIME < $CPUACC_AVERAGE_INFERENCE_TIME" | bc -l) )); then
            DELTA=`echo $CPUACC_AVERAGE_INFERENCE_TIME - $AVERAGE_INFERENCE_TIME | bc`
            echo " \"--number-of-threads $i\" resulted in a faster average inference by $DELTA ms. ($AVERAGE_INFERENCE_TIME v $CPUACC_AVERAGE_INFERENCE_TIME)"
            FASTER=1
        fi
    done
    if [ -z $FASTER ]; then
        echo "No value of \"number-of-threads\" was faster than the default."
    fi
    # Next is fp16-turbo-mode. We do both accuracy and speed on this one.
    echo
    echo -n  "Now trying to enable fp16-turbo-mode. This will only have positive results with fp32 models."
    echo
    RunAccuracyAndPerformanceWithExtraParameter CpuAcc "--fp16-turbo-mode" $CPUACC_AVERAGE_INFERENCE_TIME

    # Next is enable-fast-math. Again both accuracy and speed on this one.
    echo
    echo -n  "Now trying \"enable-fast-math\"."
    echo
    RunAccuracyAndPerformanceWithExtraParameter CpuAcc "--enable-fast-math" $CPUACC_AVERAGE_INFERENCE_TIME
fi

# GpuAcc.
# Options to check enable-fast-math, fp16-turbo-mode, and tuning-level/tuning-path.
if [[ $BACKENDS =~ "GpuAcc" ]]; then
    echo
    echo    "GpuAcc optimizations."
    echo    "============================"

    # fp16-turbo-mode. We do both accuracy and speed on this one.
    echo
    echo -n  "Now trying to enable fp16-turbo-mode. This will only have positive results with fp32 models."
    echo
    RunAccuracyAndPerformanceWithExtraParameter GpuAcc "--fp16-turbo-mode" $GPUACC_AVERAGE_INFERENCE_TIME

    # Next is enable-fast-math. Again both accuracy and speed on this one.
    echo
    echo -n  "Now trying \"enable-fast-math\"."
    echo
    RunAccuracyAndPerformanceWithExtraParameter GpuAcc "--enable-fast-math" $GPUACC_AVERAGE_INFERENCE_TIME

    # Next is tuning levels. Just speed on this one.
    echo
    echo -n  "Now trying \"tuning-level/tuning-path\"."
    echo
    for i in {1..3}
    do
        $ADB touch $EXECUTE_NETWORK_PATH/tuned-network.bin
        AssertZeroExitCode
        # Create tuned network file with the first run.
        OUTPUT=`$ADB $EXECUTE_NETWORK -m $MODEL -c $GpuAcc,CpuRef --tuning-path $EXECUTE_NETWORK_PATH/tuned-network.bin --tuning-level $i -N 2> /dev/null`
        AssertZeroExitCode
        # Now run the perforance test reusing that saved network.
        RESULT=$(RunPerformanceOnBackendWithParameters "GpuAcc,CpuRef" "--tuning-path $EXECUTE_NETWORK_PATH/tuned-network.bin")
        AVERAGE_INFERENCE_TIME=`echo $RESULT | cut -d ' ' -f 4`
        if (( $(echo "$AVERAGE_INFERENCE_TIME < $GPUACC_AVERAGE_INFERENCE_TIME" | bc -l) )); then
            DELTA=`echo $AVERAGE_INFERENCE_TIME - $GPUACC_AVERAGE_INFERENCE_TIME | bc`
            echo  " \"--tuning-level $i\" resulted in a faster average inference by $DELTA ms. ($AVERAGE_INFERENCE_TIME v $GPUACC_AVERAGE_INFERENCE_TIME)"
        else
            echo  " \"--tuning-level $i\" did not result in a faster average inference time. ($AVERAGE_INFERENCE_TIME v $GPUACC_AVERAGE_INFERENCE_TIME)"
        fi
        $ADB rm $EXECUTE_NETWORK_PATH/tuned-network.bin
    done
fi
