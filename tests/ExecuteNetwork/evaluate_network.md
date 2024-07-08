# Evaluate Tensorflow Lite script.

This script will run a TfLite model through ExecuteNetwork evaluating its performance and accuracy on all available backends and with some available performance options. This script is designed to be used on aarch64 Linux and Android targets.

## Usage
__evaluate_network.sh -e \<Path to ExecuteNetwork> -m \<Tflite model to test> [-a]__

The script takes two mandatory parameters. The first, -e, is the directory containing the prebuilt execute network binary. The second, -m, is the path to the Tf Lite model to be evaluated. A third optional parameter, -a, indicates the script should use Android debug bridge to connect to an Android device. For example:

```bash
evaluate_network.sh -e ./build/release/armnn/test -m ./my_tflite_model.tflite
```
or
```bash
evaluate_network.sh -e /data/local/tmp -m /data/local/tmp/my_tflite_model.tflite -a
```

## Prerequisites of your built execute network binary

* Built for an Aarch64 Linux or Android target
* CpuRef must be enabled (-DARMNNREF=1)
* The TfLite delegate must be enabled (-DBUILD_CLASSIC_DELEGATE=1)
* The TfLite parser must be enabled (-DBUILD_TF_LITE_PARSER=1)
* Any backend you want to test against. E.g. -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1

## Prerequisites of the model
* The model must be fully supported by Arm NN.

## Prerequisites of the Android environment

* Should be accessible via Android debug bridge.
* The adb executable should be in the path.
* The path to the ExecuteNetwork executable should
  * Also contain the Arm NN shared libraries.
  * Should be writable if you intend to use the GpuAcc backend.

## What tests are performed?

* Initial validation
  * Checks that the mandatory parameters point to valid locations.
  * Determines what backends are both built into the Execute Network binary and can execute on the target platform.
  * Checks that the TfLite delegate is supported by the binary.
  * Checks that the model is fully supported by Arm NN.
* Accuracy: for each available backend it will
  * Execute the model with input tensors set to all zeros.
  * Compare the results against running the model via the TfLite reference implementation.
  * The results are expressed as an RMS error between resultant tensors.
* Performance: for each available backend it will
  * Execute an inference 6 times.
  * Print the measured "model load" and "model optimization" times.
  * Print the execution time of the first inference. This is considered the "initial inference". Generally, this is longer as some kernel compilation may be required.
  * Average the remaining 5 inference times.
  * For the CpuAcc backend, if available, it will re-run the 6 performance inferences with:
    * "number-of-threads" set values between 1 and 12 printing the average inference times for each.
    * "fp16-turbo-mode" enabled it will print the average inference times.
    * "enable-fast-math" enabled it will print the average inference times.
  * For the GpuAcc backend, if available, it will re-run the 6 performance inferences with:
    * "fp16-turbo-mode" enabled it will print the average inference times.
    * "enable-fast-math" enabled it will print the average inference times.
    * "tuning-level/tuning-path" it will cycle through values 1 to 3 and printing average inference times.

## Worked examples

The following examples were run on an Odroid N2+ (4xCortex-A73, 2xCortex-A53, Mali-G52 GPU)

First using an int8 mobilenet v1 TfLite model:
```
~/$ ./evaluate_network.sh -e . -m ./mobilenet_v1_1.0_224_quant.tflite
Using Execute Network from			: ./ExecuteNetwork
Available backends on this executable		: GpuAcc CpuAcc CpuRef 
Is the delegate supported on this executable?	: Yes
Is the model fully supported by Arm NN?		: Yes
===================================================================================
BACKEND		ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)
GpuAcc		OK 		5.68		3.63			121.14			47.242	
CpuAcc		OK 		9.47		8.37			141.30			45.366	
CpuRef		OK 		6.54		3.21			8570.74			8585.3	

CpuAcc optimizations.
============================
The value of "number-of-threads" parameter by default is decided on by the backend.
Cycle through number-of-threads=1 -> 12 and see if any are faster than the default.

 "--number-of-threads 3" resulted in a faster average inference by 11.348 ms. (34.018 v 45.366)
 "--number-of-threads 4" resulted in a faster average inference by 6.992 ms. (38.374 v 45.366)
 "--number-of-threads 5" resulted in a faster average inference by 2.664 ms. (42.702 v 45.366)
 "--number-of-threads 6" resulted in a faster average inference by 2.060 ms. (43.306 v 45.366)
 "--number-of-threads 7" resulted in a faster average inference by 18.016 ms. (27.35 v 45.366)
 "--number-of-threads 8" resulted in a faster average inference by 18.792 ms. (26.574 v 45.366)
 "--number-of-threads 9" resulted in a faster average inference by 15.294 ms. (30.072 v 45.366)
 "--number-of-threads 10" resulted in a faster average inference by 16.820 ms. (28.546 v 45.366)
 "--number-of-threads 11" resulted in a faster average inference by 16.130 ms. (29.236 v 45.366)
 "--number-of-threads 12" resulted in a faster average inference by 16.134 ms. (29.232 v 45.366)

Now tryng to enable fp16-turbo-mode. This will only have positive results with fp32 models.
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		28.40		5.68			94.65			41.84				3.526  (41.84 v 45.366)

Now tryng "enable-fast-math".
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		61.05		5.79			92.53			42.036				3.330  (42.036 v 45.366)

GpuAcc optimizations.
============================

Now tryng to enable fp16-turbo-mode. This will only have positive results with fp32 models.
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		18.86		3.92			78.16			42.738				4.504  (42.738 v 47.242)

Now tryng "enable-fast-math".
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		6.60		3.88			78.06			43.47				3.772  (43.47 v 47.242)

Now tryng "tuning-level/tuning-path".
 "--tuning-level 1" resulted in a faster average inference by -3.652 ms. (43.59 v 47.242)
 "--tuning-level 2" resulted in a faster average inference by -3.718 ms. (43.524 v 47.242)
 "--tuning-level 3" resulted in a faster average inference by -4.624 ms. (42.618 v 47.242)
```
Looking at the results, the fastest execution mechanism for this model is using CpuAcc and setting -number-of-threads 8. The average time of this inference being almost twice as fast as the default CpuAcc execution. Unsurprisingly with an int8 model the GPU parameters didn't improve its execution times by much.

This next example is a fp32 resnet50 v2 TfLite model.
```
~/$ ./evaluate_network.sh -e . -m ./resnet50_v2_batch_fixed_fp32.tflite 
Using Execute Network from			: ./ExecuteNetwork
Available backends on this executable		: GpuAcc CpuAcc CpuRef 
Is the delegate supported on this executable?	: Yes
Is the model fully supported by Arm NN?		: Yes
===================================================================================
BACKEND		ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)
GpuAcc		OK 		144.54		31.19			779.37			220.274	
CpuAcc		OK 		152.36		28.55			1309.72			284.556	
CpuRef		OK 		5.13		8.70			39374.79			39349.9	

CpuAcc optimizations.
============================
The value of "number-of-threads" parameter by default is decided on by the backend.
Cycle through number-of-threads=1 -> 12 and see if any are faster than the default.

 "--number-of-threads 2" resulted in a faster average inference by 7.078 ms. (277.478 v 284.556)
 "--number-of-threads 3" resulted in a faster average inference by 80.326 ms. (204.23 v 284.556)
 "--number-of-threads 4" resulted in a faster average inference by 116.096 ms. (168.46 v 284.556)
 "--number-of-threads 5" resulted in a faster average inference by 64.658 ms. (219.898 v 284.556)
 "--number-of-threads 6" resulted in a faster average inference by 76.662 ms. (207.894 v 284.556)
 "--number-of-threads 7" resulted in a faster average inference by 63.524 ms. (221.032 v 284.556)
 "--number-of-threads 8" resulted in a faster average inference by 108.138 ms. (176.418 v 284.556)
 "--number-of-threads 9" resulted in a faster average inference by 117.110 ms. (167.446 v 284.556)
 "--number-of-threads 10" resulted in a faster average inference by 115.042 ms. (169.514 v 284.556)
 "--number-of-threads 11" resulted in a faster average inference by 100.866 ms. (183.69 v 284.556)
 "--number-of-threads 12" resulted in a faster average inference by 97.302 ms. (187.254 v 284.556)

Now tryng to enable fp16-turbo-mode. This will only have positive results with fp32 models.
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		184.41		37.74			1486.33			278.828				5.728  (278.828 v 284.556)

Now tryng "enable-fast-math".
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		183.09		44.90			1438.94			279.976				4.580  (279.976 v 284.556)

GpuAcc optimizations.
============================

Now tryng to enable fp16-turbo-mode. This will only have positive results with fp32 models.
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		5.20		277.20			303.70			184.028				36.246  (184.028 v 220.274)

Now tryng "enable-fast-math".
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		190.88		27.11			775.53			222.564				**No improvment**

Now tryng "tuning-level/tuning-path".
 "--tuning-level 1" did not result in a faster average inference time. (223.06 v 220.274)
 "--tuning-level 2" did not result in a faster average inference time. (222.72 v 220.274)
 "--tuning-level 3" did not result in a faster average inference time. (222.958 v 220.274)
```
Again for this model CpuAcc with --number-of-threads 9 produced the fastest inference. However, you can see how adding --fp16-turbo-mode to GpuAcc almost brings it to the same performance level as CpuAcc.

## Android worked example

The following example was run on an HiKey 960 (4 Cortex A73 + 4 Cortex A53, Mali G71 GPU).

```
~/$ ./evaluate_network_android.sh -e /data/local/tmp/ -m /data/local/tmp/resnet50_v2_batch_fixed_fp32.tflite -a
Using adb from					: /usr/bin/adb
Using Execute Network from			: /data/local/tmp/ExecuteNetwork
Available backends on this executable		: GpuAcc CpuAcc CpuRef 
Looking for 64bit libOpenCL.so			: /vendor/lib64
Looking for 64bit libGLES_mali.so		: /vendor/lib64/egl
Is the delegate supported on this executable?	: Yes
Is the model fully supported by Arm NN?		: Yes
Arm NN ABI version is				: v33.1.0
===================================================================================
BACKEND		ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)
GpuAcc		OK 		30.30		18.51			691.39			142.226	
CpuAcc		OK 		33.33		19.20			1307.41			207.134	
CpuRef		OK 		29.53		7.94			36039.98			36039.8	

CpuAcc optimizations.
============================
The value of "number-of-threads" parameter by default is decided on by the backend.
Cycle through number-of-threads=1 -> 12 and see if any are faster than the default.

No value of "number-of-threads" was faster than the default.

Now trying to enable fp16-turbo-mode. This will only have positive results with fp32 models.
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		26.63		19.34			1504.11			242.78				**No improvment**

Now trying "enable-fast-math".
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		27.43		19.41			1475.96			245.296				**No improvment**

GpuAcc optimizations.
============================

Now trying to enable fp16-turbo-mode. This will only have positive results with fp32 models.
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		30.40		302.67			192.75			136.1				6.126  (136.1 v 142.226)

Now trying "enable-fast-math".
ACCURACY	MODEL LOAD(ms)	OPTIMIZATION(ms)	INITIAL INFERENCE(ms)	AVERAGE INFERENCE(ms)		DELTA(ms)
OK 		27.21		18.58			742.81			142.628				**No improvment**

Now trying "tuning-level/tuning-path".
 "--tuning-level 1" did not result in a faster average inference time. (142.342 v 142.226)
 "--tuning-level 2" did not result in a faster average inference time. (144.034 v 142.226)
 "--tuning-level 3" resulted in a faster average inference by -.602 ms. (141.624 v 142.226)
```