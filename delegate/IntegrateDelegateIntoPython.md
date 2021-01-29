# Integrate the TfLite delegate into TfLite using Python
If you have built the TfLite delegate as a separate dynamic library then this tutorial will show you how you can
integrate it in TfLite to run models using python.

Here is an example python script showing how to do this. In this script we are making use of the 
[external adaptor](https://www.tensorflow.org/lite/performance/implementing_delegate#option_2_leverage_external_delegate) 
tool of TfLite that allows you to load delegates at runtime.
```python
import numpy as np
import tflite_runtime.interpreter as tflite

# Load TFLite model and allocate tensors.
# (if you are using the complete tensorflow package you can find load_delegate in tf.experimental.load_delegate)
armnn_delegate = tflite.load_delegate( library="<your-armnn-build-dir>/delegate/libarmnnDelegate.so",
                                       options={"backends": "CpuAcc,GpuAcc,CpuRef", "logging-severity":"info"})
# Delegates/Executes all operations supported by ArmNN to/with ArmNN
interpreter = tflite.Interpreter(model_path="<your-armnn-repo-dir>/delegate/python/test/test_data/mock_model.tflite", 
                                 experimental_delegates=[armnn_delegate])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# Print out result
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

# Prepare the environment
Pre-requisites:
 * Dynamically build Arm NN Delegate library
 * python3 (Depends on TfLite version)
 * virtualenv
 * numpy (Depends on TfLite version)
 * tflite_runtime (>=2.0, depends on Arm NN Delegate)

If you haven't built the delegate yet then take a look at the [build guide](./BuildGuideNative.md).

We recommend creating a virtual environment for this tutorial. For the following code to work python3 is needed. Please
also check the documentation of the TfLite version you want to use. There might be additional prerequisites for the python
version.
```bash
# Install python3 (We ended up with python3.5.3) and virtualenv
sudo apt-get install python3-pip
sudo pip3 install virtualenv

# create a virtual environment
cd your/tutorial/dir
# creates a directory myenv at the current location
virtualenv -p python3 myenv 
# activate the environment
source myenv/bin/activate
```

Now that the environment is active we can install additional packages we need for our example script. As you can see 
in the python script at the start of this page, this tutorial uses the `tflite_runtime` rather than the whole tensorflow 
package. The `tflite_runtime` is a package that wraps the TfLite Interpreter. Therefore it can only be used to run inferences of 
TfLite models. But since Arm NN is only an inference engine itself this is a perfect match. The 
`tflite_runtime` is also much smaller than the whole tensorflow package and better suited to run models on 
mobile and embedded devices.

At the time of writing, there are no packages of either `tensorflow` or `tflite_runtime` available on `pypi` that 
are built for an arm architecture. That means installing them using `pip` on your development board is currently not 
possible. The TfLite [website](https://www.tensorflow.org/lite/guide/python) points you at prebuilt `tflite_runtime` 
packages. However, that limits you to specific TfLite and Python versions. For this reason we will build the 
`tflite_runtime` from source.

You will have downloaded the tensorflow repository in order to build the Arm NN delegate. In there you can find further 
instructions on how to build the `tflite_runtime` under `tensorflow/lite/tools/pip_package/README.md`. This tutorial 
uses bazel to build it natively but there are scripts for cross-compilation available as well.
```bash
# Add the directory where bazel is built to your PATH so that the script can find it
PATH=$PATH:your/build/dir/bazel/output
# Run the following script to build tflite_runtime natively.
tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh
```
The execution of the script creates a `.whl` file which can be used by `pip` to install the TfLite Runtime package. 
The build-script produces some output in which you can find the location where the `.whl` file was created. Then all that is 
left to do is to install all necessary python packages with `pip`.
```bash
pip install tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.3.1-py3-none-any.whl numpy
```

Your virtual environment is now all setup. Copy the final python script into a python file e.g. 
`ExternalDelegatePythonTutorial.py`. Modify the python script above and replace `<your-armnn-build-dir>` and 
`<your-armnn-repo-dir>` with the directories you have set up. If you've been using the [native build guide](./BuildGuideNative.md) 
this will be `$BASEDIR/armnn/build` and `$BASEDIR/armnn`.

Finally, execute the script:
```bash
python ExternalDelegatePythonTutorial.py
```
The output should look similar to this:
```bash
Info: ArmNN v23.0.0

Info: Initialization time: 0.56 ms

INFO: TfLiteArmnnDelegate: Created TfLite ArmNN delegate.
[[ 12 123  16  12  11  14  20  16  20  12]]
Info: Shutdown time: 0.28 ms
```

For more details on what kind of options you can pass to the Arm NN delegate please check 
[armnn_delegate_adaptor.cpp](src/armnn_external_delegate.cpp).

You can also test the functionality of the external delegate adaptor by running some unit tests:
```bash
pip install pytest
cd armnn/delegate/python/test
# You can deselect tests that require backends that your hardware doesn't support using markers e.g. -m "not GpuAccTest"
pytest --delegate-dir="<your-armnn-build-dir>/armnn/delegate/libarmnnDelegate.so" -m "not GpuAccTest" 
```
