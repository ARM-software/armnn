#!/bin/bash
set -e

JENKINS_BUILD=0
while getopts ":j" opt; do
    case "$opt" in
        j) JENKINS_BUILD=1 ;;
    esac
done

python3.6 -m venv toxenv
source toxenv/bin/activate
pip install virtualenv==16.3.0 tox

export  ARMNN_INCLUDE=$(pwd)/../../include
python ./swig_generate.py

tox -e devenv
# If jenkins build, also run unit tests, generate docs, etc
if [ $JENKINS_BUILD == 1 ]; then
    tox
    tox -e doc
fi

deactivate
rm -rf toxenv

source env/bin/activate
