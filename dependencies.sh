#!/bin/bash

# check for the number arguments provided are as expected or not
num_of_arg=$#
if [ ${num_of_arg} -eq 0 ]
then
  echo "At least one argument stating the pip to be used is required. Given ${num_of_arg}"
  exit 1
fi

if [ "${1}" == "pip3" ]
then
  pip3 install numpy==1.18.1
  pip3 install pandas==0.25.3
  pip3 install matplotlib==2.1.1
  pip3 install seaborn==0.10.0
  pip3 install py-xgboost==0.90
  pip3 install scikit-learn==0.19.1
elif [ "${1}" == "pip" ]
then
  pip install numpy==1.18.1
  pip install pandas==0.25.3
  pip install matplotlib==2.1.1
  pip install seaborn==0.10.0
  pip install py-xgboost==0.90
  pip install scikit-learn==0.19.1
else
  echo "Unexpected argument: ${1}"
fi
