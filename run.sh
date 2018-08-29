#!/bin/bash

debug_str="import pydevd;pydevd.settrace('localhost', port=8081, stdoutToServer=True, stderrToServer=True)"
# pydevd module path
export PYTHONPATH=/home/shhs/Desktop/user/soft/pycharm-2018.1.4/debug-eggs/pycharm-debug-py3k.egg_FILES

insert_debug_string()
{
    file=$1
    line=$2
    debug_string=$3
    debug=$4

    value=`sed -n ${line}p "$file"`
    if [ "$value" != "$debug_str" ] && [ "$debug" = debug ]
    then
    echo "++Insert $debug_string in line_${line}++"
    sed -i "${line}i $debug_str" "$file"
    fi
}

delete_debug_string()
{
    file=$1
    line=$2
    debug_string=$3

    value=`sed -n ${line}p "$file"`
    if [ "$value" = "$debug_str" ]
    then
    echo "--Delete $debug_string in line_${line}--"
    sed -i "${line}d" "$file"
    fi
}

# python3.6 tf_1_6
source $HOME/anaconda3/bin/activate tf_1_6
export LD_LIBRARY_PATH=/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

if [ "$1" = build ]
then
    python setup.py build_ext --inplace

elif [ "$1" = "jupyter" ]
then
    jupyter notebook --browser google-chrome

elif [ "$1" = "examples/Non RGB Example.ipynb" ]
then
    # ./run.sh "examples/Non RGB Example.ipynb" debug

    debug=$2
    if [ $debug = debug ]
    then
        cd examples
        file="Non RGB Example.py"
        line=15
        insert_debug_string "$file" $line "$debug_str" $debug
        python "$file"
        delete_debug_string "$file" $line "$debug_str"
    else
        jupyter notebook --browser google-chrome
    fi

elif [ "$1" = "examples/inference.py" ]
then

    debug=$2
    if [ $debug = debug ]
    then
    # ./run.sh "examples/inference.py" debug build
        if [ $3 = build ]
        then
            python setup.py build_ext --inplace
        fi
        cd examples
        file="inference.py"
        line=4
        insert_debug_string "$file" $line "$debug_str" $debug
        python "$file" im2.png anno2.png test.png
        delete_debug_string "$file" $line "$debug_str"

    else
        cd examples
        python inference.py im2.png anno2.png test.png
    fi

else
    echo NoParameter
fi