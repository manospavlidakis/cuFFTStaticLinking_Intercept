#!/bin/bash

# specify the directory path
dir="."

# loop through all files in the directory
for file in "$dir"/*
do
    if [ "${file: -3}" != ".sh" ]; then
	    # append the string to the end of the file
	    #echo '.include "libcufft_static.117.sm_80.ptx"' >> "$file"
	    sed -i 's/\.include "libcufft_static\.117\.sm_80\.ptx"//g' "$file" 
    fi
done
