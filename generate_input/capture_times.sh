#!/bin/bash

# Iterate over files with .err extension
for file in ./*.err; do
    # Extract file name
    filename=$(basename "$file")
    
    # Extract and print "real" times from the file, tab-separated
    grep -hoE 'real\s+[0-9]+m[0-9.]+s' "$file" | while read -r time; do
        echo -e "$filename\t$time"
    done
done

