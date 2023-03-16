#!/bin/zsh

if [[ $# != 2 ]]; then
    echo "Usage: clean_all.zsh (directory of csv's) (destination directory)"
    exit 1
fi

directory=$1
destination=$2

# Check if the directory exists
if [[ ! -e $directory ]]; then
    echo ERROR: $directory does not exist
    exit 2
fi

if [[ ! -r $destination ]]; then
    echo ERROR: $destination does not exist
    exit 2
fi

for file in $directory/*; do 
    ./cleaning_scripts/clean_data_daily.zsh $file $destination
    echo Daily: $file
done

for file in $destination/*; do
    ./cleaning_scripts/remove_blanks.zsh $file $destination
    echo Blanks: $file
    if [[ $? == 0 ]]; then
        echo Successfully cleaned $file
    else
        echo Error cleaning $file
    fi
done

rm $destination/*daily.csv

for file in $directory/*; do
    
done
