#!/bin/zsh

if [[ $# != 2 ]]; then
    echo "Usage: remove_blanks.zsh (filename with extention) (destination directory)"
    exit 1
fi

# Check if the file is readable 
if [[ ! -r $1 ]]; then
    echo ERROR: $1 is not readable
    exit 2
fi

old_filepath=$1
destination=$2
filename=`basename $old_filepath`
new_filename="${filename%.*}_no_blanks.csv"
new_filepath="${destination}/${new_filename}"

# Check if a file exists at the destination
if [[ -e $new_filepath ]]; then
    echo ERROR: there is a file at the destination
    exit 3
fi

grep -ve '""' $old_filepath > $new_filepath