#!/bin/zsh

if [[ $# != 2 ]]; then
    echo "Usage: clean_data_daily.zsh (filename with extention) (destination directory)"
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
new_filename="${filename%.*}_daily.csv"
new_filepath="${destination}/${new_filename}"

# Check if a file exists at the destination
if [[ -e $new_filepath ]]; then
    echo ERROR: there is a file at the destination
    exit 3
fi

head -n 1 $old_filepath > $destination/temp.csv
awk -f ./cleaning_scripts/remove_hourly_lines.awk $old_filepath >> $destination/temp.csv

ROWS_TO_DROP="2,27-46"
cut -f$ROWS_TO_DROP -d, $destination/temp.csv > $new_filepath

rm $destination/temp.csv


