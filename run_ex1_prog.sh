#!/bin/bash

# If not mounted already, mount the P network drive to pull data from.
if [ "$(ls -A /mnt/p/)" ]
then
	: # do nothing
else
	printf "P: drive not mounted. Enter password if prompted.\n"
	sudo ${HOME}/pu_automount
fi


# Define path variables
LOCAL_RAW="$HOME/ex1_cvt_data_preproc/raw_data"
LOCAL_SYNC="$HOME/ex1_cvt_data_preproc/sync_data"
LOCAL_FIG="$HOME/ex1_cvt_data_preproc/figs"


# Create raw_data folders if they doesn't exist already
if [ -d "$LOCAL_RAW" ];
then
	: # do nothing
else
	mkdir "$LOCAL_RAW"
fi

if [ -d "$LOCAL_RAW/INCA/" ];
then
	: # do nothing
else
	mkdir "$LOCAL_RAW/INCA/"
fi

if [ -d "$LOCAL_RAW/eDAQ/" ];
then
	: # do nothing
else
	mkdir "$LOCAL_RAW/eDAQ/"
fi


# Purge local output folders
if [ -d ${LOCAL_SYNC} ] && [ "$(ls -A ${LOCAL_SYNC}/)" ]
then
	rm ${LOCAL_SYNC}/*
fi

if [ -d ${LOCAL_FIG} ] && [ "$(ls -A ${LOCAL_FIG}/)" ]
then
	rm ${LOCAL_FIG}/*
fi


# Copy data from P drive folder into local project's directory structure.
REMOTE_RAW="/mnt/p/Current Projects/100840 GOAT Engine/Prod Engg/Test_&_Reliab_Svcs/CVT/RAW_DATA_TO_SYNC"
REMOTE_SYNC="/mnt/p/Current Projects/100840 GOAT Engine/Prod Engg/Test_&_Reliab_Svcs/CVT/SYNCED_DATA"

if [ "$(ls -A "${REMOTE_RAW}/INCA/")" ]
then
	printf "\nCopying data from P: drive RAW_DATA_TO_SYNC/INCA folder...\n"
	rsync -aziv --delete "${REMOTE_RAW}"/INCA/ ${LOCAL_RAW}/INCA | sed  's/^/  /'
	# Rsync only copies files that have changed or don't exist already in local
	# raw_data folder. delete option removes extraneous residual files.
	# Piping to 'sed' bash program to indent rsync output to aid readability.
	printf "Finished copying INCA data.\n"
else
	printf "Nothing found in P: drive RAW_DATA_TO_SYNC/INCA/ folder. Exiting.\n"
	exit 1
fi

if [ "$(ls -A "${REMOTE_RAW}/eDAQ/")" ]
then
	printf "\nCopying data from P: drive RAW_DATA_TO_SYNC/eDAQ folder...\n"
	rsync -aziv --delete "${REMOTE_RAW}"/eDAQ/ ${LOCAL_RAW}/eDAQ | sed  's/^/  /'
	# Rsync only copies files that have changed or don't exist already in local
	# raw_data folder. delete option removes extraneous residual files.
	# Piping to 'sed' bash program to indent rsync output to aid readability.
	printf "Finished copying eDAQ data.\n"
else
	printf "Nothing found in P: drive RAW_DATA_TO_SYNC/INCA/ folder. Exiting.\n"
	exit 1
fi


# Make new folder in SYNCED_DATA every time program run.
# Needed because fig file names change each time so new ones won't always
# replace old ones.
# Also, user doesn't want extraneous data purged from folder before transfer.
TIMESTAMP="$(date "+%Y-%m-%dT%H%M%S")"
mkdir "${REMOTE_SYNC}/${TIMESTAMP}"


# Navigate into the local project then run the python script, storing its return value.
cd ~/ex1_cvt_data_preproc # Need to be in the project dir for the Python script to run its main function.

printf "\nStarting Python script:\n"
python3 run.py --auto --over --plot --ignore-warn --log-dir "${REMOTE_SYNC}/${TIMESTAMP}"
PYTHON_RETURN=$? # gets return value of last command executed.

# Make sure the program ran correctly
if [ ${PYTHON_RETURN} -ne 0 ]; then
	printf "\nSomething went wrong when executing the Python program.\n"
	exit 1
else
	printf "Finished Python script.\n"
fi


# Copy synced data to P: drive.
printf "\nCopying synced data (CSVs) to P:/.../SYNCED_DATA/${TIMESTAMP} folder...\n"
rsync -rchziv --progress ${LOCAL_SYNC}/ "${REMOTE_SYNC}/${TIMESTAMP}" | sed  's/^/  /'
printf "Finished transferring CSV data to P: drive.\n"

printf "\nCopying plots to P:/.../SYNCED_DATA/${TIMESTAMP} folder...\n"
rsync -rchziv --progress ${LOCAL_FIG}/ "${REMOTE_SYNC}/${TIMESTAMP}" | sed  's/^/  /'
printf "Finished transferring plots to P: drive.\n"
# -c: Looks at checksum to determine if file needs updating. NTFS mod times not
# preserved in transfer, so standard size-or-mod time criterion not a reliable
# indicator of difference. Not critical since exporting to new folder each time.
# Piping to 'sed' bash program to indent rsync output to aid readability.

# References:
# https://stackoverflow.com/questions/6834487/what-is-the-dollar-question-mark-variable-in-shell-scripting
# https://stackoverflow.com/questions/14561475/use-bash-home-in-shell-script#14561608
# https://stackoverflow.com/questions/4181703/how-to-concatenate-string-variables-in-bash
# https://www.cyberciti.biz/faq/linux-unix-shell-check-if-directory-empty/
# https://stackoverflow.com/questions/8352851/how-to-call-one-shell-script-from-another-shell-script
# https://serverfault.com/questions/151986/rsync-command-to-synchronize-two-ntfs-drives
# https://askubuntu.com/questions/609968/rsync-delete-option-doesnt-delete-files-in-target-directory
# https://stackoverflow.com/questions/17484774/indenting-multi-line-output-in-a-shell-script
