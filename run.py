import os
import wpfix
import csv


class FilenameError(Exception):
    pass

class DataReadError(Exception):
    pass

# This isn't used, but I found a way to digest the input string and escape the backslashes
# that I didn't know before.
def path_intake(data_type):
    """Data type should be 'eDAQ' or 'INCA'
    Returns correctly-formatted file path string."""
    data_path_win = input("Enter file name of %s data ready for sync "
                            "(include 'r' before the string):\n>" % data_type)
    # coerce Windows path into Linux convention
    data_path = wpfix.wpfix("%s" % data_path_win)
    # make sure the path exists on the system before proceeding.
    # also check that you're not importing the wrong data type.
    # Assumption is you have INCA or eDAQ in the path.
    if not os.path.exists(data_path):
        raise FilenameError("Bad path input: %s" % data_path)
    elif data_type not in data_path:
        raise FilenameError("Bad path input - not %s data: %s" % (data_type, data_path))
    else:
        return data_path


def path_find():
    """Data type should be 'eDAQ' or 'INCA'
    Returns correctly-formatted file path string."""
    user_run_num = input("Enter run num (four digits)\n> ")
    if len(user_run_num) < 4:
        raise FilenameError("Need a four-digit number")

    inca_dir = "./raw_data/INCA/"
    all_inca_runs = os.listdir(inca_dir)
    found_inca = False # initialize to false. Will change if file is found.
    for inca_run in all_inca_runs:
        run_num_i = run_name_parse(inca_run)
        if run_num_i == user_run_num:
            # break out of look while "inca_run" is set to correct filename
            found_inca = True
            break

    if found_inca:
        INCA_data_path = os.path.join(inca_dir, inca_run) # this is a relative path
    else:
        raise FilenameError("No INCA file found for run %s" % user_run_num)

    if not os.path.exists(INCA_data_path):
        raise FilenameError("Bad path: %s" % INCA_data_path)

    # Now use that run number to find the right eDAQ file
    eDAQ_file_num = user_run_num[0:2]

    eDAQ_dir = "./raw_data/eDAQ/"
    all_eDAQ_runs = os.listdir(eDAQ_dir)
    found_eDAQ = False # initialize to false. Will change if file is found.
    for eDAQ_run in all_eDAQ_runs:
        # Split the extension off the file name, then isolate the final two numbers off the date
        run_num_i = os.path.splitext(eDAQ_run)[0].split("_")[1]
        if run_num_i == eDAQ_file_num:
            # break out of look while "eDAQ_run" is set to correct filename
            found_eDAQ = True
            break
    if found_eDAQ:
        eDAQ_data_path = os.path.join(eDAQ_dir, eDAQ_run) # this is a relative path
    else:
        raise FilenameError("No eDAQ file found for run %s" % eDAQ_file_num)

    if not os.path.exists(eDAQ_data_path):
        raise FilenameError("Bad path: %s" % eDAQ_data_path)

    return user_run_num, INCA_data_path, eDAQ_data_path


def run_name_parse(filename):
    """Assuming INCA data type
    Returns run number."""
    run_num = filename.split("_")[1][0:4]
    return run_num


def find_edaq_col_offset(header_row, sub_run_num):
    """Takes in an eDAQ file's header row and finds the first column header
    containing the indicated run number.
    Returns the index of the first such column."""
    sub_run_num_edaq_format = "RN_"+str(sub_run_num)

    found_col = False
    for n, col in enumerate(header_row):
        if sub_run_num_edaq_format in col: # converting to int (abov) and back to string strips off the zero padding
            # save that index to reference for the rest of the main loop
            run_start_col = n
            found_col = True
            break

    if not found_col:
        # got to end of row and didn't find the run in any column heading
        raise DataReadError("Can't find %s in file %s" % (sub_run_num_edaq_format, eDAQ_path))

    return run_start_col


def data_read(INCA_path, eDAQ_path, run_num_text):
    # run_num = int(run_num_text)
    sub_run_num = int(run_num_text[2:4])

    INCA_data_list = []
    INCA_data_dict = {}

    # Read in both eDAQ and INCA data for specific run.
    # read INCA data first
    # Open file with read priveleges. file automatically closed at end of "with/as" block.
    with open(INCA_path, 'r') as inca_ascii_file:
        print("Reading INCA data from ASCII...") # debug
        INCA_file_in = csv.reader(inca_ascii_file, delimiter="\t")
        # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

        for i, INCA_row in enumerate(INCA_file_in):
            if i >= 5:
                INCA_data_list.append(INCA_row)

                INCA_data_dict['time'].append(float(INCA_row[0]))
                INCA_data_dict['pedal_sw'].append(float(INCA_row[1]))
                INCA_data_dict['engine_spd'].append(float(INCA_row[2]))
                INCA_data_dict['throttle'].append(float(INCA_row[3]))
    print("...done")

    # magic number for how many channels per run in the eDAQ files.
    # eDAQ_channel_count = 3

    eDAQ_data_list = []
    eDAQ_data_dict = {}
    # now read eDAQ data
    with open(eDAQ_path, 'r') as edaq_ascii_file:
        print("Reading eDAQ data from ASCII...") # debug
        eDAQ_file_in = csv.reader(edaq_ascii_file, delimiter="\t")
        # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

        for j, eDAQ_row in enumerate(eDAQ_file_in):
            if j == 0:
                # The first row is a list of channel names.
                # Loop through and find the first channel for this run.
                run_start_col = find_edaq_col_offset(eDAQ_row, sub_run_num)

            elif j > 0:
                # only add this run's channels to our data list. Don't forget the first column is always time though.
                eDAQ_data_list.append([ eDAQ_row[0], eDAQ_row[run_start_col+1], eDAQ_row[run_start_col+2] ])

                eDAQ_data_dict['time'].append(float(eDAQ_row[0]))
                # not reading in the first channel because it's pedal voltage and not needed.
                eDAQ_data_dict['gd_speed'].append(float(eDAQ_row[run_start_col+1]))
                eDAQ_data_dict['pedal_sw'].append(float(eDAQ_row[run_start_col+2]))
    print("...done")



    print("...done")
    return data_list, data_dict


# Read in data from INCA file
# INCA_data_path = path_intake("INCA")
# eDAQ_data_path = path_intake("eDAQ") # better to do this automatically
# print(INCA_data_path) # debug

[run_num, INCA_data_path, eDAQ_data_path] = path_find()
print(run_num) # debug
print(INCA_data_path) # debug
print(eDAQ_data_path) # debug

# as written, eDAQ file will have to be repeatedly opened and read for each separate INCA run.
# if this ends up too slow, program to be re-written a different way. It's probably fine now though.
data_read(INCA_data_path, eDAQ_data_path, run_num)

# Read into dictionary instead of list? Or both?
# Need to read data differently from eDAQ file.


# find first time pedal goes logical high in both.
# Delete all data before that point (later change to have a buffer before - measured in time, not data points.)
# Don't delete column headers
# Offset time values to stat at zero
