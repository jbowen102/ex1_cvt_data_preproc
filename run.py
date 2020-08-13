import os
import wpfix
import csv


class FilenameError(Exception):
    pass


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


def data_read(path, type, run_num):
    data_list = []
    data_dict = {}
    # Open file with read priveleges. file automatically closed at end of "with" block.
    with open(path, 'r') as asciifile01:
        print("Reading %s data from ASCII..." % type) # debug
        file_in = csv.reader(asciifile01, delimiter="\t")
        # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python
        if type == "eDAQ":
            offset =

        i = 0
        for row in file_in:
            if type == "INCA" and i >= 5:
                data_list.append(row)

                data_dict['time'].append(float(row[0]))
                data_dict['pedal_sw'].append(float(row[1]))
                data_dict['engine_spd'].append(float(row[2]))
                data_dict['throttle'].append(float(row[3]))

            elif type == "eDAQ" and i >= 1:
                pass
            elif type == "eDAQ" and i == 1:


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

# data_read(INCA_data_path, "INCA")
# data_read(INCA_data_path, "eDAQ")

# Read into dictionary instead of list? Or both?
# Need to read data differently from eDAQ file.


# find first time pedal goes logical high in both.
# Delete all data before that point (later change to have a buffer before - measured in time, not data points.)
# Don't delete column headers
# Offset time values to stat at zero
