import os
import wpfix
import csv


class FilenameError(Exception):
    pass

class DataReadError(Exception):
    pass

class DataSyncError(Exception):
    pass



# This isn't used, but I found a way to digest the input string and escape the
# backslashes that I didn't know before.
def path_intake(data_type):
    """Data type should be "eDAQ" or "INCA"
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
        raise FilenameError("Bad path input - not %s data: %s" %
                                                        (data_type, data_path))
    else:
        return data_path


def path_find():
    """Data type should be 'eDAQ" or 'INCA'
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
        INCA_data_path = os.path.join(inca_dir, inca_run) # relative path
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
        # Split the extension off the file name, then isolate the final two
        # numbers off the date
        run_num_i = os.path.splitext(eDAQ_run)[0].split("_")[1]
        if run_num_i == eDAQ_file_num:
            # break out of look while "eDAQ_run" is set to correct filename
            found_eDAQ = True
            break
    if found_eDAQ:
        eDAQ_data_path = os.path.join(eDAQ_dir, eDAQ_run) # relative path
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
    # converting to int (in caller function) and back to str strips zero padding

    found_col = False
    for n, col in enumerate(header_row):
        if sub_run_num_edaq_format in col:
            # save that index
            run_start_col = n
            found_col = True
            break

    if not found_col:
        # got to end of row and didn't find the run in any column heading
        raise DataReadError("Can't find %s in file %s" %
                                        (sub_run_num_edaq_format, eDAQ_path))

    return run_start_col


def find_closest(t_val, t_list):
    """
    Takes in a time value and a list of times.
    Returns the index of the closest time value in the list.
    Linear search - O(n) time complexity. Bisection search would be better but
    probably unnecessary.
    (Originally part of Alt_Test_Data program. Modified here for Python3)
    """

    time_index = 0
    smallest_diff = (t_val - t_list[time_index]) ** 2
    for i, time in enumerate(t_list):
        diff = (t_val - time) ** 2
        if diff < smallest_diff:
            smallest_diff = diff
            time_index = i

    print("Value in t_list closest to t_val (%f): %f" %
                                                    (t_val, t_list[time_index]))
    print("Index of t_list with value closest to t_val (%f): %d" %
                                                            (t_val, time_index))

    # Closest val should never be farther than half the lowest sampling rate.
    if (t_val-t_list[time_index])**2 > ((1.0/15)/2)**2:
        raise DataSyncError("Error syncing data. Can't find close "
                            "enough timestamp match.")

    return time_index


def data_read(INCA_path, eDAQ_path, run_num_text):
    sub_run_num = int(run_num_text[2:4])

    INCA_data_dict = {"time": [],
                      "pedal_sw": [],
                      "engine_spd": [],
                      "throttle": []}

    # Read in both eDAQ and INCA data for specific run.
    # read INCA data first
    # Open file with read priveleges.
    # File automatically closed at end of "with/as" block.
    with open(INCA_path, "r") as inca_ascii_file:
        print("Reading INCA data from ASCII...") # debug
        INCA_file_in = csv.reader(inca_ascii_file, delimiter="\t")
        # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

        # Save headers so we can use them when exporting synced data.
        INCA_headers = []
        for i, INCA_row in enumerate(INCA_file_in):
            if i < 5:
                INCA_headers.append(INCA_row)
            if i == 5:
                # first row of actual data - contains first time value.
                # if it's nonzero, need to force it there then offset all future
                # time values. Done in next if block.
                inca_time_offset = float(INCA_row[0])
            # using if instead of elif here because I need both if statements to
            # run in the case of i == 5. if/elif blocks are mutually exclusive.
            if i >= 5:
                # shift time values to be relative at a zero start point.
                INCA_data_dict["time"].append(
                                        float(INCA_row[0]) - inca_time_offset)
                INCA_data_dict["pedal_sw"].append(float(INCA_row[1]))
                INCA_data_dict["engine_spd"].append(float(INCA_row[2]))
                INCA_data_dict["throttle"].append(float(INCA_row[3]))
    print("...done")

    eDAQ_data_dict = {"time": [],
                      "gnd_speed": [],
                      "pedal_sw": []}

    # now read eDAQ data
    with open(eDAQ_path, "r") as edaq_ascii_file:
        print("Reading eDAQ data from ASCII...") # debug
        eDAQ_file_in = csv.reader(edaq_ascii_file, delimiter="\t")
        # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

        # Save headers so we can use them when exporting synced data.
        eDAQ_headers = []
        for j, eDAQ_row in enumerate(eDAQ_file_in):
            if j == 0:
                # The first row is a list of channel names.
                # Loop through and find the first channel for this run.
                run_start_col = find_edaq_col_offset(eDAQ_row, sub_run_num)
                # print("run_start_col = %d" % run_start_col) # debug

                # Save headers so we can use them when exporting synced data.
                # Not reading in the first channel eDAQ_row[run_start_col]
                # because it's pedal voltage and not needed.
                eDAQ_headers.append([eDAQ_row[0]] +
                                    eDAQ_row[run_start_col+1:run_start_col+3])
            elif j > 0:
                # Only add this run's channels to our data list.
                # Don't forget the first column is always time though.

                # Need to make sure we haven't reached end of channel stream.
                # Time vector may keep going past a channel's data, so look at
                # a run-specific channel to see if the run's ended.
                if eDAQ_row[run_start_col+1]:
                    eDAQ_data_dict["time"].append(float(eDAQ_row[0]))
                    # Not reading in the first channel eDAQ_row[run_start_col]
                    # because it's pedal voltage and not needed.
                    eDAQ_data_dict["gnd_speed"].append(
                                            float(eDAQ_row[run_start_col+1]))
                    eDAQ_data_dict["pedal_sw"].append(
                                            float(eDAQ_row[run_start_col+2]))
    print("...done")

    print(INCA_headers)
    print(eDAQ_headers)

    return INCA_headers, eDAQ_headers, INCA_data_dict, eDAQ_data_dict


def sync_data(INCA_data, eDAQ_data):
    """
    Takes in two dicts of data.
    Syncs data based on matching the first time the pedal switch goes high.
    Returns two dicts containing lists of synced data.
    """
    # copy() needed to prevent unwanted side effects / aliasing.
    INCA_mdata = INCA_data.copy()
    eDAQ_mdata = eDAQ_data.copy()

    # find first time pedal goes logical high in both.
    inca_pedal_high_start_i = INCA_data["pedal_sw"].index(1)
    inca_pedal_high_start_t = INCA_data["time"][inca_pedal_high_start_i]

    edaq_pedal_high_start_i = eDAQ_data["pedal_sw"].index(1)
    edaq_pedal_high_start_t = eDAQ_data["time"][edaq_pedal_high_start_i]
    print("\nINCA first pedal high: %f" % inca_pedal_high_start_t)
    print("eDAQ first pedal high: %f" % edaq_pedal_high_start_t)


    if inca_pedal_high_start_t > edaq_pedal_high_start_t:
        # remove time from beginning of INCA file
        print("Removing data from beginning of INCA channels.")
        match_time_index = find_closest(edaq_pedal_high_start_t,
                                                            INCA_data["time"])
        index_offset = inca_pedal_high_start_i - match_time_index

        # this is where the duplicate dict comes in handy.

        for k in INCA_data:
            INCA_mdata[k] = INCA_data[k][index_offset:]
        # Offset time values to still start at zero. Necessary because offset
        # stream doesn't necessarily yield start time of 0.
        # Has to be done in two places because this one doesn't always happen
        # (if eDAQ file is the one that gets modified in this function).
        # And after subtraction, zero point is thrown off again.
        start_time = INCA_data["time"][0]
        new_start_time = INCA_mdata["time"][0]
        INCA_mdata["time"] = [x - (new_start_time - start_time)
                                                    for x in INCA_mdata["time"]]
        # https://stackoverflow.com/questions/4918425/subtract-a-value-from-every-number-in-a-list-in-python

    else:
        # remove time from beginning of eDAQ file
        print("Removing data from beginning of eDAQ channels.")
        match_time_index = find_closest(inca_pedal_high_start_t,
                                                            eDAQ_data["time"])
        index_offset = edaq_pedal_high_start_i - match_time_index

        # this is where the duplicate dict comes in handy.
        for k in eDAQ_data:
            eDAQ_mdata[k] = eDAQ_data[k][index_offset:]
            # Offset time values to still start at zero
            # necessary because streams doesn't necessarily start at time 0 (dumb)
            start_time = eDAQ_data["time"][0]
            new_start_time = eDAQ_mdata["time"][0]
            eDAQ_mdata["time"] = [x - (new_start_time - start_time)
                                                    for x in eDAQ_mdata["time"]]


    # Now print new pedal-high times as a check
    inca_pedal_high_start_i = INCA_mdata["pedal_sw"].index(1)
    inca_pedal_high_start_t = INCA_mdata["time"][inca_pedal_high_start_i]

    edaq_pedal_high_start_i = eDAQ_mdata["pedal_sw"].index(1)
    edaq_pedal_high_start_t = eDAQ_mdata["time"][edaq_pedal_high_start_i]

    print("\nSynced INCA first pedal high: %f" % inca_pedal_high_start_t)
    print("Synced eDAQ first pedal high: %f" % edaq_pedal_high_start_t)

    return INCA_mdata, eDAQ_mdata


def transpose_data_lists(data_dict):
    """Reformats data dict into transposed list of lists"""
    # Take all data lists stored in dictionary by channel name and create
    # list of lists to use in writing output file.
    array = []
    for key in data_dict:
        array.append(data_dict[key])

        # Now need to transpose array to match output file format.
        array_t = list(map(list, zip(*array)))
        # https://stackoverflow.com/questions/6473679/transpose-list-of-lists

    return array_t


def write_sync_data(INCA_data, eDAQ_data, INCA_headers, eDAQ_headers,
                                                                full_run_num):
    """Writes data to file, labeled with run number."""

    INCA_array_t = transpose_data_lists(INCA_data)
    eDAQ_array_t = transpose_data_lists(eDAQ_data)

    # print(INCA_array_t[289:292][:]) # debug




run_num, INCA_data_path, eDAQ_data_path = path_find()

print("\t%s" % INCA_data_path) # debug
print("\t%s\n" % eDAQ_data_path) # debug

# as written, eDAQ file will have to be repeatedly opened and read for each
# separate INCA run. If this ends up too slow, program can be re-written a
# different way. It's probably fine now though.
INCA_headers, eDAQ_headers, INCA_data, eDAQ_data = data_read(INCA_data_path,
                                                        eDAQ_data_path, run_num)

INCA_mdata, eDAQ_mdata = sync_data(INCA_data, eDAQ_data)

write_sync_data(INCA_mdata, eDAQ_mdata, INCA_headers, eDAQ_headers, run_num)

# Write function to write synced data. Will make further testing easier.
# need to recover headers.



# Delete useless data before the first pedal actuation
# Keep first time value = 0
# truncate_data(INCA_mdata, eDAQ_mdata)







# debug
# print(INCA_data_dict["throttle"][38])
# print(INCA_data_dict["engine_spd"][315])
# print(INCA_data_dict["pedal_sw"][305])
# print(INCA_data_dict["pedal_sw"][306])
# print(eDAQ_data_dict["gnd_speed"][0])
# print(eDAQ_data_dict["gnd_speed"][1])
# print(eDAQ_data_dict["pedal_sw"][630])
# print(eDAQ_data_dict["pedal_sw"][631])
