import os
import wpfix
import csv


class FilenameError(Exception):
    pass

class DataReadError(Exception):
    pass

class DataSyncError(Exception):
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


def find_closest(t_val, t_list):
    """
    Takes in a time value and a list of times.
    Returns the index of the closest time value in the list.
    Linear search - O(n) time complexity. Bisection search would be better but probably unnecessary.
    (Originally part of Alt_Test_Data program. Modified here for Python3)
    """

    time_index = 0
    smallest_diff = (t_val - t_list[time_index]) ** 2
    for i, time in enumerate(t_list):
        # print 't = %d' % t
        diff = (t_val - time) ** 2
        # print 'diff = %f' % diff
        if diff < smallest_diff:
            smallest_diff = diff
            time_index = i

    print('Value in t_list closest to t_val (%f): %f' % (t_val, t_list[time_index]))
    print('Index of t_list with value closest to t_val (%f): %d' % (
                                                                t_val, time_index))

    # Closest val should never be farther than half the lowest sampling rate.
    if (t_val-t_list[time_index])**2 > ((1.0/15)/2)**2:
        raise DataSyncError("Error syncing data. Can't find close "
                            "enough timestamp match.")

    return time_index


def data_read(INCA_path, eDAQ_path, run_num_text):
    # run_num = int(run_num_text)
    sub_run_num = int(run_num_text[2:4])

    INCA_data_list = []
    INCA_data_dict = {'time': [],
                      'pedal_sw': [],
                      'engine_spd': [],
                      'throttle': []}

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
    eDAQ_data_dict = {'time': [],
                      'gnd_speed': [],
                      'pedal_sw': []}

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
                # print("run_start_col = %d" % run_start_col) # debug

            elif j > 0:
                # only add this run's channels to our data list. Don't forget the first column is always time though.
                eDAQ_data_list.append([ eDAQ_row[0], eDAQ_row[run_start_col+1], eDAQ_row[run_start_col+2] ])

                # Need to make sure we haven't reached end of channel stream.
                # Time vector may keep going past a channel's data
                if eDAQ_row[run_start_col+1]:
                    eDAQ_data_dict['time'].append(float(eDAQ_row[0]))
                    # not reading in the first channel because it's pedal voltage and not needed.
                    eDAQ_data_dict['gnd_speed'].append(float(eDAQ_row[run_start_col+1]))
                    eDAQ_data_dict['pedal_sw'].append(float(eDAQ_row[run_start_col+2]))
    print("...done")

    return INCA_data_list, INCA_data_dict, eDAQ_data_list, eDAQ_data_dict


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
    inca_pedal_high_start_i = INCA_data['pedal_sw'].index(1)
    inca_pedal_high_start_t = INCA_data['time'][inca_pedal_high_start_i]

    edaq_pedal_high_start_i = eDAQ_data['pedal_sw'].index(1)
    edaq_pedal_high_start_t = eDAQ_data['time'][edaq_pedal_high_start_i]
    print("\nINCA first pedal high: %f" % inca_pedal_high_start_t)
    print("eDAQ first pedal high: %f" % edaq_pedal_high_start_t)


    if inca_pedal_high_start_t > edaq_pedal_high_start_t:
        # remove time from beginning of INCA file
        print("Removing data from beginning of INCA channels.")
        match_time_index = find_closest(edaq_pedal_high_start_t, INCA_data['time'])
        index_offset = inca_pedal_high_start_i - match_time_index

        # this is where the duplicate dict comes in handy.
        # for k in INCA_data.keys():
        for k in INCA_data:
            if k == 'time':
                # Shift time vector the other way to keep starting point at 0
                INCA_mdata[k] = INCA_data[k][:-index_offset]
            else:
                INCA_mdata[k] = INCA_data[k][index_offset:]

    else:
        # remove time from beginning of eDAQ file
        print("Removing data from beginning of eDAQ channels.")
        match_time_index = find_closest(inca_pedal_high_start_t, eDAQ_data['time'])
        index_offset = edaq_pedal_high_start_i - match_time_index

        # this is where the duplicate dict comes in handy.
        # for k in eDAQ_data.keys():
        for k in eDAQ_data:
            if k == 'time':
                # Cut values off the back end to keep starting time = 0.
                eDAQ_mdata[k] = eDAQ_data[k][:-index_offset]
            else:
                # Cut values off the front end.
                eDAQ_mdata[k] = eDAQ_data[k][index_offset:]


    # Now print new pedal-high times as a check
    inca_pedal_high_start_i = INCA_mdata['pedal_sw'].index(1)
    inca_pedal_high_start_t = INCA_mdata['time'][inca_pedal_high_start_i]

    edaq_pedal_high_start_i = eDAQ_mdata['pedal_sw'].index(1)
    edaq_pedal_high_start_t = eDAQ_mdata['time'][edaq_pedal_high_start_i]

    print("\nSynced INCA first pedal high: %f" % inca_pedal_high_start_t)
    print("Synced eDAQ first pedal high: %f" % edaq_pedal_high_start_t)

    return INCA_mdata, eDAQ_mdata


run_num, INCA_data_path, eDAQ_data_path = path_find()
# print(run_num) # debug
print("\t%s" % INCA_data_path) # debug
print("\t%s\n" % eDAQ_data_path) # debug

# as written, eDAQ file will have to be repeatedly opened and read for each separate INCA run.
# if this ends up too slow, program can be re-written a different way. It's probably fine now though.
INCA_data_list, INCA_data, eDAQ_data_list, eDAQ_data = data_read(INCA_data_path, eDAQ_data_path, run_num)

INCA_mdata, eDAQ_mdata = sync_data(INCA_data, eDAQ_data)


# Delete all data before that point (later change to have a buffer before - measured in time, not data points.)
# Don't delete column headers
# Offset time values to stat at zero






# debug
# print(INCA_data_list[0][2])
# print(INCA_data_list[1][1])
# print(INCA_data_list[7][0])
# print(eDAQ_data_list[0][2])
# print(eDAQ_data_list[1][1])
# print(eDAQ_data_list[7][0])
# print(INCA_data_dict['throttle'][38])
# print(INCA_data_dict['engine_spd'][315])
# print(INCA_data_dict['pedal_sw'][305])
# print(INCA_data_dict['pedal_sw'][306])
# print(eDAQ_data_dict['gnd_speed'][0])
# print(eDAQ_data_dict['gnd_speed'][1])
# print(eDAQ_data_dict['pedal_sw'][630])
# print(eDAQ_data_dict['pedal_sw'][631])
