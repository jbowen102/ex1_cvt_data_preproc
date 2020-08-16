import sys
import os
import wpfix
import csv


class FilenameError(Exception):
    pass

class DataReadError(Exception):
    pass

class DataSyncError(Exception):
    pass

class DataTrimError(Exception):
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


def path_find(target_run_num=False):
    """Finds both file paths for either a user-specified run number or given
    run number"""

    if not target_run_num:
        # No input passed so prompt user for run num.
        run_prompt = "Enter run num (four digits)\n> "
        target_run_num = input(run_prompt)
        while len(target_run_num) != 4:
            target_run_num = input("Need a four-digit number. %s" % run_prompt)

    inca_dir = "./raw_data/INCA/"
    all_inca_runs = os.listdir(inca_dir)
    found_inca = False # initialize to false. Will change if file is found.
    for inca_run in all_inca_runs:
        run_num_i = run_name_parse(inca_run)
        if run_num_i == target_run_num:
            # break out of look while "inca_run" is set to correct filename
            found_inca = True
            break

    if found_inca:
        INCA_data_path = os.path.join(inca_dir, inca_run) # relative path
    else:
        raise FilenameError("No INCA file found for run %s" % target_run_num)

    if not os.path.exists(INCA_data_path):
        raise FilenameError("Bad path: %s" % INCA_data_path)

    # Now use that run number to find the right eDAQ file
    eDAQ_file_num = target_run_num[0:2]

    eDAQ_dir = "./raw_data/eDAQ/"
    all_eDAQ_runs = os.listdir(eDAQ_dir)
    found_eDAQ = False # initialize to false. Will change if file is found.
    for eDAQ_run in all_eDAQ_runs:
        # Split the extension off the file name, then isolate the final two
        # numbers off the date
        run_num_i = os.path.splitext(eDAQ_run)[0].split("_")[1][0:2]
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

    return target_run_num, INCA_data_path, eDAQ_data_path


def run_name_parse(filename):
    """Assuming INCA data type
    Returns run number."""

    run_num_text = filename.split("_")[1][0:4]
    return run_num_text


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

    print("\tValue in t_list closest to t_val (%f): %f" %
                                                    (t_val, t_list[time_index]))
    print("\tIndex of t_list with value closest to t_val (%f): %d" %
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
        print("Reading INCA data from %s" % INCA_path) # debug
        INCA_file_in = csv.reader(inca_ascii_file, delimiter="\t")
        # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

        # Save headers so we can use them when exporting synced data.
        INCA_headers = []
        for i, INCA_row in enumerate(INCA_file_in):
            if i == 2 or i == 4:
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
        print("Reading eDAQ data from %s" % eDAQ_path) # debug
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

    # print(INCA_headers) # debug
    # print(eDAQ_headers) # debug

    return INCA_headers, eDAQ_headers, INCA_data_dict, eDAQ_data_dict


def find_first_pedal_high(data_dict):
    """Finds first time in a data stream that the "pedal_sw" channel goes from
    0 to 1."""

    high_start_i = data_dict["pedal_sw"].index(1)
    high_start_t = data_dict["time"][high_start_i]
    return high_start_i, high_start_t


def sync_data(INCA_data, eDAQ_data):
    """
    Takes in two dicts of data.
    Syncs data based on matching the first time the pedal switch goes high.
    Returns two dicts containing lists of synced data.
    """

    # Creete new dict objects to hold modified data.
    # copy() needed to prevent unwanted side effects / aliasing.
    INCA_mdata = INCA_data.copy()
    eDAQ_mdata = eDAQ_data.copy()

    # find first time pedal goes logical high in both.
    inca_pedal_high_start_i, inca_pedal_high_start_t = find_first_pedal_high(
                                                                      INCA_data)
    edaq_pedal_high_start_i, edaq_pedal_high_start_t = find_first_pedal_high(
                                                                      eDAQ_data)
    print("\n\tINCA first pedal high: %f" % inca_pedal_high_start_t)
    print("\teDAQ first pedal high: %f" % edaq_pedal_high_start_t)


    if inca_pedal_high_start_t > edaq_pedal_high_start_t:
        # remove time from beginning of INCA file
        print("\tRemoving data from beginning of INCA channels.")
        match_time_index = find_closest(edaq_pedal_high_start_t,
                                                            INCA_data["time"])
        index_offset = inca_pedal_high_start_i - match_time_index

        # this is where the separate, new dict comes in handy.
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
        print("\tRemoving data from beginning of eDAQ channels.")
        match_time_index = find_closest(inca_pedal_high_start_t,
                                                            eDAQ_data["time"])
        index_offset = edaq_pedal_high_start_i - match_time_index

        # this is where the separate, new dict comes in handy.
        for k in eDAQ_data:
            eDAQ_mdata[k] = eDAQ_data[k][index_offset:]
            # Offset time values to still start at zero
            # necessary because streams doesn't necessarily start at time 0
            start_time = eDAQ_data["time"][0]
            new_start_time = eDAQ_mdata["time"][0]
            eDAQ_mdata["time"] = [x - (new_start_time - start_time)
                                                    for x in eDAQ_mdata["time"]]


    # Now print new pedal-high times as a check
    inca_pedal_high_start_i, inca_pedal_high_start_t = find_first_pedal_high(
                                                                     INCA_mdata)
    edaq_pedal_high_start_i, edaq_pedal_high_start_t = find_first_pedal_high(
                                                                     eDAQ_mdata)
    print("\tSynced INCA first pedal high: %f" % inca_pedal_high_start_t)
    print("\tSynced eDAQ first pedal high: %f" % edaq_pedal_high_start_t)

    return INCA_mdata, eDAQ_mdata


def left_trim_data(data_dict):
    """Truncates beginning of the file, making start point 1.0 seconds before
    first pedal-down event."""

    data_mdict = data_dict.copy()

    pedal_high_start_i, pedal_high_start_t = find_first_pedal_high(data_dict)
    if pedal_high_start_t > 1.0:
        match_time_index = find_closest(pedal_high_start_t - 1.0,
                                                            data_dict["time"])

        # Transcribe data from each channel starting at this -1.0s mark.
        for k in data_dict:
            data_mdict[k] = data_dict[k][match_time_index:]

        # offset the start time to still be zero.
        new_start_time = data_mdict["time"][0]
        data_mdict["time"] = [x - new_start_time for x in data_mdict["time"]]

    else:
        # Don't remove any data if there's already less than one second
        # before first pedal actuation.
        pass

    return data_mdict


def abbreviate_data(INCA_data, eDAQ_data, throt_thresh, thr_t_thresh):
    """Isolates important events in data by removing any long stretches of no
    pedal input of pedal events during which the throttle position >45% or
    whatever throt_thresh (throttle threshold) not sustained for >2 second or
    whatever thr_t_thresh."""

    # list of start and end times for pedal-down events with a segment of >45%
    # throttle for >2s.
    valid_event_times = []

    # maintain a buffer of candidate pedal-down and throttle time vals.
    ped_buffer = []
    high_throttle_time = [0, 0]

    pedal_down = False
    counting = False
    keep = False
    for i, ti in enumerate(INCA_data["time"]):
        # Main loop evaluates pedal-down event. Stores event start and end times
        # if inner loop determines throttle was >45% for >2s during event.
        if INCA_data["pedal_sw"][i]:
            if not pedal_down:
                print("\nPedal goes high at time %f" % ti)
            # pedal currently down
            pedal_down = True
            ped_buffer.append(ti) # add current time to pedal buffer.

            ## Calculate throttle >45% time to determine event legitimacy
            if not counting and INCA_data["throttle"][i] > throt_thresh:
                # first time throttle exceeds 45%
                print("\tThrottle > %d at time %f" % (throt_thresh, ti))
                high_throttle_time[0] = ti
                counting = True

            elif counting and INCA_data["throttle"][i] < throt_thresh:
                # throttle drops below 45%
                print("\tThrottle < %d at time %f" % (throt_thresh, ti))
                high_throttle_time[1] = INCA_data["time"][i-1] # previous time
                delta = high_throttle_time[1] - high_throttle_time[0]
                print("\tThrottle < %d total time: %f" % (throt_thresh, delta))
                # calculate if that >45% throttle event lasted longer than 2s.
                if high_throttle_time[1] - high_throttle_time[0] > thr_t_thresh:
                    keep = True
                    # now the times stored in ped_buffer constitute a valid
                    # event. As long as the pedal switch stays actuated,
                    # subsequentn time indices will be added to ped_buffer.
                counting = False # reset indicator
                high_throttle_time = [0, 0] # reset

        elif pedal_down:
            # pedal just lifted
            print("pedal lifted at time %f" % ti)
            if keep:
                valid_event_times.append( [ped_buffer[0], ped_buffer[-1]] )
            pedal_down = False
            ped_buffer = [] # flush buffer
            keep = False # reset
        else:
            # pedal is not currently down, and wasn't just lifted.
            pass

    print(valid_event_times)

    if not valid_event_times:
        # If no times were stored, then something might be wrong.
        raise DataTrimError("No valid pedal-down events found.")

    # make sure if two >45% events (w/ pedal lift between) are closer that 5s,
    # don't cut into either one. Look at each pair of end/start points, and
    # if they're closer than 5s, merge those two.
    previous_pair = valid_event_times[0]
    valid_event_times_c = valid_event_times.copy()
    for n, pair in enumerate(valid_event_times[1:]):
        print("\t%f - %f" % (pair[0], previous_pair[1]))
        if pair[0] - previous_pair[1] < 5:
            # Replace the two pairs with a single combined pair
            del valid_event_times_c[n-1]
            valid_event_times_c[n] = [ previous_pair[0], pair[1] ]
        previous_pair = pair
    print(valid_event_times_c)


    # add one second buffer to each side of valid pedal-down events.
    for n, pair in enumerate(valid_event_times_c):
        new_start = find_closest(pair[0] - 1.0, INCA_data["time"])
        print("New start: %f" % INCA_data["time"][new_start])
        pair[0] = INCA_data["time"][new_start]

        new_end = find_closest(pair[1] + 1.0, INCA_data["time"])
        print("New end: %f" % INCA_data["time"][new_end])
        pair[1] = INCA_data["time"][new_end]

        if n == 0 and new_start != 0:
            # should maintain zero start value, but this will stop it if not.
            DataTrimError("INCA time vector no longer starting at 0.")

    print(valid_event_times_c)

    # duplicate data structure

    INCA_data_a = INCA_data.copy()

    # offset INCA time vector to maintain continuity.
    # find closest times in eDAQ files.

    # make sure everything works in case of only one valid range (ex. 0105)

    # transcribe only the valid data ranges into new dicts
    return None, None


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


def combine_data_arrays(INCA_array, eDAQ_array):
    """Takes two data arrays and returns combined array.
    """
    # Base it off longer file so no data gets cut off.
    if len(INCA_array) > len(eDAQ_array):
        sync_array = INCA_array[:]
        for line_no, line in enumerate(eDAQ_array):
            sync_array[line_no].append("")
            sync_array[line_no] += line

    else:
        sync_array = eDAQ_array[:]
        for line_no, line in enumerate(INCA_array):
            sync_array[line_no].append("")
            sync_array[line_no] += line

    return sync_array


def write_sync_data(INCA_data, eDAQ_data, INCA_headers, eDAQ_headers,
                                            full_run_num, auto_overwrite=False):
    """Writes data to file, labeled with run number."""

    # Create reformatted arrays of just channel data (no headers)
    INCA_ch_array_t = transpose_data_lists(INCA_data)
    eDAQ_ch_array_t = transpose_data_lists(eDAQ_data)

    # Add in headers
    # eDAQ gets padding to line up first data row with INCA format.
    INCA_array_t = INCA_headers + INCA_ch_array_t
    eDAQ_array_t = eDAQ_headers + [["", "", ""]] + eDAQ_ch_array_t

    # Create unified array with both datasets
    sync_array = combine_data_arrays(INCA_array_t, eDAQ_array_t)
    # print(INCA_array_t[289:292][:]) # debug

    # Create new CSV file and write out. Closes automatically at end of with/as
    sync_basename = "%s_Sync.csv" % full_run_num
    sync_filename = "./sync_data/%s" % sync_basename

    # Check if file exists already. Prompt user for overwrite decision.
    if os.path.exists(sync_filename) and not auto_overwrite:
        ow_answer = ""
        while ow_answer.lower() not in ["y", "n"]:
            ow_answer = input("\n%s already exists in sync_data folder. "
                                "Overwrite? (Y/N)\n> " % sync_basename)
        if ow_answer.lower() == "n":
            return

    # This block should not run if answered no to overwrite above.
    with open(sync_filename, 'w+') as sync_file:
        sync_file_csv = csv.writer(sync_file, dialect="excel")

        print("\nWriting combined data to %s..." % sync_basename)
        sync_file_csv.writerows(sync_array)
        print("...done\n")


def main_prog():
    """Main program that runs automatically as long as cwd is correct."""
    # Define constants to use in isolating useful data.
    throttle_threshold = 45
    throttle_time_threshold = 2

    # If you pass in any arguments from the command line after "python run.py",
    # This pulls them in. If "--auto" specified, process all data.
    # If second arg is "--over", then automatically overwrite any existing
    # exports in the ./sync_data folder.
    autorun_arg = False # initializing for later conditional
    if len(sys.argv) == 2:
        prog, autorun_arg = sys.argv
        overw_arg = False
        # print(autorun_arg)
    elif len(sys.argv) == 3:
        prog, autorun_arg, overw_arg = sys.argv
        # print(autorun_arg)
        # print(overw_arg)

    if type(autorun_arg) is str and autorun_arg.lower() == "--auto":
        # loop through ordered contents of ./raw_data/INCA and process each run.
        INCA_root = "./raw_data/INCA/"
        INCA_files = os.listdir(INCA_root)
        INCA_files.sort()
        for INCA_file in INCA_files:
            INCA_run = run_name_parse(INCA_file)
            run_num, INCA_data_path, eDAQ_data_path = path_find(INCA_run)
            # as written, eDAQ file will have to be repeatedly opened and read
            # for each separate INCA run. If this ends up too slow, program can
            # be re-written a different way. It's probably fine now though.
            INCA_headers, eDAQ_headers, INCA_data, eDAQ_data = data_read(
                                        INCA_data_path, eDAQ_data_path, run_num)

            INCA_mdata, eDAQ_mdata = sync_data(INCA_data, eDAQ_data)

            INCA_mtdata = left_trim_data(INCA_mdata)
            eDAQ_mtdata = left_trim_data(eDAQ_mdata)

            INCA_data_mta, eDAQ_data_mta = abbreviate_data(INCA_mtdata,
                    eDAQ_mtdata, throttle_threshold, throttle_time_threshold)

            # If "over" specified, automatically overwrite destination files.
            if type(overw_arg) is str and overw_arg.lower() == "--over":
                auto_overwrite = True
            else:
                auto_overwrite = False

            write_sync_data(INCA_mtdata, eDAQ_mtdata, INCA_headers,
                                        eDAQ_headers, run_num, auto_overwrite)

    else:
        # run with user input for specific run to use
        run_num, INCA_data_path, eDAQ_data_path = path_find()

        print("\t%s" % INCA_data_path) # debug
        print("\t%s\n" % eDAQ_data_path) # debug

        INCA_headers, eDAQ_headers, INCA_data, eDAQ_data = data_read(
                                        INCA_data_path, eDAQ_data_path, run_num)

        INCA_mdata, eDAQ_mdata = sync_data(INCA_data, eDAQ_data)

        INCA_mtdata = left_trim_data(INCA_mdata)
        eDAQ_mtdata = left_trim_data(eDAQ_mdata)

        INCA_data_mta, eDAQ_data_mta = abbreviate_data(INCA_mtdata,
                eDAQ_mtdata, throttle_threshold, throttle_time_threshold)
        quit()
        write_sync_data(INCA_mtdata, eDAQ_mtdata, INCA_headers, eDAQ_headers,
                                                                        run_num)


if __name__ == "__main__":
    main_prog()

# Delete useless data between events.

# Store headers in dictionaries, remove from function formal params and return
# vals

# May be able to remove trim_data() to remove data at beginning of file and
# in between events all at once.

# Plot data to determine if trimming is correct.



# debug
# print(INCA_data_dict["throttle"][38])
# print(INCA_data_dict["engine_spd"][315])
# print(INCA_data_dict["pedal_sw"][305])
# print(INCA_data_dict["pedal_sw"][306])
# print(eDAQ_data_dict["gnd_speed"][0])
# print(eDAQ_data_dict["gnd_speed"][1])
# print(eDAQ_data_dict["pedal_sw"][630])
# print(eDAQ_data_dict["pedal_sw"][631])
