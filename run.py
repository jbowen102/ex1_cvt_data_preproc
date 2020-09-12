import os           # Used for analyzing file paths and directories
import csv          # Needed to read in and write out data
import argparse     # Used to parse optional command-line arguments
import math         # Using pi to convert linear speed to angular speed.
import pandas as pd # Series and DataFrame

try:
    import matplotlib
    matplotlib.use("Agg") # no UI backend for use w/ WSL
    # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
    import matplotlib.pyplot as plt # Needed for optional data plotting.
    PLOT_LIB_PRESENT = True
except ImportError:
    PLOT_LIB_PRESENT = False
# https://stackoverflow.com/questions/3496592/conditional-import-of-modules-in-python


# global constancts
RAW_INCA_ROOT = "./raw_data/INCA"
RAW_EDAQ_ROOT = "./raw_data/eDAQ"

INCA_CHANNELS = ["time", "pedal_sw", "engine_spd", "throttle"]
EDAQ_CHANNELS = ["time", "pedal_v", "gnd_speed", "pedal_sw"]

CHANNEL_UNITS = {"time": "s",
                 "pedal_sw": "off/on",
                 "pedal_v": "V",
                 "engine_spd": "rpm",
                 "gnd_speed": "mph",
                 "throttle": "deg"}

INCA_HEADER_HT = 5 # how many non-data rows at top of raw INCA file.
EDAQ_HEADER_HT = 1 # how many non-data rows at top of raw eDAQ file.

SAMPLING_FREQ = 100 # Hz

# Some case-specific constants stored in class definitions


class FilenameError(Exception):
    pass

class DataReadError(Exception):
    pass

class DataSyncError(Exception):
    pass

class TimeStampError(Exception):
    pass

class DataTrimError(Exception):
    pass

class CVTCalcError(Exception):
    pass


class RunGroup(object):
    """Represents a collection of runs from the raw_data directory."""
    def __init__(self, process_all=False):
        # create SingleRun object for each run but don't read in data yet.
        self.build_run_dict()

        if process_all:
            # automatically process all INCA runs (below)
            self.runs_to_process = self.run_dict
        else:
            # prompt user for single run to process.
            OnlyRun = self.prompt_for_run()
            self.runs_to_process = {OnlyRun.get_run_label(): OnlyRun}

        for run_num in self.runs_to_process:
            RunObj = self.runs_to_process[run_num]
            RunObj.read_data()
            RunObj.sync_data()
            RunObj.abridge_data()

    def build_run_dict(self):
        """Create dictionary with an entry for each INCA run in raw_data dir."""
        INCA_files = os.listdir(RAW_INCA_ROOT)
        INCA_files.sort()
        self.run_dict = {}
        # eliminate any directories that might be in the list
        for i, file in enumerate(INCA_files):
            if os.path.isdir(os.path.join(RAW_INCA_ROOT, file)):
                continue # ignore any directories found

            if "decel" in file.lower():
                # ThisRun = self.create_downhill_run(file)
                input("Skipping file %s because program can't process decel "
                "runs yet.\nPress Enter to acknowledge." % file)
                continue
            else:
                ThisRun = self.create_ss_run(file)

            self.run_dict[ThisRun.get_run_label()] = ThisRun

    def create_ss_run(self, filename):
        return SSRun(os.path.join(RAW_INCA_ROOT, filename))

    def create_downhill_run(self, filename):
        return DownhillRun(os.path.join(RAW_INCA_ROOT, filename))

    def plot_runs(self, overwrite=False):
        # If only one run in group is to be processed, this will only loop once.
        for run_num in self.runs_to_process:
            RunObj = self.runs_to_process[run_num]
            RunObj.plot_data(overwrite)

    def export_runs(self, overwrite=False):
        # If only one run in group is to be processed, this will only loop once.
        for run_num in self.runs_to_process:
            RunObj = self.runs_to_process[run_num]
            RunObj.export_data(overwrite)

    def prompt_for_run(self):
        """Prompts user for what run to process
        Returns SingleRun object"""
        run_prompt = "Enter run num (four digits)\n> "
        target_run_num = input(run_prompt)
        while len(target_run_num) != 4:
            target_run_num = input("Need a four-digit number. %s" % run_prompt)

        TargetRun = self.run_dict.get(target_run_num)
        if TargetRun:
            return TargetRun
        else:
            raise FilenameError("No INCA file found for run %s" %
                                                                target_run_num)


class SingleRun(object):
    """Represents a single run from the raw_data directory.
    No data is read in until read_data() called.
    """
    def __init__(self, INCA_path):
        self.INCA_path = INCA_path
        self.INCA_filename = os.path.basename(self.INCA_path)
        self.run_label = self.INCA_filename.split("_")[1][0:4]

        # put check in other functions to ensure they safely fail if read_data()
        # yet to be called.

    def read_data(self):
        """Read in both INCA and eDAQ data from raw_data directory"""
        self.find_edaq_path()

        # Read in both eDAQ and INCA data for specific run.
        # read INCA data first
        # Open file with read priveleges.
        # File automatically closed at end of "with/as" block.
        with open(self.INCA_path, "r") as inca_ascii_file:
            print("Reading INCA data from %s" % self.INCA_path) # debug
            INCA_file_in = csv.reader(inca_ascii_file, delimiter="\t")
            # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

            raw_inca_dict = {}
            for channel in INCA_CHANNELS:
                raw_inca_dict[channel] = []

            for i, INCA_row in enumerate(INCA_file_in):
                if i < INCA_HEADER_HT:
                    # ignore headers
                    continue
                else:
                    for n, channel in enumerate(INCA_CHANNELS):
                        raw_inca_dict[channel].append(float(INCA_row[n]))

        # Separate out time. Round to nearest hundredth.
        inca_time_series = raw_inca_dict["time"].copy()
        # print(inca_time_series[:20])
        del raw_inca_dict["time"]

        self.raw_inca_df = pd.DataFrame(data=raw_inca_dict,
                                                        index=inca_time_series)
        print("...done")

        # now read eDAQ data
        with open(self.eDAQ_path, "r") as edaq_ascii_file:
            print("Reading eDAQ data from %s" % self.eDAQ_path) # debug
            eDAQ_file_in = csv.reader(edaq_ascii_file, delimiter="\t")
            # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

            raw_edaq_dict = {}
            for channel in EDAQ_CHANNELS:
                raw_edaq_dict[channel] = []

            for j, eDAQ_row in enumerate(eDAQ_file_in):
                if j == 0:
                    # The first row is a list of channel names.
                    # Loop through and find the first channel for this run.

                    # converting to int and back to str strips zero padding
                    sub_run_num = int(self.run_label[2:4])
                    edaq_sub_run = "RN_"+str(sub_run_num)

                    for n, col in enumerate(eDAQ_row):
                        if edaq_sub_run in col:
                            edaq_run_start_col = n
                            break

                    if n == len(eDAQ_row) - 1:
                        # got to end of row and didn't find the run in any
                        # column heading
                        raise DataReadError("Can't find %s in any eDAQ file" %
                                                                edaq_sub_run)

                elif j > 0 and eDAQ_row[edaq_run_start_col+1]:
                    # Need to make sure we haven't reached end of channel strm.
                    # Time vector may keep going past a channel's data, so look
                    # at a run-specific channel to see if the run's ended.

                    # Only add this run's channels to our data list.
                    # Time is always in 1st column. Round to nearest hundredth.
                    raw_edaq_dict["time"].append(float(eDAQ_row[0]))
                    raw_edaq_dict["pedal_v"].append(
                                    float(eDAQ_row[edaq_run_start_col]))
                    raw_edaq_dict["gnd_speed"].append(
                                    float(eDAQ_row[edaq_run_start_col+1]))
                    raw_edaq_dict["pedal_sw"].append(
                                    float(eDAQ_row[edaq_run_start_col+2]))

        # Discard pedal voltage because it's not needed.
        del raw_edaq_dict["pedal_v"]

        # Separate out time
        edaq_time_series = raw_edaq_dict["time"].copy()
        # print(edaq_time_series[:20])
        del raw_edaq_dict["time"]

        # print(edaq_time_series[:10])
        # print(raw_edaq_dict["gnd_speed"][:10])
        # print(raw_edaq_dict["pedal_sw"][:10])

        self.raw_edaq_df = pd.DataFrame(data=raw_edaq_dict,
                                                        index=edaq_time_series)
        print("...done")

    def sync_data(self):
        self.inca_df = self.raw_inca_df.copy(deep=True)
        self.edaq_df = self.raw_edaq_df.copy(deep=True)
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html

        # Offset time series to start at zero.
        self.shift_time_series(self.inca_df, zero=True)
        self.shift_time_series(self.edaq_df, zero=True)

        # Convert index from seconds to hundredths of a second
        self.inca_df.set_index(pd.Index([int(round(ti * SAMPLING_FREQ))
                                for ti in self.inca_df.index]), inplace=True)
        self.edaq_df.set_index(pd.Index([int(round(ti * SAMPLING_FREQ))
                                for ti in self.edaq_df.index]), inplace=True)
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html

        # Check if any pedal events exist
        if self.inca_df.loc[self.inca_df["pedal_sw"] == 1].empty:
            raise DataSyncError("No pedal event found in INCA data (looking for"
            " value of 1 in pedal switch column). Check input pedal data and "
            "ordering of input file's columns.")
        if self.edaq_df.loc[self.edaq_df["pedal_sw"] == 1].empty:
            raise DataSyncError("No pedal event found in eDAQ data (looking for"
            " value of 1 in pedal switch column). Check input pedal data and "
            "ordering of input file's columns.")
        # find first pedal switch event
        # https://stackoverflow.com/questions/16683701/in-pandas-how-to-get-the-index-of-a-known-value
        inca_high_start_t = self.inca_df.loc[
                                        self.inca_df["pedal_sw"] == 1].index[0]
        edaq_high_start_t = self.edaq_df.loc[
                                        self.edaq_df["pedal_sw"] == 1].index[0]
        print("start times (inca, edaq): %f, %f" % (inca_high_start_t,
                                                        edaq_high_start_t))

        # Test first to see if either data set has first pedal event earlier
        # than 1s. If so, that's the new time for both files to line up at.
        start_buffer = min([1 * SAMPLING_FREQ, inca_high_start_t,
                                                            edaq_high_start_t])
        print("start buffer: %f" % start_buffer)

        # shift time values, leaving negative values in early part of file that
        # will be trimmed off below.
        inca_target_t = inca_high_start_t - start_buffer
        edaq_target_t = edaq_high_start_t - start_buffer

        self.shift_time_series(self.inca_df, offset_val=-inca_target_t)
        self.shift_time_series(self.edaq_df, offset_val=-edaq_target_t)

        print("new inca index start: %d" % self.inca_df.index[0])
        print("new edaq index start: %d" % self.edaq_df.index[0])

        # Unify datasets into one DataFrame
        # Slice out values before t=0 (1s before first pedal press)
        # Automatically truncates longer data set
        self.sync_df = pd.merge(self.inca_df.loc[0:],
                                self.edaq_df["gnd_speed"].loc[0:],
                                left_index=True, right_index=True)
        print(len(self.inca_df))
        print(len(self.inca_df.loc[0:]))
        print(len(self.edaq_df))
        print(len(self.edaq_df.loc[0:]))
        input(len(self.sync_df))

    def shift_time_series(self, df, zero=False, offset_val=None):
        """If offset_val param specified, add this signed value to all time
        values.
        If zero param passed, offset all such that first val is 0."""
        if zero:
            offset_val = -df.index[0]
        elif not offset_val:
            raise DataSyncError("shift_time_series needs either zero or "
                                                    "offset_val param.")

        shifted_time_series = df.index + offset_val
        df.set_index(shifted_time_series, inplace=True)

    def plot_data(self, overwrite=False):
        print("Plotting data")

        # Convert DF indices from hundredths of a second to seconds
        inca_times = [round(ti/SAMPLING_FREQ, 2)
                                            for ti in self.inca_df.index]
        # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html
        ax1 = plt.subplot(211)
        plt.plot(self.raw_inca_df.index, self.raw_inca_df["throttle"],
                                                        label="Throttle (og)")
        plt.title("INCA Throttle vs. Time (Run %s)" % self.run_label)
        plt.ylabel("Throttle (deg)")
        plt.legend()
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
        plt.plot(inca_times, self.inca_df["throttle"],
                                                    label="Throttle (synced)")
        # https://matplotlib.org/3.2.1/gallery/subplots_axes_and_figures/shared_axis_demo.html#sphx-glr-gallery-subplots-axes-and-figures-shared-axis-demo-py

        plt.xlabel("Time (s)")
        plt.ylabel("Throttle (deg)")
        plt.legend()
        # plt.legend(loc="best")

        fig_filepath = "./figs/%s_fig.png" % self.run_label

        if os.path.exists(fig_filepath) and not overwrite:
            ow_answer = ""
            while ow_answer.lower() not in ["y", "n"]:
                ow_answer = input("\n%s already exists in figs folder. "
                        "Overwrite? (Y/N)\n> " % os.path.basename(fig_filepath))
            if ow_answer.lower() == "n":
                plt.clf()
                return

        print("\nExporting plot as %s..." % os.path.basename(fig_filepath))
        plt.savefig(fig_filepath)
        print("...done")
        # plt.show() # can't use w/ WSL.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib
        plt.clf()

    def export_data(self, overwrite=False):

        # Create to list of lists for easier writing out
        # Convert time values from hundredths of a second to seconds
        inca_indices = [round(ti/SAMPLING_FREQ,2)
                                    for ti in self.inca_df.index.tolist()]
        edaq_indices = [round(ti/SAMPLING_FREQ,2)
                                    for ti in self.edaq_df.index.tolist()]
        inca_data_array = self.inca_df.values.tolist()
        edaq_data_array = self.edaq_df.values.tolist()
        # https://stackoverflow.com/questions/28006793/pandas-dataframe-to-list-of-lists

        # always put INCA data on the left.
        sync_array = inca_data_array[:]
        for line_no, inca_line in enumerate(sync_array):
            # put in time values
            sync_array[line_no].insert(0, inca_indices[line_no])
            # https://stackoverflow.com/questions/8537916/whats-the-idiomatic-syntax-for-prepending-to-a-short-python-list
        # Base it off longer file so no data gets cut off.
        if len(inca_data_array) > len(edaq_data_array):
            for line_no, edaq_line in enumerate(edaq_data_array):
                sync_array[line_no].append("")
                # sync_array[line_no] += edaq_indices[line_no]
                sync_array[line_no].append(edaq_indices[line_no])
                sync_array[line_no] += edaq_line
        else:
            for line_no, edaq_line in enumerate(edaq_data_array):
                if len(inca_data_array) >= line_no + 1:
                    # Copy INCA data unless it runs out.
                    sync_array[line_no] += [""]
                    # sync_array[line_no] += edaq_indices[line_no]
                    sync_array[line_no].append(edaq_indices[line_no])
                    sync_array[line_no] += edaq_line
                else:
                    # If eDAQ data contains more points, pad columns
                    sync_array.append([""]*len(INCA_CHANNELS) + edaq_line)

        # Format header rows
        inca_header = [INCA_CHANNELS[:],
                                    [CHANNEL_UNITS[c] for c in INCA_CHANNELS]]
        edaq_header = [EDAQ_CHANNELS[:],
                                    [CHANNEL_UNITS[c] for c in EDAQ_CHANNELS]]
        # Remove pedal voltage
        edaq_header[0].remove("pedal_v")
        edaq_header[1].remove("V")
        # https://note.nkmk.me/en/python-list-clear-pop-remove-del/

        # Add headers to array
        sync_array.insert(0, inca_header[1] + [""] + edaq_header[1])
        sync_array.insert(0, inca_header[0] + [""] + edaq_header[0])

        # Just using external function for now to get it working.
        sync_array_w_cvt = add_cvt_ratio(sync_array)

        # Create new CSV file and write out. Closes automatically at end of
        # with/as block.
        sync_basename = "%s_Sync.csv" % self.run_label
        sync_filename = "./sync_data/%s" % sync_basename

        # Check if file exists already. Prompt user for overwrite decision.
        if os.path.exists(sync_filename) and not overwrite:
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


    def find_edaq_path(self):
        """Locate path to eDAQ file corresponding to INCA run num."""
        eDAQ_file_num = self.run_label[0:2]

        all_eDAQ_runs = os.listdir(RAW_EDAQ_ROOT)
        found_eDAQ = False # initialize to false. Will change if file is found.
        for eDAQ_run in all_eDAQ_runs:
            if os.path.isdir(os.path.join(RAW_EDAQ_ROOT, eDAQ_run)):
                continue # ignore any directories found
            # Split the extension off the file name, then isolate the final two
            # numbers off the date
            run_num_i = os.path.splitext(eDAQ_run)[0].split("_")[1][0:2]
            if run_num_i == eDAQ_file_num:
                # break out of loop while "eDAQ_run" is set to correct filename
                found_eDAQ = True
                break
        if found_eDAQ:
            self.eDAQ_path = os.path.join(RAW_EDAQ_ROOT, eDAQ_run)
        else:
            raise FilenameError("No eDAQ file found for run %s" % eDAQ_file_num)

    def get_run_label(self):
        return self.run_label

    def __str__(self):
        return self.run_label

    def __repr__(self):
        return ("SingleRun object for INCA run %s" % self.run_label)


class SSRun(SingleRun):
    """Represents a single run with steady-state operation."""
    # def __init__(self, INCA_path):
        # This invokes all the actions in the parent class's __init__() method.
        # SingleRun.__init__(self, INCA_path)
        # super(SSRun, self).__init__(INCA_path)
        # https://stackoverflow.com/questions/5166473/inheritance-and-init-method-in-python
        # https://stackoverflow.com/questions/222877/what-does-super-do-in-python

    def abridge_data(self):
        """Isolates important events in data by removing any long stretches of
        no pedal input of pedal events during which the throttle position >45
        deg or whatever throttle threshold not sustained for >2 seconds (or
        whatever time threshold).
        """
        # different version of this function in SSRun vs. DownhillRun

        # Define constants used to isolating valid events.
        self.THRTL_THRESH = 45 # degrees ("throttle threshold")
        self.THRTL_T_THRESH = 2 # seconds ("throttle time threshold")

        # list of start and end times for pedal-down events with a segment of >45deg
        # throttle for >2s.
        valid_event_times = []

        # maintain a buffer of candidate pedal-down and throttle time vals.
        ped_buffer = []
        high_throttle_time = [0, 0]

        print("\nSteady-state event parsing:")
        pedal_down = False
        counting = False
        keep = False

        for i, ti in enumerate(self.inca_df.index):
            # Main loop evaluates pedal-down event. Stores event start and end
            # times if inner loop finds throttle was >45deg for >2s during event

            if self.inca_df["pedal_sw"][ti]:
                if not pedal_down:
                    print("\tPedal actuated at time\t\t%0.2fs" %
                                                        (ti / SAMPLING_FREQ))
                # pedal currently down
                pedal_down = True
                ped_buffer.append(ti) # add current time to pedal buffer.

                ## Calculate throttle >45 deg time to determine event validity
                if not counting and (self.inca_df["throttle"][ti] >
                                                            self.THRTL_THRESH):
                    # first time throttle exceeds 45 deg
                    print("\t\tThrottle >%d deg at time\t%0.2fs" %
                                        (self.THRTL_THRESH, ti / SAMPLING_FREQ))
                    high_throttle_time[0] = ti
                    counting = True

                elif counting and (self.inca_df["throttle"][ti] <
                                                            self.THRTL_THRESH):
                    # throttle drops below 45 deg
                    print("\t\tThrottle <%d deg at time\t%0.2fs" %
                                        (self.THRTL_THRESH, ti / SAMPLING_FREQ))
                    high_throttle_time[1] = self.inca_df.index[i-1] # prev. time
                    delta = high_throttle_time[1] - high_throttle_time[0]
                    print("\t\tThrottle >%d deg total t:\t%0.2fs" %
                                    (self.THRTL_THRESH, delta / SAMPLING_FREQ))
                    # calculate if that >45deg event lasted longer than 2s.
                    if (high_throttle_time[1] - high_throttle_time[0] >
                                          self.THRTL_T_THRESH * SAMPLING_FREQ):
                        # Multiplying by sampling f to get hundredths of a sec.
                        keep = True
                        # now the times stored in ped_buffer constitute a valid
                        # event. As long as the pedal switch stays actuated,
                        # subsequentn time indices will be added to ped_buffer.
                    counting = False # reset indicator
                    high_throttle_time = [0, 0] # reset

            elif pedal_down:
                # pedal just lifted
                print("\tPedal lifted at time\t\t%0.2fs\n" % (ti/SAMPLING_FREQ))
                if keep:
                    valid_event_times.append( [ped_buffer[0], ped_buffer[-1]] )
                pedal_down = False
                ped_buffer = [] # flush buffer
                keep = False # reset
            else:
                # pedal is not currently down, and wasn't just lifted.
                pass

        print("\nValid steady-state ranges:")
        for event_time in valid_event_times:
            print("\t%0.2f\t->\t%0.2f"
               % (event_time[0] / SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))

        if not valid_event_times:
            # If no times were stored, then something might be wrong.
            raise DataTrimError("No valid pedal-down events found.")

        # make sure if two >45 deg events (w/ pedal lift between) are closer
        # than 5s, don't cut into either one. Look at each pair of end/start
        # points, and if they're closer than 5s, merge those two.
        previous_pair = valid_event_times[0]
        valid_event_times_c = valid_event_times.copy()
        for n, pair in enumerate(valid_event_times[1:]):
            # print("\t%f - %f" % (pair[0], previous_pair[1]))
            if pair[0] - previous_pair[1] < (5 * SAMPLING_FREQ):
                # Replace the two pairs with a single combined pair
                del valid_event_times_c[n-1]
                valid_event_times_c[n] = [ previous_pair[0], pair[1] ]
            previous_pair = pair
        print("\nAfter any merges:")
        for event_time in valid_event_times_c:
            print("\t%0.2f\t->\t%0.2f"
                % (event_time[0]/SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))

        # print("\nLooking for real times to use after adding buffers:")
        # add one-second buffer to each side of valid pedal-down events.
        for n, pair in enumerate(valid_event_times_c):
            if n == 0 and pair[0] <= (1 * SAMPLING_FREQ):
                # Set zero as start value if first time is less than 1s.
                new_start_i = 0
            else:
                new_start_i = self.inca_df.index.get_loc(
                                                pair[0] - 1*SAMPLING_FREQ,
                                                method="nearest", tolerance=1)
            # tolerance is really (1/SAMPLING_FREQ)*SAMPLING_FREQ = 1

            # print("New start: %f" % INCA_data["time"][new_start_i])
            pair[0] = self.inca_df.index[new_start_i]

            new_end_i = self.inca_df.index.get_loc(pair[1] + 1*SAMPLING_FREQ,
                                            method="nearest", tolerance=1)
            # tolerance is really (1/SAMPLING_FREQ)*SAMPLING_FREQ = 1
            # print("New end: %f" % INCA_data["time"][new_end_i])
            pair[1] = self.inca_df.index[new_end_i]

        print("\nINCA times with 1-second buffers added:")
        for event_time in valid_event_times_c:
            print("\t%0.2f\t->\t%0.2f"
               % (event_time[0] / SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))
        print("\n")

        # Split DataFrame into valid pieces; store in lists
        valid_inca_events = []
        valid_edaq_events = []

        desired_start_t_inca = 0
        desired_start_t_edaq = 0
        for n, time_range in enumerate(valid_event_times_c):
            INCA_start_i = self.inca_df.index.get_loc(time_range[0])
            INCA_end_i = self.inca_df.index.get_loc(time_range[1])
            edaq_match_start_i = self.edaq_df.index.get_loc(time_range[0],
                                method="nearest", tolerance=1)
            edaq_match_end_i = self.edaq_df.index.get_loc(time_range[1],
                                method="nearest", tolerance=1)
            # tolerances are really (1/SAMPLING_FREQ)*SAMPLING_FREQ = 1

            # create separate DataFrames for just this event
            valid_inca_event = self.inca_df[self.inca_df.index[INCA_start_i]
                                             :self.inca_df.index[INCA_end_i]]
            valid_edaq_event = self.edaq_df[
                                       self.edaq_df.index[edaq_match_start_i]
                                       :self.edaq_df.index[edaq_match_end_i]]

            # shift time values to maintain continuity.
            INCA_shift = time_range[0] - desired_start_t_inca
            eDAQ_shift = (self.edaq_df.index[edaq_match_start_i] -
                                                        desired_start_t_edaq)
            print("Shift (event %d): %f" % (n, INCA_shift / SAMPLING_FREQ))
            # print("\nShift (eDAQ): %f" % (eDAQ_shift))
            valid_inca_event.set_index(valid_inca_event.index - INCA_shift,
                                                                 inplace=True)
            valid_edaq_event.set_index(valid_edaq_event.index - eDAQ_shift,
                                                                 inplace=True)

            # Add events to lists
            valid_inca_events.append(valid_inca_event)
            valid_edaq_events.append(valid_edaq_event)

            # define next start time to be next time value after new vector's
            # end time.
            desired_start_t_inca = self.inca_df.index[
                          self.inca_df.index.get_loc(time_range[1] - INCA_shift,
                                                method="nearest", tolerance=1)]
            desired_start_t_edaq = self.edaq_df.index[
                          self.edaq_df.index.get_loc(desired_start_t_inca,
                                                method="nearest", tolerance=1)]
            # tolerances are really (1/SAMPLING_FREQ)*SAMPLING_FREQ = 1
            # print("desired start times (INCA/eDAQ): %f, %f" %
            # (desired_start_t_inca, desired_start_t_edaq))

        # Now re-assemble the DataFrame with only valid events.
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        self.inca_df = pd.concat(valid_inca_events)
        self.edaq_df = pd.concat(valid_edaq_events)

        print("\nINCA file time span: %f -> %f (%d data points)" %
               (self.inca_df.index[0]/SAMPLING_FREQ,
                self.inca_df.index[-1]/SAMPLING_FREQ, len(self.inca_df.index)))
        print("eDAQ file time span: %f -> %f (%d data points)" %
               (self.edaq_df.index[0]/SAMPLING_FREQ,
                self.edaq_df.index[-1]/SAMPLING_FREQ, len(self.edaq_df.index)))


    def get_run_type(self):
        return "SSRun"


class DownhillRun(SingleRun):
    """Represents a single run with downhill engine-braking operation."""

    def abridge_data(self):
        pass

    def get_run_type(self):
        return "DownhillRun"


def calc_cvt_ratio(engine_spd, gnd_spd):
    """Calculate and return CVT ratio (float), using given engine speed and
    vehicle speed."""
    # from Excel forumula:
    # engine_spd /
    #     (convert(convert(gnd_spd, "mi", "in"), "min", "hr") /
    #         (pi * 0.965 * 18) *
    #         (1.95 * 11.47)
    #     )

    rolling_radius_factor = 0.965
    tire_diam_in = 18
    tire_circ = math.pi * tire_diam_in * rolling_radius_factor

    axle_ratio = 11.47
    gearbox_ratio = 1.95

    gnd_spd_in_min = gnd_spd * 5280 * 12/60 # ground speed in inches/min
    tire_ang_spd = gnd_spd_in_min / tire_circ

    input_shaft_ang_spd = tire_ang_spd * axle_ratio * gearbox_ratio
    if input_shaft_ang_spd == 0:
        # If ground speed is zero, CVT ratio is infinite
        return ""

    cvt_ratio = engine_spd / input_shaft_ang_spd
    if cvt_ratio == 0 or cvt_ratio > 5:
        # EX1 CVT can't be above 5, so if it appears to be, then clutch is
        # disengaged and CVT ratio can't be calculated this way.
        return ""
    else:
        return cvt_ratio


def add_cvt_ratio(sync_array):
    """Add a calculated CVT ratio to each row"""

    header_row_count = 2
    # Confirm assumption that only first two rows contain headers.
    if sync_array[header_row_count][0] != 0:
        raise DataReadError("Bad header row count assumption. Looking for data "
        "to start at row %d. That row starts with '%s'." % (header_row_count,
                                        str(sync_array[header_row_count][0])))

    engine_spd_col = 2
    gnd_spd_col = 6
    # Confirm assumptions of which columns contain engine speed and vehicle
    # speed.
    if not "engine_spd" in sync_array[0][engine_spd_col]:
        raise DataReadError("Bad engine speed column number assumption. "
        "Expected to see 'engine_spd' in column %d" % engine_spd_col)
    if not ("gnd_speed" in sync_array[0][gnd_spd_col]):
        raise DataReadError("Bad engine speed column number assumption. "
            "Expected to see 'gnd_speed' in column %d" % gnd_spd_col)

    sync_array_w_cvt = sync_array.copy()

    for row_num, row in enumerate(sync_array):
        if row_num == 0:
            # add new header name
            sync_array_w_cvt[row_num] += ["", "CVT ratio (calc)"]
        elif row_num == 1:
            sync_array_w_cvt[row_num] += ["", "rpm/rpm"]
        elif row_num < header_row_count:
            # for any other rows before data starts, add padding.
            sync_array_w_cvt[row_num] += ["", ""]
        else:
            # Test assumption of same 100 Hz sample rate between both data sets.
            # Refactor to allow easy use of find_closest() every time to relax
            # this requirement.
            if (row[0] - row[5])**2 > ((1.0/SAMPLING_FREQ)/2)**2:
                raise CVTCalcError("Bad sample rate agreement assumption of"
                                        "both data sets collected at 100 Hz.")
            engine_spd = row[engine_spd_col]
            gnd_spd = row[gnd_spd_col]

            sync_array_w_cvt[row_num].append("")
            sync_array_w_cvt[row_num].append(
                                    str(calc_cvt_ratio(engine_spd, gnd_spd)))

    return sync_array_w_cvt


def main_prog2():
    # Set up command-line argument parser
    # https://docs.python.org/3/howto/argparse.html
    # If you pass in any arguments from the command line after "python run.py",
    # This pulls them in. If "-a" or "--auto" specified, process all data.
    # If "-o" or "--over" specified, then overwrite any existing exports in
    # the ./sync_data folder (without prompting).
    # If "-p" or "-plot" specified, plot the data before and after syncing
    # for comparison.
    parser = argparse.ArgumentParser(description="Program to preprocess Ex1 "
                                                "CVT data for easier analysis")
    parser.add_argument("-a", "--auto", help="Automatically process all data "
                                    "in raw_data folders.", action="store_true")
    parser.add_argument("-o", "--over", help="Overwrite existing data in "
                    "sync_data folder without prompting.", action="store_true")
    parser.add_argument("-p", "--plot", help="Plot data before and after "
                                            "processing.", action="store_true")
    args = parser.parse_args()

    AllRuns = RunGroup(args.auto)

    if args.plot and PLOT_LIB_PRESENT:
        AllRuns.plot_runs(args.over)

    AllRuns.export_runs(args.over)


if __name__ == "__main__":
    main_prog2()
