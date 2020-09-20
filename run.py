print("Importing modules...")
import os           # Used for analyzing file paths and directories
import csv          # Needed to read in and write out data
import argparse     # Used to parse optional command-line arguments
import math         # Using pi to convert linear speed to angular speed.
import pandas as pd # Series and DataFrame structures
import numpy as np
import traceback
import time
from datetime import datetime
import getpass

try:
    import matplotlib
    matplotlib.use("Agg") # no UI backend for use w/ WSL
    # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
    import matplotlib.pyplot as plt # Needed for optional data plotting.
    PLOT_LIB_PRESENT = True
except ImportError:
    PLOT_LIB_PRESENT = False
# https://stackoverflow.com/questions/3496592/conditional-import-of-modules-in-python
print("...done\n")


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

class DataTrimError(Exception):
    pass


class RunGroup(object):
    """Represents a collection of runs from the raw_data directory."""
    def __init__(self, process_all=False, verbose=False):
        # create SingleRun object for each run but don't read in data yet.
        self.verbosity = verbose
        self.build_run_dict()
        self.process_runs(process_all)

    def build_run_dict(self):
        """Create dictionary with an entry for each INCA run in raw_data dir."""
        INCA_files = os.listdir(RAW_INCA_ROOT)
        INCA_files.sort()
        self.run_dict = {}
        # eliminate any directories that might be in the list
        for i, file in enumerate(INCA_files):
            if os.path.isdir(os.path.join(RAW_INCA_ROOT, file)):
                continue # ignore any directories found

            if "decel" in file.lower() or "deccel" in file.lower():
                # ThisRun = self.create_downhill_run(file)
                input("Skipping file '%s' because program can't process decel "
                "runs yet.\nPress Enter to acknowledge." % file)
                print("\n")
                continue
            else:
                # ThisRun = self.create_ss_run(file)
                try:
                    ThisRun = self.create_ss_run(file)
                except FilenameError as exception_text:
                    print(exception_text)
                    # https://stackoverflow.com/questions/1483429/how-to-print-an-exception-in-python
                    input("\nRun creation failed with file '%s'.\n"
                          "Press Enter to skip this run." % (file))
                    print("\n")
                    continue # Don't add to run dict

            if ThisRun.get_run_label() in self.run_dict:
                # catch duplicate run nums.
                input("\nMore than one run '%s' found in %s:\n"
                    "\t'%s'\n"
                    "\t'%s'\n"
                    "First one will be kept.\nPress Enter to acknowledge."
                    % (ThisRun.get_run_label(), RAW_INCA_ROOT,
                     self.run_dict[ThisRun.get_run_label()].get_inca_filename(),
                     file))
                print("\n")
                continue
            self.run_dict[ThisRun.get_run_label()] = ThisRun

    def create_ss_run(self, filename):
        return SSRun(os.path.join(RAW_INCA_ROOT, filename), self.verbosity)

    def create_downhill_run(self, filename):
        return DownhillRun(os.path.join(RAW_INCA_ROOT, filename), self.verbosity)

    def process_runs(self, process_all=False):
        if process_all:
            # automatically process all INCA runs (below)
            self.runs_to_process = self.run_dict
        else:
            # prompt user for single run to process.
            OnlyRun = self.prompt_for_run()
            self.runs_to_process = {OnlyRun.get_run_label(): OnlyRun}

        bad_runs = []
        for run_num in self.runs_to_process:
            RunObj = self.runs_to_process[run_num]
            try:
                RunObj.process_data()
            except Exception:
                exception_trace = traceback.format_exc()
                # https://stackoverflow.com/questions/1483429/how-to-print-an-exception-in-python
                out_file = log_exception(exception_trace, RunObj.get_output())
                input("\nProcessing failed on run '%s'.\nOutput and exception "
                    "trace written to '%s' on Desktop.\n"
                    "Press Enter to skip this run." % (run_num, out_file))
                print("\n")
                # Stage for removal from run dict.
                bad_runs.append(run_num)
                continue
        if bad_runs:
            for bad_run in bad_runs:
                # Remove any errored runs from run dict so they aren't included
                # in later calls.
                self.runs_to_process.pop(bad_run)

    def plot_runs(self, overwrite=False, desc_str=None):
        # If only one run in group is to be processed, this will only loop once.
        if not self.runs_to_process:
            print("\nNo valid runs to plot.\n")
            return
        bad_runs = []
        for run_num in self.runs_to_process:
            RunObj = self.runs_to_process[run_num]
            try:
                RunObj.plot_data(overwrite, desc_str)
            except Exception:
                exception_trace = traceback.format_exc()
                # https://stackoverflow.com/questions/1483429/how-to-print-an-exception-in-python
                out_file = log_exception(exception_trace, RunObj.get_output())
                input("\nPlotting failed on run '%s'.\nOutput and exception "
                  "trace written to '%s' on Desktop.\n"
                     "Press Enter to skip this run." % (run_num, out_file))
                # Stage for removal from run dict.
                bad_runs.append(run_num)
                continue
        if bad_runs:
            for bad_run in bad_runs:
                # Remove any errored runs from run dict so they aren't included
                # in later calls.
                self.runs_to_process.pop(bad_run)

    def export_runs(self, overwrite=False, desc_str=None):
        # If only one run in group is to be processed, this will only loop once.
        if not self.runs_to_process:
            print("\nNo valid runs to export.\n")
            return
        bad_runs = []
        for run_num in self.runs_to_process:
            RunObj = self.runs_to_process[run_num]
            try:
                RunObj.export_data(overwrite, desc_str)
            except Exception:
                exception_trace = traceback.format_exc()
                # https://stackoverflow.com/questions/1483429/how-to-print-an-exception-in-python
                out_file = log_exception(exception_trace, RunObj.get_output())
                input("\nExporting failed on run '%s'.\nOutput and exception "
                    "trace written to '%s' on Desktop.\n"
                    "Press Enter to skip this run." % (run_num, out_file))
                # Stage for removal from run dict.
                bad_runs.append(run_num)
                continue
        if bad_runs:
            for bad_run in bad_runs:
                # Remove any errored runs from run dict so they aren't included
                # in later calls.
                self.runs_to_process.pop(bad_run)

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
            print("No valid INCA file found matching that run.")
            return self.prompt_for_run()


class SingleRun(object):
    """Represents a single run from the raw_data directory.
    No data is read in until read_data() called.
    """
    def __init__(self, INCA_path, verbose=False):
        # Create a new object to store and print output info
        self.Doc = Output(verbose)
        self.INCA_path = INCA_path
        self.INCA_filename = os.path.basename(self.INCA_path)
        self.parse_run_num()

    def parse_run_num(self):
        try:
            self.run_label = self.INCA_filename.split("_")[1][0:4]
        except IndexError:
            raise FilenameError("INCA filename '%s' not in correct format.\n"
            "Expected format is "
            "'[pretext]_[four-digit run num][anything else]'.\nNeed the four "
            "characters that follow the first underscore to be run num."
                                                        % self.INCA_filename)
        if any(not char.isdigit() for char in self.run_label):
            raise FilenameError("INCA filename '%s' not in correct format.\n"
            "Expected format is "
            "'[pretext]_[four-digit run num][anything else]'.\nNeed the four "
            "characters that follow the first underscore to be run num."
                                                        % self.INCA_filename)

        # Metadata string to document in outuput file
        self.meta_str = "INCA_file: '%s' | " % self.INCA_filename

    def process_data(self):
        self.read_data()
        self.sync_data()
        self.abridge_data()
        self.add_math_channels()

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
            try:
                run_num_i = os.path.splitext(eDAQ_run)[0].split("_")[1][0:2]
            except IndexError:
                raise FilenameError("eDAQ filename '%s' not in correct format.\n"
                "Expected format is "
                "'[pretext]_[two-digit file num][anything else]'.\nNeed the two "
                "characters that follow the first underscore to be file num.\n"
                "This will cause problems with successive runs until you fix"
                "the filename or remove the offending file from %s."
                                                    % (eDAQ_run, RAW_EDAQ_ROOT))
            if run_num_i == eDAQ_file_num:
                # break out of loop while "eDAQ_run" is set to correct filename
                found_eDAQ = True
                break
                # There is no checking for multiple eDAQ files with same run
                # num. The first one found will be used.
        if found_eDAQ:
            self.eDAQ_path = os.path.join(RAW_EDAQ_ROOT, eDAQ_run)
            # Document in metadata string for later file output.
            self.meta_str += "eDAQ file: '%s' | " % eDAQ_run
        else:
            raise FilenameError("No eDAQ file found for run %s" % eDAQ_file_num)

    def read_data(self):
        """Read in both INCA and eDAQ data from raw_data directory"""
        self.find_edaq_path()

        # Read in both eDAQ and INCA data for specific run.
        # read INCA data first
        # Open file with read priveleges.
        # File automatically closed at end of "with/as" block.
        with open(self.INCA_path, "r") as inca_ascii_file:
            self.Doc.print("\nReading INCA data from %s" % self.INCA_path) # debug
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

        # Convert the dict to a pandas DataFrame for easier manipulation
        # and analysis
        self.raw_inca_df = pd.DataFrame(data=raw_inca_dict,
                                                    index=raw_inca_dict["time"])
        self.raw_inca_df.rename(columns = {"time": "raw_inca_time"}, inplace=True)
        # https://datatofish.com/rename-columns-pandas-dataframe/
        self.Doc.print("...done")
        self.Doc.print("\nraw_inca_df after reading in data:", True)
        self.Doc.print(self.raw_inca_df.to_string(max_rows=10, max_cols=7,
                                                show_dimensions=True), True)

        # now read eDAQ data
        with open(self.eDAQ_path, "r") as edaq_ascii_file:
            self.Doc.print("Reading eDAQ data from %s" % self.eDAQ_path) # debug
            eDAQ_file_in = csv.reader(edaq_ascii_file, delimiter="\t")
            # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

            raw_edaq_dict = {}
            for channel in EDAQ_CHANNELS:
                raw_edaq_dict[channel] = []

            for j, eDAQ_row in enumerate(eDAQ_file_in):
                if j < EDAQ_HEADER_HT-1:
                    pass
                elif j == EDAQ_HEADER_HT-1:
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

                elif eDAQ_row[edaq_run_start_col+1]:
                    # Need to make sure we haven't reached end of channel strm.
                    # Time vector may keep going past a channel's data, so look
                    # at a run-specific channel to see if the run's ended.

                    # Only add this run's channels to our data list.
                    # Time is always in 1st column. Round to nearest hundredth.
                    raw_edaq_dict["time"].append(float(eDAQ_row[0]))
                    for n, channel in enumerate(EDAQ_CHANNELS[1:]):
                        raw_edaq_dict[channel].append(
                                        float(eDAQ_row[edaq_run_start_col+n]))

        # Convert the dict to a pandas DataFrame for easier manipulation
        # and analysis
        self.raw_edaq_df = pd.DataFrame(data=raw_edaq_dict,
                                                    index=raw_edaq_dict["time"])
        self.raw_edaq_df.rename(columns = {"time": "raw_edaq_time"}, inplace=True)

        self.Doc.print("...done")
        self.Doc.print("raw_edaq_df after reading in data:", True)
        self.Doc.print(self.raw_edaq_df.to_string(max_rows=10, max_cols=7,
                                                    show_dimensions=True), True)

    def sync_data(self):
        # Create copies of the raw dfs to modify and merge.
        inca_df = self.raw_inca_df.copy(deep=True)
        edaq_df = self.raw_edaq_df.copy(deep=True)

        # Convert index from seconds to hundredths of a second
        # It's simple for eDAQ data.
        edaq_df.set_index(pd.Index([int(round(ti * SAMPLING_FREQ))
                                        for ti in edaq_df.index]), inplace=True)
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html
        # Offset time series to start at zero.
        self.shift_time_series(edaq_df, zero=True)

        # INCA time increments are slightly off, so error accumulates and
        # can eventually cause issues.
        # Convert INCA to deltas by subtracting each time value from previous
        # one, rounding the delta and adding to the previous (rounded) value.
        # Calculate the rolling difference (delta) between each pair of vals.
        deltas = inca_df["raw_inca_time"].diff()
        # https://stackoverflow.com/questions/13114512/calculating-difference-between-two-rows-in-python-pandas
        deltas[0] = 0 # first val was NaN
        # Round the series then do a cumulative summation:
        deltas = pd.Series([round(ti * SAMPLING_FREQ) for ti in deltas])
        rolling_delt_times = deltas.cumsum()
        # Assign as the index now (converting to int)
        inca_df.set_index(pd.Index(rolling_delt_times.astype(int)), inplace=True)

        # Check if any pedal events exist
        if inca_df.loc[inca_df["pedal_sw"] == 1].empty:
            raise DataSyncError("No pedal event found in INCA data (looking for"
            " value of 1 in pedal switch column). Check input pedal data and "
            "ordering of input file's columns.")
        if edaq_df.loc[edaq_df["pedal_sw"] == 1].empty:
            raise DataSyncError("No pedal event found in eDAQ data (looking for"
            " value of 1 in pedal switch column). Check input pedal data and "
            "ordering of input file's columns.")
        # find first pedal switch event
        # https://stackoverflow.com/questions/16683701/in-pandas-how-to-get-the-index-of-a-known-value
        inca_high_start_t = inca_df.loc[inca_df["pedal_sw"] == 1].index[0]
        edaq_high_start_t = edaq_df.loc[edaq_df["pedal_sw"] == 1].index[0]
        self.Doc.print("\nStart times (inca, edaq): %.2fs, %.2fs"
                                    % (inca_high_start_t / SAMPLING_FREQ,
                                       edaq_high_start_t / SAMPLING_FREQ), True)

        # Test first to see if either data set has first pedal event earlier
        # than 1s. If so, that's the new time for both files to line up at.
        start_buffer = min([1 * SAMPLING_FREQ, inca_high_start_t,
                                                            edaq_high_start_t])
        # self.Doc.print("start buffer: %f" % start_buffer)

        # shift time values, leaving negative values in early part of file that
        # will be trimmed off below.
        inca_target_t = inca_high_start_t - start_buffer
        edaq_target_t = edaq_high_start_t - start_buffer

        self.shift_time_series(inca_df, offset_val=-inca_target_t)
        self.shift_time_series(edaq_df, offset_val=-edaq_target_t)
        # self.Doc.print("new inca index start: %d" % inca_df.index[0])
        # self.Doc.print("new edaq index start: %d" % edaq_df.index[0])

        # Unify datasets into one DataFrame
        # Slice out values before t=0 (1s before first pedal press)
        # Automatically truncates longer data set
        # The only channel in eDAQ that's valuable, and unique is gnd_speed.

        self.sync_df = pd.merge(inca_df.loc[0:],
        edaq_df.loc[0:, edaq_df.columns.isin(["raw_edaq_time", "gnd_speed"])],
                                        left_index=True, right_index=True)

        self.Doc.print("\nsync_df at end of sync:", True)
        self.Doc.print(self.sync_df.to_string(max_rows=10, max_cols=7,
                                                    show_dimensions=True), True)

    def shift_time_series(self, df, zero=False, offset_val=None):
        """If offset_val param specified, add this signed value to all time
        values.
        If zero param passed, offset all such that first val is 0."""
        if zero:
            offset_val = -df.index[0]
        elif offset_val==None:
            raise DataSyncError("shift_time_series needs either zero or "
                                                    "offset_val param.")

        shifted_time_series = df.index + offset_val
        df.set_index(shifted_time_series, inplace=True)

        # Maybe could use df.shift() here instead.
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html

    def abridge_data(self):
        # Implemented in child classes
        pass

    def add_math_channels(self):
        self.math_df = pd.DataFrame(index=self.sync_df.index)
        # https://stackoverflow.com/questions/18176933/create-an-empty-data-frame-with-index-from-another-data-frame
        self.add_cvt_ratio()

    def add_cvt_ratio(self):
        ROLLING_RADIUS_FACTOR = 0.965
        TIRE_DIAM_IN = 18 # inches
        tire_circ = math.pi * TIRE_DIAM_IN * ROLLING_RADIUS_FACTOR # inches

        AXLE_RATIO = 11.47
        GEARBOX_RATIO = 1.95

        gnd_spd_in_min = self.sync_df["gnd_speed"] * 5280 * 12/60 # inches/min

        tire_ang_spd = gnd_spd_in_min / tire_circ
        self.math_df["input_shaft_ang_spd"] = tire_ang_spd * AXLE_RATIO * GEARBOX_RATIO
        self.math_df["cvt_ratio"] = (self.sync_df["engine_spd"]
                                        / self.math_df["input_shaft_ang_spd"])

        # # Remove any values that are zero or > 5 (including infinite).
        self.math_df["cvt_ratio"].mask((self.math_df["cvt_ratio"] > 5)
            | (self.math_df["cvt_ratio"] == 0), inplace=True)  # replaces w/ NaN

        # Transcribe to main DF for export
        self.sync_df["CVT_ratio_calc"] = self.math_df["cvt_ratio"].copy()
        CHANNEL_UNITS["CVT_ratio_calc"] = "rpm/rpm"

    def plot_data(self, overwrite=False, description=None):
        self.plot_abridge_compare(overwrite, description)

    def plot_abridge_compare(self, overwrite=False, description=None):
        # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html
        ax1 = plt.subplot(211)
        plt.plot(self.raw_inca_df.index, self.raw_inca_df["throttle"],
                                                        label="Throttle (og)")
        plt.title("Throttle vs. Time (Run %s)" % self.run_label)
        plt.ylabel("Throttle (deg)")
        plt.legend(loc="best")
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
        # Convert DF indices from hundredths of a second to seconds
        sync_time_series = [round(ti/SAMPLING_FREQ, 2)
                                                for ti in self.sync_df.index]
        plt.plot(sync_time_series, self.sync_df["throttle"],
                                                    label="Throttle (synced)")
        # https://matplotlib.org/3.2.1/gallery/subplots_axes_and_figures/shared_axis_demo.html#sphx-glr-gallery-subplots-axes-and-figures-shared-axis-demo-py

        plt.xlabel("Time (s)")
        plt.ylabel("Throttle (deg)")
        plt.legend(loc="best")

        if description:
            fig_filepath = "./figs/%s_abr-%s.png" % (self.run_label, description)
        else:
            fig_filepath = "./figs/%s_abr.png" % self.run_label

        if os.path.exists(fig_filepath) and not overwrite:
            ow_answer = ""
            while ow_answer.lower() not in ["y", "n"]:
                ow_answer = input("\n%s already exists in figs folder. "
                        "Overwrite? (Y/N)\n> " % os.path.basename(fig_filepath))
            if ow_answer.lower() == "n":
                plt.clf()
                return

        self.Doc.print("\nExporting plot as %s..." % fig_filepath)
        plt.savefig(fig_filepath)
        self.Doc.print("...done")
        # plt.show() # can't use w/ WSL.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib
        plt.clf()

    def export_data(self, overwrite=False, description=None):
        export_df = self.sync_df.drop(columns=["raw_inca_time", "raw_edaq_time"])
        # https://stackoverflow.com/questions/29763620/how-to-select-all-columns-except-one-column-in-pandas

        # Create to list of lists for easier writing out
        # Convert time values from hundredths of a second to seconds
        time_series = [round(ti/SAMPLING_FREQ,2)
                                    for ti in export_df.index.tolist()]

        # # Replace any NaNs with blanks
        export_df.fillna("", inplace=True)
        # https://stackoverflow.com/questions/26837998/pandas-replace-nan-with-blank-empty-string

        sync_array = export_df.values.tolist()
        # https://stackoverflow.com/questions/28006793/pandas-dataframe-to-list-of-lists

        # for line_no, inca_line in enumerate(sync_array):
        for line_no, line in enumerate(sync_array):
            # put in time values
            sync_array[line_no].insert(0, time_series[line_no])
            # https://stackoverflow.com/questions/8537916/whats-the-idiomatic-syntax-for-prepending-to-a-short-python-list

        # Format header rows
        channel_list = ["time"] + export_df.columns.tolist()
        header_rows = [channel_list, [CHANNEL_UNITS[c] for c in channel_list]]

        # Add headers to array
        sync_array.insert(0, header_rows[1])
        sync_array.insert(0, header_rows[0])

        # Add metadata string (removing final unneded separator)
        sync_array.insert(0, [self.get_meta_str()])

        if description:
            sync_basename = "%s_Sync_%s.csv" % (self.run_label, description)
        else:
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

        # Create new CSV file and write out. Closes automatically at end of
        # with/as block.
        # This block should not run if answered no to overwrite above.
        with open(sync_filename, 'w+') as sync_file:
            sync_file_csv = csv.writer(sync_file, dialect="excel")

            self.Doc.print("\nWriting combined data to %s..." % sync_filename)
            sync_file_csv.writerows(sync_array)
            self.Doc.print("...done")

    def get_run_label(self):
        return self.run_label

    def get_inca_filename(self):
        return self.INCA_filename

    def get_meta_str(self):
        return self.meta_str[:-3]

    def get_output(self):
        return self.Doc

    def __str__(self):
        return self.run_label

    def __repr__(self):
        return ("SingleRun object for INCA run %s" % self.run_label)


class SSRun(SingleRun):
    """Represents a single run with steady-state operation."""
    # def __init__(self, INCA_path):
    #     # This performs all the actions in the parent class's method.
    #     SingleRun.__init__(self, INCA_path)
    #     super(SSRun, self).__init__(INCA_path)
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

        self.Doc.print("\nSteady-state event parsing:")
        pedal_down = False
        counting = False
        keep = False

        for i, ti in enumerate(self.sync_df.index):
            # Main loop evaluates pedal-down event. Stores event start and end
            # times if inner loop finds throttle was >45deg for >2s during event

            if self.sync_df["pedal_sw"][ti] == 1:
                if not pedal_down:
                    self.Doc.print("\tPedal actuated at time\t\t%0.2fs" %
                                                        (ti / SAMPLING_FREQ))
                # pedal currently down
                pedal_down = True
                ped_buffer.append(ti) # add current time to pedal buffer.

                ## Calculate throttle >45 deg time to determine event validity
                if not counting and (self.sync_df["throttle"][ti] >
                                                            self.THRTL_THRESH):
                    # first time throttle exceeds 45 deg
                    self.Doc.print("\t\tThrottle >%d deg at time\t%0.2fs" %
                                        (self.THRTL_THRESH, ti / SAMPLING_FREQ))
                    high_throttle_time[0] = ti
                    counting = True

                elif counting and (self.sync_df["throttle"][ti] <
                                                            self.THRTL_THRESH):
                    # throttle drops below 45 deg
                    self.Doc.print("\t\tThrottle <%d deg at time\t%0.2fs" %
                                        (self.THRTL_THRESH, ti / SAMPLING_FREQ))
                    high_throttle_time[1] = self.sync_df.index[i-1] # prev. time
                    delta = high_throttle_time[1] - high_throttle_time[0]
                    self.Doc.print("\t\tThrottle >%d deg total t:\t%0.2fs" %
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
                self.Doc.print("\tPedal lifted at time\t\t%0.2fs\n" % (ti/SAMPLING_FREQ))
                if keep:
                    valid_event_times.append( [ped_buffer[0], ped_buffer[-1]] )
                pedal_down = False
                ped_buffer = [] # flush buffer
                keep = False # reset
            else:
                # pedal is not currently down, and wasn't just lifted.
                pass

        self.Doc.print("\nValid steady-state ranges:")
        for event_time in valid_event_times:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
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
            # self.Doc.print("\t%f - %f" % (pair[0], previous_pair[1]))
            if pair[0] - previous_pair[1] < (5 * SAMPLING_FREQ):
                # Replace the two pairs with a single combined pair
                del valid_event_times_c[n-1]
                valid_event_times_c[n] = [ previous_pair[0], pair[1] ]
            previous_pair = pair
        self.Doc.print("\nAfter any merges:")
        for event_time in valid_event_times_c:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
                % (event_time[0]/SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))

        # add one-second buffer to each side of valid pedal-down events.
        for n, pair in enumerate(valid_event_times_c):
            if n == 0 and pair[0] <= (1 * SAMPLING_FREQ):
                # Set zero as start value if first time is less than 1s.
                new_start_i = 0
            else:
                new_start_i = self.sync_df.index.get_loc(
                                                pair[0] - 1*SAMPLING_FREQ,
                                                method="nearest", tolerance=1)
            # tolerance is really (1/SAMPLING_FREQ)*SAMPLING_FREQ = 1

            pair[0] = self.sync_df.index[new_start_i]

            new_end_i = self.sync_df.index.get_loc(pair[1] + 1*SAMPLING_FREQ,
                                                            method="nearest")
            # If file ends less than 1s after event ends, this will return
            # the last time in the file. No tolerance specified for this reason.

            pair[1] = self.sync_df.index[new_end_i]

        self.Doc.print("\nINCA times with 1-second buffers added:")
        for event_time in valid_event_times_c:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
               % (event_time[0] / SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))
        self.Doc.print("\n")

        # Split DataFrame into valid pieces; store in lists
        valid_events = []
        desired_start_t = 0
        for n, time_range in enumerate(valid_event_times_c):
            # create separate DataFrames for just this event
            valid_event = self.sync_df[time_range[0]:time_range[1]]

            # shift time values to maintain continuity.
            shift = time_range[0] - desired_start_t
            self.Doc.print("Shift (event %d): %.2f" % (n, shift / SAMPLING_FREQ))

            self.shift_time_series(valid_event, offset_val=-shift)

            # Add events to lists
            valid_events.append(valid_event)

            # define next start time to be next time value after new vector's
            # end time.
            desired_start_t = time_range[1]-shift

        # Now re-assemble the DataFrame with only valid events.
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        self.sync_df = pd.concat(valid_events)
        self.Doc.print("\nsync_df after abridgement:", True)
        self.Doc.print(self.sync_df.to_string(max_rows=10, max_cols=7,
                                                    show_dimensions=True), True)

        self.Doc.print("\nData time span: %.2f -> %.2f (%d data points)" %
               (self.sync_df.index[0]/SAMPLING_FREQ,
                self.sync_df.index[-1]/SAMPLING_FREQ, len(self.sync_df.index)))

    def add_math_channels(self):
        # This performs all the actions in the parent class's method
        super(SSRun, self).add_math_channels()
        self.add_ss_avgs()

    def add_ss_avgs(self):
        WIN_SIZE_AVG = 201  # window size for speed rolling avg.
        WIN_SIZE_SLOPE = 21 # win size for rolling slope of speed rolling avg.

        GSPD_CR = 2.5     # mph. Ground speed (min) criterion for determining if
                          # steady-state event is moving rather than stationary.
        GS_SLOPE_CR = 0.125  # mph/s.
        # Ground-speed slope (max) criterion to est. steady-state. Abs value

        ESPD_CR = 2750    # rpm. Engine speed (min) criterion for determining if
                          # steady-state event is moving rather than stationary.
        ES_SLOPE_CR = 100  # rpm/s.
        # Engine-speed slope (max) criterion to est. steady-state. Abs value

        # Document in metadata string for output file:
        self.meta_str += ("Steady-state calc criteria: "
                          "gnd speed above %s mph, "
                          "gnd speed slope magnitude less than %s mph/s, "
                          "eng speed above %s rpm, "
                          "eng speed slope magnitude less than %s rpm/s | "
                            % (GSPD_CR, GS_SLOPE_CR, ESPD_CR, ES_SLOPE_CR))
        self.meta_str += ("Steady-state calc rolling window sizes: "
                                                "%d for avg, %d for slope | "
                                            % (WIN_SIZE_AVG, WIN_SIZE_SLOPE))

        # Create rolling average and rolling (regression) slope of rolling avg
        # for ground speed.
        self.math_df["gs_rolling_avg"] = self.sync_df.rolling(
                           window=WIN_SIZE_AVG, center=True)["gnd_speed"].mean()
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

        self.Doc.print("\nCalculating rolling regression on ground speed data...")
        self.math_df["gs_rolling_slope"] = self.math_df["gs_rolling_avg"].rolling(
                    window=WIN_SIZE_SLOPE, center=True).apply(
                        lambda x: np.polyfit(x.index/SAMPLING_FREQ, x, 1)[0])
        self.Doc.print("...done")

        # Create rolling average and rolling (regression) slope of rolling avg
        # for engine speed.
        self.math_df["es_rolling_avg"] = self.sync_df.rolling(
                          window=WIN_SIZE_AVG, center=True)["engine_spd"].mean()

        self.Doc.print("Calculating rolling regression on engine speed data...")
        self.math_df["es_rolling_slope"] = self.math_df["es_rolling_avg"].rolling(
                    window=WIN_SIZE_SLOPE, center=True).apply(
                        lambda x: np.polyfit(x.index/SAMPLING_FREQ, x, 1)[0])
        self.Doc.print("...done")

        # Apply speed and speed slope criteria to isolate steady-state events.
        # Use compound OR statement to generate a mask.
        # criteria_mask = (  (self.math_df["gs_rolling_avg"] < GSPD_CR)
        #                  | (self.math_df["gs_rolling_slope"] > GS_SLOPE_CR)
        #                  | (self.math_df["gs_rolling_slope"] < -GS_SLOPE_CR)
        #                  | (self.math_df["es_rolling_avg"] < ESPD_CR)
        #                  | (self.math_df["es_rolling_slope"] > ES_SLOPE_CR)
        #                  | (self.math_df["es_rolling_slope"] < -ES_SLOPE_CR) )
        ss_filter = (      (self.math_df["gs_rolling_avg"] > GSPD_CR)
                         & (self.math_df["gs_rolling_slope"] < GS_SLOPE_CR)
                         & (self.math_df["gs_rolling_slope"] > -GS_SLOPE_CR)
                         & (self.math_df["es_rolling_avg"] > ESPD_CR)
                         & (self.math_df["es_rolling_slope"] < ES_SLOPE_CR)
                         & (self.math_df["es_rolling_slope"] > -ES_SLOPE_CR) )
        # GS_SLOPE_CR and ES_SLOPE_CR are abs value so have to apply on high
        # and low end.
        self.Doc.print("\nTotal data points that fail steady-state criteria: %d"
                                                        % sum(~ss_filter), True)
        self.Doc.print("Total data points that meet steady-state criteria: %d"
                                                         % sum(ss_filter), True)
        # https://stackoverflow.com/questions/12765833/counting-the-number-of-true-booleans-in-a-python-list

        self.math_df["steady_state"] = ss_filter

        # "Mask off" by assigning NaN where criteria not met.
        self.math_df["gs_rol_avg_mskd"] = self.math_df["gs_rolling_avg"].mask(
                                                                    ~ss_filter)
        self.math_df["es_rol_avg_mskd"] = self.math_df["es_rolling_avg"].mask(
                                                                    ~ss_filter)
        # Masking these too to calculate avg slope off SS region later:
        self.math_df["gs_rslope_mskd"] = self.math_df["gs_rolling_slope"].mask(
                                                                    ~ss_filter)
        self.math_df["es_rslope_mskd"] = self.math_df["es_rolling_slope"].mask(
                                                                    ~ss_filter)
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-where-mask

        # No need to run rolling avg or slope on cvt_ratio since we aren't
        # applying criteria to it for purpose of determining steady state.
        self.math_df["cvt_ratio_mskd"] = self.math_df["cvt_ratio"].mask(
                                                                ~ss_filter)

        # Calculate overall (aggregate) mean of each filtereed/masked channel
        # Prefill with NaN and assign mean to first element
        self.math_df["SS_gnd_spd_avg"] = np.nan
        self.math_df.at[0, "SS_gnd_spd_avg"] = np.mean(
                                                self.math_df["gs_rol_avg_mskd"])
        # self.math_df["SS_gnd_spd_slope_avg"] = np.nan
        # self.math_df["SS_gnd_spd_slope_avg"][0] = np.mean(
        #                                        self.math_df["gs_rolling_slope"])
        self.math_df["SS_eng_spd_avg"] = np.nan
        self.math_df.at[0, "SS_eng_spd_avg"] = np.mean(
                                                self.math_df["es_rol_avg_mskd"])
        # self.math_df["SS_eng_spd_slope_avg"] = np.nan
        # self.math_df["SS_eng_spd_slope_avg"][0] = np.mean(
        #                                        self.math_df["es_rolling_slope"])
        self.math_df["SS_cvt_ratio_avg"] = np.nan
        self.math_df.at[0, "SS_cvt_ratio_avg"] = np.mean(
                                                   self.math_df["cvt_ratio_mskd"])
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.at.html

        # Transcribe to main DF for export
        # Leave out average SS slopes.
        self.sync_df["SS_gnd_spd_avg_calc"] = self.math_df["SS_gnd_spd_avg"]
        self.sync_df["SS_eng_spd_avg_calc"] = self.math_df["SS_eng_spd_avg"]
        self.sync_df["SS_cvt_ratio_avg_calc"] = self.math_df["SS_cvt_ratio_avg"]
        CHANNEL_UNITS["SS_gnd_spd_avg_calc"] = CHANNEL_UNITS["gnd_speed"]
        CHANNEL_UNITS["SS_eng_spd_avg_calc"] = CHANNEL_UNITS["engine_spd"]
        CHANNEL_UNITS["SS_cvt_ratio_avg_calc"] = CHANNEL_UNITS["CVT_ratio_calc"]

        self.Doc.print("\nsync_df after adding steady-state data:", True)
        self.Doc.print(self.sync_df.to_string(max_rows=10, max_cols=7,
                                                    show_dimensions=True), True)


        # pandas rolling(), apply(), regression references:
        # https://stackoverflow.com/questions/47390467/pandas-dataframe-rolling-with-two-columns-and-two-rows
        # https://pandas.pydata.org/pandas-docs/version/0.23.4/whatsnew.html#rolling-expanding-apply-accepts-raw-false-to-pass-a-series-to-the-function
        # https://stackoverflow.com/questions/49100471/how-to-get-slopes-of-data-in-pandas-dataframe-in-python
        # https://www.pythonprogramming.net/rolling-apply-mapping-functions-data-analysis-python-pandas-tutorial/
        # https://stackoverflow.com/questions/21025821/python-custom-function-using-rolling-apply-for-pandas
        # http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
        # https://stackoverflow.com/questions/50482884/module-pandas-has-no-attribute-rolling-mean
        # https://stackoverflow.com/questions/45254174/how-do-pandas-rolling-objects-work
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html
        # https://becominghuman.ai/linear-regression-in-python-with-pandas-scikit-learn-72574a2ec1a5
        # https://medium.com/the-code-monster/split-a-dataset-into-train-and-test-datasets-using-sk-learn-acc7fd1802e0
        # https://towardsdatascience.com/regression-plots-with-pandas-and-numpy-faf2edbfad4f
        # https://data36.com/linear-regression-in-python-numpy-polyfit/

    def plot_data(self, overwrite=False, description=None):
        # This performs all the actions in the parent class's method
        super(SSRun, self).plot_data(overwrite, description)
        self.plot_ss_range(overwrite, description)

    def plot_ss_range(self, overwrite=False, description=None):
        ax1 = plt.subplot(311)
        plt.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["gnd_speed"],
                                                        label="Ground Speed")
        plt.plot(self.sync_df.index/SAMPLING_FREQ, self.math_df["gs_rolling_avg"],
                                                        label="Rolling Avg")
        plt.plot(self.sync_df.index/SAMPLING_FREQ, self.math_df["gs_rol_avg_mskd"],
                                                        label="Steady-state")
        plt.title("Steady-state Isolation (Run %s)" % self.run_label)
        plt.ylabel("Speed (mph)")
        plt.legend(loc="best")
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(312, sharex=ax1)
        # Convert DF indices from hundredths of a second to seconds
        plt.plot(self.sync_df.index/SAMPLING_FREQ,
                        self.sync_df["engine_spd"], label="Engine Speed")
        plt.plot(self.sync_df.index/SAMPLING_FREQ,
                        self.math_df["es_rolling_avg"], label="Rolling Avg")
        plt.plot(self.sync_df.index/SAMPLING_FREQ,
                        self.math_df["es_rol_avg_mskd"], label="Steady-state")

        plt.ylabel("Engine Speed (rpm)")
        plt.legend(loc="best")
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3 = plt.subplot(313, sharex=ax1)
        # Convert DF indices from hundredths of a second to seconds
        plt.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["throttle"],
                                                        label="Throttle")

        plt.xlabel("Time (s)")
        plt.ylabel("Throttle (deg)")
        plt.legend(loc="best")

        if description:
            fig_filepath = "./figs/%s_ss-%s.png" % (self.run_label, description)
        else:
            fig_filepath = "./figs/%s_ss.png" % self.run_label

        if os.path.exists(fig_filepath) and not overwrite:
            ow_answer = ""
            while ow_answer.lower() not in ["y", "n"]:
                ow_answer = input("\n%s already exists in figs folder. "
                        "Overwrite? (Y/N)\n> " % os.path.basename(fig_filepath))
            if ow_answer.lower() == "n":
                plt.clf()
                return

        print("\nExporting plot as %s..." % fig_filepath)
        plt.savefig(fig_filepath)
        print("...done")
        # plt.show() # can't use w/ WSL.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib
        plt.clf()

    def get_run_type(self):
        return "SSRun"


class DownhillRun(SingleRun):
    """Represents a single run with downhill engine-braking operation."""

    def abridge_data(self):
        pass

    def get_run_type(self):
        return "DownhillRun"


def log_exception(excp_str, Out):
    # get date/timestamp
    # write output file

    # Find Desktop path
    username = getpass.getuser()
    # https://stackoverflow.com/questions/842059/is-there-a-portable-way-to-get-the-current-username-in-python
    home_contents = os.listdir("/mnt/c/Users/%s" % username)
    onedrive = [folder for folder in home_contents if "OneDrive -" in folder][0]
    desktop_path = "/mnt/c/Users/%s/%s/Desktop" % (username, onedrive)

    # Wait one second to prevent overwriting previous error if it occurred less
    # than one second ago.
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    # https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
    filename = timestamp + "_CVT_data_processing_error.txt"

    print(excp_str)
    time.sleep(1)
    with open(os.path.join(desktop_path, filename), "w") as log_file:
        log_file.write(Out.get_log_dump() + excp_str)

    return filename


class Output(object):
    def __init__(self, verbose):
        self.verbose = verbose
        self.log_string = ""
    def print(self, string, verbose_only=False):
        if verbose_only and not self.verbose:
            return
        else:
            self.log_string += string + "\n"
            print(string)
    def get_log_dump(self):
        return self.log_string


def main_prog():
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
    parser.add_argument("-v", "--verbose", help="Include additional output for "
                                            "diagnosis.", action="store_true")
    parser.add_argument("-d", "--desc", help="Specify a description string to "
        "append to output file names - data and plot files (if -p also used)",
                                                        type=str, default="")

    # https://www.programcreek.com/python/example/748/argparse.ArgumentParser
    args = parser.parse_args()

    AllRuns = RunGroup(args.auto, args.verbose)

    if args.plot and PLOT_LIB_PRESENT:
        AllRuns.plot_runs(args.over, args.desc)
    elif args.plot:
        print("\nFailed to import matplotlib. Cannot plot data.")

    AllRuns.export_runs(args.over, args.desc)


if __name__ == "__main__":
    main_prog()
