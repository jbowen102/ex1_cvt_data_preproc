print("Importing modules...")
import os           # Used for analyzing file paths and directories
import csv          # Needed to read in and write out data
import argparse     # Used to parse optional command-line arguments
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
import pandas as pd # Series and DataFrame structures
import numpy as np
import traceback
import time
from datetime import datetime
import getpass
from PIL import Image
import hashlib
import glob
from tqdm import tqdm
import copy

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

# https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
# https://pypi.org/project/tqdm/#pandas-integration
# Gives warning if tqdm version <4.33.0). Ignore.
# https://github.com/tqdm/tqdm/issues/780

# global constancts
RAW_INCA_DIR = "./raw_data/INCA"
RAW_EDAQ_DIR = "./raw_data/eDAQ"
SYNC_DIR = "./sync_data"
PLOT_DIR = "./figs"

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
        if not os.path.exists(RAW_INCA_DIR):
            raise DataReadError("No raw INCA directory found. Put data in this "
                                                "folder: %s" % RAW_INCA_DIR)
        INCA_files = os.listdir(RAW_INCA_DIR)
        INCA_files.sort()
        self.run_dict = {}
        # eliminate any directories that might be in the list

        decel_runs = []
        for i, file in enumerate(INCA_files):
            if os.path.isdir(os.path.join(RAW_INCA_DIR, file)):
                continue # ignore any directories found

            if "decel" in file.lower() or "deccel" in file.lower():
                # decel_runs.append(file)
                # continue
                try:
                    ThisRun = self.create_downhill_run(file)
                except FilenameError as exception_text:
                    print(exception_text)
                    # https://stackoverflow.com/questions/1483429/how-to-print-an-exception-in-python
                    input("\nRun creation failed with file '%s'.\n"
                          "Press Enter to skip this run." % (file))
                    print("")
                    continue # Don't add to run dict

            else:
                try:
                    ThisRun = self.create_ss_run(file)
                except FilenameError as exception_text:
                    print(exception_text)
                    # https://stackoverflow.com/questions/1483429/how-to-print-an-exception-in-python
                    input("\nRun creation failed with file '%s'.\n"
                          "Press Enter to skip this run." % (file))
                    print("")
                    continue # Don't add to run dict

            if ThisRun.get_run_label() in self.run_dict:
                # catch duplicate run nums.
                dup_answ = ""
                while dup_answ.lower() not in ["1", "2"]:
                    print("More than one run %s found in %s:\n"
                        "\t'%s'\n"
                        "\t'%s'\n"
                        "Which one should be used as run %s? (1/2)"
                        % (ThisRun.get_run_label(), RAW_INCA_DIR,
                         self.run_dict[ThisRun.get_run_label()].get_inca_filename(),
                         file, ThisRun.get_run_label()))
                    dup_answ = input("> ")
                if dup_answ.lower() == "1":
                    print("")
                    continue
                if dup_answ.lower() == "2":
                    # fall through
                    pass

            self.run_dict[ThisRun.get_run_label()] = ThisRun

        if decel_runs:
            print("Skipping these files because program can't process decel "
                                                                "runs yet:")
            for run in decel_runs:
                print("\t%s" % run)
            input("Press Enter to acknowledge.")

    def create_ss_run(self, filename):
        return SSRun(os.path.join(RAW_INCA_DIR, filename), self.verbosity)

    def create_downhill_run(self, filename):
        return DownhillRun(os.path.join(RAW_INCA_DIR, filename), self.verbosity)

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
                self.log_exception(RunObj, "Processing")
                # Stage for removal from run dict.
                bad_runs.append(run_num)
                continue
        if bad_runs:
            for bad_run in bad_runs:
                # Remove any errored runs from run dict so they aren't included
                # in later calls.
                self.runs_to_process.pop(bad_run)

    def plot_runs(self, overwrite=False, desc_str=""):
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
                self.log_exception(RunObj, "Plotting")
                # Stage for removal from run dict.
                bad_runs.append(run_num)
                continue
        if bad_runs:
            for bad_run in bad_runs:
                # Remove any errored runs from run dict so they aren't included
                # in later calls.
                self.runs_to_process.pop(bad_run)

    def export_runs(self, overwrite=False, desc_str=""):
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
                self.log_exception(exception_trace, RunObj, "Exporting")
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

    def log_exception(self, RunObj, operation):
        """Write output file for later debugging upon encountering exception."""
        exception_trace = traceback.format_exc()
        # https://stackoverflow.com/questions/1483429/how-to-print-an-exception-in-python

        # Find Desktop path
        username = getpass.getuser()
        # https://stackoverflow.com/questions/842059/is-there-a-portable-way-to-get-the-current-username-in-python
        home_contents = os.listdir("/mnt/c/Users/%s" % username)
        onedrive = [folder for folder in home_contents if "OneDrive -" in folder][0]
        desktop_path = "/mnt/c/Users/%s/%s/Desktop" % (username, onedrive)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        # https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
        filename = "%s_Run%s_%s_error.txt" % (timestamp, RunObj.get_run_label(),
                                                            operation.lower())
        print(exception_trace)
        # Wait one second to prevent overwriting previous error if it occurred less
        # than one second ago.
        time.sleep(1)
        Out = RunObj.get_output()
        with open(os.path.join(desktop_path, filename), "w") as log_file:
            log_file.write(Out.get_log_dump() + exception_trace)

        input("\n%s failed on run %s.\nOutput and exception "
            "trace written to '%s' on Desktop.\nPress Enter to skip this run."
                                % (operation, RunObj.get_run_label(), filename))
        print("")

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

        if not os.path.exists(RAW_EDAQ_DIR):
            raise DataReadError("No raw eDAQ directory found. Put data in this"
                                                "folder: %s" % RAW_EDAQ_DIR)
        all_eDAQ_runs = os.listdir(RAW_EDAQ_DIR)
        found_eDAQ = False # initialize to false. Will change if file is found.
        for eDAQ_run in all_eDAQ_runs:
            if os.path.isdir(os.path.join(RAW_EDAQ_DIR, eDAQ_run)):
                continue # ignore any directories found
            # Split the extension off the file name, then isolate the final two
            # numbers off the date
            try:
                run_num_i = os.path.splitext(eDAQ_run)[0].split("_")[1][0:2]
            except IndexError:
                raise FilenameError("eDAQ filename '%s' not in correct format."
                "\nExpected format is "
                "'[pretext]_[two-digit file num][anything else]'.\nNeed the "
                "two characters that follow the first underscore to be file "
                "num.\nThis will cause problems with successive runs until you "
                "fix the filename or remove the offending file from %s."
                                                    % (eDAQ_run, RAW_EDAQ_DIR))
            if run_num_i == eDAQ_file_num:
                # break out of loop while "eDAQ_run" is set to correct filename
                found_eDAQ = True
                break
                # There is no checking for multiple eDAQ files with same run
                # num. The first one found will be used.
        if found_eDAQ:
            self.eDAQ_path = os.path.join(RAW_EDAQ_DIR, eDAQ_run)
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
        # https://datatofish.com/rename-columns-pandas-dataframe/

        self.Doc.print("...done")
        self.Doc.print("\nraw_inca_df after reading in data:", True)
        self.Doc.print(self.raw_inca_df.to_string(max_rows=10, max_cols=7,
                                                show_dimensions=True), True)
        self.Doc.print("", True)

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

        self.Doc.print("...done")
        self.Doc.print("\nraw_edaq_df after reading in data:", True)
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
        deltas = inca_df["time"].diff()
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
        self.Doc.print("Start buffer: %0.2fs"
                                           % (start_buffer/SAMPLING_FREQ), True)

        # shift time values, leaving negative values in early part of file that
        # will be trimmed off below.
        inca_target_t = inca_high_start_t - start_buffer
        edaq_target_t = edaq_high_start_t - start_buffer

        self.shift_time_series(inca_df, offset_val=-inca_target_t)
        self.shift_time_series(edaq_df, offset_val=-edaq_target_t)
        self.Doc.print("First INCA sample shifted to time %0.2fs"
                                    % (inca_df.index[0]/SAMPLING_FREQ), True)
        self.Doc.print("First eDAQ sample shifted to time %0.2fs"
                                    % (edaq_df.index[0]/SAMPLING_FREQ), True)

        # Unify datasets into one DataFrame
        # Slice out values before t=0 (1s before first pedal press)
        # Truncate file with extra time vals at end. Will not happen during
        # join() because of the "outer" option creating union to catch any
        # time gaps in either dataset (have only seen it in one INCA run so far).
        end_time = min(inca_df.index[-1], edaq_df.index[-1])

        # The only channel in eDAQ that's valuable and unique is gnd_speed.
        # Also carry over raw time for debugging purposes.
        # The suffix options keep the two DFs' time columns from conflicting.
        self.sync_df = inca_df.loc[0:end_time].join(
                               edaq_df.loc[0:end_time, edaq_df.columns.isin(
                                                    ["time", "gnd_speed"])],
                         lsuffix="_raw_inca", rsuffix="_raw_edaq", how="outer")

        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html#pandas.DataFrame.join
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

    def knit_pedal_gaps(self):
        """Finds any gaps in INCA pedal channel and fills them if pedal signal
        is high before and after gap. Prevents false-negative in abridge
        algo."""
        na_times = self.sync_df["pedal_sw"][self.sync_df["pedal_sw"].isna()]

        if len(na_times) == 0:
            self.Doc.print("\nNo missing INCA times found.")
            return
        else:
            self.Doc.print("\nMissing INCA times (len: %d):" % len(na_times), True)
            self.Doc.print(na_times.to_string(max_rows=10), True)

        # Filter trivial case of gap length 1 to avoid IndexError below.
        if len(na_times) == 1:
            # set variables used in reassignment loop below
            gap_ranges = [na_times.index[0], na_times.index[0]]
        elif len(na_times) > 1:
            # Step through na_times and identify any discontinuities, indicating
            # multiple gaps in the data.
            gap_ranges = []
            current_range = [na_times.index[0]]
            for i, time in enumerate(na_times.index[1:]):
                prev_time = na_times.index[i] # i is behind by one.
                if time - prev_time > 1:
                    current_range.append(prev_time)
                    gap_ranges.append(current_range)
                    # Reset range
                    current_range = [time]
            # Add last value to end of last range
            current_range.append(time)
            gap_ranges.append(current_range)

        self.Doc.print("\nContinuous INCA sample gaps: ")
        for range in gap_ranges:
            self.Doc.print("\t" + " ->\t".join(str(t) for t in range))
        # https://stackoverflow.com/questions/973568/convert-nested-lists-to-string

        for range in gap_ranges:
            # Find time values before and after gap
            pre_time_i = self.sync_df.index.get_loc(range[0]) - 1
            post_time_i = self.sync_df.index.get_loc(range[-1]) + 1
            pre_time = self.sync_df.index[pre_time_i]
            post_time = self.sync_df.index[post_time_i]
            # https://stackoverflow.com/questions/28837633/pandas-get-position-of-a-given-index-in-dataframe
            knit = False
            if (self.sync_df.at[pre_time, "pedal_sw"]
                                 and self.sync_df.at[post_time, "pedal_sw"]):
                # Only need to knit gap if pedal was actuated before gap and
                # still actuated after. Assume no interruption during gap.
                self.Doc.print("\nKnitting pedal event in gap %d -> %d"
                                                        % (range[0], range[-1]))

                self.sync_df.at[range[0]:range[1]+1, "pedal_sw"] = 1
                knit = True

                self.Doc.print("\nsync_df after knitting pedal event:", True)
                self.Doc.print(self.sync_df[range[0]-2:range[1]+3].to_string(
                        max_rows=10, max_cols=7, show_dimensions=True), True)

            if not knit:
                self.Doc.print("\nNo pedal events to knit.")

    def abridge_data(self):
        # Implemented in child classes
        # Different version of this function in SSRun vs. DownhillRun
        pass

    def add_math_channels(self):
        # Implemented in child classes
        # Different version of this function in SSRun vs. DownhillRun
        pass

    def add_cvt_ratio(self):
        ROLLING_RADIUS_FACTOR = 0.965
        TIRE_DIAM_IN = 18 # inches
        tire_circ = np.pi * TIRE_DIAM_IN * ROLLING_RADIUS_FACTOR # inches

        AXLE_RATIO = 11.47
        GEARBOX_RATIO = 1.95

        if self.get_run_type() == "SSRun":
            gnd_spd_in_min = self.abr_df["gnd_speed"] * 5280 * 12/60 # inches/min
        elif self.get_run_type() == "DownhillRun":
            # Downhill run already has rolling avg available.
            # Using this more stable data to generate CVT ratio for plot.
            gnd_spd_in_min = self.math_df["gs_rolling_avg"] * 5280 * 12/60 # inches/min

        tire_ang_spd = gnd_spd_in_min / tire_circ
        self.math_df["input_shaft_ang_spd"] = tire_ang_spd * AXLE_RATIO * GEARBOX_RATIO
        self.math_df["cvt_ratio"] = (self.abr_df["engine_spd"]
                                        / self.math_df["input_shaft_ang_spd"])

        # # Remove any values that are zero or > 5 (including infinite).
        self.math_df["cvt_ratio_mskd"] = self.math_df["cvt_ratio"].mask(
            (self.math_df["cvt_ratio"] > 5) | (self.math_df["cvt_ratio"] <= 0))

        # Transcribe to main DF for export
        self.abr_df["CVT_ratio_calc"] = self.math_df["cvt_ratio_mskd"].copy()
        CHANNEL_UNITS["CVT_ratio_calc"] = "rpm/rpm"

    def plot_data(self, overwrite=False, description=""):
        self.overwrite = overwrite
        self.description = description
        self.Doc.print("")
        # https://stackoverflow.com/questions/18028504/python-is-adding-extra-newline-to-the-output
        self.plot_abridge_compare()
        self.plot_cvt_ratio()

    def plot_abridge_compare(self):
        # Implemented in child classes
        # Different version of this function in SSRun vs. DownhillRun
        pass

    def plot_cvt_ratio(self):
        # Plot vehicle speed, filtered speed, engine speed, CVT ratio
        ax1 = plt.subplot(311)
        # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html

        plt.plot(self.math_df.index/SAMPLING_FREQ,
                 self.math_df["gs_rolling_avg"], color="lightgrey", zorder=2)
        plt.plot(self.math_df.index/SAMPLING_FREQ,
                 self.math_df["gs_rol_avg_mskd"], color="r", zorder=3)
        ax1.set_ylabel("Speed (mph)")

        if self.get_run_type() == "SSRun":
            # plot average for steady-state run
            ax1.axhline(self.math_df.at[0, "SS_gnd_spd_avg"],
                                                   color="lightcoral", zorder=1)
        elif self.get_run_type() == "DownhillRun":
            # Plot slopes for downhill run
            # Restore existing y-axis limits after adding slopes because slopes
            # may extend beyond optimal window limits.
            ylims = ax1.get_ylim()
            plt.plot(self.math_df.index/SAMPLING_FREQ,
                    self.math_df["trendlines"], color="lightcoral", zorder=1, scaley=False)
            ax1.set_ylim(ylims)
            # https://stackoverflow.com/questions/7386872/make-matplotlib-autoscaling-ignore-some-of-the-plots
            # https://matplotlib.org/3.1.1/gallery/misc/zorder_demo.html
        plt.title("Run %s - CVT Ratio (Abridged Data)" % self.run_label, loc="left")
        plt.setp(ax1.get_xticklabels(), visible=False) # x labels only on bottom

        ax2 = plt.subplot(312, sharex=ax1)

        if self.get_run_type() == "SSRun":
            es_rolling_avg = self.math_df["es_rolling_avg"]
            engine_spd_mskd = self.math_df["es_rol_avg_mskd"]
        elif self.get_run_type() == "DownhillRun":
            es_rolling_avg = self.abr_df["engine_spd"]
            engine_spd_mskd = self.abr_df["engine_spd"].mask(~self.math_df["downhill_filter"])

        plt.plot(self.abr_df.index/SAMPLING_FREQ, es_rolling_avg, color="lightgrey", zorder=2)
        plt.plot(self.abr_df.index/SAMPLING_FREQ, engine_spd_mskd, color="tab:blue", zorder=3)
        if self.get_run_type() == "SSRun":
            # plot average for steady-state run
            ax2.axhline(self.math_df.at[0, "SS_eng_spd_avg"], color="lightsteelblue", zorder=1)
        ax2.set_ylabel("Engine Speed (rpm)")

        plt.setp(ax2.get_xticklabels(), visible=False) # x labels only on bottom

        ax3 = plt.subplot(313, sharex=ax1)

        plt.plot(self.math_df.index/SAMPLING_FREQ, self.math_df["cvt_ratio"],
                                                    color="lightgrey", zorder=2)
        plt.plot(self.math_df.index/SAMPLING_FREQ, self.math_df["cvt_ratio_mskd"],
                                                    color="tab:green", zorder=3)
        if self.get_run_type() == "SSRun":
            # plot average for steady-state run
            ax3.axhline(self.math_df.at[0, "SS_cvt_ratio_avg"],
                                                   color="lightgreen", zorder=1)
        ax3.set_ylabel("CVT Ratio Calc")
        ax3.set_ylim([-0.2, 4])
        ax3.set_yticks([0, 1, 2, 3, 4])

        # plt.show() # can't use w/ WSL. Export instead.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        self.export_plot("cvt")
        plt.clf()
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib

    def export_plot(self, type):
        """Exports plot that's already been created with another method.
        Assumes caller method will clear figure afterward."""
        if self.description:
            fig_filepath = ("%s/%s_%s-%s.png"
                            % (PLOT_DIR, self.run_label, type, self.description))
        else:
            fig_filepath = "%s/%s_%s.png" % (PLOT_DIR, self.run_label, type)

        short_hash_len = 6
        # Check for existing fig with same filename including description but
        # EXCLUDING hash.
        wildcard_filename = (os.path.splitext(fig_filepath)[0]
                            + "-#" + "?"*short_hash_len
                            + os.path.splitext(fig_filepath)[1])
        if glob.glob(wildcard_filename) and not self.overwrite:
            ow_answer = ""
            while ow_answer.lower() not in ["y", "n"]:
                self.Doc.print("\n%s already exists in figs folder. Overwrite? (Y/N)"
                                        % os.path.basename(wildcard_filename))
                ow_answer = input("> ")
            if ow_answer.lower() == "y":
                for filepath in glob.glob(wildcard_filename):
                    os.remove(filepath)
                # continue with rest of function
            if ow_answer.lower() == "n":
                # plot will be cleared in caller function.
                return
        elif glob.glob(wildcard_filename) and self.overwrite:
            for filepath in glob.glob(wildcard_filename):
                os.remove(filepath)

        plt.savefig(fig_filepath)
        # Calculate unique hash value (like a fingerprint) to output in CSV's
        # meta_str. Put in img filename too.
        img_hash = hashlib.sha1(Image.open(fig_filepath).tobytes())
        # https://stackoverflow.com/questions/24126596/print-md5-hash-of-an-image-opened-with-pythons-pil
        hash_text = img_hash.hexdigest()[:short_hash_len]
        fig_filepath_hash = (os.path.splitext(fig_filepath)[0] + "-#"
                                + hash_text + os.path.splitext(fig_filepath)[1])
        os.rename(fig_filepath, fig_filepath_hash)
        self.Doc.print("Exported plot as %s." % fig_filepath_hash)
        self.meta_str += ("Corresponding %s fig hash: '%s' | "
                                                            % (type, hash_text))

    def export_data(self, overwrite=False, description=""):
        self.overwrite = overwrite
        self.description = description
        export_df = self.abr_df.drop(columns=["time_raw_inca", "time_raw_edaq"])
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

        # Add metadata string
        sync_array.insert(0, [self.get_meta_str()])

        if self.description:
            sync_basename = "%s_Sync-%s.csv" % (self.run_label, self.description)
        else:
            sync_basename = "%s_Sync.csv" % self.run_label

        sync_filename = "%s/%s" % (SYNC_DIR, sync_basename)

        # Check if file exists already. Prompt user for overwrite decision.
        if os.path.exists(sync_filename) and not self.overwrite:
            ow_answer = ""
            while ow_answer.lower() not in ["y", "n"]:
                self.Doc.print("\n%s already exists in sync_data folder. Overwrite? (Y/N)"
                                                            % sync_basename)
                ow_answer = input("> ")
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
        # Removing trailing delimiter
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
        # Define constants used to isolating valid events.
        self.THRTL_THRESH = 45 # degrees ("throttle threshold")
        self.THRTL_T_THRESH = 2 # seconds ("throttle time threshold")

        # Need to repair any gaps in INCA samples. If pedal was actuated
        # when sampling cut out, and it was still actuated when the sampling
        # resumed, the abridge_data() algorithmm will treat that as a pedal lift
        # when it likely wasn't.
        self.knit_pedal_gaps()

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

            elif pedal_down: # pedal just lifted
                # Check if event is valid in case switch goes low before
                # throttle angle drops below its threshold.
                if counting:
                    self.Doc.print("\t(Pedal lifted before throttle dropped "
                                          "below %d deg.)" % self.THRTL_THRESH)
                    # similar to above code:
                    high_throttle_time[1] = self.sync_df.index[i-1] # prev. time
                    delta = high_throttle_time[1] - high_throttle_time[0]
                    self.Doc.print("\t\tThrottle >%d deg total t:\t%0.2fs" %
                                     (self.THRTL_THRESH, delta / SAMPLING_FREQ))
                    if (high_throttle_time[1] - high_throttle_time[0] >
                                          self.THRTL_T_THRESH * SAMPLING_FREQ):
                        keep = True
                    counting = False # reset indicator
                    high_throttle_time = [0, 0] # reset

                self.Doc.print("\tPedal lifted at time\t\t%0.2fs\n"
                                                        % (ti/SAMPLING_FREQ))
                if keep:
                    valid_event_times.append( [ped_buffer[0], ped_buffer[-1]] )
                pedal_down = False
                ped_buffer = [] # flush buffer
                keep = False # reset
            else:
                # pedal is not currently down, and wasn't just lifted.
                pass

        # One last check in case pedal-down event was ongoing when file ended.
        if counting:
            self.Doc.print("\t(File ended before throttle dropped "
                                  "below %d deg.)" % self.THRTL_THRESH)
            # similar to above code:
            high_throttle_time[1] = self.sync_df.index[i-1] # prev. time
            delta = high_throttle_time[1] - high_throttle_time[0]
            self.Doc.print("\t\tThrottle >%d deg total t:\t%0.2fs" %
                             (self.THRTL_THRESH, delta / SAMPLING_FREQ))
            if (high_throttle_time[1] - high_throttle_time[0] >
                                  self.THRTL_T_THRESH * SAMPLING_FREQ):
                keep = True
            counting = False # reset indicator
            high_throttle_time = [0, 0] # reset
        self.Doc.print("\tFile ended at time\t\t%0.2fs\n"
                                                % (ti/SAMPLING_FREQ))
        if keep:
            valid_event_times.append( [ped_buffer[0], ped_buffer[-1]] )

        if not valid_event_times:
            # If no times were stored, then alert user but continue with
            # program.
            self.Doc.print("\nNo valid pedal-down events found in run %s "
                                "(Criteria: throttle >%d deg for >%ds total)."
                    % (self.run_label, self.THRTL_THRESH, self.THRTL_T_THRESH))
            input("Press Enter to acknowledge and continue processing data without abridging.")
            self.abr_df = self.sync_df.copy(deep=True)

            self.meta_str += ("No valid pedal-down events found in run. "
                "(Criteria: throttle >%d deg for >%ds total). Data unabridged. | "
                                    % (self.THRTL_THRESH, self.THRTL_T_THRESH))
            return
        else:
            # Document in output file
            self.meta_str += ("Isolated events where throttle exceeded "
                "%d deg for >%ds. Removed extraneous surrounding events. | "
                                    % (self.THRTL_THRESH, self.THRTL_T_THRESH))

        self.Doc.print("Valid steady-state ranges:")
        for event_time in valid_event_times:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
               % (event_time[0] / SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))

        # Make sure if two >45 deg events (w/ pedal lift between) are closer
        # than 5s, don't cut into either one. Look at each pair of end/start
        # points, and if they're closer than 5s, merge those two.
        valid_event_times_c = [ valid_event_times[0] ]
        for n, pair in enumerate(valid_event_times[1:]):
            # self.Doc.print("\t%f - %f" % (pair[0], previous_pair[1]), True)
            earlier_pair = valid_event_times_c[-1]
            if pair[0] - earlier_pair[1] < (5 * SAMPLING_FREQ):
                # Replace the two pairs with a single combined pair
                del valid_event_times_c[-1]
                valid_event_times_c.append([ earlier_pair[0], pair[1] ])
            else:
                valid_event_times_c.append(pair)

        self.Doc.print("After any merges:")
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

            new_end_i = self.sync_df.index.get_loc(pair[1] + 1*SAMPLING_FREQ,
                                                            method="nearest")
            # If file ends less than 1s after event ends, this will return
            # the last time in the file. No tolerance specified for this reason.

            pair[0] = self.sync_df.index[new_start_i]
            pair[1] = self.sync_df.index[new_end_i]

        self.Doc.print("INCA times with 1-second buffers added:")
        for event_time in valid_event_times_c:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
               % (event_time[0] / SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))
        self.Doc.print("")

        # Split DataFrame into valid pieces; store in lists
        valid_events = []
        desired_start_t = 0
        for n, time_range in enumerate(valid_event_times_c):
            # create separate DataFrames for just this event
            valid_event = self.sync_df[time_range[0]:time_range[1]]

            # shift time values to maintain continuity.
            shift = time_range[0] - desired_start_t
            self.Doc.print("Shift (event %d): %.2f" % (n, shift / SAMPLING_FREQ)
                                                                         , True)

            self.shift_time_series(valid_event, offset_val=-shift)

            # Add events to lists
            valid_events.append(valid_event)

            # define next start time to be next time value after new vector's
            # end time.
            desired_start_t = time_range[1]-shift

        self.Doc.print("Shifted ranges:")
        for event in valid_events:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
              % (event.index[0]/SAMPLING_FREQ, event.index[-1] / SAMPLING_FREQ))

        # Now re-assemble the DataFrame with only valid events.
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        self.abr_df = pd.concat(valid_events)
        self.Doc.print("\nabr_df after abridgement:", True)
        self.Doc.print(self.abr_df.to_string(max_rows=10, max_cols=7,
                                                    show_dimensions=True), True)

        self.Doc.print("\nData time span: %.2f -> %.2f (%d data points)" %
          (self.abr_df.index[0]/SAMPLING_FREQ,
                 self.abr_df.index[-1]/SAMPLING_FREQ, len(self.abr_df.index)))

    def add_math_channels(self):
        self.math_df = pd.DataFrame(index=self.abr_df.index)
        # https://stackoverflow.com/questions/18176933/create-an-empty-data-frame-with-index-from-another-data-frame
        self.add_cvt_ratio()
        self.add_ss_avgs()

    def add_ss_avgs(self):
        win_size_avg = 51  # window size for speed rolling avg.
        win_size_slope = 301 # win size for rolling slope of speed rolling avg.

        gspd_cr = 2.5     # mph. Ground speed (min) criterion for determining if
                          # steady-state event is moving rather than stationary.
        gs_slope_cr = 0.25  # mph/s.
        # Ground-speed slope (max) criterion to est. steady-state. Abs value

        espd_cr = 2750    # rpm. Engine speed (min) criterion for determining if
                          # steady-state event is moving rather than stationary.
        es_slope_cr = 100  # rpm/s.
        # Engine-speed slope (max) criterion to est. steady-state. Abs value

        # Document in metadata string for output file:
        self.meta_str += ("Steady-state calc criteria: "
                          "gnd speed above %s mph, "
                          "gnd speed slope magnitude less than %s mph/s, "
                          "eng speed above %s rpm, "
                          "eng speed slope magnitude less than %s rpm/s | "
                            % (gspd_cr, gs_slope_cr, espd_cr, es_slope_cr))
        self.meta_str += ("Steady-state calc rolling window sizes: "
                                                "%d for avg, %d for slope | "
                                            % (win_size_avg, win_size_slope))

        # Create rolling average and rolling (regression) slope of rolling avg
        # for ground speed.
        self.math_df["gs_rolling_avg"] = self.abr_df.rolling(
                           window=win_size_avg, center=True)["gnd_speed"].mean()
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

        # Create and register a new tqdm instance with pandas. I don't know how this works.
        # You have to manually feed it the total iteration count first.
        tqdm.pandas(total=len(self.abr_df.index)-(win_size_avg-1)-(win_size_slope-1))
        # https://stackoverflow.com/questions/48935907/tqdm-not-showing-bar
        self.Doc.print("\nCalculating rolling regression on ground speed data...")
        self.math_df["gs_rolling_slope"] = self.math_df["gs_rolling_avg"].rolling(
                    window=win_size_slope, center=True).progress_apply(
                        lambda x: np.polyfit(x.index/SAMPLING_FREQ, x, 1)[0])
        # https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
        self.Doc.print("...done")

        # Create rolling average and rolling (regression) slope of rolling avg
        # for engine speed.
        self.math_df["es_rolling_avg"] = self.abr_df.rolling(
                          window=win_size_avg, center=True)["engine_spd"].mean()

        tqdm.pandas(total=len(self.abr_df.index)-(win_size_avg-1)-(win_size_slope-1))
        # https://stackoverflow.com/questions/48935907/tqdm-not-showing-bar
        self.Doc.print("Calculating rolling regression on engine speed data...")
        self.math_df["es_rolling_slope"] = self.math_df["es_rolling_avg"].rolling(
                    window=win_size_slope, center=True).progress_apply(
                        lambda x: np.polyfit(x.index/SAMPLING_FREQ, x, 1)[0])
        self.Doc.print("...done")

        # Apply speed and speed slope criteria to isolate steady-state events.
        ss_filter = (      (self.math_df["gs_rolling_avg"] > gspd_cr)
                         & (self.math_df["gs_rolling_slope"] < gs_slope_cr)
                         & (self.math_df["gs_rolling_slope"] > -gs_slope_cr)
                         & (self.math_df["es_rolling_avg"] > espd_cr)
                         & (self.math_df["es_rolling_slope"] < es_slope_cr)
                         & (self.math_df["es_rolling_slope"] > -es_slope_cr) )
        # gs_slope_cr and es_slope_cr are abs value so have to apply on high
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
        self.math_df["cvt_ratio_mskd"].mask(~ss_filter, inplace=True)

        # Calculate overall (aggregate) mean of each filtereed/masked channel
        # Prefill with NaN and assign mean to first element
        self.math_df["SS_gnd_spd_avg"] = np.nan
        self.math_df.at[0, "SS_gnd_spd_avg"] = np.mean(
                                                self.math_df["gs_rol_avg_mskd"])
        self.math_df["SS_eng_spd_avg"] = np.nan
        self.math_df.at[0, "SS_eng_spd_avg"] = np.mean(
                                                self.math_df["es_rol_avg_mskd"])
        self.math_df["SS_cvt_ratio_avg"] = np.nan
        self.math_df.at[0, "SS_cvt_ratio_avg"] = np.mean(
                                                   self.math_df["cvt_ratio_mskd"])
        # https://stackoverflow.com/questions/13842088/set-value-for-particular-cell-in-pandas-dataframe-using-index
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.at.html

        # Transcribe to main DF for export
        # Leave out average SS slopes.
        self.abr_df["SS_gnd_spd_avg_calc"] = self.math_df["SS_gnd_spd_avg"]
        self.abr_df["SS_eng_spd_avg_calc"] = self.math_df["SS_eng_spd_avg"]
        self.abr_df["SS_cvt_ratio_avg_calc"] = self.math_df["SS_cvt_ratio_avg"]
        CHANNEL_UNITS["SS_gnd_spd_avg_calc"] = CHANNEL_UNITS["gnd_speed"]
        CHANNEL_UNITS["SS_eng_spd_avg_calc"] = CHANNEL_UNITS["engine_spd"]
        CHANNEL_UNITS["SS_cvt_ratio_avg_calc"] = CHANNEL_UNITS["CVT_ratio_calc"]

        self.Doc.print("\nabr_df after adding steady-state data:", True)
        self.Doc.print(self.abr_df.to_string(max_rows=10, max_cols=7,
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

    def plot_data(self, overwrite=False, description=""):
        # This performs all the actions in the parent class's method
        super(SSRun, self).plot_data(overwrite, description)
        self.plot_ss_range()

    def plot_abridge_compare(self):
        ax1 = plt.subplot(211)
        # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html
        color = "tab:purple"
        ax1.plot(self.raw_inca_df.index, self.raw_inca_df["throttle"],
                                            color=color, label="Throttle (og)")
        plt.title("Run %s - Abridge Compare" % self.run_label, loc="left")
        ax1.set_ylim([-20, 80]) # Shift throttle trace up
        ax1.set_yticks([0, 20, 40, 60, 80])
        ax1.set_ylabel("Throttle (deg)", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        # plt.grid(True)

        ax2 = ax1.twinx() # second plot on same x axis
        # https://matplotlib.org/gallery/api/two_scales.html
        color = "tab:red"
        ax2.plot(self.raw_inca_df.index, self.raw_inca_df["pedal_sw"],
                                        color=color, label="Pedal Switch (og)")
        ax2.set_ylim([-.25, 8]) # scale down pedal switch
        ax2.set_yticks([0, 1])
        ax2.set_ylabel("Pedal Switch", color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        # plt.grid(True)
        plt.setp(ax1.get_xticklabels(), visible=False) # x labels only on bottom

        ax3 = plt.subplot(212, sharex=ax1, sharey=ax1)
        color = "tab:purple"
        # Convert DF indices from hundredths of a second to seconds
        sync_time_series = [round(ti/SAMPLING_FREQ, 2)
                                                for ti in self.abr_df.index]
        ax3.plot(sync_time_series, self.abr_df["throttle"],
                                        label="Throttle (synced)", color=color)
        plt.xlabel("Time (s)")
        # https://matplotlib.org/3.2.1/gallery/subplots_axes_and_figures/shared_axis_demo.html#sphx-glr-gallery-subplots-axes-and-figures-shared-axis-demo-py

        ax3.set_ylim([-20, 80]) # scale down pedal switch
        ax3.set_yticks([0, 20, 40, 60, 80])
        ax3.set_ylabel("Throttle (deg)", color=color)
        ax3.tick_params(axis="y", labelcolor=color)

        ax3_twin = ax3.twinx() # second plot on same x axis
        # https://matplotlib.org/gallery/api/two_scales.html
        color = "tab:red"
        ax3_twin.plot(sync_time_series, self.abr_df["pedal_sw"],
                                    color=color, label="Pedal Switch (synced)")
        ax3_twin.set_ylim([-.25, 8]) # scale down pedal switch
        ax3_twin.set_yticks([0, 1])
        ax3_twin.set_ylabel("Pedal Switch", color=color)
        ax3_twin.tick_params(axis="y", labelcolor=color)

        # plt.show() # can't use w/ WSL. Export instead.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        self.export_plot("abr")
        plt.clf()
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib

    def plot_ss_range(self):
        ax1 = plt.subplot(311)
        plt.plot(self.abr_df.index/SAMPLING_FREQ, self.abr_df["gnd_speed"],
                                                label="Ground Speed", color="k")
        plt.plot(self.abr_df.index/SAMPLING_FREQ, self.math_df["gs_rolling_avg"],
                                                label="Rolling Avg", color="c")
        plt.plot(self.abr_df.index/SAMPLING_FREQ, self.math_df["gs_rol_avg_mskd"],
                                                label="Steady-state", color="r")
        plt.title("Run %s - Steady-state Isolation (Abridged Data)"
                                                % self.run_label, loc="left")
        plt.ylabel("Speed (mph)")
        # plt.legend(loc="best")
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(312, sharex=ax1)
        # Convert DF indices from hundredths of a second to seconds
        plt.plot(self.abr_df.index/SAMPLING_FREQ,
                self.abr_df["engine_spd"], label="Engine Speed", color="yellowgreen")
        plt.plot(self.abr_df.index/SAMPLING_FREQ,
        self.math_df["es_rolling_avg"], label="Rolling Avg", color="tab:orange")
        plt.plot(self.abr_df.index/SAMPLING_FREQ,
        self.math_df["es_rol_avg_mskd"], label="Steady-state", color="tab:blue")

        plt.ylabel("Engine Speed (rpm)")
        # plt.legend(loc="best")
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3 = plt.subplot(313, sharex=ax1)
        color = "tab:purple"
        # Convert DF indices from hundredths of a second to seconds
        plt.plot(self.abr_df.index/SAMPLING_FREQ, self.abr_df["throttle"],
                                            label="Throttle", color=color)
        ax3.set_ylim([-20, 80]) # Shift throttle trace up
        ax3.set_yticks([0, 20, 40, 60, 80])
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Throttle (deg)", color=color)
        ax3.tick_params(axis="y", labelcolor=color)

        ax3_twin = ax3.twinx() # second plot on same x axis
        # https://matplotlib.org/gallery/api/two_scales.html
        color = "tab:red"
        ax3_twin.plot(self.abr_df.index/SAMPLING_FREQ, self.abr_df["pedal_sw"], color=color)
        ax3_twin.set_ylim([-.25, 8]) # scale down pedal switch
        ax3_twin.set_yticks([0, 1])
        ax3_twin.set_ylabel("Pedal Switch", color=color)
        ax3_twin.tick_params(axis="y", labelcolor=color)

        # plt.show() # can't use w/ WSL.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        self.export_plot("ss")
        plt.clf()
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib

    def get_run_type(self):
        return "SSRun"


class DownhillRun(SingleRun):
    """Represents a single run with downhill engine-braking operation."""

    def abridge_data(self):
        # Apply rolling avg filter to smooth data first.
        # Apply mask to data to find which sections ground speed is increasing
        # w/ pedal not pressed.

        # Need to repair any gaps in INCA samples. If pedal was actuated
        # when sampling cut out, and it was still actuated when the sampling
        # resumed, the abridge_data() algorithmm will treat that as a pedal lift
        # when it likely wasn't.
        self.knit_pedal_gaps()

        win_size_avg = 101  # window size for speed rolling avg.
        win_size_slope = 301 # win size for rolling slope of speed rolling avg.
        gspd_cr = 2.5     # mph. Ground speed (min) criterion for discerning
                          # valid downhill event.
        gs_slope_cr = +1.0  # mph/s.
        # Ground-speed slope min criterion to identify increasing speed downhill.
        gs_slope_t_cr = 3.0 # seconds. Continuous amount of time the slope
                            # criterion must be met to keep event.
        throttle_cr = 5.0 # deg. Anything below this interpreted as closed throt.

        # Create rolling average of ground speed (unabridged data).
        gs_rolling_avg = self.sync_df.rolling(
                           window=win_size_avg, center=True)["gnd_speed"].mean()

        tqdm.pandas(total=len(self.sync_df.index)-(win_size_avg-1)-(win_size_slope-1))
        # https://stackoverflow.com/questions/48935907/tqdm-not-showing-bar
        self.Doc.print("\nCalculating rolling regression on ground speed data...")
        gs_rolling_slope = gs_rolling_avg.rolling(
                            window=win_size_slope, center=True).progress_apply(
                        lambda x: np.polyfit(x.index/SAMPLING_FREQ, x, 1)[0])
        self.Doc.print("...done")

        # Apply pedal, throttle, speed, and speed slope criteria to isolate
        # downhill, pedal-up events.
        downhill_filter = (  (  (self.sync_df["pedal_sw"].isna())
                                 | (self.sync_df["throttle"] < throttle_cr)  )
                              & (gs_rolling_avg > gspd_cr)
                              & (gs_rolling_slope > gs_slope_cr)     )
        # NaNs in pedal channel treated as pedal up.

        # Mask off every data point not meeting the filter criteria
        gs_rol_avg_mskd = gs_rolling_avg.mask(~downhill_filter)
        gs_rol_slope_mskd = gs_rolling_slope.mask(~downhill_filter)
        # Convert to a list of indices.
        valid_times = gs_rol_avg_mskd[~gs_rol_avg_mskd.isna()]

        if len(valid_times) == 0:
            # If no times were stored, then alert user but continue with
            # program.
            self.Doc.print("\nNo valid downhill events found in run %s (Criteria: "
            "speed slope >%d mph/s, speed >%d mph, and throttle <%d deg)."
                % (self.run_label, gs_slope_cr, gspd_cr, throttle_cr))
            input("Press Enter to acknowledge and continue processing data without abridging.")
            # Take care of needed assignments that are typically down below.
            self.sync_df["gs_rolling_avg"] = gs_rolling_avg
            self.sync_df["gs_rolling_slope"] = gs_rolling_slope
            self.sync_df["downhill_filter"] = downhill_filter
            self.sync_df["trendlines"] = np.nan
            self.sync_df["slopes"] = np.nan
            self.abr_df = self.sync_df.copy(deep=True)

            self.meta_str += ("No valid downhill events found in run (Criteria: "
            "speed slope >%d mph/s, speed >%d mph, and throttle <%d deg). "
            "Data unabridged. | " % (gs_slope_cr, gspd_cr, throttle_cr))
            return

        # Identify separate continuous ranges.
        cont_ranges = [] # ranges w/ continuous data (no NaNs)
        current_range = [valid_times.index[0]]
        for i, time in enumerate(valid_times.index[1:]):
            prev_time = valid_times.index[i] # i is behind by one.
            if time - prev_time > 1:
                current_range.append(prev_time)
                cont_ranges.append(current_range)
                # Reset range
                current_range = [time]
        # Add last value to end of last range
        current_range.append(time)
        cont_ranges.append(current_range)

        self.Doc.print("\nDownhill ranges (before imposing length req.):", True)
        for event_range in cont_ranges:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
               % (event_range[0] / SAMPLING_FREQ, event_range[1] / SAMPLING_FREQ), True)

        valid_slopes = []
        for range in cont_ranges:
            if range[1]-range[0] > gs_slope_t_cr*SAMPLING_FREQ:
                # Must have > gs_slope_t_cr seconds to count.
                valid_slopes.append(range)
            else:
                # Adjust filter to eliminate these extraneous events.
                downhill_filter[range[0]:range[1]] = False
                pass
                # Remove from downhill_filter

        if not valid_slopes:
            # If no times were stored, then alert user but continue with
            # program.
            self.Doc.print("\nNo valid downhill events found in run %s (Criteria: "
            "speed slope >%d mph/s, speed >%d mph, and throttle <%d deg for >%ds)."
                % (self.run_label, gs_slope_cr, gspd_cr, throttle_cr, gs_slope_t_cr))
            input("Press Enter to acknowledge and continue processing data without abridging.")
            self.sync_df["gs_rolling_avg"] = gs_rolling_avg
            self.sync_df["gs_rolling_slope"] = gs_rolling_slope
            self.sync_df["downhill_filter"] = downhill_filter
            self.sync_df["trendlines"] = np.nan
            self.sync_df["slopes"] = np.nan
            self.abr_df = self.sync_df.copy(deep=True)

            self.meta_str += ("No valid downhill events found in run (Criteria: "
            "speed slope >%d mph/s, speed >%d mph, and throttle <%d deg for >%ds). "
            "Data unabridged. | " % (gs_slope_cr, gspd_cr, throttle_cr, gs_slope_t_cr))
            return
        else:
            # Document in output file
            self.meta_str += ("Isolated events where speed slope exceeded %d "
            "mph/s with speed >%d mph and throttle <%d deg for >%ds. "
            "Removed extraneous surrounding events. "
            "These same criteria were used for the downhill calcs. | "
                % (gs_slope_cr, gspd_cr, throttle_cr, gs_slope_t_cr))

        # Document window sizes in metadata string for output file:
        self.meta_str += ("Isolation and downhill calc rolling window sizes: "
                                                "%d for avg, %d for slope | "
                                            % (win_size_avg, win_size_slope))

        # Add buffers on each side - find closest point where ground speed
        # <1 mph. Add additional second beyond that.
        # Could do this with another filter. Then find closest val in filtered
        # list. Bias down for first range val, bias up for second.
        slow_filter = (gs_rolling_avg < 1) # mph

        # Mask off every data point not meeting the filter criterion
        gs_rol_avg_slow = gs_rolling_avg.mask(~slow_filter)
        # Convert to a list of indices.
        slow_times = gs_rol_avg_slow[~gs_rol_avg_slow.isna()]
        self.slow_times = slow_times

        # Now loop through event ranges and find "slow" times on either side
        # of range to expand and give context to the event.
        valid_ranges = copy.deepcopy(valid_slopes)
        # When copying list of lists, the contained lists are aliased w/
        # typical list-copy methods like [:] or .copy().
        # https://stackoverflow.com/questions/2612802/list-changes-unexpectedly-after-assignment-how-do-i-clone-or-copy-it-to-prevent
        for n, event_range in enumerate(valid_ranges):
            # Find closest neighbor value that is below 1 mph.
            try:
                new_start_i = slow_times.index[slow_times.index.get_loc(event_range[0], method="ffill")]
            except KeyError:
                # get_loc returns a KeyError if no value meeting our criteria
                # exists between start point and start/end of file
                new_start_i = 0

            try:
                new_end_i = slow_times.index[slow_times.index.get_loc(event_range[1], method="bfill")]
                # new_end_i = slow_times[slow_times.index.get_loc(event_range[1], method="bfill")]
            except KeyError:
                new_end_i = len(self.sync_df.index)-1

            event_range[0] = self.sync_df.index[new_start_i]
            event_range[1] = self.sync_df.index[new_end_i]

        # Create overall regression curve (not rolling) for each valid range.
        # Store for later plotting.
        trendlines = pd.Series(np.nan, index=self.sync_df.index)
        slopes = pd.Series(np.nan, index=self.sync_df.index)
        last_end_i = 0
        self.Doc.print("\nValid downhill ranges:")
        for n, event_range in enumerate(valid_slopes):
            # Input each event range's gs_rolling_avg values into np.polyval
            # Put them in new column. Everywhere else is NaN.
            coeff = np.polyfit(self.sync_df.index[event_range[0]:event_range[1]]/SAMPLING_FREQ,
                               gs_rolling_avg[event_range[0]:event_range[1]], 1)
            poly_fxn = np.poly1d(coeff)
            # https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs

            # Use broader range for trendline plotting so each appears extended
            # in plot used later.
            trendlines[valid_ranges[n][0]:valid_ranges[n][1]-1] = poly_fxn(
                self.sync_df.index[valid_ranges[n][0]:valid_ranges[n][1]-1]/SAMPLING_FREQ)
            # Subtracting one to end index to maintain a NaN between slopes,
            # else plot would draw line joining them.

            # Store slope value itself for later retrieval. This time in the precise window
            slopes[event_range[0]:event_range[1]] = coeff[0]

            self.Doc.print("\t%0.2f\t->\t%0.2f\t|    Slope: %+0.2f mph/s"
              % (event_range[0] / SAMPLING_FREQ, event_range[1] / SAMPLING_FREQ,
                  coeff[0]))

        self.Doc.print("After widening range to capture complete event(s):")
        for event_time in valid_ranges:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
                % (event_time[0]/SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))

        # Make sure if two valid events are closer
        # than 5s, don't cut into either one. Look at each pair of end/start
        # points, and if they're closer than 5s, merge those two.
        # This also handles cases where two or more ranges end up being
        # identical after widening window to closest low-speed areas.
        valid_ranges_c = [ valid_ranges[0] ]
        for n, pair in enumerate(valid_ranges[1:]):
            earlier_pair = valid_ranges_c[-1]
            if pair[0] - earlier_pair[1] < (5 * SAMPLING_FREQ):
                # Replace the two pairs with a single combined pair
                del valid_ranges_c[-1]
                valid_ranges_c.append([ earlier_pair[0], pair[1] ])
            else:
                valid_ranges_c.append(pair)

        self.Doc.print("After any merges:")
        for event_time in valid_ranges_c:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
                % (event_time[0]/SAMPLING_FREQ, event_time[1] / SAMPLING_FREQ))
        self.Doc.print("")

        # Split DataFrame into valid pieces; store in lists
        valid_events = []
        # Cut up and re-join rolling avg channels too for later use.
        # Piggyback on sync_df for now.
        self.sync_df["gs_rolling_avg"] = gs_rolling_avg
        self.sync_df["gs_rolling_slope"] = gs_rolling_slope
        self.sync_df["downhill_filter"] = downhill_filter
        self.sync_df["trendlines"] = trendlines
        self.sync_df["slopes"] = slopes
        desired_start_t = 0
        for n, time_range in enumerate(valid_ranges_c):
            # create separate DataFrames for just this event
            valid_event = self.sync_df[time_range[0]:time_range[1]]

            # shift time values to maintain continuity.
            shift = time_range[0] - desired_start_t
            self.Doc.print("Shift (event %d): %.2f" % (n, shift / SAMPLING_FREQ)
                                                                         , True)

            self.shift_time_series(valid_event, offset_val=-shift)

            # Add events to lists
            valid_events.append(valid_event)

            # define next start time to be next time value after new vector's
            # end time.
            desired_start_t = time_range[1]-shift

        self.Doc.print("Shifted ranges:")
        for event in valid_events:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
              % (event.index[0]/SAMPLING_FREQ, event.index[-1] / SAMPLING_FREQ))

        # Now re-assemble the DataFrame with only valid events.
        # Carries over rolling and filter channels added to sync_df above.
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        self.abr_df = pd.concat(valid_events)

        self.Doc.print("\nabr_df after abridgement:", True)
        self.Doc.print(self.abr_df.to_string(max_rows=10, max_cols=7,
                                                    show_dimensions=True), True)

        self.Doc.print("\nData time span: %.2f -> %.2f (%d data points)" %
          (self.abr_df.index[0]/SAMPLING_FREQ,
                 self.abr_df.index[-1]/SAMPLING_FREQ, len(self.abr_df.index)))

    def add_math_channels(self):
        self.math_df = pd.DataFrame(index=self.abr_df.index)
        # https://stackoverflow.com/questions/18176933/create-an-empty-data-frame-with-index-from-another-data-frame

        # Move channels calculated during abridge to the math_df.
        self.math_df["gs_rolling_avg"] = self.abr_df["gs_rolling_avg"]
        self.math_df["gs_rolling_slope"] = self.abr_df["gs_rolling_slope"]
        self.math_df["downhill_filter"] = self.abr_df["downhill_filter"]
        self.math_df["trendlines"] = self.abr_df["trendlines"]
        self.math_df["slopes"] = self.abr_df["slopes"]
        del self.abr_df["gs_rolling_avg"]
        del self.abr_df["gs_rolling_slope"]
        del self.abr_df["downhill_filter"]
        del self.abr_df["trendlines"]
        del self.abr_df["slopes"]

        self.add_cvt_ratio()
        self.add_downhill_avgs()

    def add_downhill_avgs(self):

        self.math_df["gs_rol_avg_mskd"] = self.math_df["gs_rolling_avg"].mask(~self.math_df["downhill_filter"])
        self.math_df["gs_rol_slope_mskd"] = self.math_df["gs_rolling_slope"].mask(~self.math_df["downhill_filter"])
        self.math_df["cvt_ratio_mskd"].mask(~self.math_df["downhill_filter"], inplace=True)
            # CVT values of 0 or above 5 already masked.

        self.Doc.print("\nTotal data points that fail downhill criteria: %d"
                                % sum(~self.math_df["downhill_filter"]), True)
        self.Doc.print("Total data points that meet downhill criteria: %d"
                                 % sum(self.math_df["downhill_filter"]), True)

        # Create separate channels for engine-on and engine-off segments
        engine_on = (self.abr_df["engine_spd"] > 0)
        engine_off = (self.abr_df["engine_spd"] == 0)

        self.math_df["gs_rol_avg_mskd_eng_on"] = self.math_df["gs_rol_avg_mskd"].mask(
                                                                    engine_off)
        self.math_df["gs_rol_avg_mskd_eng_off"] = self.math_df["gs_rol_avg_mskd"].mask(
                                                                    engine_on)

        self.Doc.print("\nTotal engine-on downhill data points: %d"
                        % self.math_df["gs_rol_avg_mskd_eng_on"].count(), True)
        self.Doc.print("Total engine-off downhill data points: %d"
                        % self.math_df["gs_rol_avg_mskd_eng_off"].count(), True)

        # Calculate aggregate slope (accel is positive / decel is negative)
        self.math_df["accel_avg_calc_eng_on"] = np.nan
        self.math_df.at[0, "accel_avg_calc_eng_on"] = np.mean(
                                        self.math_df["slopes"].mask(engine_off))

        self.math_df["accel_avg_calc_eng_off"] = np.nan
        self.math_df.at[0, "accel_avg_calc_eng_off"] = np.mean(
                                        self.math_df["slopes"].mask(engine_on))

        self.Doc.print("\nEngine-on downhill accel: %.2f"
                                % self.math_df.at[0, "accel_avg_calc_eng_on"])
        self.Doc.print("Engine-off downhill accel: %.2f"
                                % self.math_df.at[0, "accel_avg_calc_eng_off"])

        self.abr_df["accel_avg_calc_eng_on"] = self.math_df["accel_avg_calc_eng_on"]
        self.abr_df["accel_avg_calc_eng_off"] = self.math_df["accel_avg_calc_eng_off"]
        CHANNEL_UNITS["accel_avg_calc_eng_on"] = CHANNEL_UNITS["gnd_speed"] + "/s"
        CHANNEL_UNITS["accel_avg_calc_eng_off"] = CHANNEL_UNITS["accel_avg_calc_eng_on"]

        self.Doc.print("\nabr_df after adding steady-state data:", True)
        self.Doc.print(self.abr_df.to_string(max_rows=10, max_cols=7,
                                                    show_dimensions=True), True)

    def plot_data(self, overwrite=False, description=""):
        # This performs all the actions in the parent class's method
        super(DownhillRun, self).plot_data(overwrite, description)
        self.plot_downhill_range()

    def plot_abridge_compare(self):
        ax1 = plt.subplot(211)
        # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html
        plt.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["gnd_speed"], color="k")
        plt.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["gs_rolling_avg"], color="c")
        plt.plot(self.sync_df.index/SAMPLING_FREQ,
            self.sync_df["gs_rolling_avg"].mask(~self.sync_df["downhill_filter"]), color="r")

        plt.title("Run %s - Abridge Compare" % self.run_label, loc="left")
        ax1.set_ylabel("Speed (mph)")

        plt.setp(ax1.get_xticklabels(), visible=False) # x labels only on bottom

        ax2 = plt.subplot(212, sharex=ax1)

        plt.plot(self.abr_df.index/SAMPLING_FREQ, self.abr_df["gnd_speed"], color="k")
        plt.plot(self.math_df.index/SAMPLING_FREQ, self.math_df["gs_rolling_avg"], color="c")
        plt.plot(self.math_df.index/SAMPLING_FREQ, self.math_df["gs_rol_avg_mskd"], color="r")

        ax2.set_ylabel("Speed (mph)")

        # plt.show() # can't use w/ WSL. Export instead.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        self.export_plot("abr")
        plt.clf()
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib

    def plot_downhill_range(self):
        """Plot with downhill segments identified."""
        ax1 = plt.subplot(311)
        color = "k"
        # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        ax1.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["gnd_speed"], color=color)
        color = "c"
        ax1.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["gs_rolling_avg"], color=color)
        color = "r"
        ax1.plot(self.sync_df.index/SAMPLING_FREQ,
            self.sync_df["gs_rolling_avg"].mask(~self.sync_df["downhill_filter"]), color=color)
        ax1.set_ylabel("Speed (mph)")
        plt.setp(ax1.get_xticklabels(), visible=False) # x labels only on bottom
        plt.title("Run %s - Downhill Isolation (Unabridged Data)"
                                                % self.run_label, loc="left")

        ax2 = plt.subplot(312, sharex=ax1)
        ax2.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["engine_spd"])
        ax2.set_ylabel("Engine Speed (rpm)")
        plt.setp(ax2.get_xticklabels(), visible=False) # x labels only on bottom

        ax3 = plt.subplot(313, sharex=ax1)
        color = "tab:purple"
        ax3.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["throttle"], color=color)
        ax3.set_ylim([-20, 80]) # Shift throttle trace up
        ax3.set_yticks([0, 20, 40, 60, 80])

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Throttle (deg)", color=color)
        ax3.tick_params(axis="y", labelcolor=color)

        ax3_twin = ax3.twinx() # second plot on same x axis
        # https://matplotlib.org/gallery/api/two_scales.html
        color = "tab:red"
        ax3_twin.plot(self.sync_df.index/SAMPLING_FREQ, self.sync_df["pedal_sw"], color=color)
        ax3_twin.set_ylim([-.25, 8]) # scale down pedal switch
        ax3_twin.set_yticks([0, 1])
        ax3_twin.set_ylabel("Pedal Switch", color=color)
        ax3_twin.tick_params(axis="y", labelcolor=color)

        self.export_plot("downhill")
        plt.clf()

    def get_run_type(self):
        return "DownhillRun"


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
        if not os.path.exists(PLOT_DIR):
            # Create folder for output plots if it doesn't exist already.
            os.mkdir(PLOT_DIR)
        AllRuns.plot_runs(args.over, args.desc)
    elif args.plot:
        print("\nFailed to import matplotlib. Cannot plot data.")

    if not os.path.exists(SYNC_DIR):
        # Create folder for output data if it doesn't exist already.
        os.mkdir(SYNC_DIR)

    AllRuns.export_runs(args.over, args.desc)


if __name__ == "__main__":
    main_prog()
