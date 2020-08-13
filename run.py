import os
import wpfix

class FilenameError(Exception):
    pass

eDAQ_data_path_win = input("Enter file name of eDAQ data ready for sync "
                                        "(include 'r' before the string):\n>")

# coerce Windows path into Linux convention
eDAQ_data_path = wpfix.wpfix("%s" % eDAQ_data_path_win)

# make sure the path exists on the system before proceeding.
if not os.path.exists(eDAQ_data_path):
    raise FilenameError("Bad path input: %s" % eDAQ_data_path)
else:
    print("%s" % eDAQ_data_path)
