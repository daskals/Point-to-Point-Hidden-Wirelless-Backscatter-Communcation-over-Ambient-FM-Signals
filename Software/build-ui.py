#!/usr/bin/env python3

"""
Utility script to compile the ui and qrc file. The script will
attempt to compile themfile using the following tools:
    - pyuic5
    - pyrcc5
Delete the compiled files that you don't want to use
manually after running this script.
"""

import os

def main():
    """
    Compile UI and resource files for PyQt5
    """
    os.system("pyuic5  -o messenger.py resx/messenger.ui")
    os.system("pyrcc5 resx/resx.qrc -o resx_rc.py")

if __name__ == '__main__':
    main()
