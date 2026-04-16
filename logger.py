#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 2026

@author: jds40
"""

import os
from datetime import datetime

class Logger:
    def __init__(self, directory: str = "logCustom.txt"):
        self.directory = directory
    
    # Append time and desired details to existing log
    def add(self, string: str):
        cTime = str(datetime.now()).split(".")[0].ljust(21)
        if(os.path.exists(self.directory)==False):
            with open(self.directory, "w") as f:
                f.write(cTime + string)
            return
        with open(self.directory, "r") as f:
            text: str = f.read()
        if(len(text.split("\n")[-1])>0):
            text += "\n"
        text += cTime + string
        with open(self.directory, "w") as f:
            f.write(text)
    
    # Clear current log file to make way for more
    def clear(self):
        if(os.path.exists(self.directory)):
            os.remove(self.directory)