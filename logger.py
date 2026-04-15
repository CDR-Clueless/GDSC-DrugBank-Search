#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 2026

@author: jds40
"""

import os

class Logger:
    def __init__(self, directory: str = "logCustom.txt"):
        self.directory = directory
    
    def add(self, string: str):
        if(os.path.exists(self.directory)==False):
            with open(self.directory, "w") as f:
                f.write(string)
            return
        with open(self.directory, "r") as f:
            text: str = f.read()
        if(len(text.split("\n")[-1])>0):
            text += "\n"
        text += string
        with open(self.directory, "w") as f:
            f.write(text)