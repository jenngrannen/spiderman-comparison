#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = "docker build -t jenn-density ."
    code = os.system(cmd)
