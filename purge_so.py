#!/usr/bin/env python3
import site, glob, os, shutil, sys

for p in site.getsitepackages():
    for so in glob.glob(os.path.join(p, "whispercpp", "*.so")):
        print("Removing", so)
        os.remove(so) 