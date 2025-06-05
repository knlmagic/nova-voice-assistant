#!/usr/bin/env python3
import site, glob, sys, os, shutil

for p in site.getsitepackages():
    for so in glob.glob(os.path.join(p, "whispercpp*")):
        print("Removing:", so)
        try:
            if os.path.isdir(so):
                shutil.rmtree(so)
            else:
                os.remove(so)
        except Exception as e:
            print(f"  Failed to remove {so}: {e}") 