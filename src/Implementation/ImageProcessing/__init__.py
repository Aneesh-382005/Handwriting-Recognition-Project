from glob import glob
import multiprocessing
import functools
import sys
import os

pn_CPP_FILES = os.path.join("src", "Implementation", "ImageProcessing", "cpp")
fn_CPP_OUT = os.path.join(".", "bin", "imgproc.out")

class Image():
    def __init__(self, src, out):
        self.src = src
        self.out = os.path.join(out, os.path.basename(src))

def compile():
    cpp_files = " ".join(sorted(glob(os.path.join(pn_CPP_FILES, "*.cpp"))))
    cmd = f"g++ {cpp_files} -o {fn_CPP_OUT}"

    if sys.platform == "linux":
        cmd += " -std=c++17 -lstdc++fs `pkg-config --cflags --libs opencv4`"

    print(f"Compiling with command: {cmd}")
    
    if os.system(cmd) != 0:
        sys.exit("Error: Compilation failed.")

def execute(images, out):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(functools.partial(__execute__, out=out), images)
    pool.close()
    pool.join()

def __execute__(image, out):
    im = Image(image, out)
    cmd = f"{fn_CPP_OUT} {im.src} {im.out}"

    print(f"Running command: {cmd}")

    if os.system(cmd) != 0:
        sys.exit(f"Image process error: {im.src}")

