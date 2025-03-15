import sys
import os

from dotenv import load_dotenv

load_dotenv()


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "build/Release")))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "build/Release")))

opencv_dll_path = os.getenv('OPENCV_DLL_PATH')

os.add_dll_directory(opencv_dll_path)

import imgproc

imgproc.process_image("text.jpg", "outputText")
