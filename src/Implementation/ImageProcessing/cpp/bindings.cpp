#include <pybind11/pybind11.h>
#include "ImageProcessing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(imgproc, m) {
    m.def("process_image", &ProcessImage, "Process an image using the C++ function");
}
