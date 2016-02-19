QT       += core
QT       -= gui

TARGET    = QtCuda
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app
# C++ source code
SOURCES += main.cpp \
	   loadSaveImage.cpp \
	   reference_calc.cpp \
	   compare.cpp
HEADERS += reference_calc.h \
	   loadSaveImage.h \
	   compare.h \
	   timer.h \
	   utils.h

# project build directories
DESTDIR     = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/Obj
# C++ flags
QMAKE_CXXFLAGS_RELEASE =-O3
# Cuda sources
CUDA_SOURCES += student_func.cu \
		HW3.cu

# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
# check system architecture
contains(QMAKE_HOST.arch, x86_64) {
    INCLUDEPATH  += $$CUDA_DIR/lib64
    QMAKE_LIBDIR += $$CUDA_DIR/lib64
    SYSTEM_TYPE = 64

} else {
    INCLUDEPATH  += $$CUDA_DIR/lib
    QMAKE_LIBDIR += $$CUDA_DIR/lib
    SYSTEM_TYPE = 32
}

INCLUDEPATH += $$quote(/usr/local/include/opencv/) \
	    $$quote(/usr/local/include)

#LIBS += -L$$quote(D:/opencv/build/x86/vc11/lib) -lopencv_calib3d2410 -lopencv_contrib2410 -lopencv_core2410 -lopencv_features2d2410 -lopencv_flann2410 -lopencv_highgui2410 -lopencv_imgproc2410 -lopencv_legacy2410 -lopencv_ml2410 -lopencv_nonfree2410  -lopencv_objdetect2410 -lopencv_ocl2410 -lopencv_photo2410 -lopencv_stitching2410 -lopencv_superres2410 -lopencv_ts2410 -lopencv_video2410 -lopencv_videostab2410

LIBS += -L$$quote(/usr/local/lib) -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core -lopencv_hal

# libs used in your code
LIBS += -lcudart -lcuda
# GPU architecture: https://en.wikipedia.org/wiki/CUDA
CUDA_ARCH     = sm_21                # Adjust with your compute capability
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math
# Prepare the extra compiler configuration (taken from the nvidia forum)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -m$$SYSTEM_TYPE -O3 -D_DEBUG -arch=$$CUDA_ARCH -c $$NVCCFLAGS\
		      $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
		      2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
    cuda_d.dependency_type = TYPE_C
#    cuda_d.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    # nvcc error printout format ever so slightly different from gcc
    # http://forums.nvidia.com/index.php?showtopic=171651
    cuda.commands = $$CUDA_DIR/bin/nvcc -m$$SYSTEM_TYPE -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS\
		    $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
		    2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
    cuda.dependency_type = TYPE_C
#    cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
    # Tell Qt that we want add more stuff to the Makefile
    QMAKE_EXTRA_COMPILERS += cuda
}
