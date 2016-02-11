QT       += core
QT       -= gui

TARGET    = QtCuda
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app
# C++ source code
SOURCES += main.cpp
# project build directories
DESTDIR     = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/Obj
# C++ flags
QMAKE_CXXFLAGS_RELEASE =-O3
# Cuda sources
CUDA_SOURCES += cuda_interface.cu

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
