# Target settings
TARGET := linux
TARGET_ARCH := x86_64

# Toolchain
TOOLCHAIN := gcc
CXX_TOOL := g++
CC_TOOL := gcc
AR_TOOL := ar

# Path to your tflite-micro repo
TFLM_ROOT := /mnt/c/code_space/EmbeddedAI/tflite-micro

THIRD_PARTY := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads

# Includes
INCLUDES := -I$(TFLM_ROOT) \
  -I. \
  -I$(THIRD_PARTY)/flatbuffers/include \
  -I$(THIRD_PARTY)/gemmlowp

# Warning flags
CC_WARNINGS := \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wunused-variable \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wmissing-field-initializers \
  -Wstrict-aliasing \
  -Wno-unused-parameter

# Common flags
COMMON_FLAGS := \
  -std=c++17 \
  -fno-unwind-tables \
  -fno-asynchronous-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  $(CC_WARNINGS) \
  -DTF_LITE_USE_CTIME

CXXFLAGS := \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  -Wnon-virtual-dtor \
  $(COMMON_FLAGS) \
  -DMYNAMESTR=\"shane\ mccammon\"   # <--- your name is injected here

# Linker flags
LDFLAGS := \
  $(TFLM_ROOT)/gen/$(TARGET)_$(TARGET_ARCH)_default_gcc/lib/libtensorflow-microlite.a \
  -lm \
  -Wl,--fatal-warnings -Wl,--gc-sections

# Build rules
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

all: hello_world

all_objs := hello_world.o hello_world_float_model_data.o
# if using int8: add hello_world_int8_model_data.o

hello_world: $(all_objs)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o hello_world $(all_objs) $(LDFLAGS)

clean:
	$(RM) *.o *.d *.map hello_world