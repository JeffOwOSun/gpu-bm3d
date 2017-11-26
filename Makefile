APPNAME=bm3d
OBJS=bm3d.o denoise.o
LIBS=X11 jpeg png z cufft cudart glut GL
LDLIBS=$(addprefix -l,$(LIBS))
CXX=g++ -w -m64 -std=c++11
CXXFLAGS = -O3 -Wall -Wno-unknown-pragmas
CVFLAGS=$(shell pkg-config --cflags --libs opencv)

LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
INCLUDE=/usr/local/depot/cuda-8.0/include
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35
NVCC=nvcc

default: $(APPNAME)

opencv: opencv_test

$(APPNAME): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(LDLIBS) $(OBJS) -I /opt/X11/include -L /opt/X11/lib -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(addprefix -I,$(INCLUDE)) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(addprefix -I,$(INCLUDE)) -c $< -o $@

clean:
	rm *.o
