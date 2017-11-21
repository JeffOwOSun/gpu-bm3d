APPNAME=bm3d
OBJS=denoise.o
LIBS=X11 jpeg png z
LDLIBS=$(addprefix -l,$(LIBS))
CXX=g++ -w -m64 -std=c++11
CXXFLAGS = -I. -O3 -Wall -Wno-unknown-pragmas
CVFLAGS=$(shell pkg-config --cflags --libs opencv)


default: $(APPNAME)

opencv: opencv_test

$(APPNAME): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDLIBS) $(OBJS) -I /opt/X11/include -L /opt/X11/lib -o $@

$(OBJS): denoise.cpp
	$(CXX) $(CXXFLAGS) $(LDLIBS) -I /opt/X11/include -L /opt/X11/lib -c $< -o $@

opencv_test: opencv_test.cpp
	$(CXX) $(CVFLAGS) -o $@ opencv_test.cpp

clean:
	rm *.o
