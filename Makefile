APP_NAME=opencv_test
OBJS=opencv_test.cpp
CXX=g++ -w -m64 -std=c++11
CXXFLAGS=$(shell pkg-config --cflags --libs opencv)


opencv: $(APP_NAME)

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)


