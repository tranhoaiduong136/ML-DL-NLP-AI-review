CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
LDFLAGS = 

# Common objects
COMMON_OBJS = core/Neuron.o \
              activations/Step.o \
              activations/Sigmoid.o \
              layers/Layer.o \
              loss/MSE.o \
              models/Perceptron.o \
              models/MLP.o

# Targets
all: neural_net mlp_test

neural_net: main.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o neural_net main.o $(COMMON_OBJS) $(LDFLAGS)

mlp_test: main_MLP.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o mlp_test main_MLP.o $(COMMON_OBJS) $(LDFLAGS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(COMMON_OBJS) main.o main_MLP.o neural_net mlp_test

run: neural_net
	./neural_net

run_mlp: mlp_test
	./mlp_test

.PHONY: all clean run run_mlp
