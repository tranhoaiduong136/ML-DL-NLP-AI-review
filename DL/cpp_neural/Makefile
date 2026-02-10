CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
LDFLAGS = 

# Source files
SRCS = main.cpp \
       core/Neuron.cpp \
       models/Perceptron.cpp \
       activations/Step.cpp

# Object files (in build directory or alongside source)
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = neural_net

# Main target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: all
	./$(TARGET)

.PHONY: all clean run
