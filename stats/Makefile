# ML/DL Metrics Library Makefile
# C++17 required for structured bindings

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2
DEBUG_FLAGS := -g -DDEBUG

# Directories
SRC_DIR := .
BUILD_DIR := build
INSTALL_DIR := /usr/local/include/metrics

# Headers (header-only library)
HEADERS := IMetric.hpp \
           ClassificationMetrics.hpp \
           RegressionMetrics.hpp \
           MetricFactory.hpp \
           MetricEvaluator.hpp

# Target executable
TARGET := metrics_demo
TARGET_PATH := $(BUILD_DIR)/$(TARGET)

# Source files
SRCS := main.cpp
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Default target
.PHONY: all
all: $(TARGET_PATH)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link executable
$(TARGET_PATH): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Debug build
.PHONY: debug
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET_PATH)

# Run the demo
.PHONY: run
run: $(TARGET_PATH)
	./$(TARGET_PATH)

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

# Install headers
.PHONY: install
install:
	mkdir -p $(INSTALL_DIR)
	cp $(HEADERS) $(INSTALL_DIR)/

# Uninstall headers
.PHONY: uninstall
uninstall:
	rm -rf $(INSTALL_DIR)

# Help
.PHONY: help
help:
	@echo "ML/DL Metrics Library"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build the demo executable"
	@echo "  make debug    - Build with debug symbols"
	@echo "  make run      - Build and run the demo"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make install  - Install headers to $(INSTALL_DIR)"
	@echo "  make help     - Show this help message"
