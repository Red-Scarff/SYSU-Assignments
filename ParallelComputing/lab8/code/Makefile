CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall
TARGET = shortest_path
SOURCE = shortest_path.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	./$(TARGET) updated_flower.csv test_queries.txt 1 0
	./$(TARGET) updated_flower.csv test_queries.txt 2 0
	./$(TARGET) updated_flower.csv test_queries.txt 4 0
	./$(TARGET) updated_flower.csv test_queries.txt 8 0
	./$(TARGET) updated_flower.csv test_queries.txt 16 0

	./$(TARGET) updated_mouse.csv test_queries.txt 1 0
	./$(TARGET) updated_mouse.csv test_queries.txt 2 0
	./$(TARGET) updated_mouse.csv test_queries.txt 4 0
	./$(TARGET) updated_mouse.csv test_queries.txt 8 0
	./$(TARGET) updated_mouse.csv test_queries.txt 16 0


.PHONY: all clean test