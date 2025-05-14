CC = gcc
CFLAGS = -fPIC -Iinclude -pthread -O3
LDFLAGS = -shared

SRC = src/parallel_for.c
TARGET = libparallel_for.so

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)