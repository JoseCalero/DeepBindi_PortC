CC      := gcc
CFLAGS  := -std=c99 -O2 -Wall -Wextra -pedantic
LDFLAGS := -lm

# ── Logging ───────────────────────────────────────────────────────────────────
# By default the binary is silent (suitable for embedded targets with no UART).
# Add -DDEEPBINDI_ENABLE_LOGGING to see output on stdout/stderr.
# Use `make debug` for a one-shot debug build without changing these defaults.
# CFLAGS += -DDEEPBINDI_ENABLE_LOGGING

TARGET  := deepbindi_c_demo
SOURCES := main.c nn_runtime.c cnn_models_c.c
OBJECTS := $(SOURCES:.c=.o)

.PHONY: all debug clean run

all: $(TARGET)

# Debug build: same binary name, logging enabled. Cleans first so flag change takes effect.
debug: clean
debug: CFLAGS += -DDEEPBINDI_ENABLE_LOGGING
debug: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: debug
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
