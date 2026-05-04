# Makefile for the static (no-malloc) CNN inference port
#
# Targets:
#   make         – build deepbindi_static_demo  (silent, embedded-ready)
#   make debug   – build with console output enabled
#   make run     – build (debug) and run
#   make clean   – remove build artefacts
#   make verify  – run both static and dynamic ports, diff the checksums
#
# The static binary has no external dependencies beyond libc and libm.
# Total BSS+data ≈ 10 MB (dominated by g_weight_pool and g_act_arena).

CC      := gcc
CFLAGS  := -std=c99 -O2 -Wall -Wextra -pedantic
LDFLAGS := -lm

# ── Logging ───────────────────────────────────────────────────────────────────
# By default the binary is fully silent (no printf/fprintf anywhere).
# Add -DDEEPBINDI_ENABLE_LOGGING to enable stdout/stderr output.
# Use `make debug` for a one-shot debug build without changing these defaults.
# CFLAGS += -DDEEPBINDI_ENABLE_LOGGING

TARGET  := deepbindi_static_demo
SOURCES := main.c arena.c nn_runtime.c cnn_models_c.c
OBJECTS := $(SOURCES:.c=.o)

.PHONY: all debug run clean verify

all: $(TARGET)

# Debug build: same binary name, logging enabled. Cleans first so flag change takes effect.
debug: clean
debug: CFLAGS += -DDEEPBINDI_ENABLE_LOGGING
debug: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: debug
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

# Compare static output with the dynamic port to verify numerical equivalence.
verify: debug
	@./$(TARGET) > /tmp/static_out.txt
	@$(MAKE) -C ../c_port debug run > /tmp/dyn_out.txt 2>&1
	@echo "=== Diff (empty = identical checksums) ===" && \
	  grep "checksum" /tmp/static_out.txt > /tmp/s.tmp && \
	  grep "checksum" /tmp/dyn_out.txt   > /tmp/d.tmp && \
	  diff /tmp/s.tmp /tmp/d.tmp && echo "OK: checksums match"
