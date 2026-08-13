CC ?= cc

CPPFLAGS := -I.
BASE_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Wconversion \
	-Wsign-conversion -Werror
SANITIZER_CFLAGS := -O1 -g -fno-omit-frame-pointer
BUILD_DIR := build
LIB_SOURCE := fixed_reference.c
TEST_SOURCE := test_fixed_reference.c
FUZZ_SOURCE := fuzz_fixed_reference.c

.PHONY: all strict test sanitize ubsan asan clean

all: test

strict: $(BUILD_DIR)/test_fixed_reference $(BUILD_DIR)/fuzz_fixed_reference

test: strict
	$(BUILD_DIR)/test_fixed_reference
	$(BUILD_DIR)/fuzz_fixed_reference

sanitize: ubsan asan

ubsan: $(BUILD_DIR)/test_fixed_reference_ubsan $(BUILD_DIR)/fuzz_fixed_reference_ubsan
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 $(BUILD_DIR)/test_fixed_reference_ubsan
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 $(BUILD_DIR)/fuzz_fixed_reference_ubsan

asan: $(BUILD_DIR)/test_fixed_reference_asan $(BUILD_DIR)/fuzz_fixed_reference_asan
	ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 $(BUILD_DIR)/test_fixed_reference_asan
	ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 $(BUILD_DIR)/fuzz_fixed_reference_asan

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_fixed_reference: $(LIB_SOURCE) fixed_reference.h $(TEST_SOURCE) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(BASE_CFLAGS) -O2 $(LIB_SOURCE) $(TEST_SOURCE) -o $@

$(BUILD_DIR)/fuzz_fixed_reference: $(LIB_SOURCE) fixed_reference.h $(FUZZ_SOURCE) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(BASE_CFLAGS) -O2 $(LIB_SOURCE) $(FUZZ_SOURCE) -o $@

$(BUILD_DIR)/test_fixed_reference_ubsan: $(LIB_SOURCE) fixed_reference.h $(TEST_SOURCE) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(BASE_CFLAGS) $(SANITIZER_CFLAGS) \
		-fsanitize=undefined $(LIB_SOURCE) $(TEST_SOURCE) -o $@

$(BUILD_DIR)/fuzz_fixed_reference_ubsan: $(LIB_SOURCE) fixed_reference.h $(FUZZ_SOURCE) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(BASE_CFLAGS) $(SANITIZER_CFLAGS) \
		-fsanitize=undefined $(LIB_SOURCE) $(FUZZ_SOURCE) -o $@

$(BUILD_DIR)/test_fixed_reference_asan: $(LIB_SOURCE) fixed_reference.h $(TEST_SOURCE) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(BASE_CFLAGS) $(SANITIZER_CFLAGS) \
		-fsanitize=address $(LIB_SOURCE) $(TEST_SOURCE) -o $@

$(BUILD_DIR)/fuzz_fixed_reference_asan: $(LIB_SOURCE) fixed_reference.h $(FUZZ_SOURCE) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(BASE_CFLAGS) $(SANITIZER_CFLAGS) \
		-fsanitize=address $(LIB_SOURCE) $(FUZZ_SOURCE) -o $@

clean:
	rm -rf $(BUILD_DIR)
