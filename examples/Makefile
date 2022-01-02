
BUILD_DIR = ../build/bin
TEST_DIR = ../test

%: %.mlir
	$(BUILD_DIR)/hcl-opt -opt $(FLAG) $< > $<.opt
	$(BUILD_DIR)/hcl-translate -emit-hlscpp $<.opt > $@.cpp

.PHONY: clean
clean:
	rm -rf *.opt *.cpp