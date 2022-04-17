
BUILD_DIR = ../build/bin
TEST_DIR = ../test

%: %.mlir
	-mkdir $@
	$(BUILD_DIR)/hcl-opt -opt $(FLAG) $< > $<.opt
	$(BUILD_DIR)/hcl-translate -emit-vivado-hls $<.opt > $@.cpp
	mv $<.opt $@/
	mv $@.cpp $@/
	cp run_base.tcl $@/

# Run synthesis
# vivado_hls -f run_base.tcl [test_file]

.PHONY: clean
clean:
	rm -rf *.opt *.cpp