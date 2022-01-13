#=============================================================================
# run_base.tcl 
#=============================================================================
# @brief: A Tcl script for synthesizing the design.

# Project name
set prj_name [lindex $argv 2]
set hls_prj [lindex $argv 2]_hls.prj

# Open/reset the project
open_project ${hls_prj} -reset

# Top function of the design is "top"
set_top ${prj_name}

# Add design and testbench files
add_files ${prj_name}.cpp

open_solution "solution1"
# Use Zynq device
set_part {xczu3eg-sbva484-1-e}

# Target clock period is 10ns (100MHz)
create_clock -period 10

# Directives 

############################################

# Simulate the C++ design
# csim_design -O
# Synthesize the design
csynth_design
# Co-simulate the design
#cosim_design
# Implement the design
# export_design -format ip_catalog

exit