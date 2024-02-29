#!/usr/bin/env python

#This routine reads in an arterial tree and solves the static lung
#perfusion model as published in Burrowes et al. 2009 Ann Biomed Eng
#Vol 37, pp 2497-2509The model reads in an arterial tree only and #solves Pressure-Resistance-Flow equations within this tree for compliant vessels. Pressure is defined at terminal arteries as a function of gravitational height.

import time
import os
import shutil
from aether.diagnostics import set_diagnostics_on
from aether.indices import perfusion_indices, get_ne_radius
from aether.filenames import read_geometry_main,get_filename
from aether.geometry import append_units,define_node_geometry, define_1d_elements,define_rad_from_geom,add_matching_mesh
from aether.exports import export_1d_elem_geometry, export_node_geometry, export_1d_elem_field,export_node_field,export_terminal_perfusion
from aether.pressure_resistance_flow import evaluate_prq


def main():

   # start = time.clock()

    set_diagnostics_on(False)
    
    export_directory = 'output'
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    

    #define model geometry and indices
    perfusion_indices()

    #Read in geometry files
    define_node_geometry('../geometry/P2BRP268-H12816_Artery_Full.ipnode')
    define_1d_elements('../geometry/P2BRP268-H12816_Artery_Full.ipelem')
    append_units()
    
    
    add_matching_mesh()

    #define radius by Strahler order
    mpa_rad=11.2 #main pulmonary artery radius, needs to be unstrained, so if defining from CT scale back to zero pressure
    s_ratio=1.54 #straheler diameter ratio
    order_system = 'strahler'
    order_options = 'all'
    name = 'inlet'
    define_rad_from_geom(order_system, s_ratio, name, mpa_rad, order_options,'')
    
    s_ratio_ven=1.56
    inlet_rad_ven=14.53
    order_system = 'strahler'
    order_options = 'list'
    name = 'inlet'
    define_rad_from_geom(order_system, s_ratio_ven, '61361', inlet_rad_ven, order_options,'122720')#matched arteries

    ##Call solve
    mesh_type = 'full_plus_ladder'
    vessel_type = 'elastic_g0_beta'
    grav_dirn = 2
    grav_factor = 1.0
    bc_type = 'flow'
    inlet_bc = 83333.0
    outlet_bc = 666.0 
    
    evaluate_prq(mesh_type,vessel_type,grav_dirn,grav_factor,bc_type,inlet_bc,outlet_bc)

    ##export geometry
    group_name = 'perf_model'
    filename = export_directory + '/P2BRP268-H12816_Artery.exelem'
    export_1d_elem_geometry(filename, group_name)
    filename = export_directory + '/P2BRP268-H12816_Artery.exnode'
    export_node_geometry(filename, group_name)

    # export element field for radius
    field_name = 'radius_perf'
    ne_radius = get_ne_radius()
    filename = export_directory + '/P2BRP268-H12816_radius_perf.exelem'
    export_1d_elem_field(ne_radius, filename, name, field_name)
    
    # export flow element
    filename = export_directory + '/P2BRP268-H12816_flow_perf.exelem'
    field_name = 'flow'
    export_1d_elem_field(7,filename, group_name, field_name)


    #export node field for pressure
    filename=export_directory + '/P2BRP268-H12816_pressure_perf.exnode'
    field_name = 'pressure_perf'
    export_node_field(1, filename, group_name, field_name)
    
    # Export terminal solution
    filename = export_directory + '/P2BRP268-H12816_terminal.exnode'
    export_terminal_perfusion(filename, group_name)
  #  elapsed = time.clock()
   # elapsed = elapsed - start
    #print ("Time spent solving is: ", elapsed)

    shutil.move('micro_flow_unit.out',export_directory + '/micro_flow_unit.out')
    shutil.move('micro_flow_ladder.out',export_directory + '/micro_flow_ladder.out')


if __name__ == '__main__':
    main()
