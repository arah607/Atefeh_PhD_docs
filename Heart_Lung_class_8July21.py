import math
from math import cos
from math import floor

#Import numpy
import numpy as np
import matplotlib.pyplot as plt
from switch import Switch
from math import pi
from math import exp
from math import sqrt

#import importlib
#from subprocess import call


# !/usr/bin/env python

# This routine reads in an arterial tree and solves the static lung
# perfusion model as published in Burrowes et al. 2009 Ann Biomed Eng
# Vol 37, pp 2497-2509The model reads in an arterial tree only and #solves Pressure-Resistance-Flow equations within this tree for compliant vessels. Pressure is defined at terminal arteries as a function of gravitational height.
import os
import shutil
from aether.diagnostics import set_diagnostics_on
from aether.indices import perfusion_indices, get_ne_radius
from aether.filenames import read_geometry_main, get_filename
from aether.geometry import append_units, define_node_geometry, define_1d_elements, define_rad_from_geom, add_matching_mesh
from aether.exports import export_1d_elem_geometry, export_node_geometry, export_1d_elem_field, export_node_field, export_terminal_perfusion
from aether.pressure_resistance_flow import evaluate_prq


class Heart_Lung:
    def __init__(self):

       self.T_vc = 0.34  # The duration of ventricles contraction
       self.T_vr = 0.15  # The duration of ventricles relaxation

       self.t_ar = 0.97  # The time when the atria start to relax
       self.T_ar = 0.17  # The duration of atria relaxation
       self.t_ac = 0.80  # The time when the atria start to contraction
       self.T_ac = 0.17  # The duration of atria contraction



       # blood_press_Atria_ventricles

       self.E_ra_A = 7.998e+6  # amplitude value of the RA elastance
       self.E_ra_B = 9.331e+6  # baseline value of the RA elastance
       #self.V_ra = 4.8422067057679e-5 # blood volume of RA
       self.V_ra = 20.0e-6    # blood volume of RA
       self.V_ra_0 = 4.0e-6    # dead blood volume of RA

       self.E_rv_A = 73.315e+6 # amplitude value of the RV elastance
       self.E_rv_B = 6.665e+6 # baseline value of the RV elastance
       #self.V_rv = 0.000122489698505439 # blood volume of RV
       self.V_rv = 500.0e-6 # blood volume of RV
       self.V_rv_0 = 10.0e-6 # dead blood volume of RV

       self.E_la_A = 9.331e+6 # amplitude value of the LA elastance
       self.E_la_B = 11.997e+6 # baseline value of the LA elastance
      # self.V_la = 6.03152734220287e-5 # blood volume of LA
       self.V_la = 20.0e-6 # blood volume of LA
       self.V_la_0 = 4.0e-6 # dead blood volume of LA

       self.E_lv_A = 366.575e+6 # amplitude value of the LV elastance
       self.E_lv_B = 10.664e+6 # baseline value of the LV elastance
       #self.V_lv = 366.575e+6 # blood volume of LV
       #self.V_lv = 0.000122331670770222
       self.V_lv = 500.0e-6
       self.V_lv_0 = 5.0e-6  # dead blood volume of LV

       # blood_flow_atria_ventricles
       self.CQ_trv = 34.6427e-6 #triscupid valve coefficient
       self.CQ_puv = 30.3124e-6 # pulmonary valve coefficient
       self.P_pulmonary_artery = 4000.0  # pulmonary arteries
       self.CQ_miv = 34.6427e-6 # mitral valve coefficient
       self.CQ_aov = 30.3124e-6 #aortic valve coefficient
       #self.P_root = 10435.6603285072 #blood pressure in the aortic root
       #self.P_root = 0.4e+6
       self.P_root = 0

       # blood_volume_Atria_ventricles
       self.Q_sup_venacava = 1.21324067213557e-5 #blood flow superior vena cava
       self.Q_inf_venacava = 4.11763993353109e-5 #blood flow inferior vena cava
       self.Q_pulmonary_vein = 0 # vein flow
       #self.Q_pulmonary_vein = -1.66718121734813e-5  # vein flow

       # pulmonary circulation
       #self.Q_artery = 1.92753215344535e-5 # artery flow
       self.Q_pulmonary_artery = 0 # artery flow

       self.C_pulmonary_artery = 0.0309077e-6  # artery compliance
       self.P_pulmonary_artery = 4000.0  # artery pressure
       #self.P_pulmonary_artery = 1345.04464652372

       self.I_pulmonary_artery = 1.0e-6  # artery inductance
       self.R_pulmonary_artery = 10.664e+6  # artery resistance

       self.Q_pulmonary_vein = 0 # vein flow
       self.C_pulmonary_vein = 0.60015e-6  # vein compliance
      # self.P_vein = 0  # vein pressure
       self.P_pulmonary_vein = 1139.49261768037 #simulation value
       self.R_pulmonary_vein = 1.333e+6 # vein resistance
       self.I_pulmonary_vein = 1.0e-6  # vein inductance


       # systemic circulation
       self.Q_systemic_artery = 0 # systemic artery flow
       self.C_systemic_artery = 0.0112528e-6 # systemic artery compliance?????????
       self.P_systemic_artery = 100  # systemic artery pressure
       self.R_systemic_artery = 0.06665e+6 # systemic artery resistance?????????????
       self.I_systemic_artery = 0.1333e+6  # systemic artery inductance?????????????

       self.Q_systemic_vein = 0 # systemic vein flow
       self.C_systemic_vein = 0.5626407e-6 # systemic vein compliance
       self.P_systemic_vein = 2    # systemic vein pressure
       self.R_systemic_vein = 1.1997e+6 # systemic vein resistance
       self.I_systemic_vein = 0.06665e+6 # systemic vein inductance

       self.T = 1  # duration of a cardiac cycle


    def define_t_array(self, a, b, num_int):
        ''' This function considers the time as an array '''
        self.t = np.linspace(a, b, num_int)

    def activation_ventricles(self, mt):
        '''  This function calculates the activation ventricles '''
        if (mt >= 0) and (mt <= self.T_vc * self.T):
             self.e_v= 0.5 * (1 - cos(pi * mt / (self.T_vc * self.T)))
        elif (mt > self.T_vc * self.T) and (mt <= (self.T_vc + self.T_vr) * self.T):
            self.e_v= 0.5 * (1 + cos(pi * (mt - self.T_vc * self.T) / (self.T_vr * self.T)))
        elif (mt > (self.T_vc + self.T_vr) * self.T) and (mt < self.T):
            self.e_v = 0

        return self.e_v

    def activation_atria(self, mt):
        ''' This function calculates the  activation atria '''
        if (mt >= 0) and (mt <= (self.t_ar + self.T_ar) * self.T - self.T):
            self.e_a = 0.5 * (1 + cos(pi * (mt + self.T - self.t_ar * self.T) / (self.T_ar * self.T)))
        elif (mt > (self.t_ar + self.T_ar) * self.T - self.T) and (mt <= self.t_ac * self.T):
            self.e_a = 0
        elif (mt > self.t_ac * self.T) and (mt <= (self.t_ac + self.T_ac) * self.T):
            self.e_a = 0.5 * (1 - cos(pi * (mt - self.t_ac * self.T) / (self.T_ac * self.T)))
        elif (mt > (self.t_ac + self.T_ac) * self.T) and (mt <= self.T):
            self.e_a = 0.5 * (1 + cos(pi * (mt - self.t_ar * self.T) / (self.T_ar * self.T)))

        return self.e_a

    def Blood_Press_Atria_Ventricles(self):
        ''' This function calculates the blood pressure in atria and ventricles '''


        # pressure in RA
        self.P_ra = (self.e_a * self.E_ra_A + self.E_ra_B) * (self.V_ra - self.V_ra_0)

        # RV pressure
        self.P_rv = (self.e_v * self.E_rv_A + self.E_rv_B) * (self.V_rv - self.V_rv_0)

        # LA pressure
        self.P_la = (self.e_a * self.E_la_A + self.E_la_B) * (self.V_la - self.V_la_0)

        # LV pressure
        self.P_lv = (self.e_v * self.E_lv_A + self.E_lv_B) * (self.V_lv - self.V_lv_0)

        return (self.P_ra, self.P_rv, self.P_la, self.P_lv)

    def Blood_Flow_Atria_ventricles(self):
        '''   This function calculates the blood flow in atria and ventricles '''
        # RA blood flow
        if self.P_ra >= self.P_rv:
            self.Q_ra = self.CQ_trv * sqrt(self.P_ra - self.P_rv)
        else:
            self.Q_ra = 0

        # RV blood flow
        if self.P_rv >= self.P_rv:    #self.P_artery
            self.Q_rv = self.CQ_puv * sqrt(abs(self.P_rv - self.P_pulmonary_artery))
        else:
            self.Q_rv = 0

        # LA blood flow
        if self.P_la >= self.P_lv:
            self.Q_la = self.CQ_miv * sqrt(self.P_la - self.P_lv)
        else:
            self.Q_la = 0

        # LV blood flow
        if self.P_lv >= self.P_root:
            self.Q_lv = self.CQ_aov * sqrt(self.P_lv - self.P_root)
        else:
            self.Q_lv = 0
        return (self.Q_ra, self.Q_rv, self.Q_la, self.Q_lv)


    def Blood_Volume_Atria_ventricles(self):
        ''' This function calculates the blood volume changes in atria and ventricles '''

        # blood volume changes in RA
        der_volume_RA = self.Q_sup_venacava + self.Q_inf_venacava - self.Q_ra

        # blood volume changes in RV
        der_volume_RV = self.Q_ra - self.Q_rv

        # blood volume changes of LA
        der_volume_LA = self.Q_pulmonary_vein - self.Q_la

        # blood volume changes of LV
        der_volume_LV = self.Q_la - self.Q_lv

        return (der_volume_RA, der_volume_RV, der_volume_LA, der_volume_LV)

    def systemic_circulation(self):
        '''' This function calculates the blood pressure and volume changes in artery and vein in systemic circulation'''

        der_systemic_press_artery = (self.Q_lv - self.Q_systemic_artery) / self.C_systemic_artery  # systemic pressure changes in artery

        der_systemic_press_vein = (self.Q_systemic_artery - self.Q_systemic_vein) / self.C_systemic_vein  # systemic pressure changes in vein

        der_systemic_flow_artery = (self.P_systemic_artery - self.P_systemic_vein - self.Q_systemic_artery * self.R_systemic_artery) / self.I_systemic_artery  # systemic flow changes in artery

        der_systemic_flow_vein = (self.P_systemic_vein - self.P_ra - self.Q_systemic_vein * self.R_systemic_vein) / self.I_systemic_vein  # systemic flow changes in vein

        return (der_systemic_press_artery, der_systemic_press_vein, der_systemic_flow_artery, der_systemic_flow_vein)



    #def pulmonary_circulation(self):
     #   ''' This function calculates the blood pressure and volume changes in artery and vein in pulmonary circulation'''

      #  der_press_artery = (self.Q_rv - self.Q_artery) / self.C_artery  # pressure changes in artery

       # der_press_vein = (self.Q_artery - self.Q_vein) / self.C_vein  # pressure changes in vein

        #der_flow_artery = (self.P_artery - self.P_vein - self.Q_artery * self.R_artery) / self.I_artery  # flow changes in artery

        #der_flow_vein = (self.P_vein - self.P_la - self.Q_vein * self.R_vein) / self.I_vein  # flow changes in vein

        #return (der_press_artery, der_press_vein, der_flow_artery, der_flow_vein)


# Call Perfusion code here
#def call_perfusion_file(self):
#    call(["python", "{}".format(self.path)])
def perfusion(P_rv,P_la,i):
    set_diagnostics_on(False)

    export_directory = 'output'

    if not os.path.exists(export_directory):
        os.makedirs(export_directory)

    # define model geometry and indices
    perfusion_indices()
    if i == 1:
        # Read in geometry files
        define_node_geometry('../geometry/P2BRP268-H12816_Artery_Full.ipnode')
        define_1d_elements('../geometry/P2BRP268-H12816_Artery_Full.ipelem')
        append_units()

        add_matching_mesh()

    # define radius by Strahler order
    mpa_rad = 11.2  # main pulmonary artery radius, needs to be unstrained, so if defining from CT scale back to zero pressure
    s_ratio = 1.54  # straheler diameter ratio
    order_system = 'strahler'
    order_options = 'all'
    name = 'inlet'
    define_rad_from_geom(order_system, s_ratio, name, mpa_rad, order_options, '')

    s_ratio_ven = 1.56
    inlet_rad_ven = 14.53
    order_system = 'strahler'
    order_options = 'list'
    name = 'inlet'
    define_rad_from_geom(order_system, s_ratio_ven, '61361', inlet_rad_ven, order_options, '122720')  # matched arteries

    ##Call solve
    mesh_type = 'full_plus_ladder'
    vessel_type = 'elastic_g0_beta'
    grav_dirn = 2
    grav_factor = 1.0
    bc_type = 'pressure'
    inlet_bc = P_rv       #749.7438405387509
    outlet_bc = P_la      # 1161.738365326976

    evaluate_prq(mesh_type, vessel_type, grav_dirn, grav_factor, bc_type, inlet_bc, outlet_bc)

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
    export_1d_elem_field(7, filename, group_name, field_name)

    # export node field for pressure
    filename = export_directory + '/P2BRP268-H12816_pressure_perf.exnode'
    field_name = 'pressure_perf'
    export_node_field(1, filename, group_name, field_name)

    # Export terminal solution
    filename = export_directory + '/P2BRP268-H12816_terminal.exnode'
    export_terminal_perfusion(filename, group_name)

    shutil.move('micro_flow_unit.out', export_directory + '/micro_flow_unit.out')
    shutil.move('micro_flow_ladder.out', export_directory + '/micro_flow_ladder.out')
    ne_radius = []



def plot_results(dic_array, x_lable, y_lable, plot_title, dic_array_2={}):
    ''' This function plot the result found solving systems'''

    if dic_array_2 == {}:
        dic_vals = dic_array.items()
        x, y = zip(*dic_vals)
        plt.plot(x, y)
        plt.xlabel(x_lable)
        plt.ylabel(y_lable)
        plt.title(plot_title)
    else:
        dic_vals = dic_array.items()
        x, y = zip(*dic_vals)
        dic_vals_2 = dic_array_2.items()
        x_2, y_2 = zip(*dic_vals_2)
        plt.plot(x, y)
        plt.plot(x_2, y_2)
        plt.xlabel(x_lable)
        plt.ylabel(y_lable)
        plt.title(plot_title)




    # plt.legend()

    plt.show()

def print_results(*arg):
    ''' This function print output values'''
    print("e_a:" + str([(k, arg[0][k]) for k in arg[0]]))
    print("e_v:" + str([(k, arg[1][k]) for k in arg[1]]))

    print("P_la:" + str([(k, arg[2][k]) for k in arg[2]]))
    print("P_lv:" + str([(k, arg[3][k]) for k in arg[3]]))
    print("P_ra" + str([(k, arg[4][k]) for k in arg[4]]))
    print("P_rv:" + str([(k, arg[5][k]) for k in arg[5]]))

    print("Q_la:" + str([(k, arg[6][k]) for k in arg[6]]))
    print("Q_lv:" + str([(k, arg[7][k]) for k in arg[7]]))
    print("Q_ra:" + str([(k, arg[8][k]) for k in arg[8]]))
    print("Q_rv:" + str([(k, arg[9][k]) for k in arg[9]]))

    print("Q_la_So:" + str([(k, arg[10][k]) for k in arg[10]]))
    print("Q_lv_So:" + str([(k, arg[11][k]) for k in arg[11]]))
    print("Q_ra_So:" + str([(k, arg[12][k]) for k in arg[12]]))
    print("Q_rv_So:" + str([(k, arg[13][k]) for k in arg[13]]))

    print("der_volume_RA:" + str([(k, arg[14][k]) for k in arg[14]]))
    print("der_volume_RV:" + str([(k, arg[15][k]) for k in arg[15]]))
    print("der_volume_LA:" + str([(k, arg[16][k]) for k in arg[16]]))
    print("der_volume_LV:" + str([(k, arg[17][k]) for k in arg[17]]))

    #print("der_press_artery:" + str([(k, arg[14][k]) for k in arg[14]]))
    #print("der_press_vein:" + str([(k, arg[15][k]) for k in arg[15]]))
    #print("der_flow_artery:" + str([(k, arg[16][k]) for k in arg[16]]))
    #print("der_flow_vein:" + str([(k, arg[17][k]) for k in arg[17]]))

    print("der_systemic_press_artery:" + str([(k, arg[18][k]) for k in arg[18]]))
    print("der_systemic_press_vein:" + str([(k, arg[19][k]) for k in arg[19]]))
    print("der_systemic_flow_artery:" + str([(k, arg[20][k]) for k in arg[20]]))
    print("der_systemic_flow_vein:" + str([(k, arg[21][k]) for k in arg[21]]))

def read_file(path="/home/arah607/lung-group-examples/perfusion_Clark2011/output/" + "P2BRP268-H12816_flow_perf.exelem"):
   '''This function read the txt file from output of perfusion_clark2011 code '''
   flow_file = open(path)
   read_line = [10]  #This line shows the 11th line value in P2BRP268-H12816_flow_perf.exelem (output)

   for position, line in enumerate(flow_file):  #position shows the row and line shows the value in that row
        if position in read_line:
            flow = line
            break

   flow = flow.split()   #if the value has some strings, this line slipt them
   flow = float(flow[0])
   #flow = flow.split(" ")
   return flow

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # define an object regarding to heart_lung class
    heart_lung_obj = Heart_Lung()

    # activation_ventricles and activation_atria
    e_v = {}
    e_a = {}
    # stream = open("perfusion_Clark2011.py")
    # read_file = stream.read()
    # exec(read_file)

    # blood_press_Atria_ventricles
    P_ra = {}
    P_rv = {}
    P_la = {}
    P_lv = {}

    # P_ra_So = {}
    # P_rv_So = {}
    # P_la_So = {}
    # P_lv_So = {}

    # blood_flow_atria_ventricles
    Q_ra = {}
    Q_rv = {}
    Q_la = {}
    Q_lv = {}

    Q_ra_So = {}
    Q_rv_So = {}
    Q_la_So = {}
    Q_lv_So = {}

    # blood_volume_Atria_ventricles
    der_volume_RA = {}
    der_volume_RV = {}
    der_volume_LA = {}
    der_volume_LV = {}

    # pulmonary circulation
    # der_press_artery = {}
    #der_press_vein = {}
    #der_flow_artery = {}
    #der_flow_vein = {}

    # systemic circulation
    der_systemic_press_artery = {}  # systemic pressure changes in artery
    der_systemic_press_vein = {}  # systemic pressure changes in vein
    der_systemic_flow_artery = {}  # systemic flow changes in artery
    der_systemic_flow_vein = {}  # systemic flow changes in vein

    
    # receive an the first and the last element of interval and number of equal parts from user to create an array
    first_int = int(input("please enter the first value of interval :"))
    end_int = int(input("please enter the last value of interval :"))
    num_split = int(input("please enter the number of splitting interval :"))

    heart_lung_obj.define_t_array(first_int, end_int, num_split)

    #print(flow_file.readline())
    for i in range(0, len(heart_lung_obj.t)):
      if i == 0:  #In this loope we want to run the heart model
        # define time parameter
        mt = heart_lung_obj.t[i] - heart_lung_obj.T * floor(heart_lung_obj.t[i] / heart_lung_obj.T)

        # Calculate activation ventricles value regarding its time
        e_v[i] = heart_lung_obj.activation_ventricles(mt)

        # Calculate activation atria value regarding its time
        e_a[i] = heart_lung_obj.activation_atria(mt)

        # Calculate blood pressure in atria and ventricles regarding their time
        [P_ra[i], P_rv[i], P_la[i], P_lv[i]] = heart_lung_obj.Blood_Press_Atria_Ventricles()
        # P_ra_So[i] = P_ra[i]
        # P_rv_So[i] = P_rv[i]
        # P_la_So[i] = P_la[i]
        # P_lv_So[i] = P_lv[i]
         # if i == 1:
        #    P_vein = 1161.738365326976
         #   P_la[i] = P_vein

        # Calculate blood flow in atria and ventricles regarding their time
        [Q_ra[i], Q_rv[i], Q_la[i], Q_lv[i]] = heart_lung_obj.Blood_Flow_Atria_ventricles()
        Q_ra_So[i] = Q_ra[i]
        Q_rv_So[i] = Q_rv[i]
        Q_la_So[i] = Q_la[i]
        Q_lv_So[i] = Q_lv[i]

        # Calculate blood volume changes in atria and ventricles regarding their time
        [der_volume_RA[i], der_volume_RV[i], der_volume_LA[i], der_volume_LV[i]] = heart_lung_obj.Blood_Volume_Atria_ventricles()

        # Calculate blood pressure and volume in pulmonary circulation regarding its time
       # [der_press_artery[i], der_press_vein[i], der_flow_artery[i], der_flow_vein[i]] = heart_lung_obj.pulmonary_circulation()
       # if i == 1:
        #    P_artery = 749.7438405387509
         #   P_vein = 1161.738365326976
          #  Q_vein = -0.29308E+05
           # Q_la[i] = Q_vein
            #Q_rv[i] = Q_vein

        # Calculate blood pressure and volume in systemic circulation regarding its time
        [der_systemic_press_artery[i], der_systemic_press_vein[i], der_systemic_flow_artery[i], der_systemic_flow_vein[i]] = heart_lung_obj.systemic_circulation()
      else:
          #In these loops first we get the privious P_rv and P_la and put them in perfusion function. Then we run the heart model, but aftre running the perfusion model we received the flow and we replace the flow from perfusion function with Q_rv and Q_la in heart
          #P_rv[0] = 11929.443538438134
          #P_la[0] = 202.33952809536052
          perfusion(P_rv[i - 1], P_la[i - 1], i)
          flow = read_file()
          Q_vein = flow

          mt = heart_lung_obj.t[i] - heart_lung_obj.T * floor(heart_lung_obj.t[i] / heart_lung_obj.T)

          # Calculate activation ventricles value regarding its time
          e_v[i] = heart_lung_obj.activation_ventricles(mt)

          # Calculate activation atria value regarding its time
          e_a[i] = heart_lung_obj.activation_atria(mt)

          # Calculate blood pressure in atria and ventricles regarding their time
          [P_ra[i], P_rv[i], P_la[i], P_lv[i]] = heart_lung_obj.Blood_Press_Atria_Ventricles()
          # P_ra_So[i] = P_ra[i]
          # P_rv_So[i] = P_rv[i]
          # P_la_So[i] = P_la[i]
          # P_lv_So[i] = P_lv[i]

          # Calculate blood flow in atria and ventricles regarding their time
          [Q_ra[i], Q_rv[i], Q_la[i], Q_lv[i]] = heart_lung_obj.Blood_Flow_Atria_ventricles()
          Q_ra_So[i] = Q_ra[i]
          Q_rv_So[i] = Q_rv[i]
          Q_la_So[i] = Q_la[i]
          Q_lv_So[i] = Q_lv[i]

          Q_rv[i] = flow
          Q_la[i] = flow


          heart_lung_obj.Q_rv = Q_rv[i] #replace new values in flow function
          heart_lung_obj.Q_la = Q_la[i] #replace new values in flow function

          # Calculate blood volume changes in atria and ventricles regarding their time
          [der_volume_RA[i], der_volume_RV[i], der_volume_LA[i], der_volume_LV[i]] = heart_lung_obj.Blood_Volume_Atria_ventricles()

          # Calculate blood pressure and volume in systemic circulation regarding its time
          [der_systemic_press_artery[i], der_systemic_press_vein[i], der_systemic_flow_artery[i], der_systemic_flow_vein[i]] = heart_lung_obj.systemic_circulation()

    plot_results(e_a, 'Time', 'Value', 'Activation Atria')
    plot_results(e_v, 'Time', 'Value', 'Activation Ventricle')

    plot_results(P_la, 'Time', 'Value', 'Left Atrium Pressure')
    plot_results(P_lv, 'Time', 'Value', 'Left Ventricle Pressure')
    plot_results(P_ra, 'Time', 'Value', 'Right Atrium Pressure')
    plot_results(P_rv, 'Time', 'Value', 'Right Ventricle Pressure')

    # plot_results(P_la_So, 'Time', 'Value', 'Left Atrium Pressure')
    # plot_results(P_lv_So, 'Time', 'Value', 'Left Ventricle Pressure')
    # plot_results(P_ra_So, 'Time', 'Value', 'Right Atrium Pressure')
    # plot_results(P_rv_So, 'Time', 'Value', 'Right Ventricle Pressure')


    plot_results(Q_la, 'Time', 'Value', 'Left Atrium Flow', Q_la_So)
    plot_results(Q_lv, 'Time', 'Value', 'Left Ventricle Flow', Q_lv_So)
    plot_results(Q_ra, 'Time', 'Value', 'Right atrium Flow', Q_ra_So)
    plot_results(Q_rv, 'Time', 'Value', 'Right Ventricle Flow', Q_rv_So)

    # plot_results(Q_la, 'Time', 'Value', 'Left Atrium Flow')
    # plot_results(Q_lv, 'Time', 'Value', 'Left Ventricle Flow')
    # plot_results(Q_ra, 'Time', 'Value', 'Right atrium Flow')
    # plot_results(Q_rv, 'Time', 'Value', 'Right Ventricle Flow')

    plot_results(der_volume_RA, 'Time', 'Value', 'blood volume changes in RA')
    plot_results(der_volume_RV, 'Time', 'Value', 'blood volume changes in RV')
    plot_results(der_volume_LA, 'Time', 'Value', 'blood volume changes in LA')
    plot_results(der_volume_LV, 'Time', 'Value', 'blood volume changes in LV')


    # plot_results(der_press_artery, 'Time', 'Value', 'Pulmonary Artery Pressure')
    # plot_results(der_press_vein, 'Time', 'Value', 'Pulmonary Vein Pressure')
    # plot_results(der_flow_artery, 'Time', 'Value', 'Pulmonary Artery Flow')
    # plot_results(der_flow_vein, 'Time', 'Value', 'Pulmonary Vein Flow')

    plot_results(der_systemic_press_artery, 'Time', 'Value', 'Systemic Artery Pressure')
    plot_results(der_systemic_press_vein, 'Time', 'Value', 'Systemic Vein Pressure')
    plot_results(der_systemic_flow_artery, 'Time', 'Value', 'Systemic Artery Flow')
    plot_results(der_systemic_flow_vein, 'Time', 'Value', 'Systemic Vein Flow')





    print_results(e_a, e_v, P_la, P_lv, P_ra, P_rv, Q_la, Q_lv, Q_ra, Q_rv, Q_la_So, Q_lv_So, Q_ra_So, Q_rv_So, der_volume_LA, der_volume_LV, der_volume_RA, der_volume_RV,
                  der_systemic_press_artery, der_systemic_press_vein,
                  der_systemic_flow_artery, der_systemic_flow_vein)