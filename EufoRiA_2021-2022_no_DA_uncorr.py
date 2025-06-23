import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nquality import QualityBuilder
from pipedream_solver.simulation import Simulation
from pipedream_solver.wq_reaction import WQ_reaction
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import time
sns.set_palette('viridis')
import warnings
warnings.filterwarnings("ignore")
import math

###############################################################################
#   Reading the model input files
###############################################################################
input_path = './model/EufoRiA_model'

superjunctions = pd.read_csv(f'{input_path}/superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/superjunction_wq_params.csv')
weirs = pd.read_csv(f'{input_path}/weirs.csv')

wq_para_new = pd.read_csv(f'{input_path}/WQ_reaction_params_new.csv', index_col=0)  # spatially different parameters

# If you want change the dataset, you need to change the following four lines. And you should change the 'date' part.
modeling_years = '2021-2022'
data_path = f'{input_path}/{modeling_years}'
H_weir =  pd.read_csv(f'{data_path}/H_weir_{modeling_years}.csv')
Q_in_file = pd.read_csv(f'{data_path}/Q_flow_{modeling_years}.csv', index_col=0)
WQ_df = pd.read_csv(f'{data_path}/WQ_data_{modeling_years}.csv')

# Applying the multiple MET stations
MET_data2 = pd.read_csv(f'{data_path}/met_hourly_4stations_2021-2022.csv')
MET_data2.fillna(0, inplace=True)       
MET_data2 = MET_data2.drop('date', axis=1)
MET_data_N = {key: MET_data2[MET_data2['station'] == key] for key in MET_data2['station'].unique()}
key_MET = MET_data2['station'].unique()
key_MET = key_MET.tolist()
for key_ in key_MET:
    MET_data_N[key_] = np.array(MET_data_N[key_])
    
superlinks_New = superlinks.copy()
sj1_to_met_station = superjunction_wq_params['met_station'].to_dict()
superlinks_New['met_station'] = superlinks_New['sj_1'].map(sj1_to_met_station)
superlinks_New['met_station'] = superlinks_New.apply(
    lambda row: sj1_to_met_station.get(row['sj_0'], row['met_station']) if pd.isnull(row['met_station']) else row['met_station'], axis=1 )

weirs_New = weirs.copy()
weirs_New['met_station'] = weirs_New['sj_1'].map(sj1_to_met_station)
weirs_New['met_station'] = weirs_New.apply(
    lambda row: sj1_to_met_station.get(row['sj_0'], row['met_station']) if pd.isnull(row['met_station']) else row['met_station'], axis=1 )

###############################################################################
#   Set the general options and parameters
###############################################################################
# Set the start and end model time(Julian day)
Start_J_day = 0
Start_date = pd.to_datetime('2021-01-01 00:00')  # Model start time: for only input data interpolation process
Base_date =  pd.to_datetime('2021-01-01 00:00')  # J_day = 0 at this time: for only input data interpolation process
End_J_day = 730
End_date = pd.to_datetime('2023-01-02 00:00')    # add 1 day: for only data interpolation process

# Adaptive time stepping parameters
Coefficient_dt = 0.15    # This coefficient is multiplied to the calculated dt based on the mean velocity by mean percentile.
mean_percentile = 60
# Mean percentile: included percent of model elements to calculate the average velocity
# The smaller percent makes the dt more sensitive to higher velocity.
dt_min = 60
dt_max = 300

da_hour = np.linspace(0,1,9)       # data assimilation J_day list < 1 day
save_hour = np.linspace(0,1,25)    # output saving J_day list < 1 day

# Modeling options
screen_output = 1         # 1: print screen output during the modeling,              # 0: no screen output
saved_interpolated_wq = 0 # 1: read from the pre-interpolated data,                  # 0: interpolate them again using 'WQ_data.csv' file
c_in_interpolation = 0    # 1: interpolate the c_in values by real J_day.            # 0: use the c_in data at int(J_day)
KF_option = 0             # 1: Kalman filter On,                                     # 0: Kalman filter OFF (No data assimilation)
spatial_corr_on = 0       # 1: spatial correlation in the process noise cov matrix,  # other numbers: no correlation
saved_spatial_corr = 0    # 1: read from the pre_calculated correlation data,        # 0: re-calculate them

# Weir control variables
weir_sj = [16,20,23,28,32,36,42,53,67]
control_K1 = 0.000008#0.000002     # orifice control 1 for error at the current time
control_K2 = 0.000002#0.0000005    # orifice control 2 for error sum  = current + previous
control_K3 = 0.005#0.005         # orifice control 3 for error change

# Defining the length(dx) of the internal links
# If dx_option = 0: Calculate the number of internal links(minimum > 3) based on the 'desired_dx' length.
# If dx_option = other number: Read the user defined 'internallinks' for each superlink from 'superlinks.csv'
dx_option = 0   

if dx_option == 0:
    desired_dx = 400
    superlinks['internal_links_k'] = round(superlinks['dx'] / desired_dx, 0)
    superlinks['internal_links_k'] = np.where(superlinks['internal_links_k'] > 3,
                                              superlinks['internal_links_k'], 3)
    superlinks['internal_links_k'] = superlinks['internal_links_k'].astype(int)
else:
    pass

# Make a list for constituents
wq_name = ['Temp', 'DO', 'LDOC', 'LPOC', 'RDOC',  'RPOC', 'PIN', 'NH4N', 'NO3N', 'LDON',
           'LPON', 'RDON', 'RPON', 'PIP','PO4P',  'LDOP', 'LPOP', 'RDOP', 'RPOP', 'ALG1',
           'ALG2', 'ALG3', 'ISS']
da_obs = ['Temp', 'DO', 'TOC', 'TN', 'NH4N', 'NO3N', 'TP','PO4P', 'CHLA', 'ISS']
da_target = ['Temp', 'DO', 'LDOC', 'LPOC', 'RDOC', 'RPOC', 'PIN', 'NH4N', 'NO3N', 
         'LDON', 'LPON', 'RDON', 'RPON', 'PIP','PO4P', 'LDOP', 'LPOP', 'RDOP', 'RPOP',
         'ALG1', 'ALG2', 'ALG3', 'ISS']
out_name = ['Temp','DO', 'NH4N', 'NO3N', 'PO4P', 'ALG1', 'ALG2', 'ALG3', 'ISS']


###############################################################################
#  Multiple scenarios running
###############################################################################

scenario_list = ['S00_no_DA']

for sc_nm in scenario_list:
    scenario_path = f'{input_path}/Uncorr_data/{sc_nm}'
    WQ_match = pd.read_csv(f'{scenario_path}/wq_locations.csv')
    Q_in_file = pd.read_csv(f'{data_path}/Q_flow_{modeling_years}.csv', index_col=0)  # re-load as a dataframe

    ###############################################################################
    #  Hydraulic model initializing
    ###############################################################################
    SL = SuperLink(superlinks, superjunctions, weirs=weirs, internal_links = 0, bc_method = 'b', 
                   min_depth = 1e-4, mobile_elements = True)
    
    WQ_R = WQ_reaction(SL, superjunction_params=superjunction_wq_params,
                       superlink_params=superlink_wq_params, N_constituent=len(wq_name), wq_para_new = wq_para_new,
                       met_data=MET_data_N, met_key=key_MET, superlinks = superlinks_New, weirs=weirs_New)
    
    SL.reposition_junctions()         
    
    # weir control vector u : fully opened = 1
    u = np.zeros(len(weirs))   # Closed
    dt = 150 # time step only for initializing process
    # Initial Height of the weirs from observation
    now_H_weir = WQ_R.calculate_H_weir(Start_J_day, H_weir)
    Q_in_file, volume_initial, u = WQ_R.Initialize_Hydraulics_New(SL, Simulation, 
                                    Start_J_day, End_J_day, dt, u, Q_in_file, now_H_weir,
                                    weir_sj, control_K1, control_K2, control_K3)
    
    print('▷ Initial H_j of weirs (EL.m)', f'\t = {SL.H_j[16] : 9.3f}{SL.H_j[20] : 9.3f}{SL.H_j[23] : 9.3f}', 
          f'{SL.H_j[28] : 9.3f}{SL.H_j[32] : 9.3f}{SL.H_j[36] : 9.3f}',
          f'{SL.H_j[42] : 9.3f}{SL.H_j[53] : 9.3f}{SL.H_j[67] : 9.3f}' )
    _ = SL.plot_profile([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                         23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,
                         46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67], width = 10)
    
    ###############################################################################
    #   Water qualtity model initializing
    ###############################################################################
    print('\n▶▶ Started: Water Quality Model Initialization.')
    WQ = {}
    out_H_j = []
    out_c_j = defaultdict(list)
    out_c_Ik = defaultdict(list)
    out_c_ik = defaultdict(list)
    
    # New WQ data reading from the input files / WQ data conversion and calculate
    for i in range(0,len(wq_name)):
        WQ[wq_name[i]] = QualityBuilder(SL, superjunction_params=superjunction_wq_params,
                                     superlink_params=superlink_wq_params)
    if saved_interpolated_wq == 0:   
        # Water quality data preprocessing
        saved_preprocess = WQ_R.WQ_data_preprocessing(WQ_df, WQ_match, Start_J_day, End_J_day)
        with open(f'{data_path}/pre_calculated_WQ_preprocessed_data.pkl', 'wb') as fp:
            pickle.dump(saved_preprocess, fp)
        # Input water quality data and observed data for DA interpolation
        WQ_in_data, observed_data = WQ_R.WQ_data_interpolation_new(wq_name, da_obs, Start_date, End_date, Base_date)
        with open(f'{data_path}/pre_calculated_WQ_data.pkl', 'wb') as fp:
            pickle.dump(WQ_in_data, fp)
        #observed_data = WQ_R.DA_data_interpolation(Start_J_day, End_J_day, da_obs)
        with open(f'{data_path}/observation_data.pkl', 'wb') as fp:
            pickle.dump(observed_data, fp)
    elif saved_interpolated_wq == 1:
        print('▷ Loading the saved preprocessed WQ input files.')
        # Load the pre-calculated preprocessed data and save them to WQ_R instance
        with open(f'{data_path}/pre_calculated_WQ_preprocessed_data.pkl', 'rb') as fp:
            saved_preprocess = pickle.load(fp)
        WQ_R.apply_saved_preprocess(saved_preprocess)
        # Load the pre-calculated daily interpolated WQ data based on 'input_WQ_data.csv'.
        print('▷ Loading the pre-interpolated WQ input files.')
        with open(f'{data_path}/pre_calculated_WQ_data.pkl', 'rb') as fp:
            WQ_in_data = pickle.load(fp)
        with open(f'{data_path}/observation_data.pkl', 'rb') as fp:
            observed_data = pickle.load(fp)
    else:
        print("Error: [saved_in_obs_option] = 0 or 1 is required.")
        
    print("▷ Applying the initial water quality data.")
    # Calculate(interpolate) the initial WQ data and applythem considering the simulation start day.
    WQ_initial = WQ_R.WQ_data_initial(Start_J_day, Start_J_day, wq_name)
    WQ, out_c_j, out_c_Ik, out_c_ik = WQ_R.applying_initial_values(wq_name, WQ, 
                               WQ_initial, superlinks, out_c_j, out_c_Ik, out_c_Ik)
    
    J_day = Start_J_day
    WQ_R.WQ_reactions(dt, WQ, wq_name, J_day)     # Initialzing the internal variables in WQ_R
    print('▶▶ Finished: Water Quality Model Initialization. \n')
    
    ###############################################################################
    #   Kalman Filtering initializing
    ###############################################################################
    if KF_option == 1:
        WQ_A_k = {}
        WQ_B_k = {}
        if spatial_corr_on == 1:
            if saved_spatial_corr == 0:
                Q_spatial = WQ_R.make_Q_spatial(observed_data, da_obs, superjunctions, WQ_match)
                with open(f'{data_path}/saved_spatial_corr.pkl', 'wb') as fp:
                    pickle.dump(Q_spatial, fp)
            elif saved_spatial_corr == 1:
                with open(f'{data_path}/saved_spatial_corr.pkl', 'rb') as fp:
                    Q_spatial = pickle.load(fp)
            else:
                raise ValueError("Error: [saved_spatial_corr] = 1 or 0 is required.")
        elif spatial_corr_on == 0:
            Q_spatial = {}
            for i in range(0,len(wq_name)):
                Q_spatial[wq_name[i]] = np.eye(WQ_R._N_j,WQ_R._N_j)
        else:
            raise ValueError("Error: [spatial_corr_on] = 1 or 0 is required.")
            
        # Basically, there are no correlations between water quality species in this application
        Q_species = {}
        for i in range(0,WQ_R._N_j):
            Q_species[i] = np.zeros((len(wq_name),len(wq_name)))
            
        WQ_R.make_QR_cov_All_New(Q_spatial, Q_species, da_obs, da_target)
    elif KF_option == 0:
        pass
    else:
        raise ValueError("Error: [KF_option] = 1 or 0 is required.")
    
    ###############################################################################
    #   Reset model time and variables
    ###############################################################################
    J_day = Start_J_day
    SL.t = 0
    t_end = (End_J_day - Start_J_day)*86400
    
    H_error_prev = np.zeros(len(weir_sj))
    out_H_weir = []
    out_H_weir.append(SL.H_j[weir_sj].copy())
    out_H_error = []
    out_H_error.append(np.zeros(len(weir_sj)))
    
    out_Q_weir = []
    out_Q_weir.append(SL.Q_w.copy())
    out_H_SJ = []
    out_J_day = []
    out_J_day.append(J_day)
    
    start_time = time.time()
    
    H_error_prev_1 = np.zeros(len(weir_sj))
    H_error_prev_2 = np.zeros(len(weir_sj))
    
    ###############################################################################
    #   Simulate unsteady hydraulics, WQ transport, Kalman Filter, and WQ reactions
    ###############################################################################
    with Simulation(SL, dt=dt, t_end=t_end,
                    interpolation_method='linear') as simulation:
        while SL.t < t_end + 0.001:
            #######################################################################
            # calculate new dt
            #######################################################################
            dt = Coefficient_dt*np.mean(np.percentile(np.abs(SL._dx_ik/(SL.Q_ik/SL.A_ik)),mean_percentile))
            if dt < dt_min:
                dt = dt_min
            elif dt > dt_max:
                dt = dt_max
            dt = round(dt,0)
            
            #######################################################################
            # Hydraulics modeling
            #######################################################################
            # interpolate the input Q flow rate at the current time (J_day) from the daily timeseries
            Q_in = Q_in_file[int(J_day),1:WQ_R._N_j+1] +( (J_day - Q_in_file[int(J_day),0])*(Q_in_file[int(J_day)+1,1:WQ_R._N_j+1]
                     - Q_in_file[int(J_day),1:WQ_R._N_j+1] ) / (Q_in_file[int(J_day)+1,0] - Q_in_file[int(J_day),0]) )
            
            # control the orifices based on the observed H depth of weirs
            now_H_weir = WQ_R.calculate_H_weir(J_day, H_weir)
            H_error = (SL.H_j[weir_sj] - now_H_weir.T[:,0]).T
            for i_ in range(len(now_H_weir.T)):
                '''
                u[i_] = u[i_] + (control_K1*(SL.H_j[weir_sj[i_]] - now_H_weir.T[i_,0])
                                 + control_K2*dt*0.5*(H_error[i_] + H_error_prev[i_])
                                 + control_K3*(H_error[i_] - H_error_prev[i_])/dt )
                '''
                u[i_] = u[i_] + (
                                 control_K1*(H_error[i_] - H_error_prev_1[i_])
                                 + control_K2*H_error[i_]*dt
                                 + control_K3*(H_error[i_] - 2*H_error_prev_1[i_] + H_error_prev_2[i_])/dt
                                 )
            u = np.where(u<0,0,u)
            u = np.where(u>1,1,u)
            H_error_prev_2 = H_error_prev_1
            H_error_prev_1 = H_error
            
            # [Hydraulic model calculation]
            simulation.step(dt=dt, u_w=u, Q_in = Q_in)
    
            #######################################################################
            # WQ transport modeling
            #######################################################################
            # [WQ Transport] For the first constituent in the transport solver
            # interpolate the input WQ concentrations at the current time (J_day) from the daily timeseries
            for wq_nm in wq_name:
                if c_in_interpolation == 0:
                    c_in = WQ_in_data[wq_nm][int(J_day),1:]
                elif c_in_interpolation == 1:
                    c_in = WQ_in_data[wq_nm][int(J_day),1:WQ_R._N_j+1] +( (J_day
                                - WQ_in_data[wq_nm][int(J_day),0])*(WQ_in_data[wq_nm][int(J_day)+1,1:WQ_R._N_j+1] 
                                - WQ_in_data[wq_nm][int(J_day),1:WQ_R._N_j+1] ) / (WQ_in_data[wq_nm][int(J_day)+1,0]
                                - WQ_in_data[wq_nm][int(J_day),0]) )
                if wq_nm == wq_name[0]:
                    WQ[wq_nm].step(dt = dt, c_0j = c_in)     # calculate the superlink algorithm coefficients for the 1st constituent
                else:
                    WQ[wq_nm].step_2nd(WQ[wq_name[0]], dt = dt, c_0j = c_in)  # re-use the common coefficients for the other constituents
            
            #######################################################################
            # Kalman filter application to the WQ constituents
            #######################################################################
            # DA time calculate
            hour_now = J_day - math.trunc(J_day) 
            hour_diff = np.abs(hour_now - da_hour)
            min_diff = np.min(hour_diff)
            
            if(min_diff < dt/86400/2):
                if KF_option == 1:
                    before_c_j_NO3N = WQ['NO3N']._c_j
                    before_WQ_ik_0 = WQ['NO3N'].c_ik[0:9].copy()
                    # Calculate the A_k and B_k matrices for the Kalman Filtering
                    for i in range(0,len(wq_name)):
                        A_k, B_k = WQ[wq_name[i]].KF_Multi_WQ(dt)
                        WQ_A_k[wq_name[i]] = A_k
                        WQ_B_k[wq_name[i]] = B_k
    
                    # Apply the Kalman Filering and data assimilation
                    x_hat = WQ_R.KF_DA_All_New(dt, WQ_A_k, WQ_B_k, WQ,
                            observed_data, int(J_day), da_obs, da_target, Q_spatial, Q_species, J_day)
                    pre_WQ_c_j = {}
                    out_upper_temp = {}
                    out_lower_temp = {}
                    for i in range (0,len(da_target)):
                        # save the WQ values (before DA update)
                        pre_WQ_c_j[da_target[i]] = WQ[da_target[i]]._c_j.copy()
                        WQ[da_target[i]]._c_j = x_hat[i*WQ_R._N_j:(i+1)*WQ_R._N_j]
                    
                    # Recalculate the all internals after KF and DA(update)
                    for i in range(0,len(da_target)):
                        WQ[da_target[i]].solve_boundary_states()
                        WQ[da_target[i]].solve_internals_backwards()
                       
            #######################################################################
            # WQ reaction modeling
            #######################################################################
            # [WQ reaction calculation]
            WQ = WQ_R.WQ_reactions(dt, WQ, wq_name, J_day)
    
            #######################################################################
            # Screen output / saving output
            #######################################################################
            # save output hourly when DA applided
            hour_now = J_day - math.trunc(J_day) 
            hour_diff = np.abs(hour_now - save_hour)
            min_diff = np.min(hour_diff)
            if(min_diff < dt/86400/2):
                out_J_day.append(J_day)
                out_c_j = WQ_R.print_screen_and_save_output(dt, WQ, 
                    wq_name, screen_output, J_day, Start_J_day, End_J_day, start_time, out_name, out_c_j)
                out_Q_weir.append(SL.Q_w.copy())
                out_H_weir.append(SL.H_j[weir_sj])
                out_H_error.append(np.array(H_error, dtype = np.float64))
            
                print('Scenario = ', sc_nm)
            
            simulation.model.reposition_junctions()
            J_day = J_day + dt/86400
            
        # Save the WQ constituents results and weir outflow rate at the superjunction points
        with open(f'{scenario_path}/c_j_output.pkl', 'wb') as fp:
            pickle.dump(out_c_j, fp)
        with open(f'{scenario_path}/output_J_day.pkl', 'wb') as fp:
            pickle.dump(out_J_day, fp)
        with open(f'{scenario_path}//out_Q_weir.pkl', 'wb') as fp:
            pickle.dump(out_Q_weir, fp)
        with open(f'{scenario_path}//out_H_weir.pkl', 'wb') as fp:
            pickle.dump(out_H_weir, fp)
        with open(f'{scenario_path}//out_H_error.pkl', 'wb') as fp:
            pickle.dump(out_H_error, fp)
        
        ###############################################################################        
        #   Plot the outputs
        ###############################################################################
        WQ_weir_out_nm = ['Donam', 'Nakdan', 'Sunsan', 'Chilgok', 'Dasa', 'Nongong', 'Dukgok', 'Haman']
        SJ_out_no = 5
        for weir_out_nm in WQ_weir_out_nm:
            WQ_R.print_WQ_combined_output([weir_out_nm], out_J_day, out_c_j, data_path, SJ_out_no, compare_hour = 0.0, base_year=2021)
            SJ_out_no += 1