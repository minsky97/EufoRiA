# Definition of the reaction functions for EufoRiA
from numba import njit
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
#import cupy as cp

class WQ_reaction():
    def __init__(self, hydraulics, superjunction_params, superlink_params, N_constituent, wq_para_new, met_data, met_key, superlinks, weirs,
                 junction_params=None, link_params=None):
        self.hydraulics = hydraulics
        #self._J_dk = self.hydraulics._J_dk            # Index of superjunction downstream of superlink k: this variable has a problem when the index is inversed in superlinks.py
        #self._J_uk = self.hydraulics._J_uk            # Index of superjunction upstream of superlink k: this variable has a problem when the index is inversed in superlinks.py
        self._J_uk_ix = superlinks['sj_0']
        self._J_dk_ix = superlinks['sj_1']
        
        self._I_1k = self.hydraulics._I_1k            # Index of first junction in superlink k
        self._I_Np1k = self.hydraulics._I_Np1k        # Index of last junction in superlink k
        self._i_1k = self.hydraulics._i_1k            # Index of first link in superlink k
        self._i_nk = self.hydraulics._i_nk            # Index of last link in superlink k
        self._N_j = len(self.hydraulics.H_j)          # Number of superjunctions
        self._N_Ik = len(self.hydraulics.h_Ik)        # Number of junctions
        self._N_ik = len(self.hydraulics.Q_ik)        # Number of links
        self._N_uk = len(self.hydraulics.Q_uk)        # Number of upstream superlink
        self._N_dk = len(self.hydraulics.Q_dk)        # Number of downstream superlink
        self.WQ_bc_data = {}
        self.WQ_out_data = {}
        self.met_data = met_data
        self.met_key = met_key
        self.MET_i = 0
        
        ###### WQ parameters and coefficients #########################################
        # All reaction coefficients have unit of "1/day".
        # This part should be moved to the nquality.py with parameter input from the file.
        self.gravity = 9.81
        
        ##################
        wq_para_dic = {}
        wq_para_list = wq_para_new.columns[1:].tolist()

        self.element_N = int(self._N_j + self._N_Ik + self._N_ik + self._N_uk + self._N_dk)

        for wq_para_nm in wq_para_list:
            wq_para_dic[wq_para_nm] = np.zeros(self.element_N)
            wq_para_dic[wq_para_nm][0:self._N_j] = np.array(wq_para_new[wq_para_nm])
            for j in range(self._N_uk):
                wq_para_dic[wq_para_nm][self._N_j+ self._I_1k[j] :self._N_j + self._I_Np1k[j] +1] =  \
                    wq_para_new[wq_para_nm][self._J_dk_ix[j]]*np.ones(self._I_Np1k[j] - self._I_1k[j] + 1)
                wq_para_dic[wq_para_nm][self._N_j+self._N_Ik + self._i_1k[j] :self._N_j+self._N_Ik + self._i_nk[j] +1] =   \
                    wq_para_new[wq_para_nm][self._J_dk_ix[j]]*np.ones(self._i_nk[j] - self._i_1k[j] + 1)
                wq_para_dic[wq_para_nm][self._N_j+self._N_Ik+self._N_ik + j] = \
                    wq_para_new[wq_para_nm][self._J_dk_ix[j]]
                wq_para_dic[wq_para_nm][self._N_j+self._N_Ik+self._N_ik+self._N_uk + j] = \
                    wq_para_new[wq_para_nm][self._J_dk_ix[j]]
        
        # save the parameter arrays for all elements (=self.element_N) in the model
        for key in wq_para_list:
            setattr(self, key, np.array(wq_para_dic[key]))

        # Temperature multipliers for different reactions
        self.gamma_SOD_1 = (1/(self.T2_SOD - self.T1_SOD))*np.log((self.K2_SOD*(1-self.K1_SOD))/(self.K1_SOD*(1-self.K2_SOD)))
        self.gamma_OC_1 = (1/(self.T2_OC - self.T1_OC))*np.log((self.K2_OC*(1-self.K1_OC))/(self.K1_OC*(1-self.K2_OC)))
        self.gamma_ON_1 = (1/(self.T2_ON - self.T1_ON))*np.log((self.K2_ON*(1-self.K1_ON))/(self.K1_ON*(1-self.K2_ON)))
        self.gamma_OP_1 = (1/(self.T2_OP - self.T1_OP))*np.log((self.K2_OP*(1-self.K1_OP))/(self.K1_OP*(1-self.K2_OP)))
        
        self.gamma1_ALG1 = (1/(self.T2_ALG1-self.T1_ALG1))*np.log((self.K2_ALG1*(1-self.K1_ALG1))/(self.K1_ALG1*(1-self.K2_ALG1)))
        self.gamma2_ALG1 = (1/(self.T4_ALG1-self.T3_ALG1))*np.log((self.K3_ALG1*(1-self.K4_ALG1))/(self.K4_ALG1*(1-self.K3_ALG1)))
        self.gamma1_ALG2 = (1/(self.T2_ALG2-self.T1_ALG2))*np.log((self.K2_ALG2*(1-self.K1_ALG2))/(self.K1_ALG2*(1-self.K2_ALG2)))
        self.gamma2_ALG2 = (1/(self.T4_ALG2-self.T3_ALG2))*np.log((self.K3_ALG2*(1-self.K4_ALG2))/(self.K4_ALG2*(1-self.K3_ALG2)))
        self.gamma1_ALG3 = (1/(self.T2_ALG3-self.T1_ALG3))*np.log((self.K2_ALG3*(1-self.K1_ALG3))/(self.K1_ALG3*(1-self.K2_ALG3)))
        self.gamma2_ALG3 = (1/(self.T4_ALG3-self.T3_ALG3))*np.log((self.K3_ALG3*(1-self.K4_ALG3))/(self.K4_ALG3*(1-self.K3_ALG3)))
        self.gamma_NH4_1 = (1/(self.T2_NH4 - self.T1_NH4))*np.log((self.K2_NH4*(1-self.K1_NH4))/(self.K1_NH4*(1-self.K2_NH4)))
        self.gamma_NO3_1 = (1/(self.T2_NO3 - self.T1_NO3))*np.log((self.K2_NO3*(1-self.K1_NO3))/(self.K1_NO3*(1-self.K2_NO3)))
        
        self.P_k = np.zeros((self.hydraulics.M*N_constituent,self.hydraulics.M*N_constituent)) # for initial P_k size in the Kalman filter
        #self.K_k = np.zeros((self.hydraulics.M*N_constituent,self.hydraulics.M*N_constituent))
        #self.H_k = np.zeros((self.hydraulics.M*N_constituent,self.hydraulics.M*N_constituent))
        self.N_constituent = N_constituent
        
        # Divide some groups of model elements by allocated met stations.
        self.index_j = {met_station: [] for met_station in superlinks['met_station'].unique()}
        self.index_j_weir = {met_station: [] for met_station in superlinks['met_station'].unique()}
        self.index_I = {met_station: [] for met_station in superlinks['met_station'].unique()}
        self.index_i = {met_station: [] for met_station in superlinks['met_station'].unique()}
        
        
        # Classify the superjunctions by the met stations. This index also can be used for uk, dk.
        for index, row in superlinks.iterrows():
            self.index_j[row['met_station']].append(index)
        
        for index, row in weirs.iterrows():
            self.index_j_weir[row['met_station']].append(index)
            
        # Classify the internal links by the met stations
        for key_ in met_key:
            for i in range(len(self.index_j[key_])):
                start = hydraulics._I_1k[self.index_j[key_][i]]
                end = hydraulics._I_Np1k[self.index_j[key_][i]]
                indices = np.linspace(start, end, end - start + 1).astype(int).tolist()
                self.index_I[key_].extend(indices)
                
                start = hydraulics._i_1k[self.index_j[key_][i]]
                end = hydraulics._i_nk[self.index_j[key_][i]]
                indices = np.linspace(start, end, end - start + 1).astype(int).tolist()
                self.index_i[key_].extend(indices)
                
        C_all_index = {met_station: [] for met_station in superlinks['met_station'].unique()}
        
        for key_ in met_key:
            C_all_index[key_].extend(self.index_j[key_])
            C_all_index[key_].extend([x + len(superlinks) for x in self.index_j_weir[key_]])
            C_all_index[key_].extend([x + self._N_j for x in self.index_I[key_]])
            C_all_index[key_].extend([x + self._N_j + self._N_Ik for x in self.index_i[key_]])
            C_all_index[key_].extend([x + self._N_j + self._N_Ik + self._N_ik for x in self.index_j[key_]])
            C_all_index[key_].extend([x + self._N_j + self._N_Ik + self._N_ik + self._N_uk for x in self.index_j[key_]])
        self.C_all_index = C_all_index
        #self.T_si = self.T_si_ini*np.ones(self._N_j + self._N_Ik + self._N_ik + self._N_uk + self._N_dk)
        
    def calculate_H_weir(self, Start_J_day, H_weir):
        df = H_weir
        exact_match_row = df[df['J_day'] == Start_J_day]
        now_H_weirs = exact_match_row

        if exact_match_row.empty:
            interpolate_f = {col: interp1d(df['J_day'], df[col], kind='linear', fill_value='extrapolate') for col in df.columns}
            interpolated_values = {col: interpolate_f[col](Start_J_day) for col in df.columns}
            now_H_weirs = pd.DataFrame([interpolated_values])
        else:
            pass
        now_H_weirs = np.array(now_H_weirs.drop('J_day', axis=1))
        return now_H_weirs
    
    def Initialize_Hydraulics(self, SL, Simulation, Start_J_day, End_J_day, dt, u, Q_in_file):
        print('\n▶▶ Started: Hydraulic Model Initialization')
        # input flow rate conversion: Now, this code support only "daily flow rate" input data.
        Q_in_file2 = np.array(Q_in_file.iloc[Start_J_day:End_J_day+1])
        Q_in_days = np.array(np.linspace(start=Start_J_day, stop=End_J_day, num=End_J_day-Start_J_day+1))
        Q_in_days = Q_in_days.reshape(int(End_J_day-Start_J_day+1),1)
        Q_in_file = np.block([Q_in_days, Q_in_file2])
        
        print('▷ [Step 1] Filling the weirs')
        initializing_days = 3
        t_end = initializing_days*86400
        SL.t = 0
        with Simulation(SL, dt=dt, t_end=t_end,
                        interpolation_method='linear') as simulation:
            while SL.t < t_end:
                Q_in = 60
                Q_in = np.array(Q_in)
                simulation.step(dt=dt, u_w=u, Q_in = Q_in)
                simulation.model.reposition_junctions()
                simulation.print_progress()

        print('\n▷ [Step 2] Stabilizing the water levels for all elements with the initial flow rates')
        initializing_days = 20
        t_end = initializing_days*86400
        J_day = Start_J_day
        SL.t = 0
        Q_i = 0
        N_j = self._N_j

        with Simulation(SL, dt=dt, t_end=t_end,
                        interpolation_method='linear') as simulation:
            for _i in range(Q_i,len(Q_in_file)):
                if Q_in_file[_i,0] > J_day:
                    _i = _i-1
                    Q_i = _i
                    break
            Q_in = Q_in_file[_i,1:N_j+1] +( (J_day - Q_in_file[_i,0])*(Q_in_file[_i+1,1:N_j+1] - Q_in_file[_i,1:N_j+1] )
                                           / (Q_in_file[_i+1,0] - Q_in_file[_i,0]) )
            while SL.t < t_end:
                Q_in = np.array(Q_in)
                simulation.step(dt=dt, u_w=u, Q_in = Q_in)
                simulation.model.reposition_junctions()
                simulation.print_progress()

        # calculate the initial volume
        volume_initial = ( (SL.A_sj*(SL.H_j-SL.z_inv_j)).sum()
                         + (SL.A_ik*SL._dx_ik).sum()
                         + (SL._A_SIk*SL.h_Ik).sum()
                         + (SL.A_uk*SL._dx_uk).sum()
                         + (SL.A_dk*SL._dx_dk).sum() )
        print('\n▶▶ Finished : Hydraulic Model Initialization')
        return Q_in_file, volume_initial
    
    def Initialize_Hydraulics_New(self, SL, Simulation, Start_J_day, End_J_day, dt, u, Q_in_file,
                                  now_H_weir, weir_sj, control_K1, control_K2, control_K3):
        print('\n▶▶ Started: Hydraulic Model Initialization')
        # input flow rate conversion: Now, this code support only "daily flow rate" input data.
        Q_in_file2 = np.array(Q_in_file.iloc[Start_J_day:End_J_day+1])
        Q_in_days = np.array(np.linspace(start=Start_J_day, stop=End_J_day, num=End_J_day-Start_J_day+1))
        Q_in_days = Q_in_days.reshape(int(End_J_day-Start_J_day+1),1)
        Q_in_file = np.block([Q_in_days, Q_in_file2])
        
        print('▷ [Step 1] Filling the weirs')
        initializing_days = 2
        t_end = initializing_days*86400
        SL.t = 0
        
        H_error_prev_1 = np.zeros(len(weir_sj))
        H_error_prev_2 = np.zeros(len(weir_sj))
        with Simulation(SL, dt=dt, t_end=t_end,
                        interpolation_method='linear') as simulation:
            while SL.t < t_end:
                
                # control the orifices based on the observed H depth of weirs
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
                
                Q_in = 80
                Q_in = np.array(Q_in)
                simulation.step(dt=dt, u_w=u, Q_in = Q_in)
                simulation.model.reposition_junctions()
                simulation.print_progress()

        print('')
        print('▷ [Step 2] Stabilizing the water levels for all elements with the initial flow rates')
        initializing_days = 20
        t_end = initializing_days*86400
        J_day = Start_J_day
        SL.t = 0
        Q_i = 0
        N_j = self._N_j
        
        H_error_prev = np.zeros(len(weir_sj))
        with Simulation(SL, dt=dt, t_end=t_end,
                        interpolation_method='linear') as simulation:
            for _i in range(Q_i,len(Q_in_file)):
                if Q_in_file[_i,0] > J_day:
                    _i = _i-1
                    Q_i = _i
                    break
            Q_in = Q_in_file[_i,1:N_j+1] +( (J_day - Q_in_file[_i,0])*(Q_in_file[_i+1,1:N_j+1] - Q_in_file[_i,1:N_j+1] )
                                           / (Q_in_file[_i+1,0] - Q_in_file[_i,0]) )
            while SL.t < t_end:
                # control the orifices based on the observed H depth of weirs
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
                
                Q_in = np.array(Q_in)
                simulation.step(dt=dt, u_w=u, Q_in = Q_in)
                simulation.model.reposition_junctions()
                simulation.print_progress()

        # calculate the initial volume
        volume_initial = ( (SL.A_sj*(SL.H_j-SL.z_inv_j)).sum()
                         + (SL.A_ik*SL._dx_ik).sum()
                         + (SL._A_SIk*SL.h_Ik).sum()
                         + (SL.A_uk*SL._dx_uk).sum()
                         + (SL.A_dk*SL._dx_dk).sum() )
        print('')
        print('▶▶ Finished : Hydraulic Model Initialization')
        return Q_in_file, volume_initial, u
    
    def WQ_data_preprocessing(self, WQ_df, WQ_match, Start_J_day, End_J_day):
        print('▷ Preprocessing and converting the WQ input files.')
        WQ_df = WQ_df.fillna(0)
        
        # Data pre-processing for input boundary timeseries
        is_ = WQ_match['classification'] == 'bc_in'
        WQ_data_nm = WQ_match[is_]
        WQ_bc_nm = list(WQ_data_nm['station'])
        WQ_bc_data = {}
        for i in range(len(WQ_bc_nm)):
            is_ = WQ_df['name'] == WQ_bc_nm[i]
            WQ_bc_data[WQ_bc_nm[i]] = WQ_df[is_]
            
            if WQ_bc_data[WQ_bc_nm[i]]['J_day'].iloc[0] > Start_J_day:
                first_data = pd.DataFrame(WQ_bc_data[WQ_bc_nm[i]].iloc[0]).T
                new_data = pd.concat([first_data,  WQ_bc_data[WQ_bc_nm[i]]], ignore_index = True)
                WQ_bc_data[WQ_bc_nm[i]] = new_data
                WQ_bc_data[WQ_bc_nm[i]]['J_day'].iloc[0] = Start_J_day
            else:
                new_data = WQ_bc_data[WQ_bc_nm[i]].reset_index(drop=True)
                WQ_bc_data[WQ_bc_nm[i]] = new_data                
            if WQ_bc_data[WQ_bc_nm[i]]['J_day'].iloc[-1] < End_J_day:
                last_data = pd.DataFrame(WQ_bc_data[WQ_bc_nm[i]].iloc[-1]).T
                new_data = pd.concat([WQ_bc_data[WQ_bc_nm[i]], last_data], ignore_index = True)
                WQ_bc_data[WQ_bc_nm[i]] = new_data
                WQ_bc_data[WQ_bc_nm[i]]['J_day'].iloc[-1] = End_J_day+1
            else:
                new_data = WQ_bc_data[WQ_bc_nm[i]].reset_index(drop=True)
                WQ_bc_data[WQ_bc_nm[i]] = new_data
        
        # data for initial condition
        is_ = WQ_match['classification'] == 'initial_value'
        WQ_data_nm = WQ_match[is_]
        WQ_ic_nm = list(WQ_data_nm['station'])
        WQ_ic_data = {}
        for i in range(len(WQ_ic_nm)):
            is_ = WQ_df['name'] == WQ_ic_nm[i]
            WQ_ic_data[WQ_ic_nm[i]] = WQ_df[is_]
            
            if WQ_ic_data[WQ_ic_nm[i]]['J_day'].iloc[0] > Start_J_day:
                first_data = pd.DataFrame(WQ_ic_data[WQ_ic_nm[i]].iloc[0]).T
                new_data = pd.concat([first_data,  WQ_ic_data[WQ_ic_nm[i]]], ignore_index = True)
                WQ_ic_data[WQ_ic_nm[i]] = new_data
                WQ_ic_data[WQ_ic_nm[i]]['J_day'].iloc[0] = Start_J_day
            else:
                new_data = WQ_ic_data[WQ_ic_nm[i]].reset_index(drop=True)
                WQ_ic_data[WQ_ic_nm[i]] = new_data   
            if WQ_ic_data[WQ_ic_nm[i]]['J_day'].iloc[-1] < End_J_day:
                last_data = pd.DataFrame(WQ_ic_data[WQ_ic_nm[i]].iloc[-1]).T
                new_data = pd.concat([WQ_ic_data[WQ_ic_nm[i]], last_data], ignore_index = True)
                WQ_ic_data[WQ_ic_nm[i]] = new_data
                WQ_ic_data[WQ_ic_nm[i]]['J_day'].iloc[-1] = End_J_day+1
            else:
                new_data = WQ_ic_data[WQ_ic_nm[i]].reset_index(drop=True)
                WQ_ic_data[WQ_ic_nm[i]] = new_data
        
        # data for the data assimilation
        is_ = WQ_match['classification'] == 'data_assimilation'
        WQ_data_nm = WQ_match[is_]
        WQ_da_nm = list(WQ_data_nm['station'])
        WQ_da_data = {}
        for i in range(len(WQ_da_nm)):
            is_ = WQ_df['name'] == WQ_da_nm[i]
            WQ_da_data[WQ_da_nm[i]] = WQ_df[is_]
            
            if WQ_da_nm[i] == "No_DA":
                pass
            else:
                if WQ_da_data[WQ_da_nm[i]]['J_day'].iloc[0] > Start_J_day:
                    first_data = pd.DataFrame(WQ_da_data[WQ_da_nm[i]].iloc[0]).T
                    new_data = pd.concat([first_data,  WQ_da_data[WQ_da_nm[i]]], ignore_index = True)
                    WQ_da_data[WQ_da_nm[i]] = new_data
                    WQ_da_data[WQ_da_nm[i]]['J_day'].iloc[0] = Start_J_day
                else:
                    new_data = WQ_da_data[WQ_da_nm[i]].reset_index(drop=True)
                    WQ_da_data[WQ_da_nm[i]] = new_data
                if WQ_da_data[WQ_da_nm[i]]['J_day'].iloc[-1] < End_J_day:
                    last_data = pd.DataFrame(WQ_da_data[WQ_da_nm[i]].iloc[-1]).T
                    new_data = pd.concat([WQ_da_data[WQ_da_nm[i]], last_data], ignore_index = True)
                    WQ_da_data[WQ_da_nm[i]] = new_data
                    WQ_da_data[WQ_da_nm[i]]['J_day'].iloc[-1] = End_J_day+1        
                else:
                    new_data = WQ_da_data[WQ_da_nm[i]].reset_index(drop=True)
                    WQ_da_data[WQ_da_nm[i]] = new_data

        # data for output and calibration check
        is_ = WQ_match['classification'] == 'obs_cal'
        WQ_data_nm = WQ_match[is_]
        WQ_out_nm = list(WQ_data_nm['station'])
        WQ_SJ_out = list(WQ_data_nm['sj_num'])
        self.WQ_SJ_out = WQ_SJ_out
        WQ_out_data = {}
        for i in range(len(WQ_out_nm)):
            is_ = WQ_df['name'] == WQ_out_nm[i]
            WQ_out_data[WQ_out_nm[i]] = WQ_df[is_]
        
        # For input data conversion, I don't apply the spatial varying parameters. (It's impossible.) They should be constant.
        # So I just use the first number of them
        LPOC_to_TOC = self.LPOC_to_TOC[0]
        RPOC_to_TOC = self.RPOC_to_TOC[0]
        LDOC_to_TOC = self.LDOC_to_TOC[0]
        RDOC_to_TOC = self.RDOC_to_TOC[0]

        L_portion_in_PON = self.L_portion_in_PON[0]
        R_portion_in_PON = self.R_portion_in_PON[0]
        L_portion_in_DON = self.L_portion_in_DON[0]
        R_portion_in_DON = self.R_portion_in_DON[0]
        L_portion_in_POP = self.L_portion_in_POP[0]
        R_portion_in_POP = self.R_portion_in_POP[0]
        L_portion_in_DOP = self.L_portion_in_DOP[0]
        R_portion_in_DOP = self.R_portion_in_DOP[0]

        PIP_to_Particulate_P = self.PIP_to_Particulate_P[0]
        PIN_to_Particulate_N = self.PIN_to_Particulate_N[0]
        #DIC_to_COD = 0.02
        
        # Algae species biomass concentration conversion from chl-a data  : seasonal portions are considered
        for i in range(len(WQ_bc_nm)):
            WQ_bc_data[WQ_bc_nm[i]]['ALG1'] = np.zeros(len(WQ_bc_data[WQ_bc_nm[i]])) #initialize
            WQ_bc_data[WQ_bc_nm[i]]['ALG2'] = np.zeros(len(WQ_bc_data[WQ_bc_nm[i]])) #initialize
            WQ_bc_data[WQ_bc_nm[i]]['ALG3'] = np.zeros(len(WQ_bc_data[WQ_bc_nm[i]])) #initialize
            for j in range(len(WQ_bc_data[WQ_bc_nm[i]])):
                day_ = WQ_bc_data[WQ_bc_nm[i]]['J_day'].iloc[j]
                ALG1_portion, ALG2_portion, ALG3_portion = ALG_ratio_by_period(day_)
                
                ALG1 = WQ_bc_data[WQ_bc_nm[i]]['CHLA'][j] * self.Chl_to_ALG_ratio_1[0] * ALG1_portion
                ALG2 = WQ_bc_data[WQ_bc_nm[i]]['CHLA'][j] * self.Chl_to_ALG_ratio_1[0] * ALG2_portion
                ALG3 = WQ_bc_data[WQ_bc_nm[i]]['CHLA'][j] * self.Chl_to_ALG_ratio_1[0] * ALG3_portion

                WQ_bc_data[WQ_bc_nm[i]]['ALG1'].iloc[j] = ALG1
                WQ_bc_data[WQ_bc_nm[i]]['ALG2'].iloc[j] = ALG2
                WQ_bc_data[WQ_bc_nm[i]]['ALG3'].iloc[j] = ALG3
        
            # Calculate the Organic Carbon constituents    
            LPOC = ( (WQ_bc_data[WQ_bc_nm[i]]['TOC']
                      - (ALG1*self.delta_C_ALG1[0] + ALG2*self.delta_C_ALG2[0] 
                         + ALG3*self.delta_C_ALG3[0]) ) * LPOC_to_TOC )
            RPOC = ( (WQ_bc_data[WQ_bc_nm[i]]['TOC']
                      - (ALG1*self.delta_C_ALG1[0] + ALG2*self.delta_C_ALG2[0] 
                         + ALG3*self.delta_C_ALG3[0]) ) * RPOC_to_TOC )
            LDOC = ( (WQ_bc_data[WQ_bc_nm[i]]['TOC']
                      - (ALG1*self.delta_C_ALG1[0] + ALG2*self.delta_C_ALG2[0] 
                         + ALG3*self.delta_C_ALG3[0]) ) * LDOC_to_TOC )
            RDOC = ( (WQ_bc_data[WQ_bc_nm[i]]['TOC']
                      - (ALG1*self.delta_C_ALG1[0] + ALG2*self.delta_C_ALG2[0] 
                         + ALG3*self.delta_C_ALG3[0]) ) * RDOC_to_TOC )
            #DIC = WQ_bc_data[WQ_bc_nm[i]]['COD'] * DIC_to_COD
            np.where(LPOC<0,0,LPOC)
            np.where(RPOC<0,0,RPOC)
            np.where(LDOC<0,0,LDOC)
            np.where(RDOC<0,0,RDOC)
            #np.where(DIC<0,0,DIC)
            WQ_bc_data[WQ_bc_nm[i]]['LPOC'] = LPOC
            WQ_bc_data[WQ_bc_nm[i]]['RPOC'] = RPOC
            WQ_bc_data[WQ_bc_nm[i]]['LDOC'] = LDOC
            WQ_bc_data[WQ_bc_nm[i]]['RDOC'] = RDOC
            #WQ_bc_data[WQ_bc_nm[i]]['DIC'] = DIC
            
            # Calculate the Organic Nitrogen constituents
            '''
            Org_N = WQ_bc_data[WQ_bc_nm[i]]['TN'] - WQ_bc_data[WQ_bc_nm[i]]['NO3N'] - WQ_bc_data[WQ_bc_nm[i]]['NH4N']
            Particulate_portion = (WQ_bc_data[WQ_bc_nm[i]]['TN'] - WQ_bc_data[WQ_bc_nm[i]]['DTN']) / WQ_bc_data[WQ_bc_nm[i]]['TN']
            Dissolved_portion = 1 - Particulate_portion
            LPON = L_portion_in_PON * (Particulate_portion * Org_N)
            RPON = R_portion_in_PON * (Particulate_portion * Org_N)
            LDON = L_portion_in_DON * (Dissolved_portion * Org_N)
            RDON = R_portion_in_DON * (Dissolved_portion * Org_N)
            '''
            DON = WQ_bc_data[WQ_bc_nm[i]]['DTN'] - WQ_bc_data[WQ_bc_nm[i]]['NO3N'] - WQ_bc_data[WQ_bc_nm[i]]['NH4N']
            ALG_N = (WQ_bc_data[WQ_bc_nm[i]]['ALG1']*self.delta_N_ALG1[0] 
                     + WQ_bc_data[WQ_bc_nm[i]]['ALG2']*self.delta_N_ALG2[0] 
                     + WQ_bc_data[WQ_bc_nm[i]]['ALG3']*self.delta_N_ALG3[0] )
            PON = WQ_bc_data[WQ_bc_nm[i]]['TN'] - WQ_bc_data[WQ_bc_nm[i]]['DTN'] - ALG_N
            LPON = L_portion_in_PON * PON
            RPON = R_portion_in_PON * PON
            LDON = L_portion_in_DON * DON
            RDON = R_portion_in_DON * DON
            
            PIN = PIN_to_Particulate_N * WQ_bc_data[WQ_bc_nm[i]]['TN']
            LPON[LPON<0] = 0
            RPON[RPON<0] = 0
            LDON[LDON<0] = 0
            RDON[RDON<0] = 0
            PIN[PIN<0] = 0
            WQ_bc_data[WQ_bc_nm[i]]['LPON'] = LPON
            WQ_bc_data[WQ_bc_nm[i]]['RPON'] = RPON
            WQ_bc_data[WQ_bc_nm[i]]['LDON'] = LDON
            WQ_bc_data[WQ_bc_nm[i]]['RDON'] = RDON
            WQ_bc_data[WQ_bc_nm[i]]['PIN'] = PIN

            # Calculate the Organic Phosphorus constituents
            '''
            Org_P = WQ_bc_data[WQ_bc_nm[i]]['TP'] - WQ_bc_data[WQ_bc_nm[i]]['PO4P']
            WQ_bc_data[WQ_bc_nm[i]]['TP'] = WQ_bc_data[WQ_bc_nm[i]]['TP'].replace(0,0.00001)
            Particulate_portion = (WQ_bc_data[WQ_bc_nm[i]]['TP'] - WQ_bc_data[WQ_bc_nm[i]]['DTP']) / WQ_bc_data[WQ_bc_nm[i]]['TP']
            Dissolved_portion = 1 - Particulate_portion
            LPOP = L_portion_in_POP * (Particulate_portion * Org_P)
            RPOP = R_portion_in_POP * (Particulate_portion * Org_P)
            LDOP = L_portion_in_DOP * (Dissolved_portion * Org_P)
            RDOP = R_portion_in_DOP * (Dissolved_portion * Org_P)
            '''
            DOP = WQ_bc_data[WQ_bc_nm[i]]['DTP'] - WQ_bc_data[WQ_bc_nm[i]]['PO4P']
            ALG_P = (WQ_bc_data[WQ_bc_nm[i]]['ALG1']*self.delta_P_ALG1[0] 
                     + WQ_bc_data[WQ_bc_nm[i]]['ALG2']*self.delta_P_ALG2[0] 
                     + WQ_bc_data[WQ_bc_nm[i]]['ALG3']*self.delta_P_ALG3[0] )
            POP = WQ_bc_data[WQ_bc_nm[i]]['TP'] - WQ_bc_data[WQ_bc_nm[i]]['DTP'] - ALG_P
            LPOP = L_portion_in_POP * POP
            RPOP = R_portion_in_POP * POP
            LDOP = L_portion_in_DOP * DOP
            RDOP = R_portion_in_DOP * DOP
            
            PIP = PIP_to_Particulate_P * WQ_bc_data[WQ_bc_nm[i]]['TP']
            LPOP[LPOP<0] = 0
            RPOP[RPOP<0] = 0
            LDOP[LDOP<0] = 0
            RDOP[RDOP<0] = 0
            PIP[PIP<0] = 0
            WQ_bc_data[WQ_bc_nm[i]]['LPOP'] = LPOP
            WQ_bc_data[WQ_bc_nm[i]]['RPOP'] = RPOP
            WQ_bc_data[WQ_bc_nm[i]]['LDOP'] = LDOP
            WQ_bc_data[WQ_bc_nm[i]]['RDOP'] = RDOP
            WQ_bc_data[WQ_bc_nm[i]]['PIP'] = PIP
        
        # calculate the unobserved constituents for the initial condition
        for i in range(len(WQ_ic_nm)):
            WQ_ic_data[WQ_ic_nm[i]]['ALG1'] = np.zeros(len(WQ_ic_data[WQ_ic_nm[i]])) #initialize_test
            WQ_ic_data[WQ_ic_nm[i]]['ALG2'] = np.zeros(len(WQ_ic_data[WQ_ic_nm[i]])) #initialize_test
            WQ_ic_data[WQ_ic_nm[i]]['ALG3'] = np.zeros(len(WQ_ic_data[WQ_ic_nm[i]])) #initialize_test
            
            # Algae species biomass concentration from chl-a data  : seasonal portion should be considered later
            for j in range(len(WQ_ic_data[WQ_ic_nm[i]])):
                day_ = WQ_ic_data[WQ_ic_nm[i]]['J_day'].iloc[j]
                ALG1_portion, ALG2_portion, ALG3_portion = ALG_ratio_by_period(day_)
                
                ALG1 = WQ_ic_data[WQ_ic_nm[i]]['CHLA'][j] * self.Chl_to_ALG_ratio_1[0] * ALG1_portion
                ALG2 = WQ_ic_data[WQ_ic_nm[i]]['CHLA'][j] * self.Chl_to_ALG_ratio_2[0] * ALG2_portion
                ALG3 = WQ_ic_data[WQ_ic_nm[i]]['CHLA'][j] * self.Chl_to_ALG_ratio_3[0] * ALG3_portion

                WQ_ic_data[WQ_ic_nm[i]]['ALG1'].iloc[j] = ALG1
                WQ_ic_data[WQ_ic_nm[i]]['ALG2'].iloc[j] = ALG2
                WQ_ic_data[WQ_ic_nm[i]]['ALG3'].iloc[j] = ALG3
            
            # Calculate the Organic Carbon constituents    
            LPOC = WQ_ic_data[WQ_ic_nm[i]]['TOC'] * LPOC_to_TOC
            RPOC = WQ_ic_data[WQ_ic_nm[i]]['TOC'] * RPOC_to_TOC
            LDOC = WQ_ic_data[WQ_ic_nm[i]]['TOC'] * LDOC_to_TOC
            RDOC = WQ_ic_data[WQ_ic_nm[i]]['TOC'] * RDOC_to_TOC
            #DIC = WQ_ic_data[WQ_ic_nm[i]]['COD'] * DIC_to_COD
            np.where(LPOC<0,0,LPOC)
            np.where(RPOC<0,0,RPOC)
            np.where(LDOC<0,0,LDOC)
            np.where(RDOC<0,0,RDOC)
            #np.where(DIC<0,0,DIC)
            WQ_ic_data[WQ_ic_nm[i]]['LPOC'] = LPOC
            WQ_ic_data[WQ_ic_nm[i]]['RPOC'] = RPOC
            WQ_ic_data[WQ_ic_nm[i]]['LDOC'] = LDOC
            WQ_ic_data[WQ_ic_nm[i]]['RDOC'] = RDOC
            #WQ_ic_data[WQ_ic_nm[i]]['DIC'] = DIC
            
            # Calculate the Organic Nitrogen constituents
            '''
            Org_N = WQ_ic_data[WQ_ic_nm[i]]['TN'] - WQ_ic_data[WQ_ic_nm[i]]['NO3N'] - WQ_ic_data[WQ_ic_nm[i]]['NH4N']
            Particulate_portion = (WQ_ic_data[WQ_ic_nm[i]]['TN'] - WQ_ic_data[WQ_ic_nm[i]]['DTN']) / WQ_ic_data[WQ_ic_nm[i]]['TN']
            Dissolved_portion = 1 - Particulate_portion
            LPON = L_portion_in_PON * (Particulate_portion * Org_N)
            RPON = R_portion_in_PON * (Particulate_portion * Org_N)
            LDON = L_portion_in_DON * (Dissolved_portion * Org_N)
            RDON = R_portion_in_DON * (Dissolved_portion * Org_N)
            '''
            DON = WQ_ic_data[WQ_ic_nm[i]]['DTN'] - WQ_ic_data[WQ_ic_nm[i]]['NO3N'] - WQ_ic_data[WQ_ic_nm[i]]['NH4N']
            ALG_N = (WQ_ic_data[WQ_ic_nm[i]]['ALG1']*self.delta_N_ALG1[0] 
                     + WQ_ic_data[WQ_ic_nm[i]]['ALG2']*self.delta_N_ALG2[0] 
                     + WQ_ic_data[WQ_ic_nm[i]]['ALG3']*self.delta_N_ALG3[0] )
            PON = WQ_ic_data[WQ_ic_nm[i]]['TN'] - WQ_ic_data[WQ_ic_nm[i]]['DTN'] - ALG_N
            LPON = L_portion_in_PON * PON
            RPON = R_portion_in_PON * PON
            LDON = L_portion_in_DON * DON
            RDON = R_portion_in_DON * DON
            
            PIN = PIN_to_Particulate_N * WQ_ic_data[WQ_ic_nm[i]]['TN']
            LPON[LPON<0] = 0
            RPON[RPON<0] = 0
            LDON[LDON<0] = 0
            RDON[RDON<0] = 0
            PIN[PIN<0] = 0
            WQ_ic_data[WQ_ic_nm[i]]['LPON'] = LPON
            WQ_ic_data[WQ_ic_nm[i]]['RPON'] = RPON
            WQ_ic_data[WQ_ic_nm[i]]['LDON'] = LDON
            WQ_ic_data[WQ_ic_nm[i]]['RDON'] = RDON
            WQ_ic_data[WQ_ic_nm[i]]['PIN'] = PIN

            # Calculate the Organic Phosphorus constituents
            '''
            Org_P = WQ_ic_data[WQ_ic_nm[i]]['TP'] - WQ_ic_data[WQ_ic_nm[i]]['PO4P']
            WQ_ic_data[WQ_ic_nm[i]]['TP'] = WQ_ic_data[WQ_ic_nm[i]]['TP'].replace(0,0.00001)
            Particulate_portion = (WQ_ic_data[WQ_ic_nm[i]]['TP'] - WQ_ic_data[WQ_ic_nm[i]]['DTP']) / WQ_ic_data[WQ_ic_nm[i]]['TP']
            Dissolved_portion = 1 - Particulate_portion
            LPOP = L_portion_in_POP * (Particulate_portion * Org_P)
            RPOP = R_portion_in_POP * (Particulate_portion * Org_P)
            LDOP = L_portion_in_DOP * (Dissolved_portion * Org_P)
            RDOP = R_portion_in_DOP * (Dissolved_portion * Org_P)
            '''
            DOP = WQ_ic_data[WQ_ic_nm[i]]['DTP'] - WQ_ic_data[WQ_ic_nm[i]]['PO4P']
            ALG_P = (WQ_ic_data[WQ_ic_nm[i]]['ALG1']*self.delta_P_ALG1[0] 
                     + WQ_ic_data[WQ_ic_nm[i]]['ALG2']*self.delta_P_ALG2[0] 
                     + WQ_ic_data[WQ_ic_nm[i]]['ALG3']*self.delta_P_ALG3[0] )
            POP = WQ_ic_data[WQ_ic_nm[i]]['TP'] - WQ_ic_data[WQ_ic_nm[i]]['DTP'] - ALG_P
            LPOP = L_portion_in_POP * POP
            RPOP = R_portion_in_POP * POP
            LDOP = L_portion_in_DOP * DOP
            RDOP = R_portion_in_DOP * DOP
            
            PIP = PIP_to_Particulate_P * WQ_ic_data[WQ_ic_nm[i]]['TP']
            LPOP[LPOP<0] = 0
            RPOP[RPOP<0] = 0
            LDOP[LDOP<0] = 0
            RDOP[RDOP<0] = 0
            PIP[PIP<0] = 0
            WQ_ic_data[WQ_ic_nm[i]]['LPOP'] = LPOP
            WQ_ic_data[WQ_ic_nm[i]]['RPOP'] = RPOP
            WQ_ic_data[WQ_ic_nm[i]]['LDOP'] = LDOP
            WQ_ic_data[WQ_ic_nm[i]]['RDOP'] = RDOP
            WQ_ic_data[WQ_ic_nm[i]]['PIP'] = PIP
        
        self.WQ_bc_data = WQ_bc_data
        self.WQ_ic_data = WQ_ic_data
        self.WQ_da_data = WQ_da_data
        self.WQ_out_data = WQ_out_data
        self.WQ_bc_nm = WQ_bc_nm
        self.WQ_ic_nm = WQ_ic_nm
        self.WQ_da_nm = WQ_da_nm
        self.WQ_out_nm = WQ_out_nm
        self.WQ_SJ_out = WQ_SJ_out
        
        saved_preprocess = {}
        saved_preprocess[0] = WQ_bc_data
        saved_preprocess[1] = WQ_ic_data
        saved_preprocess[2] = WQ_da_data
        saved_preprocess[3] = WQ_out_data
        saved_preprocess[4] = WQ_bc_nm
        saved_preprocess[5] = WQ_ic_nm
        saved_preprocess[6] = WQ_da_nm
        saved_preprocess[7] = WQ_out_nm
        saved_preprocess[8] = WQ_SJ_out
        
        return saved_preprocess
    
    def apply_saved_preprocess(self, saved_preprocess):
        self.WQ_bc_data = saved_preprocess[0]
        self.WQ_ic_data = saved_preprocess[1]
        self.WQ_da_data = saved_preprocess[2]
        self.WQ_out_data = saved_preprocess[3]
        self.WQ_bc_nm = saved_preprocess[4]
        self.WQ_ic_nm = saved_preprocess[5]
        self.WQ_da_nm = saved_preprocess[6]
        self.WQ_out_nm = saved_preprocess[7]
        self.WQ_SJ_out = saved_preprocess[8]
    
    def WQ_data_interpolation_new(self, wq_name, da_obs, Start_date, End_date, Base_date):
        
        def convert_j_day_to_date(j_day, Base_date):
            return Base_date + pd.to_timedelta(j_day, unit='D')
        # interpolation for the input bc water quality data
        test_dic = self.WQ_bc_data
        df_list = []
        for key, df in test_dic.items():
            df['name'] = key
            df_list.append(df)
        
        combined_df = pd.concat(df_list)
        for col in combined_df.columns:
            try:
                combined_df[col] = combined_df[col].astype(float)
            except ValueError:
                pass
        combined_df['date'] = combined_df['J_day'].apply(lambda x: convert_j_day_to_date(x, Base_date))
        
        result_df = pd.DataFrame()
        for name, group in combined_df.groupby('name'):
            group.set_index('date', inplace=True)
            extended_index = pd.date_range(start=Start_date, end=End_date, freq='D')
            group_reindexed = group.reindex(extended_index)
            group_interpolated = group_reindexed.interpolate(method='linear')
            group_interpolated['name'] = name
            result_df = pd.concat([result_df, group_interpolated])
        result_df.reset_index(inplace=True)
        result_df.rename(columns={'index': 'date'}, inplace=True)
        
        WQ_bc_nm_ = self.WQ_bc_nm
        WQ_in_data = {}
        bc_0 = WQ_bc_nm_[0]
        for wq_nm in wq_name:
            WQ_in_data[wq_nm] = result_df[result_df['name'] == bc_0].reset_index(drop=True)
            WQ_in_data[wq_nm] = WQ_in_data[wq_nm]['J_day']
            for bc_nm in WQ_bc_nm_:
                bc_nm_data = result_df[result_df['name'] == bc_nm].reset_index(drop=True)
                wq_bc_data = bc_nm_data[wq_nm]
                WQ_in_data[wq_nm] = pd.concat([WQ_in_data[wq_nm], wq_bc_data], axis = 1, ignore_index=True)
            WQ_in_data[wq_nm] = np.array(WQ_in_data[wq_nm])
        
        # interpolation for water quality observed data for DA
        test_dic = self.WQ_da_data
        df_list = []
        for key, df in test_dic.items():
            df['name'] = key
            df_list.append(df)
        
        combined_df = pd.concat(df_list)
        for col in combined_df.columns:
            try:
                combined_df[col] = combined_df[col].astype(float)
            except ValueError:
                pass
        combined_df['date'] = combined_df['J_day'].apply(lambda x: convert_j_day_to_date(x, Base_date))
        
        result_df = pd.DataFrame()
        for name, group in combined_df.groupby('name'):
            group.set_index('date', inplace=True)
            extended_index = pd.date_range(start=Start_date, end=End_date, freq='D')
            group_reindexed = group.reindex(extended_index)
            group_interpolated = group_reindexed.interpolate(method='linear')
            group_interpolated['name'] = name
            result_df = pd.concat([result_df, group_interpolated])
        result_df.reset_index(inplace=True)
        result_df.rename(columns={'index': 'date'}, inplace=True)
        
        # Build the WQ_in_data Dataframe
        WQ_da_nm_ = self.WQ_da_nm
        observed_data = {}
        for da_nm in WQ_da_nm_:
            if da_nm != 'No_DA':
                da_0 = da_nm
        for wq_nm in da_obs:
            observed_data[wq_nm] = result_df[result_df['name'] == da_0]['J_day'].reset_index(drop=True)
            for da_nm in WQ_da_nm_:
                if da_nm == 'No_DA':
                    observed_da_data = pd.DataFrame(-999 * np.ones((End_date-Start_date).days + 1))
                else:
                    da_nm_data = result_df[result_df['name'] == da_nm].reset_index(drop=True)
                    observed_da_data = da_nm_data[wq_nm]
                observed_data[wq_nm] = pd.concat([observed_data[wq_nm], observed_da_data], axis = 1, ignore_index=True)
            observed_data[wq_nm] = np.array(observed_data[wq_nm])
        
        return WQ_in_data, observed_data
    
    def WQ_data_initial(self, Start_J_day, End_J_day, name):
        dt_step = 86400
        num_steps = int( (End_J_day - Start_J_day)*86400/dt_step + 1)
        WQ_ic_data = self.WQ_ic_data
        WQ_ic_nm = self.WQ_ic_nm
        
        WQ_initial_data = {}
        for i in range(0,len(name)):
            WQ_initial_data[name[i]] = np.zeros((num_steps, self._N_j+1))

        for m in range(int(len(name))):
            J_day = Start_J_day
            for k in range(num_steps):
                c_in = []
                c_N_j_pre = []
                WQ_i = np.zeros(self._N_j)
                c_in.append(J_day)
                for i in range(self._N_j):
                    if i > 0:
                        if WQ_ic_nm[i] == WQ_ic_nm[i-1]:
                            c_in.append(c_N_j_pre)
                        else:    
                            a = WQ_ic_nm[i]
                            for j in range(int(WQ_i[i]), len(WQ_ic_data[a]) ):
                                if WQ_ic_data[a]['J_day'].iloc[j] > J_day:
                                   _j = j - 1
                                   WQ_i[i] = _j
                                   break
                            b = name[m]
                            c_N_j = WQ_ic_data[a][b].iloc[_j] + ( (J_day - WQ_ic_data[a]['J_day'].iloc[_j])*(WQ_ic_data[a][b].iloc[_j+1]
                                      - WQ_ic_data[a][b].iloc[_j] ) / (WQ_ic_data[a]['J_day'].iloc[_j+1] 
                                          - WQ_ic_data[a]['J_day'].iloc[_j] ))   
                            c_in.append(c_N_j)
                            c_N_j_pre = c_N_j
                    else:
                        a = WQ_ic_nm[i]
                        for j in range(int(WQ_i[i]), len(WQ_ic_data[a]) ):
                            if WQ_ic_data[a]['J_day'].iloc[j] > J_day:
                               _j = j - 1
                               WQ_i[i] = _j
                               break
                        b = name[m]
                        c_N_j = WQ_ic_data[a][b].iloc[_j] + ( (J_day - WQ_ic_data[a]['J_day'].iloc[_j])*(WQ_ic_data[a][b].iloc[_j+1]
                                  - WQ_ic_data[a][b].iloc[_j] ) / (WQ_ic_data[a]['J_day'].iloc[_j+1] 
                                      - WQ_ic_data[a]['J_day'].iloc[_j] ))   
                        c_in.append(c_N_j)
                        c_N_j_pre = c_N_j
                WQ_initial_data[name[m]][k] = c_in
                J_day = J_day + dt_step/86400
        return WQ_initial_data
    
    def applying_initial_values(self, name, WQ, WQ_initial, superlinks, out_c_j, out_c_Ik, out_c_ik):
        for i in range(0,len(name)):
            WQ[name[i]]._c_Ik = np.zeros(len(WQ[name[i]].c_Ik))  # zero volume: ignore
            WQ[name[i]]._c_uk = np.zeros(len(WQ[name[i]].c_uk))  # very small volume: ignore
            WQ[name[i]]._c_dk = np.zeros(len(WQ[name[i]].c_dk))  # very small volume: ignore
            out_c_j[name[i]].append(WQ[name[i]].c_j)
            out_c_Ik[name[i]].append(WQ[name[i]].c_Ik)
            out_c_ik[name[i]].append(WQ[name[i]].c_ik)
        
        # Setting the initial concentrations for superjunctions and internal links
        Num_interval = self._i_nk - self._i_1k + 1
        for k in range(len(name)):
            for i_1 in range(self._N_j):
                WQ[name[k]]._c_j[i_1] = WQ_initial[name[k]][0,i_1 + 1]
            for i in range(len(superlinks)-1):
                for j in range(Num_interval[i]):
                    WQ[name[k]].c_ik[self._i_1k[i] + j] = (WQ_initial[name[k]][0,self._J_uk_ix[i]+1]
                                    + (j+0.5)*(WQ_initial[name[k]][0,self._J_dk_ix[i]+1]
                                    - WQ_initial[name[k]][0,self._J_uk_ix[i]+1])/Num_interval[i] )
        # derived constituents: intial value
        out_c_j['TN'].append(WQ['NO3N'].c_j.copy() + WQ['NH4N'].c_j.copy()
                              + WQ['LPON'].c_j.copy() + WQ['LDON'].c_j.copy()
                              + WQ['RPON'].c_j.copy() + WQ['RDON'].c_j.copy()
                              + WQ['PIN'].c_j.copy()
                              + WQ['ALG1'].c_j.copy()*self.delta_N_ALG1[0] 
                              + WQ['ALG2'].c_j.copy()*self.delta_N_ALG2[0]
                              + WQ['ALG3'].c_j.copy()*self.delta_N_ALG3[0]  )
        out_c_j['TOC'].append(WQ['LPOC'].c_j.copy() + WQ['LDOC'].c_j.copy()
                              + WQ['RPOC'].c_j.copy()+ WQ['RDOC'].c_j.copy() 
                              + WQ['ALG1'].c_j.copy()*self.delta_C_ALG1[0] 
                              + WQ['ALG2'].c_j.copy()*self.delta_C_ALG2[0]
                              + WQ['ALG3'].c_j.copy()*self.delta_C_ALG3[0] )
        out_c_j['TP'].append(WQ['PO4P'].c_j.copy() + WQ['LPOP'].c_j.copy() 
                              + WQ['LDOP'].c_j.copy() + WQ['RPOP'].c_j.copy()
                              + WQ['RDOP'].c_j.copy() + WQ['PIP'].c_j.copy()
                              + WQ['ALG1'].c_j.copy()*self.delta_P_ALG1[0] 
                              + WQ['ALG2'].c_j.copy()*self.delta_P_ALG2[0]
                              + WQ['ALG3'].c_j.copy()*self.delta_P_ALG3[0] )
        out_c_j['CHLA'].append(WQ['ALG1'].c_j.copy()/self.Chl_to_ALG_ratio_1[0]
                              + WQ['ALG2'].c_j.copy()/self.Chl_to_ALG_ratio_2[0]
                              + WQ['ALG3'].c_j.copy()/self.Chl_to_ALG_ratio_3[0] )
        
        return WQ, out_c_j, out_c_Ik, out_c_ik
    
    def print_screen_and_save_output(self, dt, WQ, name, screen_output, J_day, Start_J_day, End_J_day, start_time, out_name, out_c_j):
        # save results for post-processing(plotting)
        for i in range(0,len(out_name)):
            out_c_j[out_name[i]].append(WQ[out_name[i]].c_j.copy())
            
        # calculate the derived constituents
        temp_value = {}
        temp_value['TN'] = (WQ['NO3N'].c_j.copy() + WQ['NH4N'].c_j.copy()
                             + WQ['LPON'].c_j.copy() + WQ['LDON'].c_j.copy()
                             + WQ['RPON'].c_j.copy() + WQ['RDON'].c_j.copy()
                             + WQ['PIN'].c_j.copy()
                             + WQ['ALG1'].c_j.copy()*self.delta_N_ALG1[0] 
                             + WQ['ALG2'].c_j.copy()*self.delta_N_ALG2[0]
                             + WQ['ALG3'].c_j.copy()*self.delta_N_ALG3[0] 
                             )
        out_c_j['TN'].append(temp_value['TN'])
        
        temp_value['TOC'] = (WQ['LPOC'].c_j.copy() + WQ['LDOC'].c_j.copy()
                             + WQ['RPOC'].c_j.copy()+ WQ['RDOC'].c_j.copy() 
                             + WQ['ALG1'].c_j.copy()*self.delta_C_ALG1[0] 
                             + WQ['ALG2'].c_j.copy()*self.delta_C_ALG2[0]
                             + WQ['ALG3'].c_j.copy()*self.delta_C_ALG3[0] 
                             )
        out_c_j['TOC'].append(temp_value['TOC'])
        
        temp_value['TP'] = (WQ['PO4P'].c_j.copy() + WQ['LPOP'].c_j.copy() 
                             + WQ['LDOP'].c_j.copy() + WQ['RPOP'].c_j.copy()
                             + WQ['RDOP'].c_j.copy() + WQ['PIP'].c_j.copy()
                             + WQ['ALG1'].c_j.copy()*self.delta_P_ALG1[0] 
                             + WQ['ALG2'].c_j.copy()*self.delta_P_ALG2[0]
                             + WQ['ALG3'].c_j.copy()*self.delta_P_ALG3[0] 
                             )
        out_c_j['TP'].append(temp_value['TP'])
        
        temp_value['CHLA'] = (WQ['ALG1'].c_j.copy()/self.Chl_to_ALG_ratio_1[0]
                             + WQ['ALG2'].c_j.copy()/self.Chl_to_ALG_ratio_2[0]
                             + WQ['ALG3'].c_j.copy()/self.Chl_to_ALG_ratio_3[0]
                             )
        out_c_j['CHLA'].append(temp_value['CHLA'])
        
        name2 = ['TN', 'TP', 'TOC', 'CHLA']
        
        if screen_output == 1:
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f'J_Day = {J_day:.3f}')
            print("Total progress = " , f'{100*(J_day - Start_J_day)/(End_J_day - Start_J_day) : 5.2f} %')
            print("Total elapsed time = ", f'{time.time() - start_time : 5.1f} sec')
            print("Applied time step =", f'{dt: 5.1f} sec')
            print("------------------------------------------------------------------------------------------------")
            print("▶▶ Water Quality constituents")
            for i in range(0,len(name)):
                print( f'{name[i] : >5} \t = {WQ[name[i]].c_j[16] : 9.3f}{WQ[name[i]].c_j[20] : 9.3f}{WQ[name[i]].c_j[23] : 9.3f}',
                      f'{WQ[name[i]].c_j[28] : 9.3f}{WQ[name[i]].c_j[32] : 9.3f}{WQ[name[i]].c_j[36] : 9.3f}',
                      f'{WQ[name[i]].c_j[42] : 9.3f}{WQ[name[i]].c_j[53] : 9.3f}{WQ[name[i]].c_j[67] : 9.3f}' )
            
            for i in range(0,len(name2)):
                print( f'{name2[i] : >5} \t = {temp_value[name2[i]][16] : 9.3f}{temp_value[name2[i]][20] : 9.3f}{temp_value[name2[i]][23] : 9.3f}',
                      f'{temp_value[name2[i]][28] : 9.3f}{temp_value[name2[i]][32] : 9.3f}{temp_value[name2[i]][36] : 9.3f}',
                      f'{temp_value[name2[i]][42] : 9.3f}{temp_value[name2[i]][53] : 9.3f}{temp_value[name2[i]][67] : 9.3f}' )    
            
            print("▶▶ Hydraulics")
            print('Q_weir', f'\t = {self.hydraulics.Q_w[0] : 9.2f}{self.hydraulics.Q_w[1] : 9.2f}{self.hydraulics.Q_w[2] : 9.2f}', 
                  f'{self.hydraulics.Q_w[3] : 9.2f}{self.hydraulics.Q_w[4] : 9.2f}{self.hydraulics.Q_w[5] : 9.2f}',
                  f'{self.hydraulics.Q_w[6] : 9.2f}{self.hydraulics.Q_w[7] : 9.2f}{self.hydraulics.Q_w[8] : 9.2f}' )
            print('H_weir', f'\t = {self.hydraulics.H_j[16] : 9.2f}{self.hydraulics.H_j[20] : 9.2f}{self.hydraulics.H_j[23] : 9.2f}', 
                  f'{self.hydraulics.H_j[28] : 9.2f}{self.hydraulics.H_j[32] : 9.2f}{self.hydraulics.H_j[36] : 9.2f}',
                  f'{self.hydraulics.H_j[42] : 9.2f}{self.hydraulics.H_j[53] : 9.2f}{self.hydraulics.H_j[67] : 9.2f}' )
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return out_c_j
    
    def print_WQ_output(self, WQ_weir_out_nm, out_J_day, out_c_j, data_path):
        p_J_day = np.array(out_J_day)
        font = font_manager.FontProperties(family='Arial', style='normal', size=15)
        WQ_data = self.WQ_out_data
        WQ_SJ_out = self.WQ_SJ_out[5:13]
        
        ax = ['ax1','ax2','ax3','ax4','ax5','ax6','ax7','ax8']
        plot_nm = ['Temp', 'ISS', 'DO', 'CHLA', 'TN', 'NO3N', 'NH4N', 'TP', 'PO4P', 'TOC']
        obs_col_num = [2, 15, 3, 6, 7, 12, 11, 8, 14, 9]
        plot_ylim = [50, 50, 20, 100, 8, 8, 0.5, 0.2, 0.2, 10]
        
        for wq_i_ in range(len(plot_nm)):
            out_Conc = np.array(out_c_j[plot_nm[wq_i_]]) 
            fig, ((ax[0], ax[1]), (ax[2], ax[3]), (ax[4],ax[5]), (ax[6], ax[7])) = plt.subplots(4, 2, figsize=(22,16))
            for i in range(len(WQ_weir_out_nm)):
                measure = np.array(WQ_data[WQ_weir_out_nm[i]])
                p_Conc = out_Conc[0:len(out_J_day),int(WQ_SJ_out[i])]
                ax[i].scatter(measure[:,1],measure[:,obs_col_num[wq_i_]], color = 'b', alpha = 0.5, s = 50, label= 'Measurment')
                ax[i].plot(p_J_day, p_Conc, c = 'black', alpha = 0.5, label='EufoRiA')
                ax[i].set_ylabel(f'{plot_nm[wq_i_]}', font=font)
                ax[i].set_xlabel('Julian day', font=font)
                ax[i].set_ylim(0,plot_ylim[wq_i_])
                ax[i].set_title(WQ_weir_out_nm[i], font=font)
                ax[i].legend()
                ax[i].grid(alpha=0.35)
            fig.tight_layout()
            plt.savefig(f'{data_path}/output_pic1_{plot_nm[wq_i_]}.jpg', dpi=180)
            plt.savefig(f'{data_path}/output_pic1_{plot_nm[wq_i_]}.pdf')
            plt.show()
            
    def print_WQ_output_2(self, WQ_weir_out_nm, out_J_day, out_c_j, SJ_out_no, data_path):
        
        p_J_day = np.array(out_J_day)
        font = font_manager.FontProperties(family='Arial', style='normal', size=15)
        WQ_data = self.WQ_out_data
        WQ_SJ_out = [self.WQ_SJ_out[SJ_out_no]]
        
        plot_nm = ['Temp', 'DO', 'CHLA', 'TN', 'TP',  'TOC']
        obs_col_num = [2, 3, 6, 7,  8, 9]
        plot_ylim = [50, 20, 100, 8,  0.2, 10]
    
        for i in range(len(WQ_weir_out_nm)):
            n_rows = (len(plot_nm) + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3.5 * n_rows))
            axes = axes.flatten()
            
            for wq_i_ in range(len(plot_nm)):
                if wq_i_ < len(axes):
                    ax = axes[wq_i_]
                    measure = np.array(WQ_data[WQ_weir_out_nm[i]])
                    out_Conc = np.array(out_c_j[plot_nm[wq_i_]])
                    p_Conc = out_Conc[0:len(out_J_day), int(WQ_SJ_out[i])]
                    
                    ax.scatter(measure[:, 1], measure[:, obs_col_num[wq_i_]], color='b', alpha=0.4, s=50, label='Measurement')
                    ax.plot(p_J_day, p_Conc, c='black', alpha=0.7, label='EufoRiA')
                    
                    ax.set_ylabel(f'{plot_nm[wq_i_]}', font=font)
                    ax.set_xlabel('Julian day', font=font)
                    ax.set_ylim(0, plot_ylim[wq_i_])
                    ax.set_title(f'{plot_nm[wq_i_]}', font=font)
                    ax.legend()
                    ax.grid(alpha=0.35)
            
            for j in range(len(plot_nm), len(axes)):
                fig.delaxes(axes[j])
            
            fig.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            plt.savefig(f'{data_path}/output_pic1_station_{WQ_weir_out_nm[i]}.jpg', dpi=180)
            plt.savefig(f'{data_path}/output_pic1_station_{WQ_weir_out_nm[i]}.pdf')
            plt.show()
            
    def print_WQ_output_new(self, WQ_weir_out_nm, out_J_day, out_c_j, data_path, compare_hour):
        font = font_manager.FontProperties(family='Arial', style='normal', size=15)
        WQ_data = self.WQ_out_data
        for out_nm in WQ_weir_out_nm:
            WQ_data[out_nm]['TP'].replace(0, np.nan, inplace=True)
            WQ_data[out_nm]['TP'].fillna(method='ffill', inplace=True)
            
        WQ_SJ_out = self.WQ_SJ_out[5:13]
        
        ax = ['ax1','ax2','ax3','ax4','ax5','ax6','ax7','ax8']
        plot_nm = ['Temp', 'ISS', 'DO', 'CHLA', 'TN', 'NO3N', 'NH4N', 'TP', 'PO4P', 'TOC']
        obs_col_num = [2, 15, 3, 6, 7, 12, 11, 8, 14, 9]
        plot_xy_max = [35, 50, 20, 100, 5, 4, 0.5, 0.1,  8]
        
        for wq_i_ in range(len(plot_nm)):
            out_Conc = np.array(out_c_j[plot_nm[wq_i_]]) 
            fig, ((ax[0], ax[1], ax[2], ax[3]), (ax[4],ax[5], ax[6], ax[7])) = plt.subplots(2, 4, figsize=(16,8))
            for i in range(len(WQ_weir_out_nm)):
                measure = np.array(WQ_data[WQ_weir_out_nm[i]])
                j_ = []
                for j in range(len(measure)):
                    for k in range(len(out_J_day)):
                        if out_J_day[k] > measure[j,1] + compare_hour:
                            j_.append(k)
                            break
                        else:
                            pass
                out_Conc2 = out_Conc[j_]
                p_Conc = np.array(out_Conc2[:,int(WQ_SJ_out[i])])
                y_true = np.array(measure[:,obs_col_num[wq_i_]] , dtype='float64') 
                rmse = np.sqrt(mean_squared_error(y_true, p_Conc))

                # Calculating NSE
                NSE = 1 - np.sum( (p_Conc - y_true)**2 ) / np.sum( 
                                  (y_true - np.mean(y_true))**2)
                # Calculating KGE
                KGE = 1 - np.sqrt( (np.corrcoef(p_Conc, y_true)[0,1] - 1)**2 
                                  + (np.std(p_Conc)/np.std(y_true) - 1)**2
                                  + (np.mean(p_Conc)/np.mean(y_true) - 1)**2  )
                # Calculating PBIAS
                PBIAS = 100*np.sum(y_true - p_Conc)/np.sum(y_true)
                
                xy_max = plot_xy_max[wq_i_]
                ax[i].plot([0, xy_max], [0, xy_max], 'k--')
                ax[i].scatter(y_true, p_Conc, color = 'b', alpha = 0.5, s = 50, label= 'Measurment')
                ax[i].text(0.05*xy_max, 0.90*xy_max, f'▶ NSE: {NSE: 3.3f}', fontsize = 12)
                ax[i].text(0.05*xy_max, 0.82*xy_max, f'▶ KGE: {KGE: 3.3f}', fontsize = 12)
                ax[i].text(0.05*xy_max, 0.74*xy_max, f'▶ RMSE: {rmse: 3.3f}', fontsize = 12)
                ax[i].text(0.05*xy_max, 0.66*xy_max, f'▶ PBIAS: {PBIAS: 3.1f}%', fontsize = 12)
                ax[i].set_xlabel('Observation', font=font)
                ax[i].set_ylabel('Simulation', font=font)
                ax[i].set_xlim(0,xy_max)
                ax[i].set_ylim(0,xy_max)
                ax[i].set_title(WQ_weir_out_nm[i], font=font)
                ax[i].grid(alpha=0.35)
            fig.suptitle(f'{plot_nm[wq_i_]}: observation vs. simulation', fontsize=17)
            fig.tight_layout()
            plt.savefig(f'{data_path}/output_pic2_{plot_nm[wq_i_]}.jpg', dpi=180)
            plt.savefig(f'{data_path}/output_pic2_{plot_nm[wq_i_]}.pdf')
            plt.show()
    
    def print_WQ_output_new2(self, WQ_weir_out_nm, out_J_day, out_c_j, SJ_out_no, data_path, compare_hour):
        
        font = font_manager.FontProperties(family='Arial', style='normal', size=15)
        WQ_data = self.WQ_out_data
        for out_nm in WQ_weir_out_nm:
            WQ_data[out_nm]['TP'].replace(0, np.nan, inplace=True)
            WQ_data[out_nm]['TP'].fillna(method='ffill', inplace=True)
            
        WQ_SJ_out = [self.WQ_SJ_out[SJ_out_no]]
        
        plot_nm = ['Temp', 'DO', 'CHLA', 'TN',  'TP', 'TOC']
        obs_col_num = [2, 3, 6, 7, 8, 9]
        plot_xy_max = [35, 20, 100, 5,  0.1,  8]
        
        for i in range(len(WQ_weir_out_nm)):
            n_rows = 2
            n_cols = 3
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
            axes = axes.flatten()
            
            for wq_i_ in range(len(plot_nm)):
                if wq_i_ < len(axes):
                    ax = axes[wq_i_]
                    
                    out_Conc = np.array(out_c_j[plot_nm[wq_i_]])
                    measure = np.array(WQ_data[WQ_weir_out_nm[i]])
                    
                    j_ = []
                    for j in range(len(measure)):
                        for k in range(len(out_J_day)):
                            if out_J_day[k] > measure[j,1] + compare_hour:
                                j_.append(k)
                                break
                            else:
                                pass
                    
                    out_Conc2 = out_Conc[j_]
                    p_Conc = np.array(out_Conc2[:,int(WQ_SJ_out[i])])
                    y_true = np.array(measure[:,obs_col_num[wq_i_]], dtype='float64')
                    
                    rmse = np.sqrt(mean_squared_error(y_true, p_Conc))
                    NSE = 1 - np.sum((p_Conc - y_true)**2) / np.sum(
                                 (y_true - np.mean(y_true))**2)
                    KGE = 1 - np.sqrt((np.corrcoef(p_Conc, y_true)[0,1] - 1)**2 
                                    + (np.std(p_Conc)/np.std(y_true) - 1)**2
                                    + (np.mean(p_Conc)/np.mean(y_true) - 1)**2)
                    PBIAS = 100*np.sum(y_true - p_Conc  )/np.sum(y_true)
                    
                    xy_max = plot_xy_max[wq_i_]
                    ax.plot([0, xy_max], [0, xy_max], 'k--')
                    ax.scatter(y_true, p_Conc, color='b', alpha=0.3, s=50, label='Measurement')
                    
                    ax.text(0.05*xy_max, 0.90*xy_max, f'▶ NSE: {NSE: 3.3f}', fontsize=10)
                    ax.text(0.05*xy_max, 0.82*xy_max, f'▶ KGE: {KGE: 3.3f}', fontsize=10)
                    ax.text(0.05*xy_max, 0.74*xy_max, f'▶ RMSE: {rmse: 3.3f}', fontsize=10)
                    ax.text(0.05*xy_max, 0.66*xy_max, f'▶ PBIAS: {PBIAS: 3.1f}%', fontsize=10)
                    
                    ax.set_xlabel('Observation', fontsize=12)
                    ax.set_ylabel('Simulation', fontsize=12)
                    ax.set_xlim(0, xy_max)
                    ax.set_ylim(0, xy_max)
                    ax.set_title(f'{plot_nm[wq_i_]}', fontsize=12)
                    ax.grid(alpha=0.35)
        
        for j in range(len(plot_nm), len(axes)):
            fig.delaxes(axes[j])
        
        fig.tight_layout()
        
        plt.savefig(f'{data_path}/output_pic2_station_{WQ_weir_out_nm[i]}.jpg', dpi=180)
        plt.savefig(f'{data_path}/output_pic2_station_{WQ_weir_out_nm[i]}.pdf')
        plt.show() 
    
    def print_WQ_combined_output(self, WQ_weir_out_nm, out_J_day, out_c_j, data_path, SJ_out_no, compare_hour, base_year):

        import matplotlib.font_manager as font_manager
        from sklearn.metrics import mean_squared_error
        print(WQ_weir_out_nm)
        
        p_J_day = np.array(out_J_day)
        font = font_manager.FontProperties(family='Arial', style='normal', size=14)
        WQ_data = self.WQ_out_data
        
        # Replace zeros with NaN and forward fill for TP
        for out_nm in WQ_weir_out_nm:
            WQ_data[out_nm]['TP'].replace(0, np.nan, inplace=True)
            WQ_data[out_nm]['TP'].fillna(method='ffill', inplace=True)
        
        WQ_SJ_out = [self.WQ_SJ_out[SJ_out_no]]
        
        plot_nm = ['Temp', 'DO', 'CHLA', 'TN', 'TP', 'TOC']
        abc = ['a)', 'b)','c)','d)','e)','f)',]
        obs_col_num = [2, 3, 6, 7, 8, 9]
        plot_ylim = [50, 20, 100, 8, 0.2, 10]
        plot_xy_max = [35, 20, 100, 5, 0.1, 8]
        
        for i in range(len(WQ_weir_out_nm)):
            n_cols = 3  # Number of columns for both graphs
            
            # Create a figure with 2 rows of subplots (top: time series, bottom: 1:1 plots)
            fig = plt.figure(figsize=(16, 18))
            
            height_ratios = [0.05, 2, 2, 0.05, 2, 2] 
            
            # Define GridSpec for more control over subplot layout with specified height ratios
            gs = fig.add_gridspec(6, n_cols, height_ratios=height_ratios, hspace=0.4) 
            
            title_ax1 = fig.add_subplot(gs[0, :])
            title_ax1.text(0.5, 0.5, 'Time Series', fontsize=16, fontweight='bold', 
                          ha='center', va='center', transform=title_ax1.transAxes)
            title_ax1.axis('off')
            
            title_ax2 = fig.add_subplot(gs[3, :])
            title_ax2.text(0.5, 0.5, 'Observation vs. Simulation', fontsize=16, fontweight='bold',
                          ha='center', va='center', transform=title_ax2.transAxes)
            title_ax2.axis('off')
            
            # Loop through water quality parameters for time series plots (top half)
            for wq_i_ in range(len(plot_nm)):
                # Create subplot in the top section
                if wq_i_ < n_cols:
                    row_idx = 1
                else:
                    row_idx = 2
                col_idx = wq_i_ % n_cols
                
                # Create time series plot
                ax_ts = fig.add_subplot(gs[row_idx, col_idx])
                
                measure = np.array(WQ_data[WQ_weir_out_nm[i]])
                out_Conc = np.array(out_c_j[plot_nm[wq_i_]])
                p_Conc = out_Conc[0:len(out_J_day), int(WQ_SJ_out[i])]
                
                ax_ts.scatter(measure[:, 1], measure[:, obs_col_num[wq_i_]], 
                             color='b', alpha=0.4, s=50, label='Measurement')
                ax_ts.plot(p_J_day, p_Conc, c='black', alpha=0.7, label='EufoRiA')
                
                ax_ts.set_ylabel(f'{plot_nm[wq_i_]}', font=font)
                ax_ts.set_xlabel('Time (days)', font=font)
                ax_ts.set_ylim(0, plot_ylim[wq_i_])
                ax_ts.set_title(f'{abc[wq_i_]} {plot_nm[wq_i_]}', font=font)
                ax_ts.legend()
                ax_ts.grid(alpha=0.35)
                ax_ts.tick_params(axis='both', which='major', labelsize=12)
            
            # Loop through water quality parameters for 1:1 comparison plots (bottom half)
            for wq_i_ in range(len(plot_nm)):
                # Create subplot in the bottom section
                if wq_i_ < n_cols:
                    row_idx = 4
                else:
                    row_idx = 5
                col_idx = wq_i_ % n_cols
                
                # Create 1:1 comparison plot
                ax_comp = fig.add_subplot(gs[row_idx, col_idx])
                
                out_Conc = np.array(out_c_j[plot_nm[wq_i_]])
                measure = np.array(WQ_data[WQ_weir_out_nm[i]])
                
                j_ = []
                for j in range(len(measure)):
                    for k in range(len(out_J_day)):
                        if out_J_day[k] > measure[j,1] + compare_hour:
                            j_.append(k)
                            break
                        else:
                            pass
                
                out_Conc2 = out_Conc[j_]
                p_Conc = np.array(out_Conc2[:,int(WQ_SJ_out[i])])
                y_true = np.array(measure[:,obs_col_num[wq_i_]], dtype='float64')
                
                # Calculate statistics
                rmse = np.sqrt(mean_squared_error(y_true, p_Conc))
                NSE = 1 - np.sum((p_Conc - y_true)**2) / np.sum(
                             (y_true - np.mean(y_true))**2)
                KGE = 1 - np.sqrt((np.corrcoef(p_Conc, y_true)[0,1] - 1)**2 
                                + (np.std(p_Conc)/np.std(y_true) - 1)**2
                                + (np.mean(p_Conc)/np.mean(y_true) - 1)**2)
                PBIAS = 100*np.sum(y_true - p_Conc )/np.sum(y_true)
                
                xy_max = plot_xy_max[wq_i_]
                ax_comp.plot([0, xy_max], [0, xy_max], 'k--')
                ax_comp.scatter(y_true, p_Conc, color='b', alpha=0.3, s=50, label='Measurement')
                
                ax_comp.text(0.05*xy_max, 0.90*xy_max, f'▶ NSE: {NSE: 3.3f}', font=font)
                ax_comp.text(0.05*xy_max, 0.82*xy_max, f'▶ KGE: {KGE: 3.3f}', font=font)
                ax_comp.text(0.05*xy_max, 0.74*xy_max, f'▶ RMSE: {rmse: 3.3f}', font=font)
                ax_comp.text(0.05*xy_max, 0.66*xy_max, f'▶ PBIAS: {PBIAS: 3.1f}%', font=font)
                
                ax_comp.set_xlabel('Observation', font=font)
                ax_comp.set_ylabel('Simulation', font=font)
                ax_comp.set_xlim(0, xy_max)
                ax_comp.set_ylim(0, xy_max)
                ax_comp.set_title(f'{abc[wq_i_]} {plot_nm[wq_i_]}', font=font)
                ax_comp.grid(alpha=0.35)
                ax_comp.tick_params(axis='both', which='major', labelsize=12)
            
            plt.subplots_adjust(top=0.97, bottom=0.05, hspace=0.5, wspace=0.25)
            
            #plt.savefig(f'{data_path}/out_CH_{WQ_weir_out_nm[i]}.jpg', dpi=180)
            plt.savefig(f'{data_path}/out_{WQ_weir_out_nm[i]}_all.pdf', bbox_inches='tight', pad_inches=0.1)
            plt.show()
            
            self._create_monthly_boxplot_comparison(WQ_weir_out_nm[i], out_J_day, out_c_j, 
                                                   WQ_data, plot_nm, obs_col_num, int(WQ_SJ_out[i]), 
                                                   data_path, base_year)
            
    def _create_monthly_boxplot_comparison(self, weir_name, out_J_day, out_c_j, WQ_data, 
                                          plot_nm, obs_col_num, station_idx, 
                                          data_path, base_year):

        from datetime import datetime, timedelta
        import matplotlib.font_manager as font_manager
        import seaborn as sns
        
        font = font_manager.FontProperties(family='Arial', style='normal', size=14)
        
        start_date = datetime(base_year, 1, 1)
        model_dates = [start_date + timedelta(days=int(day)) for day in out_J_day]
        
        model_year_months = [f"{date.year}-{date.month:02d}" for date in model_dates]
        
        measure = np.array(WQ_data[weir_name])
        measure_dates = [start_date + timedelta(days=int(day)) for day in measure[:, 1]]
        measure_year_months = [f"{date.year}-{date.month:02d}" for date in measure_dates]
        
        for param_idx, param_name in enumerate(plot_nm):
            out_Conc = np.array(out_c_j[param_name])
            model_values = out_Conc[0:len(out_J_day), int(station_idx)]
            
            obs_values = measure[:, obs_col_num[param_idx]]
            
            model_df = pd.DataFrame({
                'YearMonth': model_year_months,
                'Value': model_values,
                'Type': 'EufoRiA',
                'Year': [date.year for date in model_dates],
                'Month': [date.month for date in model_dates]
            })
            
            obs_df = pd.DataFrame({
                'YearMonth': measure_year_months,
                'Value': obs_values,
                'Type': 'Measurement',
                'Year': [date.year for date in measure_dates],
                'Month': [date.month for date in measure_dates]
            })
            
            combined_df = pd.concat([model_df, obs_df])
            
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            combined_df['Month_Name'] = combined_df['Month'].map(month_names)
            combined_df['YearMonth_Name'] = combined_df['Year'].astype(str) + '-' + combined_df['Month_Name']
            combined_df['Sort_Order'] = combined_df['Year'] * 100 + combined_df['Month']
            combined_df = combined_df.sort_values('Sort_Order')
            
            plt.figure(figsize=(16, 8))
            ax = sns.boxplot(x='YearMonth_Name', y='Value', hue='Type', data=combined_df,
                            palette={'EufoRiA': '#BEBEBE', 'Measurement': '#1E90FF'})
            
            plt.title(f'Monthly Comparison of {param_name} at {weir_name}', fontsize=16)
            plt.xlabel('Year-Month', fontsize=14)
            plt.ylabel(f'{param_name}', fontsize=14)
            
            plt.xticks(rotation=270, ha='center', va='top')
            
            for i, tick in enumerate(ax.get_xticklabels()):
                if i > 0:
                    current_month = tick.get_text().split('-')[1]
                    prev_month = ax.get_xticklabels()[i-1].get_text().split('-')[1]
                    if current_month != prev_month:
                        ax.axvline(x=i-0.5, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            plt.grid(alpha=0.3, axis='y')
            plt.legend()
            
            plt.tight_layout()
            #plt.savefig(f'{data_path}/box_{weir_name}_{param_name}.jpg', dpi=180)
            plt.savefig(f'{data_path}/box_{weir_name}_{param_name}.pdf', bbox_inches='tight', pad_inches=0.1)
            plt.show()
            
            monthly_stats = combined_df.groupby(['YearMonth_Name', 'Type'])['Value'].agg(['mean', 'std', 'min', 'max']).reset_index()
            monthly_stats.to_csv(f'{data_path}/monthly_stats_{weir_name}_{param_name}.csv', index=False)
            
        plt.figure(figsize=(20, 15))
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        for param_idx, param_name in enumerate(plot_nm):
            row = param_idx // 2
            col = param_idx % 2
            
            out_Conc = np.array(out_c_j[param_name])
            model_values = out_Conc[0:len(out_J_day), int(station_idx)]            

            obs_values = measure[:, obs_col_num[param_idx]]
            
            model_df = pd.DataFrame({
                'YearMonth': model_year_months,
                'Value': model_values,
                'Type': 'EufoRiA',
                'Year': [date.year for date in model_dates],
                'Month': [date.month for date in model_dates]
            })
            
            obs_df = pd.DataFrame({
                'YearMonth': measure_year_months,
                'Value': obs_values,
                'Type': 'Measurement',
                'Year': [date.year for date in measure_dates],
                'Month': [date.month for date in measure_dates]
            })
            
            combined_df = pd.concat([model_df, obs_df])
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            combined_df['Month_Name'] = combined_df['Month'].map(month_names)
            
            combined_df['YearMonth_Name'] = combined_df['Year'].astype(str) + '-' + combined_df['Month_Name']
            combined_df['Sort_Order'] = combined_df['Year'] * 100 + combined_df['Month']
            combined_df = combined_df.sort_values('Sort_Order')
            
            sns.boxplot(x='YearMonth_Name', y='Value', hue='Type', data=combined_df,
                       palette={'EufoRiA': '#BEBEBE', 'Measurement': '#1E90FF'}, ax=axes[row, col])
            
            axes[row, col].set_title(f'{param_name}', fontsize=14)
            axes[row, col].set_xlabel('Year-Month', fontsize=12)
            axes[row, col].set_ylabel(f'{param_name}', fontsize=12)
            axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), 
                                          rotation=270, ha='center', va='top')
            
            for i, tick in enumerate(axes[row, col].get_xticklabels()):
                if i > 0:
                    current_month = tick.get_text().split('-')[1]
                    prev_month = axes[row, col].get_xticklabels()[i-1].get_text().split('-')[1]
                    if current_month != prev_month:
                        axes[row, col].axvline(x=i-0.5, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            axes[row, col].grid(alpha=0.3, axis='y')
            axes[row, col].legend()
        
        #plt.suptitle(f'Monthly Comparison of All Parameters at {weir_name}', fontsize=18)
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.subplots_adjust(bottom=0.2)
        #plt.savefig(f'{data_path}/box_{weir_name}_all.jpg', dpi=180)
        plt.savefig(f'{data_path}/box_{weir_name}_all.pdf', bbox_inches='tight', pad_inches=0.1)
        plt.show()
    
    def WQ_reactions(self, dt, WQ, name, J_day):
        # interpolate the MET data for the current time from hourly data
        MET_i = self.MET_i
        met_key = self.met_key
        for _i in range(MET_i, len(self.met_data[self.met_key[0]])):
            if self.met_data[self.met_key[0]][_i,0] > J_day:
                _i = _i -1
                MET_i = _i
                break
        MET_in_all = np.zeros((len(self.met_key),5))
        # interpolate the meteorological data at the current time (J_day) from the daily timeseries
        st_N = 0
        for key_ in met_key:
            MET_in_all[st_N] = self.met_data[key_][MET_i,1:6]
            #MET_in[st_N] = MET_data[key_][MET_i,1:6] +( (J_day - MET_data[key_][MET_i,0])*(MET_data[key_][MET_i + 1,1:6] 
            #        - MET_data[key_][MET_i,1:6] )/(MET_data[key_][MET_i + 1,0] - MET_data[key_][MET_i,0]) )
            st_N +=1

        # Calculating the hydraulic parameters
        SL = self.hydraulics
        # Merging the state varibles of all elements for WQ reaction calculation
        C_all = {}
        for i in range(0,len(name)):
            C_all[name[i]] = np.concatenate([WQ[name[i]].c_j, WQ[name[i]].c_Ik,
                           WQ[name[i]].c_ik, WQ[name[i]].c_uk, WQ[name[i]].c_dk])
            #C_new[name[i]] = np.zeros(len(C_all[name[i]]))*np.concatenate([WQ[name[i]].c_j, WQ[name[i]].c_Ik,
            #               WQ[name[i]].c_ik, WQ[name[i]].c_uk, WQ[name[i]].c_dk])
        
        # Calculate the hydraulic parameters for all elements
        self.A_sur = np.concatenate([SL.A_sj, SL._A_SIk, SL.B_ik*SL._dx_ik,
                                SL._B_uk*SL._dx_uk, SL._B_dk*SL._dx_dk])
        self.A_bot = self.A_sur
        H_dep = np.concatenate([SL.H_j-SL._z_inv_j, SL._h_Ik, 
                                SL.A_ik/SL._B_ik, SL._h_uk, SL._h_dk])
        self.H_dep = np.where(H_dep<0.2,0.2,H_dep)
        U_vel = np.concatenate([0*SL.H_j, 0*SL._h_Ik, SL.Q_ik/SL.A_ik,
                                SL.Q_uk/SL.A_uk, SL.Q_dk/SL.A_dk])
        self.U_vel = np.where(U_vel<0.02,0.02,U_vel)
        
        ######################################################################################
        c_min = 0
        c_max = [40,40,8,8,8, 8,10,1,10,2, 2,2,2,1,1, 1,1,1,1,20, 20,20,100]
        
        MET_atemp = np.zeros(self.element_N)   # just copy the size of all elements.
        MET_dtemp = np.zeros(self.element_N)
        MET_atemp = np.zeros(self.element_N)
        MET_windspd  = np.zeros(self.element_N)
        MET_solar = np.zeros(self.element_N)
        MET_cloud  = np.zeros(self.element_N)
        
        i_met = 0        
        for key_ in met_key:
            MET_atemp[self.C_all_index[key_]] = np.ones(len(self.C_all_index[key_]))*MET_in_all[i_met,0]
            MET_dtemp[self.C_all_index[key_]] = np.ones(len(self.C_all_index[key_]))*MET_in_all[i_met,1]
            MET_windspd[self.C_all_index[key_]] = np.ones(len(self.C_all_index[key_]))*MET_in_all[i_met,2]
            MET_solar[self.C_all_index[key_]] = np.ones(len(self.C_all_index[key_]))*MET_in_all[i_met,3]
            MET_cloud[self.C_all_index[key_]] = np.ones(len(self.C_all_index[key_]))*MET_in_all[i_met,4]
            i_met +=1
            
        C_new = {}
        ###################################################################
        #   Temperature
        ###################################################################
        C_new['Temp'] = RK4_Temp2(dt, C_all['Temp'], MET_solar, 
                   MET_atemp, MET_dtemp, MET_windspd, MET_cloud, self.T_si, self.H_dep, self.alpha_temp)

        ###################################################################
        #   Numba is applied for all other constituents
        ###################################################################
        C_new['DO'], C_new['LDOC'], C_new['LPOC'],\
        C_new['RDOC'], C_new['RPOC'], C_new['PIN'],\
        C_new['NH4N'], C_new['NO3N'], C_new['LDON'],\
        C_new['LPON'], C_new['RDON'], C_new['RPON'],\
        C_new['PIP'], C_new['PO4P'], C_new['LDOP'],\
        C_new['LPOP'], C_new['RDOP'], C_new['RPOP'],\
        C_new['ALG1'], C_new['ALG2'], C_new['ALG3'], C_new['ISS']\
            = numba_all(self.K1_SOD, self.T1_SOD, self.gamma_SOD_1,
                self.K1_OC, self.T1_OC, self.gamma_OC_1, self.K1_ON, self.T1_ON, self.gamma_ON_1,
                self.K1_OP, self.T1_OP, self.gamma_OP_1, self.K1_NH4, self.T1_NH4, self.gamma_NH4_1,
                self.K1_NO3, self.T1_NO3, self.gamma_NO3_1,
                self.K1_ALG1, self.K2_ALG1, self.K3_ALG1, self.K4_ALG1, self.T1_ALG1, self.T2_ALG1,
                self.T3_ALG1, self.T4_ALG1, self.gamma1_ALG1, self.gamma2_ALG1,
                self.K1_ALG2, self.K2_ALG2, self.K3_ALG2, self.K4_ALG2, self.T1_ALG2, self.T2_ALG2,
                self.T3_ALG2, self.T4_ALG2, self.gamma1_ALG2, self.gamma2_ALG2,
                self.K1_ALG3, self.K2_ALG3, self.K3_ALG3, self.K4_ALG3, self.T1_ALG3, self.T2_ALG3,
                self.T3_ALG3, self.T4_ALG3, self.gamma1_ALG3, self.gamma2_ALG3,
                dt, self.K_h_N_ALG1, self.K_h_N_ALG2, self.K_h_N_ALG3,
                self.K_h_P_ALG1, self.K_h_P_ALG2, self.K_h_P_ALG3,self.keb, self.alpha_ISS, self.alpha_POM,
                self.alpha_ALG, MET_solar, MET_windspd, self.K_Lp, self.H_dep,
                self.K_ag_max_ALG1, self.K_ag_max_ALG2, self.K_ag_max_ALG3, 
                self.K_ar_max_ALG1, self.K_ar_max_ALG2, self.K_ar_max_ALG3, 
                self.K_ae_max_ALG1, self.K_ae_max_ALG2, self.K_ae_max_ALG3, 
                self.K_am_max_ALG1, self.K_am_max_ALG2, self.K_am_max_ALG3, 
                self.K_s_ALG1, self.K_s_ALG2, self.K_s_ALG3, 
                self.delta_O_ALG1_ag, self.delta_O_ALG2_ag, self.delta_O_ALG3_ag, 
                self.delta_O_ALG1_ar, self.delta_O_ALG2_ar, self.delta_O_ALG3_ar,
                self.delta_O_OC, self.K_LDOC, self.K_LPOC, self.K_RDOC, self.K_RPOC,
                self.K_NH4, self. delta_O_NH4, self.K_SOD,
                self.P_d_C, self.delta_C_ALG1, self.delta_C_ALG2, self.delta_C_ALG3,
                self.K_LDOC_to_RDOC, self.K_LPOC_to_RPOC, self.K_LDOP_to_RDOP, 
                self.K_LPOP_to_RPOP, self.K_LDON_to_RDON, self.K_LPON_to_RPON, 
                self.K_s_LPOC,  self.K_s_RPOC,
                self.K_ads_PIN, self.K_h_PIN, self.K_h_NH4, 
                self.delta_N_ALG1, self.delta_N_ALG2, self.delta_N_ALG3, 
                self.delta_N_ON, self.K_LDON, self.K_LPON, self.K_RDON, self.K_RPON,
                self.P_d_N, self.K_des_PIN, self.K_NO3, self.K_s_NO3, self.K_s_LPON, self.K_s_RPON, self.K_s_PIN,
                self.K_ads_PIP, self.K_h_PIP, 
                self.delta_P_ALG1, self.delta_P_ALG2, self.delta_P_ALG3,
                self.delta_P_OP, self.K_LDOP, self.K_LPOP, self.K_RDOP, self.K_RPOP,
                self.P_d_P, self.K_des_PIP, self.K_s_PIP, self.K_s_LPOP, self.K_s_RPOP, self.K_s_ISS, self.U_vel,
                C_all['Temp'], C_all['DO'], C_all['LDOC'], C_all['LPOC'],
                C_all['RDOC'], C_all['RPOC'], C_all['PIN'],
                C_all['NH4N'], C_all['NO3N'], C_all['LDON'],
                C_all['LPON'], C_all['RDON'], C_all['RPON'],
                C_all['PIP'], C_all['PO4P'], C_all['LDOP'], 
                C_all['LPOP'], C_all['RDOP'], C_all['RPOP'], 
                C_all['ALG1'], C_all['ALG2'], C_all['ALG3'], C_all['ISS'])
            
        for i in range(len(name)):
            C_new[name[i]]  = np.where(C_new[name[i]]<c_min,c_min,C_new[name[i]])
            C_new[name[i]]  = np.where(C_new[name[i]]>c_max[i],c_max[i],C_new[name[i]])
            # if nan, use the previous values
            nan_indices = np.isnan(C_new[name[i]])
            C_new[name[i]][nan_indices] = C_all[name[i]][nan_indices]
                
        # saving the reaction results to WQ
        for i in range (0,len(name)):
            WQ[name[i]]._c_j = C_new[name[i]][0:self._N_j]
            WQ[name[i]]._c_Ik = C_new[name[i]][self._N_j  :  self._N_j+self._N_Ik]
            WQ[name[i]]._c_ik = C_new[name[i]][self._N_j+self._N_Ik  :  self._N_j+self._N_Ik+self._N_ik]
            WQ[name[i]]._c_uk = C_new[name[i]][self._N_j+self._N_Ik+self._N_ik  :  self._N_j+self._N_Ik+self._N_ik+self._N_uk]
            WQ[name[i]]._c_dk = C_new[name[i]][self._N_j+self._N_Ik+self._N_ik+self._N_uk  :  self._N_j+self._N_Ik+self._N_ik+self._N_uk+self._N_dk]    
        return WQ
    
    def make_Q_spatial(self, observed_data, da_obs, superjunctions, WQ_match):
        Q_spatial = {}
        corr_matrix = {}
        for i in range(0, int(len(da_obs))):
            observed_data[da_obs[i]] = np.array(observed_data[da_obs[i]])
            df = pd.DataFrame(observed_data[da_obs[i]])
            df = df.drop(0, axis=1)
            df.iloc[:,0] = df.iloc[:,2]
            
            # interpolate the missing data
            data = np.array(df)
            data = np.where(data<0,np.nan,data)
            
            # Create arrays of row and column indices
            rows, cols = np.indices(data.shape)
            
            # Find the indices of missing values
            missing_indices = np.isnan(data)
            
            # Get the indices of non-missing values
            valid_indices = ~missing_indices
            
            # Extract the non-missing data points
            valid_data = data[valid_indices]
            valid_rows = rows[valid_indices]
            valid_cols = cols[valid_indices]
            
            # Define the grid for interpolation
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
            
            # Interpolate missing values using griddata
            interpolated_data = griddata((valid_cols, valid_rows), valid_data, (x, y), method='cubic')
            
            # Update the original data with interpolated values
            data[missing_indices] = interpolated_data[missing_indices]
            
            df = pd.DataFrame(data)
            df = df.applymap(lambda x: 0 if x < 0 else x)
            corr_matrix[da_obs[i]] = df.corr()
            
        # distance matrix
        distance_matrix = np.zeros((len(superjunctions),len(superjunctions)))
        
        for i in range(0,len(superjunctions)):
            for j in range(0,len(superjunctions)):
                distance_matrix[i][j] = np.sqrt(
                    (superjunctions['map_x'].iloc[i]-superjunctions['map_x'].iloc[j])**2 
                    + (superjunctions['map_y'].iloc[i]-superjunctions['map_y'].iloc[j])**2   )
                
        # Update the correlation by the distances: element-wise multiplication
        corr_by_distance = {}
        distance_coeff = 20000
        for i in range(0, int(len(da_obs))):
            corr_by_distance[da_obs[i]] = corr_matrix[da_obs[i]] * np.exp(-distance_matrix/distance_coeff)
            Q_spatial[da_obs[i]] = corr_by_distance[da_obs[i]].applymap(lambda x: 0 if x < 0 else x)
        
        Q_spatial['ALG1'] = Q_spatial['CHLA']
        Q_spatial['ALG2'] = Q_spatial['CHLA']
        Q_spatial['ALG3'] = Q_spatial['CHLA']
        
        Q_spatial['LDOC'] = Q_spatial['TOC']
        Q_spatial['RDOC'] = Q_spatial['TOC']
        Q_spatial['LPOC'] = Q_spatial['TOC']
        Q_spatial['RPOC'] = Q_spatial['TOC']
        
        Q_spatial['PIN'] = Q_spatial['TN']
        Q_spatial['LDON'] = Q_spatial['TN']
        Q_spatial['RDON'] = Q_spatial['TN']
        Q_spatial['LPON'] = Q_spatial['TN']
        Q_spatial['RPON'] = Q_spatial['TN']
        
        Q_spatial['PIP'] = Q_spatial['TP']
        Q_spatial['LDOP'] = Q_spatial['TP']
        Q_spatial['RDOP'] = Q_spatial['TP']
        Q_spatial['LPOP'] = Q_spatial['TP']
        Q_spatial['RPOP'] = Q_spatial['TP']
        
        # Set zero correlation between the different rivers
        is_ = WQ_match['classification'] == 'data_assimilation'
        WQ_data_nm = WQ_match[is_]
        WQ_river_no = list(WQ_data_nm['river_no'])
        
        for k in range(0,int(len(da_obs))):
            for i in range(0, len(superjunctions)):
                for j in range(0, len(superjunctions)):
                    if WQ_river_no[i] == WQ_river_no[j]:
                        pass
                    else:
                        Q_spatial[da_obs[k]].iloc[i][j] = 0
        
        ALG_nm = [ 'ALG1', 'ALG2', 'ALG3']
        for k in ALG_nm:
            for i in range(0, len(superjunctions)):
                for j in range(0, len(superjunctions)):
                    if WQ_river_no[i] == WQ_river_no[j]:
                        pass
                    else:
                        Q_spatial[k].iloc[i][j] = 0
        return Q_spatial
    
    def make_QR_cov_All_New(self, Q_spatial, Q_species, da_obs, da_target):
        # Define the measurement noise covariance matrix: The measurement noise is the same at all superjunctions and all WQ species.
        R_cov = (self.N_measure_sigma[0]**2)*np.eye(self.hydraulics.M*len(da_obs),self.hydraulics.M*len(da_obs))
        self.R_cov = R_cov
       
        # Make the spatial correlation matrices
        sc = np.zeros((int(len(da_target)),int(len(da_target)),self.hydraulics.M,self.hydraulics.M))
        for i in range(0,int(len(da_target))):
            for j in range(0,int(len(da_target))):
                for k in range(0,self.hydraulics.M):
                    sc[i][j][k][k] = Q_species[k][i][j]
        self.sc = sc            
       
        # Define the process noise covariance matrix based on spatial correlation for each constituent and inter-species correlation for each superjunction
        Q_cov = np.block([[Q_spatial['Temp'],sc[0][1],sc[0][2],sc[0][3],sc[0][4],sc[0][5],sc[0][6],sc[0][7],sc[0][8],sc[0][9],sc[0][10],sc[0][11],sc[0][12],sc[0][13],
                           sc[0][14],sc[0][15],sc[0][16],sc[0][17],sc[0][18],sc[0][19],sc[0][20],sc[0][21],sc[0][22]],
                        [sc[1][0],Q_spatial['DO'],sc[1][2],sc[1][3],sc[1][4],sc[1][5],sc[1][6],sc[1][7],sc[1][8],sc[1][9],sc[1][10],sc[1][11],sc[1][12],sc[1][13],
                         sc[1][14],sc[1][15],sc[1][16],sc[1][17],sc[1][18],sc[1][19],sc[1][20],sc[1][21],sc[1][22]],
                        [sc[2][0],sc[2][1],Q_spatial['LDOC'],sc[2][3],sc[2][4],sc[2][5],sc[2][6],sc[2][7],sc[2][8],sc[2][9],sc[2][10],sc[2][11],sc[2][12],sc[2][13],
                         sc[2][14],sc[2][15],sc[2][16],sc[2][17],sc[2][18],sc[2][19],sc[2][20],sc[2][21],sc[2][22]],
                        [sc[3][0],sc[3][1],sc[3][2],Q_spatial['LPOC'],sc[3][4],sc[3][5],sc[3][6],sc[3][7],sc[3][8],sc[3][9],sc[3][10],sc[3][11],sc[3][12],sc[3][13],
                         sc[3][14],sc[3][15],sc[3][16],sc[3][17],sc[3][18],sc[3][19],sc[3][20],sc[3][21],sc[3][22]],
                        [sc[4][0],sc[4][1],sc[4][2],sc[4][3],Q_spatial['RDOC'],sc[4][5],sc[4][6],sc[4][7],sc[4][8],sc[4][9],sc[4][10],sc[4][11],sc[4][12],sc[4][13],
                         sc[4][14],sc[4][15],sc[4][16],sc[4][17],sc[4][18],sc[4][19],sc[4][20],sc[4][21],sc[4][22]],
                        [sc[5][0],sc[5][1],sc[5][2],sc[5][3],sc[5][4],Q_spatial['RPOC'],sc[5][6],sc[5][7],sc[5][8],sc[5][9],sc[5][10],sc[5][11],sc[5][12],sc[5][13],
                         sc[5][14],sc[5][15],sc[5][16],sc[5][17],sc[5][18],sc[5][19],sc[5][20],sc[5][21],sc[5][22]],
                        [sc[6][0],sc[6][1],sc[6][2],sc[6][3],sc[6][4],sc[6][5],Q_spatial['PIN'],sc[6][7],sc[6][8],sc[6][9],sc[6][10],sc[6][11],sc[6][12],sc[6][13],
                         sc[6][14],sc[6][15],sc[6][16],sc[6][17],sc[6][18],sc[6][19],sc[6][20],sc[6][21],sc[6][22]],
                        [sc[7][0],sc[7][1],sc[7][2],sc[7][3],sc[7][4],sc[7][5],sc[7][6],Q_spatial['NH4N'],sc[7][8],sc[7][9],sc[7][10],sc[7][11],sc[7][12],sc[7][13],
                         sc[7][14],sc[7][15],sc[7][16],sc[7][17],sc[7][18],sc[7][19],sc[7][20],sc[7][21],sc[7][22]],
                        [sc[8][0],sc[8][1],sc[8][2],sc[8][3],sc[8][4],sc[8][5],sc[8][6],sc[8][7],Q_spatial['NO3N'],sc[8][9],sc[8][10],sc[8][11],sc[8][12],sc[8][13],
                         sc[8][14],sc[8][15],sc[8][16],sc[8][17],sc[8][18],sc[8][19],sc[8][20],sc[8][21],sc[8][22]],
                        [sc[9][0],sc[9][1],sc[9][2],sc[9][3],sc[9][4],sc[9][5],sc[9][6],sc[9][7],sc[9][8],Q_spatial['LDON'],sc[9][10],sc[9][11],sc[9][12],sc[9][13],
                         sc[9][14],sc[9][15],sc[9][16],sc[9][17],sc[9][18],sc[9][19],sc[9][20],sc[9][21],sc[9][22]],
                        [sc[10][0],sc[10][1],sc[10][2],sc[10][3],sc[10][4],sc[10][5],sc[10][6],sc[10][7],sc[10][8],sc[10][9],Q_spatial['LPON'],sc[10][11],sc[10][12],sc[10][13],
                         sc[10][14],sc[10][15],sc[10][16],sc[10][17],sc[10][18],sc[10][19],sc[10][20],sc[10][21],sc[10][22]],
                        [sc[11][0],sc[11][1],sc[11][2],sc[11][3],sc[11][4],sc[11][5],sc[11][6],sc[11][7],sc[11][8],sc[11][9],sc[11][10],Q_spatial['RDON'],sc[11][12],sc[11][13],
                         sc[11][14],sc[11][15],sc[11][16],sc[11][17],sc[11][18],sc[11][19],sc[11][20],sc[11][21],sc[11][22]],
                        [sc[12][0],sc[12][1],sc[12][2],sc[12][3],sc[12][4],sc[12][5],sc[12][6],sc[12][7],sc[12][8],sc[12][9],sc[12][10],sc[12][11],Q_spatial['RPON'],sc[12][13],
                         sc[12][14],sc[12][15],sc[12][16],sc[12][17],sc[12][18],sc[12][19],sc[12][20],sc[12][21],sc[12][22]],
                        [sc[13][0],sc[13][1],sc[13][2],sc[13][3],sc[13][4],sc[13][5],sc[13][6],sc[13][7],sc[13][8],sc[13][9],sc[13][10],sc[13][11],sc[13][12],Q_spatial['PIP'],
                         sc[13][14],sc[13][15],sc[13][16],sc[13][17],sc[13][18],sc[13][19],sc[13][20],sc[13][21],sc[13][22]],
                        [sc[14][0],sc[14][1],sc[14][2],sc[14][3],sc[14][4],sc[14][5],sc[14][6],sc[14][7],sc[14][8],sc[14][9],sc[14][10],sc[14][11],sc[14][12],sc[14][13]
                         ,Q_spatial['PO4P'],sc[14][15],sc[14][16],sc[14][17],sc[14][18],sc[14][19],sc[14][20],sc[14][21],sc[14][22]],
                        [sc[15][0],sc[15][1],sc[15][2],sc[15][3],sc[15][4],sc[15][5],sc[15][6],sc[15][7],sc[15][8],sc[15][9],sc[15][10],sc[15][11],sc[15][12],sc[15][13],
                         sc[15][14],Q_spatial['LDOP'],sc[15][16],sc[15][17],sc[15][18],sc[15][19],sc[15][20],sc[15][21],sc[15][22]],
                        [sc[16][0],sc[16][1],sc[16][2],sc[16][3],sc[16][4],sc[16][5],sc[16][6],sc[16][7],sc[16][8],sc[16][9],sc[16][10],sc[16][11],sc[16][12],sc[16][13],
                         sc[16][14],sc[16][15],Q_spatial['LPOP'],sc[16][17],sc[16][18],sc[16][19],sc[16][20],sc[16][21],sc[16][22]],
                        [sc[17][0],sc[17][1],sc[17][2],sc[17][3],sc[17][4],sc[17][5],sc[17][6],sc[17][7],sc[17][8],sc[17][9],sc[17][10],sc[17][11],sc[17][12],sc[17][13],
                         sc[17][14],sc[17][15],sc[17][16],Q_spatial['RDOP'],sc[17][18],sc[17][19],sc[17][20],sc[17][21],sc[17][22]],
                        [sc[18][0],sc[18][1],sc[18][2],sc[18][3],sc[18][4],sc[18][5],sc[18][6],sc[18][7],sc[18][8],sc[18][9],sc[18][10],sc[18][11],sc[18][12],sc[18][13],
                         sc[18][14],sc[18][15],sc[18][16],sc[18][17],Q_spatial['RPOP'],sc[18][19],sc[18][20],sc[18][21],sc[18][22]],
                        [sc[19][0],sc[19][1],sc[19][2],sc[19][3],sc[19][4],sc[19][5],sc[19][6],sc[19][7],sc[19][8],sc[19][9],sc[19][10],sc[19][11],sc[19][12],sc[19][13],
                         sc[19][14],sc[19][15],sc[19][16],sc[19][17],sc[19][18],Q_spatial['ALG1'],sc[19][20],sc[19][21],sc[19][22]],
                        [sc[20][0],sc[20][1],sc[20][2],sc[20][3],sc[20][4],sc[20][5],sc[20][6],sc[20][7],sc[20][8],sc[20][9],sc[20][10],sc[20][11],sc[20][12],sc[20][13],
                         sc[20][14],sc[20][15],sc[20][16],sc[20][17],sc[20][18],sc[20][19],Q_spatial['ALG2'],sc[20][21],sc[20][22]],
                        [sc[21][0],sc[21][1],sc[21][2],sc[21][3],sc[21][4],sc[21][5],sc[21][6],sc[21][7],sc[21][8],sc[21][9],sc[21][10],sc[21][11],sc[21][12],sc[21][13],
                         sc[21][14],sc[21][15],sc[21][16],sc[21][17],sc[21][18],sc[21][19],sc[21][20],Q_spatial['ALG3'],sc[21][22]],
                        [sc[22][0],sc[22][1],sc[22][2],sc[22][3],sc[22][4],sc[22][5],sc[22][6],sc[22][7],sc[22][8],sc[22][9],sc[22][10],sc[22][11],sc[22][12],sc[22][13],
                         sc[22][14],sc[22][15],sc[22][16],sc[22][17],sc[22][18],sc[22][19],sc[22][20],sc[22][21],Q_spatial['ISS']]         ] )
        
        Q_cov = self.N_process_sigma[0]**2 * Q_cov
        self.Q_cov = Q_cov
        
    def KF_DA_All_New(self, dt, WQ_A_k, WQ_B_k, WQ, observed_data, DA_i, da_obs, da_target, Q_spatial, Q_species, J_day):
        x_hat = np.block([WQ['Temp'].c_j.T, WQ['DO'].c_j.T, WQ['LDOC'].c_j.T, WQ['LPOC'].c_j.T, WQ['RDOC'].c_j.T,
                        WQ['RPOC'].c_j.T, WQ['PIN'].c_j.T, WQ['NH4N'].c_j.T,   WQ['NO3N'].c_j.T, WQ['LDON'].c_j.T, 
                        WQ['LPON'].c_j.T, WQ['RDON'].c_j.T, WQ['RPON'].c_j.T, WQ['PIP'].c_j.T, WQ['PO4P'].c_j.T, 
                        WQ['LDOP'].c_j.T, WQ['LPOP'].c_j.T, WQ['RDOP'].c_j.T, WQ['RPOP'].c_j.T, WQ['ALG1'].c_j.T,
                        WQ['ALG2'].c_j.T, WQ['ALG3'].c_j.T, WQ['ISS'].c_j.T])
        self.x_hat2 = x_hat
        z0 = np.zeros((self._N_j, self._N_j))
        TA_k = np.block([
                        [WQ_A_k['Temp'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,WQ_A_k['DO'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,WQ_A_k['LDOC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,WQ_A_k['LPOC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,WQ_A_k['RDOC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,WQ_A_k['RPOC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,WQ_A_k['PIN'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,WQ_A_k['NH4N'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['NO3N'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['LDON'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['LPON'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['RDON'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['RPON'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['PIP'],z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['PO4P'],z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['LDOP'],z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['LPOP'],z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['RDOP'],z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['RPOP'],z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ALG1'],z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ALG2'],z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ALG3'],z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ISS']],
                         
                        ])

        TB_k = np.block([WQ_B_k['Temp'].T, WQ_B_k['DO'].T, WQ_B_k['LDOC'].T, WQ_B_k['LPOC'].T, WQ_B_k['RDOC'].T,
                        WQ_B_k['RPOC'].T, WQ_B_k['PIN'].T, WQ_B_k['NH4N'].T,  WQ_B_k['NO3N'].T, WQ_B_k['LDON'].T, 
                        WQ_B_k['LPON'].T, WQ_B_k['RDON'].T, WQ_B_k['RPON'].T, WQ_B_k['PIP'].T, WQ_B_k['PO4P'].T,
                        WQ_B_k['LDOP'].T, WQ_B_k['LPOP'].T, WQ_B_k['RDOP'].T, WQ_B_k['RPOP'].T, WQ_B_k['ALG1'].T,
                        WQ_B_k['ALG2'].T, WQ_B_k['ALG3'].T, WQ_B_k['ISS'].T])
        
        # interpolate the DA concentrations
        da_c_in = {}
        for i in range(0,len(da_obs)):
            da_c_in[da_obs[i]] = observed_data[da_obs[i]][DA_i,1:self._N_j+1] 
            '''
            +( (J_day
                   - observed_data[da_obs[i]][DA_i,0])*(observed_data[da_obs[i]][DA_i+1,1:self._N_j+1] 
                   - observed_data[da_obs[i]][DA_i,1:self._N_j+1] ) / (observed_data[da_obs[i]][DA_i+1,0]
                   - observed_data[da_obs[i]][DA_i,0]) )
            '''                                                           
        T_obs = np.block([da_c_in['Temp'], da_c_in['DO'], da_c_in['TOC'], da_c_in['TN'], da_c_in['NH4N'],
                          da_c_in['NO3N'], da_c_in['TP'], da_c_in['PO4P'], da_c_in['CHLA'], da_c_in['ISS']])
  
        # Make the observation matrix
        H_k_1 = {}
        for wq_nm in da_obs:
            H_k_1[wq_nm] = np.zeros((self._N_j, self._N_j))
        # If there is observed data at specific superjunction for specific constituent, it will be 1
            for i in range(0,self.hydraulics.M):
                if da_c_in[wq_nm][i] < 0:
                    H_k_1[wq_nm][i,i] = 0
                else:
                    H_k_1[wq_nm][i,i] = 1
        
        c1 = self.delta_C_ALG1[0]
        c2 = self.delta_C_ALG2[0]
        c3 = self.delta_C_ALG3[0]
        
        n1 = self.delta_N_ALG1[0]
        n2 = self.delta_N_ALG2[0]
        n3 = self.delta_N_ALG3[0]
        
        p1 = self.delta_P_ALG1[0]
        p2 = self.delta_P_ALG2[0]
        p3 = self.delta_P_ALG3[0]
               
        chl1 = 1/self.Chl_to_ALG_ratio_1[0]
        chl2 = 1/self.Chl_to_ALG_ratio_2[0]
        chl3 = 1/self.Chl_to_ALG_ratio_3[0]
        
        H_k = np.block([
                        [H_k_1['Temp'],z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0],
                        [z0,H_k_1['DO'],z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0],
                        [z0,z0,H_k_1['TOC'],H_k_1['TOC'],H_k_1['TOC'], H_k_1['TOC'],z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,c1*H_k_1['TOC'], c2*H_k_1['TOC'],c3*H_k_1['TOC'],z0],
                        [z0,z0,z0,z0,z0, z0,H_k_1['TN'],H_k_1['TN'],H_k_1['TN'],H_k_1['TN'], H_k_1['TN'],H_k_1['TN'],H_k_1['TN'],z0,z0, z0,z0,z0,z0,n1*H_k_1['TN'], n2*H_k_1['TN'],n3*H_k_1['TN'],z0],
                        [z0,z0,z0,z0,z0, z0,z0,H_k_1['NH4N'],z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0],
                        [z0,z0,z0,z0,z0, z0,z0,z0,H_k_1['NO3N'],z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0],
                        [z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,H_k_1['PO4P'],H_k_1['PO4P'], H_k_1['PO4P'],H_k_1['PO4P'],H_k_1['PO4P'],H_k_1['PO4P'],p1*H_k_1['PO4P'], p2*H_k_1['PO4P'],p3*H_k_1['PO4P'],z0],
                        [z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0, H_k_1['PO4P'], z0,z0,z0,z0,z0, z0,z0,z0],
                        [z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,chl1*H_k_1['CHLA'], chl2*H_k_1['CHLA'],chl3*H_k_1['CHLA'],z0],
                        [z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,z0,z0,z0, z0,z0,H_k_1['ISS']] 
                        ])
        Q_cov = self.Q_cov
        R_cov = self.R_cov
        P_k = self.P_k
        
        # Kalman Filtering Algorithm
        x_hat_k = TA_k @ x_hat + TB_k
        P_k = TA_k @ P_k @ TA_k.T + Q_cov
        K_k = P_k @ H_k.T @ np.linalg.inv(H_k @ P_k @ H_k.T + R_cov)
        P_k = P_k - K_k @ H_k @ P_k
        x_hat = x_hat_k + K_k @ (T_obs - H_k @ x_hat_k)
        
        x_hat[x_hat < 0.0] = 10e-7
        self.P_k = P_k
        
        return x_hat

def ALG_ratio_by_period(number):
    # Algae ratio by period(month): start, end, ALG1, ALG2, ALG3 (Diatom, Cyano, Green)
    while number > 365:
        number -= 365
    ranges_and_values = [
        (0, 31, 0.953, 0.0, 0.047),
        (31.000001, 59, 0.956, 0.004, 0.040),
        (59.000001, 90, 0.951, 0.000, 0.049),
        (90.000001, 120, 0.790, 0.027, 0.183),
        (120.000001, 151, 0.587, 0.057, 0.356),
        (151.000001, 181, 0.352, 0.180, 0.468),
        (181.000001, 212, 0.179, 0.356, 0.465),
        (212.000001, 243, 0.273, 0.511, 0.216),
        (243.000001, 273, 0.194, 0.222, 0.584),
        (273.000001, 304, 0.731, 0.046, 0.223),
        (304.000001, 334, 0.660, 0.095, 0.245),
        (334.000001, 380, 0.864, 0.033, 0.103),
        ]
    for start, end, ALG1, ALG2, ALG3 in ranges_and_values:
        if start <= number <= end:
            return ALG1, ALG2, ALG3

##############################################################################
#   Temperature
##############################################################################
@njit
def F_Temp(Temp, J_solar, Tair, Tdew, windspd, cloud, T_si, H_dep):
#    Temp = np.where(Temp<0,0,Temp)
    Fw = 19.0 + 0.95*np.power(windspd*np.power(7/10, 0.15),2)
    e_s = 4.596*np.exp( 17.27*Temp/(237.3 + Temp))
    e_air = 4.596*np.exp( 17.27*Tdew/(237.3 + Tdew))
    F_answer = ( 23.9*J_solar
                + 11.7*0.00000001*np.power(Tair + 273.15, 4)*(1+0.17*np.power(cloud,2))
                - 0.97*11.7*0.00000001*np.power(Temp + 273.15, 4)
                - 0.47*Fw*(Temp - Tair) 
                - Fw*(e_s - e_air) + 1.6*0.4*(T_si-Temp)/(0.5*50))  /(100*H_dep) 
    # In the above equation, I used the QUAL2K's heat balance expression based on cal and cm units
    # 1.6 = sediment density, 0.4 = specific heat of sediment, 10cm = effective sediment depth
    return F_answer

@njit
def RK4_Temp(dt, A, B, C, D, E, F, G, H):
    K1=(dt/86400)*F_Temp(A, B, C, D, E, F, G, H)
    K2=(dt/86400)*F_Temp(A+K1/2, B, C, D, E, F, G, H)
    K3=(dt/86400)*F_Temp(A+K2/2, B, C, D, E, F, G, H)
    K4=(dt/86400)*F_Temp(A+K3, B, C, D, E, F, G, H)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    # G_new: new sediment bed tamperature. 
    # where, 0.0126 is thermal diffusivity of sediment bed (cm^2/sec). 10 (cm) is the effective sediment depth.
    G_new = G - 2*0.005*86400*(G - A)/(50**2)
    return RK4_Answer, G_new

@njit
def F_Temp2(Temp, J_solar, Tair, Tdew, windspd, cloud, T_si, H_dep, alpha_temp):
#    Temp = np.where(Temp<0,0,Temp)
    Fw = 19.0 + 0.95*np.power(windspd*np.power(7/10, 0.15),2)
    e_s = 4.596*np.exp( 17.27*Temp/(237.3 + Temp))
    e_air = 4.596*np.exp( 17.27*Tdew/(237.3 + Tdew))
    F_answer = ( 23.9*J_solar
                + 11.7*0.00000001*np.power(Tair + 273.15, 4)*(1+0.17*np.power(cloud,2))*alpha_temp
                - 0.97*11.7*0.00000001*np.power(Temp + 273.15, 4)
                - 0.47*Fw*(Temp - Tair) 
                - Fw*(e_s - e_air) - 0.3*0.485*(Temp - T_si))/(100*H_dep) 
    # In this part we use the unit of cm, cal, by referring QUAL2K
    # Constant 23.9 is an unit conversion parameter from MJ/m^2(unit of the observed data) to cal/cm^2(unit in the model)
    return F_answer

@njit
def RK4_Temp2(dt, A, B, C, D, E, F, G, H, I):
    K1=(dt/86400)*F_Temp2(A, B, C, D, E, F, G, H, I)
    K2=(dt/86400)*F_Temp2(A+K1/2, B, C, D, E, F, G, H, I)
    K3=(dt/86400)*F_Temp2(A+K2/2, B, C, D, E, F, G, H, I)
    K4=(dt/86400)*F_Temp2(A+K3, B, C, D, E, F, G, H, I)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   Dissolved Oxygen
##############################################################################
@njit
def K_L_cal(U_vel, H_dep, windspd):
    # just copy the variable's dimension
    K_L = 0*U_vel
    K_ah = 0*U_vel
    K_w =  0*U_vel
    for i in range(0,len(U_vel)):
        if H_dep[i]<0.6:
            K_ah[i] = 5.32*np.power(U_vel[i],0.67)/np.power(H_dep[i], 1.85)
        elif (H_dep[i]>0.6 and H_dep[i] > 3.45*np.power(U_vel[i],2.5)):
            K_ah[i] = 3.93*np.power(U_vel[i],0.2)/np.power(H_dep[i], 1.67)
        else:
            K_ah[i] = 5.026*U_vel[i]/np.power(H_dep[i], 1.67)
    # K_w equation should be added here
    K_w = 0.0986*np.power(windspd, 1.64) + K_w
    K_L = K_ah + K_w
    return K_L

@njit
def F_DO(DO, Temp, K_L, DO_1, DO_2, DO_3):
    DO_sat = np.exp( (1.575701*1e5)/(Temp+273.15)-(6.642308*1e7)/np.power(Temp+273.15,2)
    +(1.2438*1e10)/np.power(Temp+273.15,3)-(8.621949*1e11)/np.power(Temp+273.15,4)-139.34411)
    F_answer = K_L*(DO_sat - DO) + DO_1 + DO_2 + DO_3
    return F_answer

@njit
def RK4_DO(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_DO(A, B, C, D, E, F)
    K2=(dt/86400)*F_DO(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_DO(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_DO(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   Dissolved Inorganic Carbon
##############################################################################
@njit
def F_DIC(DIC, K_L, CO2_air, DIC_1, DIC_2):
    F_answer = K_L*(CO2_air - DIC) + DIC_1 + DIC_2
    return F_answer

@njit
def RK4_DIC(dt, A, B, C, D, E):
    K1=(dt/86400)*F_DIC(A, B, C, D, E)
    K2=(dt/86400)*F_DIC(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_DIC(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_DIC(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LDOC: Labile Dissolved Organic Carbon
##############################################################################
@njit
def F_LDOC(LDOC, K_LDOC, K_LDOC_to_RDOC, gamma_OC, LDOC_1):
    F_answer = -K_LDOC*gamma_OC*LDOC - K_LDOC_to_RDOC*gamma_OC*LDOC + LDOC_1
    return F_answer

@njit
def RK4_LDOC(dt, A, B, C, D, E):
    K1=(dt/86400)*F_LDOC(A, B, C, D, E)
    K2=(dt/86400)*F_LDOC(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_LDOC(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_LDOC(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LPOC: Labile Particulate Organic Carbon
##############################################################################
@njit
def F_LPOC(LPOC, K_LPOC, K_LPOC_to_RPOC, K_s_LPOC, H_dep, gamma_OC, LPOC_1):
    F_answer = ( -K_LPOC*gamma_OC*LPOC - K_LPOC_to_RPOC*gamma_OC*LPOC 
                 - (K_s_LPOC/H_dep)*LPOC + LPOC_1 )
    return F_answer

@njit
def RK4_LPOC(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_LPOC(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_LPOC(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_LPOC(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_LPOC(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RDOC: Refractory Dissolved Organic Carbon
##############################################################################
@njit
def F_RDOC(RDOC, K_RDOC, gamma_OC, RDOC_1):
    F_answer = -K_RDOC*gamma_OC*RDOC + RDOC_1
    return F_answer

@njit
def RK4_RDOC(dt, A, B, C, D):
    K1=(dt/86400)*F_RDOC(A, B, C, D)
    K2=(dt/86400)*F_RDOC(A+K1/2, B, C, D)
    K3=(dt/86400)*F_RDOC(A+K2/2, B, C, D)
    K4=(dt/86400)*F_RDOC(A+K3, B, C, D)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RPOC: Refractory Particulate Organic Carbon
##############################################################################
@njit
def F_RPOC(RPOC, K_RPOC, gamma_OC, K_s_RPOC, H_dep, RPOC_1):
    F_answer = -K_RPOC*gamma_OC*RPOC - (K_s_RPOC/H_dep)*RPOC + RPOC_1
    return F_answer

@njit
def RK4_RPOC(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_RPOC(A, B, C, D, E, F)
    K2=(dt/86400)*F_RPOC(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_RPOC(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_RPOC(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   PIN: Particulate Inorganic Nitrogen
##############################################################################
@njit
def F_PIN(PIN, K_des_PIN, K_s_PIN, H_dep, PIN_1):
    F_answer = -K_des_PIN*PIN - (K_s_PIN/H_dep)*PIN + PIN_1
    return F_answer

@njit
def RK4_PIN(dt, A, B, C, D, E):
    K1=(dt/86400)*F_PIN(A, B, C, D, E)
    K2=(dt/86400)*F_PIN(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_PIN(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_PIN(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   NH4: Ammonium Nitrogen
##############################################################################
@njit
def F_NH4(NH4, K_NH4, gamma_NH4, PIN_ads, NH4_1, NH4_2):
    F_answer = -K_NH4*gamma_NH4*NH4 - PIN_ads*NH4 + NH4_1 + NH4_2
    return F_answer

@njit
def RK4_NH4(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_NH4(A, B, C, D, E, F)
    K2=(dt/86400)*F_NH4(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_NH4(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_NH4(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   NO3: Nitrate Nitrogen
##############################################################################
@njit
def F_NO3(NO3, K_NO3, gamma_NO3, K_s_NO3, H_dep, NO3_1):
    F_answer = - K_NO3*gamma_NO3*NO3 - (K_s_NO3/H_dep)*NO3 + NO3_1
    return F_answer

@njit
def RK4_NO3(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_NO3(A, B, C, D, E, F)
    K2=(dt/86400)*F_NO3(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_NO3(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_NO3(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LDON: Labile Dissolved Organic Nitrogen
##############################################################################
@njit
def F_LDON(LDON, K_LDON, K_LDON_to_RDON, gamma_ON, LDON_1):
    F_answer = -K_LDON*gamma_ON *LDON - K_LDON_to_RDON*gamma_ON*LDON + LDON_1
    return F_answer

@njit
def RK4_LDON(dt, A, B, C, D, E):
    K1=(dt/86400)*F_LDON(A, B, C, D, E)
    K2=(dt/86400)*F_LDON(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_LDON(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_LDON(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LPON: Labile Particulate Organic Nitrogen
##############################################################################
@njit
def F_LPON(LPON, K_LPON, K_LPON_to_RPON, K_s_LPON, H_dep, gamma_ON, LPON_1):
    F_answer = ( -K_LPON*gamma_ON*LPON - K_LPON_to_RPON*gamma_ON*LPON
                 - (K_s_LPON/H_dep)*LPON + LPON_1 )
    return F_answer

@njit
def RK4_LPON(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_LPON(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_LPON(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_LPON(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_LPON(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RDON: Refractory Dissolved Organic Nitrogen
##############################################################################
@njit
def F_RDON(RDON, K_RDON, gamma_ON, RDON_1):
    F_answer = -K_RDON*gamma_ON*RDON + RDON_1
    return F_answer

@njit
def RK4_RDON(dt, A, B, C, D):
    K1=(dt/86400)*F_RDON(A, B, C, D)
    K2=(dt/86400)*F_RDON(A+K1/2, B, C, D)
    K3=(dt/86400)*F_RDON(A+K2/2, B, C, D)
    K4=(dt/86400)*F_RDON(A+K3, B, C, D)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RPON: Refractory Particulate Organic Nitrogen
##############################################################################
@njit
def F_RPON(RPON, K_RPON, gamma_ON, K_s_RPON, H_dep, RPON_1):
    F_answer = -K_RPON*gamma_ON*RPON - (K_s_RPON/H_dep)*RPON + RPON_1
    return F_answer

@njit
def RK4_RPON(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_RPON(A, B, C, D, E, F)
    K2=(dt/86400)*F_RPON(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_RPON(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_RPON(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   PIP: Particulate Inorganic Phosphorus
##############################################################################
@njit
def F_PIP(PIP, K_des_PIP, K_s_PIP, H_dep, PIP_1):
    F_answer = -K_des_PIP*PIP - (K_s_PIP/H_dep)*PIP + PIP_1
    return F_answer

@njit
def RK4_PIP(dt, A, B, C, D, E):
    K1=(dt/86400)*F_PIP(A, B, C, D, E)
    K2=(dt/86400)*F_PIP(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_PIP(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_PIP(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   PO4: Phosphate
##############################################################################
@njit
def F_PO4(PO4, PO4_des, PO4_1, PO4_2):
    F_answer = -PO4_des*PO4 + PO4_1 + PO4_2
    return F_answer

@njit
def RK4_PO4(dt, A, B, C, D):
    K1=(dt/86400)*F_PO4(A, B, C, D)
    K2=(dt/86400)*F_PO4(A+K1/2, B, C, D)
    K3=(dt/86400)*F_PO4(A+K2/2, B, C, D)
    K4=(dt/86400)*F_PO4(A+K3, B, C, D)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LDOP: Labile Dissolved Organic Nitrogen
##############################################################################
@njit
def F_LDOP(LDOP, K_LDOP, K_LDOP_to_RDOP, gamma_OP, LDOP_1):
    F_answer = -K_LDOP*gamma_OP *LDOP - K_LDOP_to_RDOP*gamma_OP*LDOP + LDOP_1
    return F_answer

@njit
def RK4_LDOP(dt, A, B, C, D, E):
    K1=(dt/86400)*F_LDOP(A, B, C, D, E)
    K2=(dt/86400)*F_LDOP(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_LDOP(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_LDOP(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LPOP: Labile Particulate Organic Nitrogen
##############################################################################
@njit
def F_LPOP(LPOP, K_LPOP, K_LPOP_to_RPOP, K_s_LPOP, H_dep, gamma_OP, LPOP_1):
    F_answer = ( -K_LPOP*gamma_OP*LPOP - K_LPOP_to_RPOP*gamma_OP*LPOP
                 - (K_s_LPOP/H_dep)*LPOP + LPOP_1 )
    return F_answer

@njit
def RK4_LPOP(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_LPOP(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_LPOP(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_LPOP(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_LPOP(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RDOP: Refractory Dissolved Organic Nitrogen
##############################################################################
@njit
def F_RDOP(RDOP, K_RDOP, gamma_OP, RDOP_1):
    F_answer = -K_RDOP*gamma_OP*RDOP + RDOP_1
    return F_answer

@njit
def RK4_RDOP(dt, A, B, C, D):
    K1=(dt/86400)*F_RDOP(A, B, C, D)
    K2=(dt/86400)*F_RDOP(A+K1/2, B, C, D)
    K3=(dt/86400)*F_RDOP(A+K2/2, B, C, D)
    K4=(dt/86400)*F_RDOP(A+K3, B, C, D)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RPOP: Refractory Particulate Organic Nitrogen
##############################################################################
@njit
def F_RPOP(RPOP, K_RPOP, gamma_OP, K_s_RPOP, H_dep, RPOP_1):
    F_answer = -K_RPOP*gamma_OP*RPOP - (K_s_RPOP/H_dep)*RPOP + RPOP_1
    return F_answer

@njit
def RK4_RPOP(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_RPOP(A, B, C, D, E, F)
    K2=(dt/86400)*F_RPOP(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_RPOP(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_RPOP(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   ISS: Inorganic Suspended Solids: ISS
##############################################################################
@njit
def F_ISS(ISS, K_s_ISS, H_dep):
    F_answer = -(K_s_ISS/H_dep)*ISS
    return F_answer

@njit
def RK4_ISS(dt, A, B, C):
    K1=(dt/86400)*F_ISS(A, B, C)
    K2=(dt/86400)*F_ISS(A+K1/2, B, C)
    K3=(dt/86400)*F_ISS(A+K2/2, B, C)
    K4=(dt/86400)*F_ISS(A+K3, B, C)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   Algae: Phytoplankton
##############################################################################
@njit
def F_ALG(ALG, K_ag, K_ar, K_ae, K_am, K_s_ALG, H_dep):
    F_answer = K_ag*ALG - (K_ar + K_ae + K_am)* ALG - (K_s_ALG/H_dep)*ALG
    return F_answer

@njit
def RK4_ALG(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_ALG(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_ALG(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_ALG(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_ALG(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   Find minimum values
##############################################################################
@njit
def min_lambda3(lambda_N, lambda_P, lambda_light):
    minimum = lambda_N
    for i in range(0, len(lambda_N)):
        if lambda_P[i] < lambda_N[i]:
            minimum[i] = lambda_P[i]
        elif lambda_light[i] < minimum[i]:
            minimum[i] = lambda_light[i]
        else:
            pass
    return minimum

@njit
def min_lambda2(lambda_N, lambda_P):
    minimum = lambda_N
    for i in range(0, len(lambda_N)):
        if lambda_P[i] < lambda_N[i]:
            minimum[i] = lambda_P[i]
        else:
            pass
    return minimum

##############################################################################
#   Input WQ data interpolations : not applied
##############################################################################
@njit
def numba_WQ_interpolation(J_day, name_i, WQ_i, WQ_bc_data, WQ_bc_nm, c_in):
    for i in range(len(WQ_bc_nm)):
        for j in range(int(WQ_i[i]), len(WQ_bc_data[WQ_bc_nm[i]])):
            if WQ_bc_data[WQ_bc_nm[i]]['J_day'][j] > J_day:
               _j = j - 1
               WQ_i[i] = _j
               break
        c_in[i] = ( WQ_bc_data[WQ_bc_nm[i]][name_i][_j] +
                    ( (J_day - WQ_bc_data[WQ_bc_nm[i]]['J_day'][_j])*(WQ_bc_data[WQ_bc_nm[i]][name_i][_j+1]
                           - WQ_bc_data[WQ_bc_nm[i]][name_i][_j] ) / (WQ_bc_data[WQ_bc_nm[i]]['J_day'][_j+1] 
                           - WQ_bc_data[WQ_bc_nm[i]]['J_day'][_j] ))   )

@njit
def numba_WQ_interpolation2(J_day, WQ_i, jdays, conc, i):
    for j in range(int(WQ_i[i]), len(jdays)):
        if jdays[j] > J_day:
           _j = j - 1
           WQ_i_numba = _j
           break
    c_in_numba = ( conc[_j] + ( (J_day - jdays[_j])*(conc[_j+1] - conc[_j] ) / (jdays[_j+1] - jdays[_j] ))   )
    return c_in_numba, WQ_i_numba

##############################################################################
#   Numba for all other constituents except temperature
##############################################################################
@njit
def numba_all(K1_SOD, T1_SOD, gamma_SOD_1,
        K1_OC, T1_OC, gamma_OC_1, K1_ON, T1_ON, gamma_ON_1, K1_OP, T1_OP, gamma_OP_1, 
        K1_NH4, T1_NH4, gamma_NH4_1, K1_NO3, T1_NO3, gamma_NO3_1, K1_ALG1, K2_ALG1, 
        K3_ALG1, K4_ALG1, T1_ALG1, T2_ALG1,
        T3_ALG1, T4_ALG1, gamma1_ALG1, gamma2_ALG1,
        K1_ALG2, K2_ALG2, K3_ALG2, K4_ALG2, T1_ALG2, T2_ALG2,
        T3_ALG2, T4_ALG2, gamma1_ALG2, gamma2_ALG2,
        K1_ALG3, K2_ALG3, K3_ALG3, K4_ALG3, T1_ALG3, T2_ALG3,
        T3_ALG3, T4_ALG3, gamma1_ALG3, gamma2_ALG3,
        dt, K_h_N_ALG1, K_h_N_ALG2, K_h_N_ALG3,
        K_h_P_ALG1, K_h_P_ALG2, K_h_P_ALG3, keb, alpha_ISS, alpha_POM, alpha_ALG, MET_solar, windspd, K_Lp, H_dep,
        K_ag_max_ALG1, K_ag_max_ALG2, K_ag_max_ALG3,
        K_ar_max_ALG1, K_ar_max_ALG2, K_ar_max_ALG3,
        K_ae_max_ALG1, K_ae_max_ALG2, K_ae_max_ALG3,
        K_am_max_ALG1, K_am_max_ALG2, K_am_max_ALG3,
        K_s_ALG1, K_s_ALG2, K_s_ALG3, 
        delta_O_ALG1_ag, delta_O_ALG2_ag, delta_O_ALG3_ag,
        delta_O_ALG1_ar, delta_O_ALG2_ar, delta_O_ALG3_ar,
        delta_O_OC, K_LDOC, K_LPOC, K_RDOC, K_RPOC,
        K_NH4,  delta_O_NH4, K_SOD,
        P_d_C, delta_C_ALG1, delta_C_ALG2, delta_C_ALG3,
        K_LDOC_to_RDOC, K_LPOC_to_RPOC, K_LDOP_to_RDOP, K_LPOP_to_RPOP, K_LDON_to_RDON, K_LPON_to_RPON, 
        K_s_LPOC, K_s_RPOC,
        K_ads_PIN, K_h_PIN, K_h_NH4, 
        delta_N_ALG1, delta_N_ALG2, delta_N_ALG3, 
        delta_N_ON, K_LDON, K_LPON, K_RDON, K_RPON,
        P_d_N, K_des_PIN, K_NO3, K_s_NO3, K_s_LPON, K_s_RPON, K_s_PIN,
        K_ads_PIP, K_h_PIP, 
        delta_P_ALG1, delta_P_ALG2, delta_P_ALG3,
        delta_P_OP, K_LDOP, K_LPOP, K_RDOP, K_RPOP,
        P_d_P, K_des_PIP, K_s_PIP, K_s_LPOP,  K_s_RPOP, K_s_ISS, U_vel,
        C_all_Temp, C_all_DO, C_all_LDOC, C_all_LPOC,C_all_RDOC, C_all_RPOC,
        C_all_PIN ,C_all_NH4N, C_all_NO3N, C_all_LDON, C_all_LPON, C_all_RDON, 
        C_all_RPON, C_all_PIP, C_all_PO4P, C_all_LDOP, C_all_LPOP, C_all_RDOP,
        C_all_RPOP, C_all_ALG1, C_all_ALG2, C_all_ALG3, C_all_ISS):
    
    ###################################################################
    #   Temperature multipliers
    ###################################################################
    gamma_OC = ( (K1_OC*np.exp(gamma_OC_1*(C_all_Temp - T1_OC))) 
                /(1+K1_OC*np.exp(gamma_OC_1*(C_all_Temp - T1_OC))-K1_OC) )
    gamma_SOD = ( (K1_SOD*np.exp(gamma_SOD_1*(C_all_Temp - T1_SOD))) 
                /(1+K1_SOD*np.exp(gamma_SOD_1*(C_all_Temp - T1_SOD))-K1_SOD) )
    gamma_ON = ( (K1_ON*np.exp(gamma_ON_1*(C_all_Temp-T1_ON)))
                /(1+K1_ON*np.exp(gamma_ON_1*(C_all_Temp-T1_ON))-K1_ON) )
    gamma_OP = ( (K1_OP*np.exp(gamma_OP_1*(C_all_Temp-T1_OP)))
                /(1+K1_OP*np.exp(gamma_OP_1*(C_all_Temp-T1_OP))-K1_OP) )
    gamma_NH4 =  ( (K1_NH4*np.exp(gamma_NH4_1*(C_all_Temp-T1_NH4)))
                  /(1+K1_NH4*np.exp(gamma_NH4_1*(C_all_Temp-T1_NH4))-K1_NH4) )
    gamma_NO3 =  ( (K1_NO3*np.exp(gamma_NO3_1*(C_all_Temp-T1_NO3)))
                  /(1+K1_NO3*np.exp(gamma_NO3_1*(C_all_Temp-T1_NO3))-K1_NO3) )
    gamma_r_ALG1 = ((K1_ALG1*np.exp(gamma1_ALG1*(C_all_Temp - T1_ALG1)))
                /(1+K1_ALG1*np.exp(gamma1_ALG1*(C_all_Temp - T1_ALG1))-K1_ALG1) )
    gamma_f_ALG1 = ((K4_ALG1*np.exp(gamma2_ALG1*(T4_ALG1 - C_all_Temp)))
                /(1+K4_ALG1*np.exp(gamma2_ALG1*(T4_ALG1 - C_all_Temp))-K4_ALG1) )
    gamma_ALG1 = gamma_r_ALG1*gamma_f_ALG1
    gamma_r_ALG2 = ((K1_ALG2*np.exp(gamma1_ALG2*(C_all_Temp - T1_ALG2)))
                /(1+K1_ALG2*np.exp(gamma1_ALG2*(C_all_Temp - T1_ALG2))-K1_ALG2) )
    gamma_f_ALG2 = ((K4_ALG2*np.exp(gamma2_ALG2*(T4_ALG2 - C_all_Temp)))
                /(1+K4_ALG2*np.exp(gamma2_ALG2*(T4_ALG2 - C_all_Temp))-K4_ALG2) )
    gamma_ALG2 = gamma_r_ALG2*gamma_f_ALG2
    gamma_r_ALG3 = ((K1_ALG3*np.exp(gamma1_ALG3*(C_all_Temp - T1_ALG3)))
                /(1+K1_ALG3*np.exp(gamma1_ALG3*(C_all_Temp - T1_ALG3))-K1_ALG3) )
    gamma_f_ALG3 = ((K4_ALG3*np.exp(gamma2_ALG3*(T4_ALG3 - C_all_Temp)))
                /(1+K4_ALG3*np.exp(gamma2_ALG3*(T4_ALG3 - C_all_Temp))-K4_ALG3) )
    gamma_ALG3 = gamma_r_ALG3*gamma_f_ALG3
    
    ###################################################################
    #   Limiting factors
    ###################################################################
    lambda_N_1 = (C_all_NH4N +  C_all_NO3N)/(K_h_N_ALG1 + C_all_NH4N +  C_all_NO3N)
    lambda_N_2 = (C_all_NH4N +  C_all_NO3N)/(K_h_N_ALG2 + C_all_NH4N +  C_all_NO3N)
    lambda_N_3 = (C_all_NH4N +  C_all_NO3N)/(K_h_N_ALG3 + C_all_NH4N +  C_all_NO3N)
    
    lambda_P_1 = (C_all_PO4P)/(K_h_P_ALG1 + C_all_PO4P)
    lambda_P_2 = (C_all_PO4P)/(K_h_P_ALG2 + C_all_PO4P)
    lambda_P_3 = (C_all_PO4P)/(K_h_P_ALG3 + C_all_PO4P)
    
    C_all_POM = C_all_LPOC + C_all_RPOC + C_all_LPON + C_all_RPON + C_all_LPOP + C_all_RPOP
    C_all_ALG = C_all_ALG1 + C_all_ALG2 + C_all_ALG3
    k_e = keb + alpha_ISS*C_all_ISS + alpha_POM*C_all_POM + alpha_ALG*C_all_ALG

    lambda_light = (2.7182/(k_e*H_dep))*(np.exp((-0.285*24*MET_solar/K_Lp)*np.exp(-k_e*H_dep)) - np.exp(-0.285*24*MET_solar/K_Lp))
    lambda_min_ALG1 = min_lambda2(lambda_N_1, lambda_P_1)*lambda_light
    lambda_min_ALG2 = min_lambda2(lambda_N_2, lambda_P_2)*lambda_light
    lambda_min_ALG3 = min_lambda2(lambda_N_3, lambda_P_3)*lambda_light
        
    K_ag_ALG1 = gamma_ALG1*lambda_min_ALG1*K_ag_max_ALG1
    K_ag_ALG2 = gamma_ALG2*lambda_min_ALG2*K_ag_max_ALG2
    K_ag_ALG3 = gamma_ALG3*lambda_min_ALG3*K_ag_max_ALG3
        
    K_ar_ALG1 = gamma_ALG1*K_ar_max_ALG1
    K_ar_ALG2 = gamma_ALG2*K_ar_max_ALG2
    K_ar_ALG3 = gamma_ALG3*K_ar_max_ALG3
        
    K_ae_ALG1 = (1-lambda_light)*gamma_ALG1*K_ae_max_ALG1
    K_ae_ALG2 = (1-lambda_light)*gamma_ALG2*K_ae_max_ALG2
    K_ae_ALG3 = (1-lambda_light)*gamma_ALG3*K_ae_max_ALG3
        
    K_am_ALG1 = gamma_ALG1*K_am_max_ALG1
    K_am_ALG2 = gamma_ALG2*K_am_max_ALG2
    K_am_ALG3 = gamma_ALG3*K_am_max_ALG3
        
    C_new_ALG1 = RK4_ALG(dt, C_all_ALG1, K_ag_ALG1, K_ar_ALG1, K_ae_ALG1, K_am_ALG1, K_s_ALG1, H_dep)
    C_new_ALG2 = RK4_ALG(dt, C_all_ALG2, K_ag_ALG2, K_ar_ALG2, K_ae_ALG2, K_am_ALG2, K_s_ALG2, H_dep)
    C_new_ALG3 = RK4_ALG(dt, C_all_ALG3, K_ag_ALG3, K_ar_ALG3, K_ae_ALG3, K_am_ALG3, K_s_ALG3, H_dep)
        
    ###################################################################
    #   Dissolved Oxygen
    ###################################################################
    K_L = K_L_cal(U_vel, H_dep, windspd)
    
    DO_1 = (K_ag_ALG1*delta_O_ALG1_ag*C_all_ALG1
            + K_ag_ALG2*delta_O_ALG2_ag*C_all_ALG2
            + K_ag_ALG3*delta_O_ALG3_ag*C_all_ALG3
            - K_ar_ALG1*delta_O_ALG1_ar*C_all_ALG1
            - K_ar_ALG2*delta_O_ALG2_ar*C_all_ALG2
            - K_ar_ALG3*delta_O_ALG3_ar*C_all_ALG3 )
    DO_2 = delta_O_OC*gamma_OC*(- K_LDOC*C_all_LDOC
                                - K_LPOC*C_all_LPOC 
                                - K_RDOC*C_all_RDOC 
                                - K_RPOC*C_all_RPOC)
    #DO_3 = - K_NH4*delta_O_NH4*gamma_NH4*C_all_NH4N - K_SOD*gamma_SOD/H_dep
    DO_3 = - K_NH4*delta_O_NH4*gamma_NH4*C_all_NH4N - K_SOD*gamma_SOD
    
    C_new_DO = RK4_DO(dt, C_all_DO, C_all_Temp, K_L, DO_1, DO_2, DO_3)
    
    ###################################################################
    #   Carbon group: LDOC, LPOC, RDOC, RPOC
    ###################################################################
    LDOC_1 = P_d_C*(K_am_ALG1*delta_C_ALG1*C_all_ALG1 
                    + K_am_ALG2*delta_C_ALG2*C_all_ALG2
                    + K_am_ALG3*delta_C_ALG3*C_all_ALG3 
                    + K_ae_ALG1*delta_C_ALG1*C_all_ALG1
                    + K_ae_ALG2*delta_C_ALG2*C_all_ALG2
                    + K_ae_ALG3*delta_C_ALG3*C_all_ALG3 )
    LPOC_1 = (1-P_d_C)*(LDOC_1/P_d_C)
    RDOC_1 = K_LDOC_to_RDOC*gamma_OC*C_all_LDOC
    RPOC_1 = K_LPOC_to_RPOC*gamma_OC*C_all_LPOC
    C_new_LDOC = RK4_LDOC(dt, C_all_LDOC, K_LDOC,
                                K_LDOC_to_RDOC, gamma_OC, LDOC_1)
    C_new_LPOC = RK4_LPOC(dt, C_all_LPOC, K_LPOC, K_LPOC_to_RPOC,
                                K_s_LPOC, H_dep, gamma_OC, LPOC_1)
    C_new_RDOC = RK4_RDOC(dt, C_all_RDOC, K_RDOC, gamma_OC, RDOC_1)
    C_new_RPOC = RK4_RPOC(dt, C_all_RPOC, K_RPOC, gamma_OC, K_s_RPOC, H_dep, RPOC_1)
    
    ###################################################################
    #   Nitrogen group: PIN, NH4, NO3, LDON, LPON, RDON, RPON
    ###################################################################
    PIN_ads = K_ads_PIN*(C_all_ISS/(K_h_PIN + C_all_ISS))
    PIN_1 = PIN_ads*C_all_NH4N

    P_NH4 = ( C_all_NH4N*(C_all_NO3N/((K_h_NH4 + C_all_NH4N)*(K_h_NH4 + C_all_NO3N))) 
             + C_all_NH4N*(K_h_NH4/((C_all_NH4N + C_all_NO3N)*(K_h_NH4 + C_all_NO3N))) )
    NH4_1 = (K_ar_ALG1*delta_N_ALG1*C_all_ALG1 
             + K_ar_ALG2*delta_N_ALG2*C_all_ALG2
             + K_ar_ALG3*delta_N_ALG3*C_all_ALG3 
             - P_NH4*(K_ag_ALG1*delta_N_ALG1*C_all_ALG1 
             + K_ag_ALG2*delta_N_ALG2*C_all_ALG2
             + K_ag_ALG3*delta_N_ALG3*C_all_ALG3 ) )
    NH4_2 = (delta_N_ON*gamma_ON*(K_LDON*C_all_LDON 
                                  + K_LPON*C_all_LPON
                                  + K_RDON*C_all_RDON 
                                  + K_RPON*C_all_RPON))
    NO3_1 = ( -(1-P_NH4)*(K_ag_ALG1*delta_N_ALG1*C_all_ALG1
                        + K_ag_ALG2*delta_N_ALG2*C_all_ALG2
                        + K_ag_ALG3*delta_N_ALG3*C_all_ALG3)
                     + K_NH4*gamma_NH4*C_all_NH4N )
    LDON_1 = P_d_N*(K_am_ALG1*delta_N_ALG1*C_all_ALG1 
                    + K_am_ALG2*delta_N_ALG2*C_all_ALG2
                    + K_am_ALG3*delta_N_ALG3*C_all_ALG3 
                    + K_ae_ALG1*delta_N_ALG1*C_all_ALG1
                    + K_ae_ALG2*delta_N_ALG2*C_all_ALG2
                    + K_ae_ALG3*delta_N_ALG3*C_all_ALG3 )
    LPON_1 = (1-P_d_N)*LDON_1
    RDON_1 = K_LDON_to_RDON*gamma_ON*C_all_LDON        
    RPON_1 = K_LPON_to_RPON*gamma_ON*C_all_LPON        
    
    C_new_PIN = RK4_PIN(dt, C_all_PIN, K_des_PIN, K_s_PIN, H_dep, PIN_1)
    C_new_NH4N = RK4_NH4(dt, C_all_NH4N, K_NH4, gamma_NH4, PIN_ads, NH4_1, NH4_2)
    C_new_NO3N = RK4_NO3(dt, C_all_NO3N, K_NO3, gamma_NO3, K_s_NO3, H_dep, NO3_1)
    C_new_LDON = RK4_LDON(dt, C_all_LDON, K_LDON,
                                K_LDON_to_RDON, gamma_ON, LDON_1)
    C_new_LPON = RK4_LPON(dt, C_all_LPON, K_LPON, K_LPON_to_RPON,
                                K_s_LPON, H_dep, gamma_ON, LPON_1)
    C_new_RDON = RK4_RDON(dt, C_all_RDON, K_RDON, gamma_ON, RDON_1)
    C_new_RPON = RK4_RPON(dt, C_all_RPON, K_RPON, gamma_ON, K_s_RPON, H_dep, RPON_1)
    
    ###################################################################
    #   Phosphorus group: PIP, PO4, LDOP, LPOP, RDOP, RPOP
    ###################################################################
    PO4_ads = K_ads_PIP*(C_all_ISS/(K_h_PIP + C_all_ISS))
    PO4_1 = (K_ar_ALG1*delta_P_ALG1*C_all_ALG1 
             + K_ar_ALG2*delta_P_ALG2*C_all_ALG2
             + K_ar_ALG3*delta_P_ALG3*C_all_ALG3 
             - K_ag_ALG1*delta_P_ALG1*C_all_ALG1 
             - K_ag_ALG2*delta_P_ALG2*C_all_ALG2
             - K_ag_ALG3*delta_P_ALG3*C_all_ALG3 )
    PO4_2 = ( K_des_PIP*C_all_PIP 
             + delta_P_OP*gamma_OP*(K_LDOP*C_all_LDOP
                                         + K_LPOP*C_all_LPOP
                                         + K_RDOP*C_all_RDOP 
                                         + K_RPOP*C_all_RPOP) )
    PIP_1 = PO4_ads*C_all_PO4P
    C_new_PIP = RK4_PIP(dt, C_all_PIP, K_des_PIP, K_s_PIP, H_dep, PIP_1)
    
    LDOP_1 = P_d_P*(K_am_ALG1*delta_P_ALG1*C_all_ALG1 
                    + K_am_ALG2*delta_P_ALG2*C_all_ALG2
                    + K_am_ALG3*delta_P_ALG3*C_all_ALG3 
                    + K_ae_ALG1*delta_P_ALG1*C_all_ALG1
                    + K_ae_ALG2*delta_P_ALG2*C_all_ALG2
                    + K_ae_ALG3*delta_P_ALG3*C_all_ALG3 )
    LPOP_1 = (1-P_d_P)*LDOP_1
    RDOP_1 = K_LDOP_to_RDOP*gamma_OP*C_all_LDOP
    RPOP_1 = K_LPOP_to_RPOP*gamma_OP*C_all_LPOP
    
   
    C_new_PO4P = RK4_PO4(dt, C_all_PO4P, PO4_ads, PO4_1, PO4_2)
    C_new_LDOP = RK4_LDOP(dt, C_all_LDOP, K_LDOP,
                                K_LDOP_to_RDOP, gamma_OP, LDOP_1)
    C_new_LPOP = RK4_LPOP(dt, C_all_LPOP, K_LPOP, K_LPOP_to_RPOP,
                                K_s_LPOP, H_dep, gamma_OP, LPOP_1)
    C_new_RDOP = RK4_RDOP(dt, C_all_RDOP, K_RDOP, gamma_OP, RDOP_1)
    C_new_RPOP = RK4_RPOP(dt, C_all_RPOP, K_RPOP, gamma_OP, K_s_RPOP, H_dep, RPOP_1)
    
    ###################################################################
    #   Inorganic Suspended Solids
    ###################################################################
    C_new_ISS = RK4_ISS(dt, C_all_ISS, K_s_ISS, H_dep)
    
    return C_new_DO, C_new_LDOC, C_new_LPOC, C_new_RDOC, C_new_RPOC, \
        C_new_PIN, C_new_NH4N, C_new_NO3N, C_new_LDON, C_new_LPON, C_new_RDON, \
        C_new_RPON, C_new_PIP, C_new_PO4P, C_new_LDOP, C_new_LPOP, C_new_RDOP, \
        C_new_RPOP, C_new_ALG1, C_new_ALG2, C_new_ALG3, C_new_ISS
