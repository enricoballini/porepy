

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import sys 
import math
from pathlib import Path
import pickle
import matplotlib.lines as mlines
import datetime 
plt.rcParams.update({'font.size': 25})
plt.style.use('seaborn')
sys.path.append('/scratch1/Diana/tesi_polito/Guendalina/Scripts/mylib')
import newecl as ecl
def load_case(fname):
      out=ecl.EclSummary(fname)
      out.load()
      return out

import datetime
def serial_date_to_string(srl_no):
    new_date = datetime.datetime(2016,9,27,00,00,00) + datetime.timedelta(srl_no )
    return new_date    




#labels = ['Initial ensemble','Matched ensemble','Observed data']
#grey = mlines.Line2D([],[],color='lightgray', label='Initial ensemble')
#grey2 = mlines.Line2D([],[],color='gray', label='GPS data')
#green = mlines.Line2D([],[],color='limegreen', label='p50')
#black = mlines.Line2D([],[],color='k', linestyle ='dashdot',label='Max/Min')
#green4 = mlines.Line2D([],[],color='lime', linestyle ='dotted',label='Max/Min')
#green3 = mlines.Line2D([],[],color='lime',linestyle= 'dashdot', label='p10/p90')
#red = mlines.Line2D([],[],color='r', marker = '*',markersize=10, label='Observed data')
#black2 = mlines.Line2D([],[],color='k', marker = '*',markersize=10, label='Observed data')        
#blu = mlines.Line2D([],[],color='b', markersize=10, label='Ref. Model')  
#green2 = mlines.Line2D([],[],linestyle='dashed',color='g',markersize=10, label='Start Forecast')  
#violet = mlines.Line2D([],[],color='violet', markersize=10, label='GPS Trend')  
#violet2 = mlines.Line2D([],[],color='violet', markersize=10, label='Start Forecast') 
#PLOT WWCT

#m = a[191:].shape[0]
##valori osservati acqua
#Ecl_Restart=np.zeros(7+m)
#dates = []
#Ecl_Restart[0] = a.iloc[26,14] 
## Ecl_Restart[1] = a.iloc[79,14]
#Ecl_Restart[1] = a.iloc[112,14]
#Ecl_Restart[2] = a.iloc[142,14]
#Ecl_Restart[3] = a.iloc[161,14] 
#Ecl_Restart[4] = a.iloc[174,14]
#Ecl_Restart[5] = a.iloc[188,14] 
#Ecl_Restart[6] = a.iloc[190,14]
#Ecl_Restart[7:]=a.iloc[191:,14]
##tempi corrispondenti (fino a fine HM 31-12-2020)
#dates.append(a.index[26])
## dates.append(a.index[79])
#dates.append(a.index[112])
#dates.append(a.index[142])
#dates.append(a.index[161])
#dates.append(a.index[174])
#dates.append(a.index[188]) 
#dates.append(a.index[190])
#for i in range(m):
#    dates.append(a.index[190+i])
#obs_wwpr = Ecl_Restart

#plt.figure(figsize=(15,10))

#y_cur = np.zeros((216,Ne))
#for i in range(Ne):
#        workdir='/scratch1/Diana/tesi_lmplus/Naide/ValidazioneHM_laura/RUN_'+str(i+1)
#        os.chdir(workdir)
#        flist = os.listdir('.')
#        cases =[ s.split('.')[0] for s in flist if s.endswith('.SMSPEC') ]
#        case_db = [ load_case(s) for s in cases ]
#        a = case_db[0].get_by_WellName('N4:CQFGB')
#        y = a.iloc[:,13].values
#        y_cur[:,i]=a.iloc[:,13].values
#        #plt.plot(a.index,y,c ='lime')
#percentiles=[10,50,90]
#y_cur_post = np.percentile(y_cur/np.max(y_cur), percentiles, axis=1)   
#y_cur_massimo = np.max(y_cur/np.max(y_cur), axis=1)
#y_cur_minimo = np.min(y_cur/np.max(y_cur), axis=1) 
#plt.plot(a.index, y_cur_post[0], color='lime', linestyle='--')[0]
#plt.plot(a.index, y_cur_post[2], color='lime', linestyle='--')[0]
#plt.plot(a.index, y_cur_post[1], color='lime', linestyle='-')[0]
#
#plt.plot(a.index,y_cur_massimo, color = 'black', linestyle ='dashdot' )
#plt.plot(a.index,y_cur_minimo, color = 'black', linestyle ='dashdot' )
    
#plt.plot(a.index,y_cur/np.max(y_cur),c='lime')
#plt.plot(a.index,y_cur_post[0,:],c='lime')
#dates_w = dates
#plt.errorbar(dates_w,obs_wwpr/np.max(y_cur),yerr = np.dot(obs_wwpr/np.max(y_cur), 0.10),fmt='o',color='r',linewidth = 2, capsize=6,zorder=10, markersize=5)
#plt.title('P2C', fontsize=30)
#plt.xlabel('Dates', fontsize=25)
#plt.ylabel('Normalized water production', fontsize=25)
#leg2 = plt.legend(handles=[green, green3, black, red, green2], fontsize = 20)
#leg2.get_lines()[1].set_linewidth(0)
#Add manual matched model
#workdir='/scratch1/Diana/tesi_lmplus/Naide/abaqus_Anna'
#os.chdir(workdir)
#flist = os.listdir('.')
#base_case =['NAIDE_2021_HM_FC_CM19_TESI']
#case_manualhm = [ load_case(s) for s in base_case ]
#df_n4 = case_manualhm[0].get_by_WellName('N4:CQFGB')
#y = df_n4.iloc[:,13].values
##plt.plot(df_n4.index[0:229],y[0:229],c ='b',label='Manual match') #ylim at 2024
#plt.vlines(df_n4.index[190],ymin = 0, ymax = 8/np.max(y_cur), linestyles='dashed', colors='g')
#


# static pressure
press_obs = pd.read_fwf('/scratch1/Diana/tesi_polito/Guendalina/Scripts/obs_press.vol', 
                        header=None, names=['Well','Date','WBHP'])
press_obs['Date']= pd.to_datetime(press_obs['Date'], format='%d-%m-%Y')

wellstot = ['2C_J3','3C_I1','3L_J3','3L_J','2AC_I1','2AL_J123']
keyw = ['WGPR','WWPR']
keyh = ['WGPRH','WWPRH']
wellspr = ['3C_I1','3L_J3','3L_J']
keypr = ['WBP9','WBP9','WBP']
field = ['FGPR','FWPR']
fieldh = ['FGPRH','FWPRH']
Nw = len(wellstot)
Nwp = len(wellspr)

wbp = np.zeros((170,Nwp))

#%%
# BASE CASE
workdir='/scratch1/Diana/tesi_polito/Guendalina/BaseCase/RUN_0'
os.chdir(workdir)
flist = os.listdir('.')
cases =[ s.split('.')[0] for s in flist if s.endswith('.SMSPEC') ]
case_db = [ load_case(s) for s in cases ]
a = case_db[0].get_by_WellName('2C_J3')
b = case_db[0].get_by_FieldKey('FGPR')
c = case_db[0].get_by_FieldKey('FWPR')

# Field quantities
fig1, ax1 = plt.subplots(1,len(field),figsize=(20,7))
for k in range(len(field)):
    flist = os.listdir('.')
    cases =[ s.split('.')[0] for s in flist if s.endswith('.SMSPEC') ]
    case_db = [ load_case(s) for s in cases ]
    sim = case_db[0].get_by_FieldKey(field[k])
    hist = case_db[0].get_by_FieldKey(fieldh[k])     
    ax1[k].plot(sim.index, sim['FIELD'].values, color='red')
    ax1[k].plot(hist.index, hist['FIELD'].values,'k.',markersize=7)
    ax1[k].legend()
    ax1[k].set_title( str(field[k]))

# Pressure
fig2, ax2 = plt.subplots(1,Nwp,figsize=(20,7))
for k in range(Nwp):
    n= press_obs[press_obs['Well']== wellspr[k]]['Date']
    x1 = pd.to_datetime(n, errors='coerce', format='%Y%d%m').to_list()
    historical1 = (press_obs[press_obs['Well']== wellspr[k]]['WBHP']).values
    flist = os.listdir('.')
    cases =[ s.split('.')[0] for s in flist if s.endswith('.SMSPEC') ]
    case_db = [ load_case(s) for s in cases ]
    a = case_db[0].get_by_WellName(wellspr[k])     
    ax2[k].plot(a.index, a[keypr[k]].values, color='red')
    ax2[k].plot(x1,historical1,'k.',markersize=12,label='WBHPH')
    ax2[k].legend()
    ax2[k].set_title('Well '+ str(wellspr[k]))

# Water and gas production
for k in range(Nw):
    fig3, ax3 = plt.subplots(1,len(keyw),figsize=(17,7))
    flist = os.listdir('.')
    cases =[ s.split('.')[0] for s in flist if s.endswith('.SMSPEC') ]
    case_db = [ load_case(s) for s in cases ]
    sim = case_db[0].get_by_WellName(wellstot[k])     
    for j in range(len(keyw)):        
        ax3[j].plot(sim.index, sim[keyw[j]].values, color='red')
        ax3[j].plot(sim.index, sim[keyh[j]].values,'k.',markersize=7)
        ax3[j].legend()
        ax3[j].set_title( str(wellstot[k])+' '+str(keyh[j]))
        
# subsidence
file = workdir + '/Node_0.txt'
subs = pd.read_fwf(file, header=None,
        names=['Node', 'Coord x', 'Coord y', 'Displ x', 'Displ y', 'Displ z'])
dates_rpt = ['2011-09-01','2012-09-01','2013-09-01','2014-09-01','2015-09-01',
'2016-09-01','2017-09-01','2018-09-01','2018-09-01','2020-01-01','2021-01-01','2022-01-01','2022-09-02']
dates_rpt = np.array(dates_rpt, dtype=np.datetime64)
fig, a = plt.subplots(figsize=(15,12))
a.plot(dates_rpt, subs['Displ z'].values,'--bo')
 

#%%
# MULTIPLE CASES (all curves plotted)
ensdir='/scratch1/Diana/tesi_polito/Guendalina/Ensemble1/RUN_'

Ne = 3


fig1, ax1 = plt.subplots(1,len(field),figsize=(20,7))
fig2, ax2 = plt.subplots(1,Nwp,figsize=(20,7))
for i in range(Ne):
    workdir = ensdir + str(i+1)
    os.chdir(workdir)
    flist = os.listdir('.')
    cases =[ s.split('.')[0] for s in flist if s.endswith('.SMSPEC') ]
    case_db = [ load_case(s) for s in cases ]
    
    # Field quantities
    for k in range(len(field)):    
        sim = case_db[0].get_by_FieldKey(field[k])
        hist = case_db[0].get_by_FieldKey(fieldh[k])     
        ax1[k].plot(sim.index, sim['FIELD'].values)
        ax1[k].plot(hist.index, hist['FIELD'].values,'k.',markersize=7)
        ax1[k].set_title( str(field[k]))

    # Pressure
    for k in range(Nwp):
        n= press_obs[press_obs['Well']== wellspr[k]]['Date']
        x1 = pd.to_datetime(n, errors='coerce', format='%Y%d%m').to_list()
        historical1 = (press_obs[press_obs['Well']== wellspr[k]]['WBHP']).values
        a = case_db[0].get_by_WellName(wellspr[k])     
        ax2[k].plot(a.index, a[keypr[k]].values)
        ax2[k].plot(x1,historical1,'k.',markersize=12)
        ax2[k].legend()
        ax2[k].set_title('Well '+ str(wellspr[k]))

# Water and gas production
for k in range(Nw):
    fig3, ax3 = plt.subplots(1,len(keyw),figsize=(17,7))
    for i in range(Ne):
        workdir = ensdir + str(i+1)
        os.chdir(workdir)
        flist = os.listdir('.')
        cases =[ s.split('.')[0] for s in flist if s.endswith('.SMSPEC') ]
        case_db = [ load_case(s) for s in cases ]
        sim = case_db[0].get_by_WellName(wellstot[k])     
        for j in range(len(keyw)):        
            ax3[j].plot(sim.index, sim[keyw[j]].values)
            ax3[j].plot(sim.index, sim[keyh[j]].values,'k.',markersize=7)
            ax3[j].legend()
            ax3[j].set_title( str(wellstot[k])+' '+str(keyh[j]))



# GPS
fig, a = plt.subplots(figsize=(15,12))
    
gps_tab = pd.read_csv('/scratch1/Diana/tesi_polito/Guendalina/GUEN_ITRF2014.csv', sep=';', encoding='cp1252')
all_dates = np.asarray(gps_tab.iloc[1:,0])
y_obs = np.array([float(s) for s in gps_tab.iloc[1:,10]])

from statsmodels.tsa.seasonal import STL
obs = pd.Series(y_obs, index=[datetime.datetime.strptime(s, '%Y-%m-%d') for s in gps_tab.iloc[1:,0]])
obs_fill = obs.resample('D').interpolate()

#fig, a = plt.subplots(figsize=(15,12))
#a.plot(obs_fill.index, (obs_fill.values-obs_fill[55])/10, label='raw data')

obs_month = obs_fill.resample('MS').mean()
a.plot(obs_month.index, (obs_month.values-obs_month[0])/10, label='month mean')

#stl_dec = STL(obs_month, robust=True)
#res = stl_dec.fit()
#gps_des = res.trend+res.resid
#a.plot(obs_month.index, (res.trend-res.trend[0])/10, label='trend def')
#a.plot(obs_month.index, (gps_des-gps_des[0])/10, label='des def')

#stl_dec = STL(obs_month,  trend = 23, robust=True)
#res = stl_dec.fit()
#gps_des = res.trend+res.resid
#a.plot(obs_month.index, (res.trend-res.trend[0])/10, label='trend 23')
#a.plot(obs_month.index, (gps_des-gps_des[0])/10, label='des 23')

stl_dec = STL(obs_month,  trend = 13, robust=True)
res = stl_dec.fit()
gps_des = res.trend+res.resid
a.plot(obs_month.index, (res.trend-res.trend[0])/10, label='trend 13')
a.plot(obs_month.index, (gps_des-gps_des[0])/10, label='des 13')


# subsidence
dates_rpt = ['2011-09-01','2012-09-01','2013-09-01','2014-09-01','2015-09-01',
'2016-09-01','2017-09-01','2018-09-01','2019-09-01','2020-01-01','2021-01-01','2022-01-01','2022-09-02']
dates_rpt = np.array(dates_rpt, dtype=np.datetime64)

for i in range(Ne):
    file = ensdir + str(i+1)+ '/Node_0.txt'
    subs = pd.read_fwf(file, header=None,
        names=['Node', 'Coord x', 'Coord y', 'Displ x', 'Displ y', 'Displ z'])
    a.plot(dates_rpt, subs['Displ z'].values*100,'--o')
a.legend()

#y_p1 = y_p[:,0,:]
#y_p2 = y_p[:,1,:]
#max_p1 =  np.max(y_p1)
#max_p2 =  np.max(y_p2)

#percentiles=[10,50,90]
#y_p1_post = np.percentile(y_p1/max_p1, percentiles, axis=1)   
#y_p1_massimo = np.max(y_p1/max_p1, axis=1)
#y_p1_minimo = np.min(y_p1/max_p1, axis=1) 
#
#y_p2_post = np.percentile(y_p2/max_p2, percentiles, axis=1)   
#y_p2_massimo = np.max(y_p2/max_p2, axis=1)
#y_p2_minimo = np.min(y_p2/max_p2, axis=1)
   
fig2, ax2 = plt.subplots(figsize=(15,10))        
#l2 = ax2.plot(a.index,y_p1,c ='lime')[0]
ax2.plot(a.index, y_p1_post[0], color='lime', linestyle='--')[0]
ax2.plot(a.index, y_p1_post[2], color='lime', linestyle='--')[0]
ax2.plot(a.index, y_p1_post[1], color='lime', linestyle='-')[0]

ax2.plot(a.index,y_p1_massimo, color = 'black', linestyle ='dashdot' )
ax2.plot(a.index,y_p1_minimo, color = 'black', linestyle ='dashdot' )
ax2.vlines(df_n4.index[190],ymin = 0.4, ymax = 1, linestyles='dashed', colors='g')
ax2.errorbar(x1,historical1/max_p1,yerr= 5/max_p1,color='r',fmt='*',capsize=5,zorder=10, markersize=10)[0]
ax2.set_title('P2C',fontsize = 30)
ax2.set_xlabel('Dates',fontsize=25)
ax2.set_ylabel('Normalized pressure',fontsize=25)

leg2 = plt.legend(handles=[green, green3, black, red, green2], fontsize = 20)

fig2, ax3 = plt.subplots(figsize=(15,10))        
#l2 = ax2.plot(a.index,y_p1,c ='lime')[0]
ax3.plot(a.index, y_p2_post[0], color='lime', linestyle='--')[0]
ax3.plot(a.index, y_p2_post[2], color='lime', linestyle='--')[0]
ax3.plot(a.index, y_p2_post[1], color='lime', linestyle='-')[0]

ax3.plot(a.index,y_p2_massimo, color = 'black', linestyle ='dashdot' )
ax3.plot(a.index,y_p2_minimo, color = 'black', linestyle ='dashdot' )
ax3.vlines(df_n4.index[190],ymin = 0.4, ymax = 1, linestyles='dashed', colors='g')
ax3.errorbar(x2,historical2/max_p2,yerr= 5/max_p2,color='r',fmt='*',capsize=5,zorder=10, markersize=10)[0]
ax3.set_title('P1C',fontsize = 30)
ax3.set_xlabel('Dates',fontsize=25)
ax3.set_ylabel('Normalized pressure',fontsize=25)

leg2 = plt.legend(handles=[green, green3, black, red, green2], fontsize = 20)
  


#dati osservati GPS
Nt = 21

Dates_rpt = ['2005-04-01','2005-06-01','2005-07-01','2006-01-01','2007-01-01', '2008-01-01',
'2009-01-01','2010-01-01','2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01',
'2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01','2021-01-01',
'2022-01-01','2022-12-31']

Dates_rpt = np.array(Dates_rpt,dtype=np.datetime64)

M = pd.read_pickle('/scratch1/Diana/tesi_lmplus/Naide/ValidazioneHM/obs_2022.pkl')
obs_gps = M['Values']/100 #To have metres
dates = np.array(pd.DataFrame(M.index))
comm_rpt,comm1, comm2 = np.intersect1d(Dates_rpt,dates,return_indices=True)
obs_gps_comm = obs_gps[comm2] 
    
#ax.plot(dates,obs_gps,'violet')

y_gps = np.zeros((21,Ne))
for i in range(Ne):
    data = pd.read_fwf('/scratch1/Diana/tesi_lmplus/Naide/ValidazioneHM_laura/RUN_'+str(i+1)+'/Node2_'+str(i+1)+'.txt', header=None, names=['Nodo','Coord x','Coord y','Spostamento lungo x','Spostamento lungo y','Spostamento lungo z'])
    y_gps[:,i] = np.array(data.iloc[:,-1].values)
    #ax.plot(Dates_rpt, y,c ='lime')
    # y_comm = y[comm1]
    # ax.plot(comm_rpt, y_comm,c ='lime')

y_gps_max = 0.03 #np.min(y_gps)
y_gps_post = np.percentile(y_gps/y_gps_max, percentiles, axis=1)
gps_massimo = np.max(y_gps/y_gps_max, axis=1)
gps_minimo = np.min(y_gps/y_gps_max, axis=1)


Dates_rpt_manual = ['2005-04-01','2006-01-01','2007-01-01', '2008-01-01',
'2009-01-01','2010-01-01','2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01',
'2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01','2020-12-31','2022-01-01',
'2023-01-01','2024-01-01','2025-01-01','2026-01-01','2027-01-01','2032-01-01','2037-01-01','2042-01-01',
'2047-01-01','2052-01-01']

Dates_rpt_manual = np.array(Dates_rpt_manual,dtype=np.datetime64)

node_manual = pd.read_fwf('/scratch1/Diana/tesi_lmplus/Naide/abaqus_Anna/nodo.txt', header=None, names=['Nodo','Coord x','Coord y','Spostamento lungo x','Spostamento lungo y','Spostamento lungo z'])
y_manual = np.array(node_manual.iloc[:,-1].values)


fig,ax = plt.subplots(figsize=(15,10))
ax.plot(Dates_rpt, y_gps_post[0], color='lime', linestyle='--')[0]
ax.plot(Dates_rpt, y_gps_post[2], color='lime', linestyle='--')[0]
ax.plot(Dates_rpt, y_gps_post[1], color='lime', linestyle='-')[0]

ax.plot(Dates_rpt,gps_massimo, color = 'black', linestyle ='dashdot' )
ax.plot(Dates_rpt,gps_minimo, color = 'black', linestyle ='dashdot' )
#ax.plot(Dates_rpt_manual[0:19], y_manual[0:19],c ='b')
ax.errorbar(comm_rpt,obs_gps_comm/y_gps_max,yerr=1/100/y_gps_max,color='r',fmt='*',capsize=5,zorder=10)
ax.vlines(Dates_rpt_manual[16],ax.get_ylim()[0], ymax =0.01, linestyles='dashed', colors='green',label='Start Forecast')
plt.xlabel('Dates', fontsize=25)
plt.ylabel('Normalized vertical displacement', fontsize=25)
plt.title('GPS', fontsize=30)
leg2 = plt.legend(handles=[green, green3, black, red, green2], fontsize = 20)
# leg = plt.legend(handles=[green, red,blu,green2,violet], fontsize = 20)
# leg.get_lines()[2].set_linewidth(0)



workdir='/scratch1/Diana/tesi_lmplus/Naide/test_laura'
os.chdir(workdir)

Naid_gps = pd.read_excel('final_table_per_Naide.xlsx', sheet_name='NAID GPS')

fig,ax = plt.subplots(figsize=(15,10))
ax.plot(Naid_gps['data'][0:4972],Naid_gps[Naid_gps.columns[2]][0:4972]/y_gps_max,color='grey',label='GPS data')
ax.plot(dates[0:186],obs_gps[0:186]/y_gps_max,linewidth=4,color='violet')
#ax.errorbar(comm_rpt,obs_gps_comm/y_gps_max,yerr=1/100/y_gps_max,color='k',fmt='*',capsize=5,zorder=10)
ax.errorbar(comm_rpt[0:17],obs_gps_comm[0:17]/y_gps_max,color='k',fmt='*',capsize=5,markersize=10,zorder=10)
plt.xlabel('Dates', fontsize=25)
plt.ylabel('Normalized vertical displacement', fontsize=25)
plt.title('GPS', fontsize=30)

leg2 = plt.legend(handles=[grey2, black2, violet], fontsize = 20)



#%% Faccio un'unica figure

fig,ax = plt.subplots(2,2,figsize=(20,18))
ax[0][0].plot(a.index, y_p1_post[0], color='lime', linestyle='dashdot')[0]
ax[0][0].plot(a.index, y_p1_post[2], color='lime', linestyle='dashdot')[0]
ax[0][0].plot(a.index, y_p1_post[1], color='limegreen', linestyle='-')[0]

ax[0][0].plot(a.index,y_p1_massimo, color = 'lime', linestyle ='dotted' )
ax[0][0].plot(a.index,y_p1_minimo, color = 'lime', linestyle ='dotted' )
ax[0][0].vlines(df_n4.index[190],ymin = 0.4, ymax = 1, linestyles='-', colors='violet',linewidth=4)
ax[0][0].errorbar(x1,historical1/max_p1,yerr= 5/max_p1,color='k',fmt='*',capsize=5,zorder=10, markersize=10)[0]
ax[0][0].set_title('P2C',fontsize = 30)
ax[0][0].set_xlabel('Dates',fontsize=25)
ax[0][0].set_ylabel('Normalized pressure',fontsize=25)
ax[0][0].set_ylim([0.6,1.02])

ax[0][1].plot(a.index, y_p2_post[0], color='lime', linestyle='dashdot')[0]
ax[0][1].plot(a.index, y_p2_post[2], color='lime', linestyle='dashdot')[0]
ax[0][1].plot(a.index, y_p2_post[1], color='limegreen', linestyle='-')[0]

ax[0][1].plot(a.index,y_p2_massimo, color = 'lime', linestyle ='dotted' )
ax[0][1].plot(a.index,y_p2_minimo, color = 'lime', linestyle ='dotted' )
ax[0][1].vlines(df_n4.index[190],ymin = 0.4, ymax = 1, linestyles='-', colors='violet',linewidth=4)
ax[0][1].errorbar(x2,historical2/max_p2,yerr= 5/max_p2,color='k',fmt='*',capsize=5,zorder=10, markersize=10)[0]
ax[0][1].set_title('P1C',fontsize = 30)
ax[0][1].set_xlabel('Dates',fontsize=25)
ax[0][1].set_ylabel('Normalized pressure',fontsize=25)
ax[0][1].set_ylim([0.6,1.02])


ax[1][0].plot(a.index, y_cur_post[0], color='lime', linestyle='dashdot')[0]
ax[1][0].plot(a.index, y_cur_post[2], color='lime', linestyle='dashdot')[0]
ax[1][0].plot(a.index, y_cur_post[1], color='limegreen', linestyle='-')[0]

ax[1][0].plot(a.index,y_cur_massimo, color = 'lime', linestyle ='dotted' )
ax[1][0].plot(a.index,y_cur_minimo, color = 'lime', linestyle ='dotted' )
    
#plt.plot(a.index,y_cur/np.max(y_cur),c='lime')
#plt.plot(a.index,y_cur_post[0,:],c='lime')
ax[1][0].errorbar(dates_w,obs_wwpr/np.max(y_cur),yerr = np.dot(obs_wwpr/np.max(y_cur), 0.10),fmt='o',color='k',linewidth = 2, capsize=6,zorder=10, markersize=5)
ax[1][0].set_title('P2C', fontsize=30)
ax[1][0].set_xlabel('Dates', fontsize=25)
ax[1][0].set_ylabel('Normalized water production', fontsize=25)
workdir='/scratch1/Diana/tesi_lmplus/Naide/abaqus_Anna'
os.chdir(workdir)
flist = os.listdir('.')
base_case =['NAIDE_2021_HM_FC_CM19_TESI']
case_manualhm = [ load_case(s) for s in base_case ]
df_n4 = case_manualhm[0].get_by_WellName('N4:CQFGB')
y = df_n4.iloc[:,13].values
#plt.plot(df_n4.index[0:229],y[0:229],c ='b',label='Manual match') #ylim at 2024
ax[1][0].vlines(df_n4.index[190],ymin = 0, ymax = 8/np.max(y_cur), linestyles='-', colors='violet',linewidth=4)
ax[1][0].set_ylim([0,1.2])

ax[1][1].plot(Dates_rpt, y_gps_post[0], color='lime', linestyle='dashdot')[0]
ax[1][1].plot(Dates_rpt, y_gps_post[2], color='lime', linestyle='dashdot')[0]
ax[1][1].plot(Dates_rpt, y_gps_post[1], color='limegreen', linestyle='-')[0]

ax[1][1].plot(Dates_rpt,gps_massimo, color = 'lime', linestyle ='dotted' )
ax[1][1].plot(Dates_rpt,gps_minimo, color = 'lime', linestyle ='dotted' )
#ax.plot(Dates_rpt_manual[0:19], y_manual[0:19],c ='b')
ax[1][1].errorbar(comm_rpt,obs_gps_comm/y_gps_max,yerr=1/100/y_gps_max,color='k',fmt='*',capsize=5,zorder=10)
ax[1][1].vlines(Dates_rpt_manual[16],ax[1][1].get_ylim()[0], ymax =-1, linestyles='-', colors='violet',linewidth=4,label='Start Forecast')
ax[1][1].set_xlabel('Dates', fontsize=25)
ax[1][1].set_ylabel('Normalized vertical displacement', fontsize=25)
ax[1][1].set_title('GPS', fontsize=30)
ax[1][1].set_ylim([-2.5,0.47])
leg2 = plt.legend(handles=[green, green3, green4, black2, violet2], fontsize = 18)

plt.savefig('/scratch1/Diana/tesi_lmplus/PUNQ/test_laura/figures/Forecast_Naide')
