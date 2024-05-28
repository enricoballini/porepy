#!/usr/bin/env python2.7
#import struct
"""
ecl is a package to rerad and do som basic operation on Eclipse binary data
Currently we support binary vectors - namely time-dependant quantities available throughout
SMSPEC and UNSMRY files
 
"""
import numpy as np
#import time
import os.path
import pandas as pd
from  datetime import date
import time
from  datetime import datetime
import matplotlib.pylab as plt
from datetime import timedelta
import scipy.interpolate
#import docstring
def read_header(fn):
    try:
        t=np.fromfile(fn,dtype='>i',count=1)
        dt=np.dtype('>c')
        x=np.fromfile(fn,dtype=dt,count=8)
        recname = x.tostring().decode('ascii')
        ndata = np.fromfile(fn,dtype='>i4',count=1)
        x=np.fromfile(fn,dtype=dt,count=4)
        rectype=x.tostring().decode('ascii')
        t=np.fromfile(fn,dtype='>i',count=1)
        if len(ndata)==0:
            return [],[],[],False
        ndata = ndata[0]

        return recname,ndata,rectype,True
    except EOFError:
        print("EOF")
        return [],[],[],False
    


def read_inte(fn,ndata):
    nblk = 1000
    ib = 0
#array of zeros of shape ndata
    vv = np.zeros(ndata,dtype=np.int)
    while ib<ndata:
        nblksize  = min(nblk,ndata-ib)  
        t  =np.fromfile(fn,dtype='>i',count=1)
        v  =np.fromfile(fn,dtype='>i',count=nblksize)
        vv[ib:ib+nblksize] = v[:] 
        t=np.fromfile(fn,dtype='>i',count=1)
        ib = ib + nblksize
    return vv
def read_real(fn,ndata):
    nblk = 1000
    ib = 0
    vv = np.zeros(ndata,dtype=np.float32)
    while ib<ndata:
        nblksize  = min(nblk,ndata-ib)  
        t =np.fromfile(fn,dtype='>i',count=1)
        v =np.fromfile(fn,dtype='>f4',count=nblksize)
        u=v.shape
        vv[ib:ib+nblksize] = v[:]
        t=np.fromfile(fn,dtype='>i',count=1)
        ib = ib + nblksize
    return vv

def read_doub(fn,ndata):
    nblk = 1000
    ib = 0
    vv = np.zeros(ndata,dtype=np.float64)
    while ib<ndata:
        nblksize  = min(nblk,ndata-ib)  
        t  =np.fromfile(fn,dtype='>i',count=1)
        v  =np.fromfile(fn,dtype='>f8',count=nblksize)
        vv[ib:ib+nblksize] = v[:] 
        ib = ib + nblksize
    return vv


def read_char(fn,ndata,nchar):
    nblk = 105
    ib = 0
    vv=np.chararray((ndata*nchar))
    while ib<ndata:
        nblksize  = min(nblk,ndata-ib)  
        t  =np.fromfile(fn,dtype='>i',count=1)
        v  =np.fromfile(fn,dtype='>c',count=nblksize*nchar)
        vv[ib*nchar:(ib+nblksize)*nchar] = v[:]
        t = np.fromfile(fn,dtype='>i',count=1)
        ib = ib + nblksize
    return vv
#def read_logi(fn,ndata):
#    return False

        
class EclipseFile:
    """
    EclipseFile is a generic class to store information stored in Eclipse binary files
    The content is simply loaded in Eclipsefile.Content list
    """
    def __init__(self,path,verbose=False):
        self.basename=path.partition('.')[0]
        self.extension=path.partition('.')[2]
        self.Content=[]
        self.eof = False
        self.Exist = False
        if not(os.path.exists(path)):
            print("error %s does not exist " % path)
            
            return None
        else:
            self.Exist = True
        self.f = open(path,"rb")
        self.verbose = verbose

    def ReadRecord(self):
        fn = self.f
        recname,ndata,rectype,ok = read_header(fn)
        if ok:
            if ndata>0:
                #print('rectype= ',rectype)
                if rectype=="INTE":
                    data = read_inte(fn,ndata)
                elif rectype=="REAL":
                    data = read_real(fn,ndata)
                elif rectype=="CHAR":
                    data = read_char(fn,ndata,8)
                    #if recname=='KEYWORDS':
                    #    for i in data:
                    #        print 'data:'+i
                elif rectype=="LOGI":
                    data = read_inte(fn,ndata)
                elif rectype=="DOUB":
                    data = read_doub(fn,ndata)
                elif rectype[0]=="C":
#                    print 'here line 122'
                    nchars = np.int(rectype[1:].lstrip('0'))
                    data = read_char(fn,ndata,nchars)
                else:
                    print(rectype[0])
                    print(" do not recognize this data: recname = ", recname," rectype = ", rectype)
                    return ['',[],[],[],False]
                return [recname,ndata,rectype,data,True]
            else:
                self.eof=False
                return [recname,ndata,rectype,[],True]
        else:
            self.eof=True
            return ['',[],[], [],False]

    def ResetVerbos(self):
        if self.verbose:
            self.verbose = False
        else:
            self.verbose = True
            
    def ReadContent(self):
        nrecord = 0
        self.eof=False
        try:
            while not self.eof:
                #ReadRecord: return [recname,ndata,rectype,data,True]
                x = self.ReadRecord()
                #if x[0]=='KEYWORDS':
                #    for i in x[3]:
                #        print 'data:'+i
                nrecord = nrecord +1
#                if (nrecord%5000==0):
#                    print " read %d records " % nrecord
                if x[4]:
                    self.Content.append([x[0],x[3],x[2]])
                    #if x[0]=='KEYWORDS':
                    #    for i in x[3]:
                    #        print 'x:'+i
                    if self.verbose:
                        s_top = "Read " + str(x[0])
                        print("%s" % s_top)
                else:
                    print("read file content")
                    if self.verbose:
                        print("%s" %("  end of file ->>>> "+ str(x[0])))
                    return
        except EOFError:
            print("END of file ")
            return

    def RestartData(self,listdata):
        if (not(self.extension == "UNRST")):
            print("this is not a restarts file")
            return
        if (len(self.Content)==0):
            print(" read file to get restart data ")
            return


        iseqnum =0
        isol = 0 
        gridName = 'COARSE'
        results=dict()
        readProp = False
        for var in listdata:
            results[var]=dict()
            
        for objects in self.Content:
            recc = objects[0].strip()
            if objects[0].strip()=='SEQNUM':
                iseqnum = iseqnum +1
                continue
            if objects[0].strip()=='STARTSOL':
                isol = isol +1
                readProp = True
                continue
            if objects[0].strip()=='ENDSOL':
                readProp = False
                gridName='COARSE'
                continue
            if objects[0].strip()=='LGR':
                gridName=objects[1].tostring()
                continue
            if recc=='INTEHEAD':
                nx,ny,nz = objects[1][8:11]
                continue
            if recc=='DOUBHEAD':
                simTime = objects[1][0]
                continue
            if (readProp):
                if objects[0].strip() in listdata:
                    t = [objects[1],simTime,[nx,ny,nz]]
                    if gridName in results[recc].keys():
                        results[recc][gridName].append(t)
                    else:
                        results[recc][gridName]=[t]
            continue
            print(iseqnum,isol)
        return results    

    def closefile(self):
        self.f.close()

class EclSummary:

    def __init__(self,path,verbose=False):
        """
        Return an EclSummary Object, the only argument is full path to model_name; usage:
            >>> ecl_summary = EclSummary('BASE') 
            if BASE.
        """

        self.basename = path
        #print('basename=',self.basename)
        smspec = self.basename + ".SMSPEC"
        unsmry = self.basename + ".UNSMRY"
        self.data = 0
        #alessia:ask non ho tanto capito wgnames non l'ho trovato nello smspec
        #Alessia:very important about wgnames
        self.listalabel = { 'KEYWORDS': 'KEYWORD' , 'WGNAMES': 'WGNAME' , 'NAMES' : 'NAMES', 'NUMS': 'NUMS','UNITS' :'UNITS'}
        #EclipseFile: it opens the file if they are present
        self.smspec = EclipseFile(smspec,verbose)
        self.unsmry = EclipseFile(unsmry,verbose)
        if (self.smspec.Exist is False or self.unsmry.Exist is None):
            print("Errori in %s EclSummary() object can not be created" % func_name())
            self.CanLoad = False
        else:
            self.CanLoad = True
        self.startdate = 0
        self.labels = dict()
        self.nlist=0
        self.nx=0
        self.ny=0
        self.nz=0
        self.reportstep = None
        self.tempo = None
        self.Wells = pd.DataFrame()
        self.Seg = pd.DataFrame()
        self.Field = pd.DataFrame()
        self.Group = pd.DataFrame()
        self.BGrid = pd.DataFrame()
        self.Connections = pd.DataFrame()
        #empy list, but then it will bocome an array
        self.dates = []
        self.verbose=False

    def load(self,verbose=False):
        """
        Read the content of smspec and unsmry and write to a set of pandas dataframe;
        Wells, Field, Group, Connections and BGroup dataframes; usage:
            >>> ecl_summary.load(verbose)
            verbose is by default False
        """
        if verbose:
            self.verbose=True
        seqhdr = False
        if (not self.CanLoad):
            print("\n\n Error in %s \n UNSMRY/SMSPEC files not available " 
                  % func_name()  )
            return None
        time_now= time.time()
        time_now1 = time_now
        self.smspec.ReadContent()
        time_now = time.time()
        elapsed = time_now - time_now1
        time_now1 = time_now
        print(" elapsed time to read smspec %g " % elapsed)        
        #print 'len(smspec):' +str(len(self.smspec.Content))
        #for i in self.smspec.Content:
        #    print 'i:'+i[0]
        self.unsmry.ReadContent()
        time_now = time.time()
        elapsed = time_now - time_now1
        time_now1 = time_now
        if (verbose):
            print(" elapsed time to read unsmry %g " % elapsed)  
        #print 'len(unsmry):'+ str(len(self.unsmry.Content))
        nsteps =0
        nparam =[]
        for obj in self.unsmry.Content:
            #Content.append([x[0],x[3],x[2]]):x[0] recname, x[3] data,x[2] rectype
            if obj[0].strip()=="PARAMS":
                nsteps = nsteps+1
                a=obj[1].shape
                #a[0] is the number of rows
                nparam.append(a[0])
        if nsteps==0:
            print("Error in the content of unsmry file ")
            return
        #nparam counts the number of data printed at each timestep
        nparam = np.array(nparam,dtype=np.int)
        nblk = nparam[0]
        #print('nblk=', nblk)
        data = np.zeros((nsteps, nblk), dtype=np.float32)
        i = -1
        ireport = 0
        tab_report=[]
        for obj in self.unsmry.Content:
           
            if obj[0].strip()=="SEQHDR":
                seqhdr = True
                
            if obj[0].strip()=="PARAMS":
                i=i+1
                data[i][:]=obj[1].copy()
                if (seqhdr):
                    tab_report.append(i)
                    ireport=ireport +1
                    seqhdr = False
        
        self.data = data
        lista = self.listalabel
        listaKeys = list(lista.keys())
        #for l_tmp in listaKeys:
        #    print 'line 293 l_tmp:'+l_tmp
        #smspec
        for obj in self.smspec.Content:
            if obj[0].strip()=="STARTDAT":
                if len(obj[1])==3:
                    day,month,year  = obj[1].tolist()
                    hour = 0
                    minute = 0
                    second = 0
                    self.startdate = datetime(year,month,day,hour,minute,second)
#  remember in python first day - day 1 is 1-1-0001
                if len(obj[1])==6:
                    day,month,year,hour,minute,second  = obj[1].tolist()
                    # second are in microseconds
                    seconds = second * 1.e-6
                    seconds = np.int(seconds)
                    self.startdate = datetime(year,month,day,hour,minute,seconds)
                if self.verbose:
                    s_toprint = "simulations begins on " + str(self.startdate)
                    print(s_toprint)
            if obj[0].strip()=="DIMENS":
                self.nlist = obj[1][0]
                self.nx = obj[1][1]
                self.ny = obj[1][2]
                self.nz = obj[1][3]
            if len(listaKeys)==0:
                break
            if obj[0].strip() in listaKeys:
                listaKeys.remove(obj[0].strip())
                if obj[0].strip()=="NUMS":
                    self.labels["NUMS"] = obj[1].copy()
                else:
                    if obj[2]=='CHAR':
                        nchars = 8
                    else:
                        nchars = np.int(obj[2][1:].lstrip('0'))
                    v = np.chararray((len(obj[1]),nchars))
                    self.labels[obj[0].strip()]=[]
                    #if obj[0]=='KEYWORDS':
                    #    print 'shape obj[1]:'+str(obj[1].shape)
                    #    for l_tmp in obj[1]:
                    #        print 'l_tmp:' + l_tmp
                    v =np.reshape(obj[1], (self.nlist, nchars))
                    #if obj[0]=='KEYWORDS':
                    #    print 'shape of v:'+str(v.shape)
                    #    print 'v:'+str(v)
                    tmp_list =[]
                    #print 'obj[0].strip():'+obj[0].strip()
                    for i in range(self.nlist):
                        s = v[i,:].tostring().strip().decode()
                        #print 'line 339 s:'+s
                        #print('i:' + str(i) + 'v:' + v[i,:])
                        #alessia:I do not think this is necessary
                        #s=s.strip('\x00')
                        tmp_list.append(s)
                    mylabel = lista[obj[0].strip()]
                    #print 'line 345 mylabel:'+ mylabel
                    self.labels[mylabel] = np.array(tmp_list)
                    #if mylabel == 'KEYWORD':
                        #for l_tmp in self.labels[mylabel]:
                            #print 'line 350 l_tmp='+l_tmp
        #preparing a datetime list
        l = [timedelta(np.float(x)) for x in self.data[:,0]]
        self.tempo =  self.data[:,0].copy()
        l = [self.startdate + x for x in l]
        self.dates = np.array(l)
        self.reportstep = self.dates[tab_report]
        if (True):
            print(" found %d report steps and %d time-step records - this does not make sense for Echelon" % (len(tab_report), len(self.dates)))
        
        #print("Cleaning file content ")
        #closing file
        self.smspec.closefile()
        self.unsmry.closefile()
        del self.smspec
        del self.unsmry
        self.Wells = self.makeDF('W')
        self.Field = self.makeDF('F')
        self.Group = self.makeDF('G')
        self.BGrid = self.makeDF('B')
        self.Region = self.makeDF('R')
        self.Seg = self.makeDF('S')
        self.Connections = self.makeDF('C')
        time_now = time.time()
        elapsed = time_now - time_now1
        time_now1 = time_now
        print(" elapsed time to arrange data structures %g " % elapsed) 
        if verbose:
            KwListDict=dict()
            #printing the keywords for each DataFrame
            L = [self.Wells, self.Group, self.Field,self.BGrid,self.Connections]
            L_name=['Wells','Group','Field','BGrid','Connections']
            for i,f in enumerate(L):
                if len(f)>0:
                    
                    to_print='Possible keywords for '+L_name[i]+':'
                    print(to_print)
                    s_k = set(f.columns.get_level_values(1))
                    for s_s in s_k:
                        print("%s " %s_s,end='')
                        
                    print('\n')
                    KwListDict[L_name[i]] = s_k
            return(KwListDict)
            
    def well_names(self):
        lista = list(self.Wells.columns.levels[0])
        return lista
    def index_tempo(self,df):
        df.index = self.tempo
        df.index.name ='time (days)'
        return df
    
    def get_calendar_rate(self,kw,freq=None):
        """
      
        given an eclipse keyword (kw) ending with T this ,ethod return the correponding calendar rate
        by default the time-line is the keyword timeline, but a frequency (y/m/d) can be specified
        admissible values for freq are the ones accepted by pd.date_range(start,end,freq=freq)
        
        """
        # first check if keyword ends with T 
        T=kw.endswith('T')
        ok = False
        df = None
        if (not T):
            print("sorry can not make calendar for %s \n" % kw)
            return None
        beg = kw.startswith('F')
        if (beg):
            df = self.get_by_FieldKey(kw)
            ok = True
        beg = kw.startswith('W')              
        if (beg):
            df = self.get_by_WellKey(kw)
            ok = True
        beg = kw.startswith('R')
        if (beg):
            df = self.get_by_RegiondKey(kw)
            ok = True
        beg = kw.startswith('G')
        if (beg):
            df = self.get_by_GoupKey(kw)
            ok = True
        if ( df is None):
            ok = False
        if (not ok ):
            print("sorry can not make calendar for %s \n" % kw)
            return None
        if (not freq is None):
            start = df.index[0]
            end =  df.index[-1]
        
            print(freq,start,end)
            calendar = pd.date_range(start,end,freq=freq)
            if (start<calendar[0]):
                calendar=np.insert(calendar,0,start)
            if (end > calendar[-1]):
                calendar = np.insert(calendar,calendar.shape[0],end)  
            
        else:
            calendar = None
        
        df1 = avg_rate(df,calendar)
        return df1
    
    def active_well_tab_summary(self,tipo='PROD',freq='all'):
        
        df = self.active_well_tab(tipo,freq)
        
        df = df.sum(axis=1)
        return df
        
        
    def active_well_tab(self,tipo='PROD',freq='all',thresh=10):
        """
    this method compute a well activity table for the simulation wells
    the table consists of 0/1 (0 if it is inactive, 1 if it is active)
    activity is given by cumulative rates. It is possible to define:
    well type - tipo can be 'PROD', 'WINJ', 'GINJ'
    freq can be 'all' or a frequency 
    thresh a threshold values to evaluate if in atime period the well is active or not
    Note: if freq='all' time lins is simulation timeline - vectors index - at time=0 all wells 
    are inactive, then activity is given for time period between index[time] and index[time-1]
    
    
       """
        if (tipo=='PROD'):
            df = self.get_calendar_rate('WOPT')
        elif (tipo=='WINJ'):
            df = self.get_calendar_rate('WWIT')
        elif (tipo=='GINJ'):
            df = self.get_calendar_rate('WGIT')
        else: 
            print(" do not kno how to deal with %s frequency " % freq)
            return None
        
        if (not isinstance(df,pd.DataFrame)):
            print(" error for well type: %s - no cumulative keyword to find data " % tipo)
            return None
        df=df.loc[:,df.sum()>thresh]
        df = df.iloc[:-1,:]
        df = df >0.
        df = df.astype(np.int)
        print(freq)
        if (freq=='all'):
            return df
        elif (freq=='year'):
           df = df.groupby([df.index.year]).max()  
        elif (freq=='month'):
           df = df.groupby([df.index.month]).max() 
        
        return df
    
             
    def well_life(self,tip1='PROD'):
        
        df = self.active_well_tab(tipo=tip1)
        if (not isinstance(df,pd.DataFrame)):
            print(" error in method well_life: cannot compute data for %ss wells " % tip1)
            return None
        nc = df.shape[1]
        init = np.zeros(nc,dtype='datetime64[ns]')
        fine = np.zeros(nc,dtype='datetime64[ns]')
        for i in range(nc):
            init[i]=df.index[df.iloc[:,i]>0][0]
            fine[i]=df.index[df.iloc[:,i]>0][-1]
        dff = pd.DataFrame({'init': init, 'fine' : fine }, index = df.columns)
        return dff
        
    def get_by_key(self,df,key,levelname):
        
        """low level function that return a dataframe with the values of key using levelname that can be KEYWORDS or NANES (
     me)"""
        Tab = [(i) for (i,x) in enumerate(df.columns.names) if x == levelname]
        if len(Tab)==0:
            print('levelname=', levelname, ' not found')
            return None
        if len(Tab)>1:
            print('error, Tab has length bigger than 1')
            return None
        #since you know that Tab will have length 1
        i = Tab[0]
        df.columns = df.columns.swaplevel(0, i)
        #alessia:why this sort is necessary?
        df.sort_index(level=0, axis=1, inplace=True)
        if key not in df.columns.levels[0]:
            print("Error - %s  non in SUMMARY keys list:" % key)
            for ss in  df.columns.levels[0]:
                print("%s " % ss)
            print("\n")
            df.columns = df.columns.swaplevel(0, i)
            df.sort_index(level=0, axis=1, inplace=True)
#            return pd.DataFrame()
            return None
#        elif( not key[0:0] == 'R'):
            
#        W = df[key].copy()
        #Lista =df.columns.levels[0]==key
        #W = df.iloc[:,Lista]
        W = df[key]
        if ( not isinstance(W,pd.core.frame.DataFrame) ):
            W = pd.DataFrame(data=W.iloc[:])
#        W.columns = W.columns.droplevel(1)
        df.columns = df.columns.swaplevel(0, i)
        #alessia:why this sort is necessary?
        df.sort_index(level=0, axis=1, inplace=True)
        
        return W
    def __str__(self):
        s = ""
        for w in self.get_WellsName():
            s = s + w + "\n"
        return s
    def WellsNames(self):
        """ print available wells in a list:
            >>> EclSummary.print_WelsName()
        
        """

        for w in self.get_WellsName():
            print("%s" % w)
        print( '\n')

        
    def get_FieldProps(self,verbose=False):
        """ returns available field properties in a list:
            >>> EclSummary.get_FieldProps()
        
        """
        ia =[ i for (i,s) in enumerate(self.Field.columns.levels) if s.name== "KEYWORDS" ]
        ia = ia[0]
        w =  set(self.Field.columns.get_level_values(ia))
        w = list(w)
        if (verbose):
            for s in w:
                print("%s" % s)
                
        return w  
    def get_WellsName(self):
        """print all well names:
            >>> ecl_summary.print_WelsName()"""
        w =  set(self.Wells.columns.get_level_values(0))
        w = list(w)
        return w
            
    def get_by_WellName(self,name):
        """ return a pandas dataframe with all available vectors for wells name
        EclSummary.get_by_WellName(wellname)
        
        """
        if len(self.Wells)>0:
            W = self.get_by_key(self.Wells, name, 'NAMES')
        else:
            print('no data for Wells')
            W = pd.DataFrame()
        return W
            
    def get_by_WellKey(self,key):
        """load results for key in a dataframe, one column for each well"""
        if len(self.Wells)>0:
            W = self.get_by_key(self.Wells, key, 'KEYWORDS')
            
        else:
            print('no data for Wells')
            W = None
        return W

    def get_by_FieldKey(self,key):
        """load results for key in a dataframe"""
        if len(self.Field)>0:
            f = self.get_by_key(self.Field, key, 'KEYWORDS')
            if (f is None):
                return None
            if len(f.shape) > 1:
                if (f.shape[1]>1):
                    f=f.iloc[:,0]
        else:
            print('no data for Field key %s '% key)
            f =None
        return f



    def get_by_RegKey(self,key):
        """load results for key in a dataframe"""
        if len(self.Field)>0:
            f = self.get_by_key(self.Region, key, 'KEYWORDS')
            nreg = f.shape[1]
            print(" available %d regions with indices" % nreg )
            for s in f.columns:
                ie = s[0]
                print("%d " %ie,end='')
            print("\n")
        else:
            print('no data for Field')
            f = None
        return f


    def get_by_BGridKey(self,keyword=None,  wname=None,ijk=None ):
        """load results for key in a dataframe, one column for each requested cell
        >>> ecl_summary.get_by_BGridKey('BPR') """
        if len(self.BGrid)>0:
            b_g = self.getConData(wname=None,keyword=keyword,ijk=ijk,datatype='blockdata')
#            b_g = self.get_by_key(self.BGrid, key, 'KEYWORDS')
            #renominate the columns starting from NUMS
        else:
            print('no data for BGrid')
            b_g = pd.DataFrame()
        return b_g

    def get_by_GroupKey(self,key):
        """load results for key in a dataframe, one column for each group"""
        if len(self.Group)>0:
            g = self.get_by_key(self.Group, key, 'KEYWORDS')
        else:
            print('no data for Group')
            g = pd.DataFrame()
        return g
    
    def list_keywords(self,s_char=None):
        """information of all possible keywords starting with s_char"""
        if (s_char==None):
            s_char_list= [ 'W', 'G', 'F', 'B','C']
        elif s_char not in ['W', 'G', 'F', 'B','C']:
            print( "  keywords beginning with %s non considered " % s_char)
            return None
        else:
            s_char_list = list(s_char)
        L = [self.Wells, self.Group, self.Field, self.BGrid,self.Connections]
        out_list = dict()
        for s_char in s_char_list:
            
            i = ['W', 'G', 'F', 'B','C'].index(s_char)
            f = L[i]
            if len(f)>0:
                to_print='Possible keywords for '+s_char+':'
                print(to_print)
                s_k = set(f.columns.get_level_values(1))
                out_list[s_char]=s_k
            else:
                print("%s\n" %( 'no x keywords starting with ' + s_char))
        return out_list
    
    
    def print_keywords_startingwith(self,s_char=None):
        """print information of all possible keywords starting with W/G/F/B """
        out=self.list_keywords(s_char)
        if (not out == None):
            k = out.keys()
            for s in k:
                for s1 in out[s]:
                    print("%s" % s1)

#alessia:cosa fa questo info?
    def info(self, wname=None, wprop=None, verbose=True):
        outinfo = dict()
        if (wname==None):
            wname=list(set(self.labels['WGNAMES']))
        
        if (wprop==None):
            wprop=list(set(self.labels['KEYWORDS']))


        if (not(isinstance(wname,(list,tuple)))):
            wname=[wname]
        if (not(isinstance(wprop,(list,tuple)))):
            wprop=[wprop]
        
        selected_prop = [i for i, x in enumerate(self.labels['KEYWORDS']) if x in wprop]


        for w in wname:

            selw=[i for i, x in enumerate(self.labels['WGNAMES']) if x==w]
            if (w==":+:+:+:+"):
                # print(self.labels['KEYWORDS'])
                selp=[i for i, x in enumerate(self.labels['KEYWORDS']) if x[0] not in ['B','W','F', 'L','R'] and x != "TIME" ]
                selw = [val for val in selp if val in selw]
            
            selw = [val for val in selected_prop if val in selw]
            if(verbose):
                for i in selw:
                    print("well %s  with prop %s " % (w,self.labels['KEYWORDS'][i]))
            if (len(selw)>0):
                outinfo [w] = selw
        timepos = [i for i, x in enumerate(self.labels['KEYWORDS']) if x == "TIME" ]
        if len(timepos) != 1:
            print(" error in smspec file - no TIME keyword available ")
            return None

        return outinfo,timepos[0]

    def makeDF(self,starting_char):
        if starting_char=='W':
            df=self.makeWellDF()
        elif starting_char=='F':
            df=self.makeFieldDF()
        elif starting_char=='G':
            df=self.makeGroupDF()
        elif starting_char=='B':
            df=self.makeBGridDF()
        elif starting_char =='R':
            df=self.makeBGridDF('R')
        elif starting_char =='C':
            df=self.makeBGridDF('C')
        elif starting_char =='S':
            df=self.makeBGridDF('S')

        else:
            to_s = "char:" + starting_char+ " not recognized"
            df=pd.DataFrame()
            
        return df

    def makeWellDF(self):
        Kw, Units,Names,Data , Nums =self.getTableByProp('W')
        if Data.shape[1]>0:
            DF = pd.DataFrame(Data)
            DF.index = self.dates
            DF.columns = pd.MultiIndex.from_tuples(list(zip(Names,Kw,Units)))
            DF.columns.names=['NAMES','KEYWORDS','UNITS']
# to fix - it is useful but nowt always
##            DF=DF.T.drop_duplicates().T

        else:
            if self.verbose:
                print('no Keyword starting with W WellDF is empty')
            DF = pd.DataFrame()
        return DF

    def makeFieldDF(self):
        Kw, Units,Names,Data, Nums =self.getTableByProp('F')
        if Data.shape[1]>0:
            FieldDF = pd.DataFrame(Data)
            FieldDF.index = self.dates
           #print('shape FieldDF=', FieldDF.shape)
           #print('lis_zip:', list(zip(Names,Kw,Units)))
            FieldDF.columns = pd.MultiIndex.from_tuples(list(zip(Names, Kw, Units)))
           #print('index=', FieldDF.columns)
            FieldDF.columns.names = ['NAMES', 'KEYWORDS', 'UNITS']
            #print('FieldDF:')
            #print(FieldDF)
            #print(FieldDF['FIELD']['FOPT'])
        else:
            if self.verbose:
                print('no Keyword starting with F Field is empty')
            FieldDF = pd.DataFrame()
        return FieldDF

    def makeBGridDF(self,proptype='B'):
        
        Kw, Units,Names,Data,Nums =self.getTableByProp(proptype)
        
        #print(Names)
        if Data.shape[1]>0:
            BGridDF = pd.DataFrame(Data)
            BGridDF.index = self.dates
            #print(Names)
            print(proptype)
            if (proptype=='C' or  proptype == 'B' or proptype=='S' ):

                BGridDF.columns = pd.MultiIndex.from_tuples(list(zip(Names, Kw, Nums,Units)))   
                BGridDF.columns.names = ['NAMES', 'KEYWORDS', 'NUMS','UNITS']
            else:    
                BGridDF.columns = pd.MultiIndex.from_tuples(list(zip(Names, Kw, Units)))
            #print(BGridDF.columns)
           #print('index=', FieldDF.columns)
                BGridDF.columns.names = ['NAMES', 'KEYWORDS', 'UNITS']
                
        else:
            if self.verbose:
                print('no Keyword starting with %s - return empty data-frame' % proptype)
            BGridDF = pd.DataFrame()
        return BGridDF

    def get_table(self,kw,select=False):
 
        Kw, Units,Names,Data ,Nums =self.getTableByProp(kw[0])

        if Data.shape[1]>0:
            DF = pd.DataFrame(Data)
            DF.index = self.dates
            DF.columns = pd.MultiIndex.from_tuples(list(zip(Names,Kw,Units)))
            DF.columns.names = ['NAMES', 'KEYWORDS', 'UNITS']
            DF.columns=DF.columns.swaplevel(0,1)
            
            try:
                DF = DF[kw]
            except:
                print("problem with keyword %s  most likely it is not in the summary" % kw)
                return None
        else:
            if self.verbose:
                print('no Keyword starting with %s ' % kw[0])
            return None
        if (select):
            if (len(DF.shape)>1):
                DF = DF.iloc[:,0]
            
        return DF


    def makeGroupDF(self):
        Kw, Units,Names,Data,Nums =self.getTableByProp('G')
        if Data.shape[1]>0:
            GroupDF = pd.DataFrame(Data)
            GroupDF.index = self.dates
            GroupDF.columns = pd.MultiIndex.from_tuples(list(zip(Names,Kw,Units)))
            GroupDF.columns.names = ['NAMES', 'KEYWORDS', 'UNITS']
        else:
            if self.verbose:
                print('no Keyword starting with G GroupDF is empty')
            GroupDF = pd.DataFrame()
        return GroupDF
     
    def getTableByProp(self,prop):
        
        if prop not in ['E','F','G','W','B','R','N','T','M','S','C']:
            return None
        List1 =[x.startswith(prop) for x in self.labels['KEYWORD']]

        ListKw = List1
        Nums = None
        if (prop in ['G','W','S']):
            if 'NAMES' in self.labels.keys():
                ListNoblanks = [x.startswith(':+:+:+:+') for x in self.labels['NAMES']]    
                ListNoblanks = np.array(ListNoblanks)
                ListNoblanks = np.logical_not(ListNoblanks)
                ListKw = np.logical_and(ListNoblanks, ListKw)

        if len(ListKw) > 0:
            ListKw = np.array(ListKw)
            

            Kw     = self.labels['KEYWORD'][ListKw]
            Units  = self.labels['UNITS'][ListKw]
            Data   = self.data[:, ListKw]
            
#            if prop == 'B' :
            if prop == 'X' :
                Names = self.labels['NUMS'][ListKw]
                #print(Names.shape)
                Names_l = list()
                #converting the numbers in nums in i j k: Names becomes a list of lists
                for i,n in enumerate(Names):
                    #to_p = 'Names['+str(i)+']='+str(Names[i])
                    #print(to_p)
                    k = int(n/(self.nx*self.ny))
                    r = int(n%(self.nx*self.ny))
                    if r>0:
                        k = k+1
                    j = int(r/self.nx)
                    i_j = int(r%self.nx)
                    if i_j>0:
                        j=j+1
                    if j==0:
                        j=self.ny
                    if i_j==0:
                        i_j=self.nx
                    #print(str(i_j)+' '+str(j+1)+' '+str(k+1))
                    Names_l.append(str(i_j)+' '+str(j)+' '+str(k))
                    #print('names'+Names_l[i])
                #print(Names_l)
                Names = np.array(Names_l)
            elif prop == 'R':
                Names = self.labels['NUMS'][ListKw]
            elif prop == 'W':
                if 'WGNAME' in self.labels.keys():
                    Names = self.labels['WGNAME'][ListKw]
                elif 'NAMES' in self.labels.keys():
                    Names = self.labels['NAMES'][ListKw]
            elif prop == 'F':
                if 'NAMES' in self.labels.keys():
                    Names = self.labels['NAMES'][ListKw]
                else:
                    Names = ['FIELD']*len(ListKw)
            elif prop == 'C' or prop == 'B' or prop== 'S':
                if 'WGNAME' in self.labels.keys():
                    Names = self.labels['WGNAME'][ListKw]
                elif 'NAMES' in self.labels.keys():
                    Names = self.labels['NAMES'][ListKw]
                Nums =  self.labels['NUMS'][ListKw]
                Filtro = Nums >0
                Data= Data[:,Filtro]
                Kw = Kw[Filtro]
                Names = Names[Filtro]
                Units=Units[Filtro]
                Nums = Nums[Filtro]
            else:
                if 'WGNAME' in self.labels.keys():
                    Names = self.labels['WGNAME'][ListKw]
                elif 'NAMES' in self.labels.keys():
                    Names = self.labels['NAMES'][ListKw]
        else:
            Kw=[]
            Units=[]
            Data=[]
            Names=[]
            Nums = None
        return Kw, Units,Names,Data,Nums
        
    def getConData(self,wname=None,keyword=None,ijk=None,datatype='Connection'):
             if (datatype=='Connection'):
                 X = self.Connections.copy()
             elif (datatype =='blockdata'):
                 X = self.BGrid.copy()
             else:
                 print(" error in method getCondata: datatype %s not availale " % datatype)
                 return None
             if (not wname is None):
                 
                 if (wname not in X.columns.levels[0]):
                     print(" Error in function %s:  %s is not a well where connections are available "
                           % (func_name(),wname))
                     print(" available wells are:")
                     for s in X.columns.levels[0]:
                         print("%s" % s)
                     return None
                 print("The available wells are:")
                 for s in X.columns.levels[0]:
                      print("%s" % s)
                 X=X[wname]

             if (not keyword is None):
                 levelname = 'KEYWORDS'
                 print(X.columns.names)
                 Tab = [(i) for (i,x) in enumerate(X.columns.names) if x == levelname]
                 ie = Tab[0]
                 X.columns = X.columns.swaplevel(0, ie)
                 try:
                     X=X[keyword]
                     
                 except:
                    print("keyword %s  not found\n available %s  are: " % (keyword,datatype))
                    V = X.columns.levels[0].copy()
                    
                    for x in V:
                        print('%s'  % x)
                    return None
             if (not ijk is None):
                levelname = 'NUMS'
                i,j,k = ijk
                l = i + self.nx*(j-1)+(k-1)*self.nx*self.ny
                Tab = [(i) for (i,x) in enumerate(X.columns.names) if x == levelname ]
                ie = Tab[0]
                X.columns = X.columns.swaplevel(0, ie)
                try:
                    X=X[l]
                except:
                    print("%s not found\n available connections are: " % datatype)
                    V = X.columns.levels[0].copy()
                    V =self.ijk(V)
                    for x in V:
                        print('%d %d %d \n' % (x[0],x[1],x[2]))
                    return None
             if (datatype=='blockdata' or datatype=='Connection'):
                
                Tab = [(i) for (i,x) in enumerate(X.columns.names) if x == "NUMS"]
                ie = Tab[0]
                # recreate vectors 
                nd = len(X.columns.names)
                L=[]
                for i in range(nd):
                    L.append(list())
                j=0
                for cols in X:
                    
                    for i in range(nd):
                        L[i].append(cols[i])
                numeri = np.array(L[ie])
                
                V = self.ijk(numeri)
                V = [(tuple(s)) for s in V]
                L[ie] = V
                y = L
                if (nd==3):
                    C = pd.MultiIndex.from_tuples(list(zip(y[0],y[1],y[2])))
                elif (nd==2):
                    C = pd.MultiIndex.from_tuples(list(zip(y[0],y[1])))
                elif (nd==1):
                    C = pd.MultiIndex.from_tuples(list(zip(y[0])))
                else:
                     print(" errr in %s nd is %d " % (func_name(),nd ))
                     return None
                C.names = X.columns.names
                X.columns = C
                return X
                
#                 return V,X
#                 W= X.columns.set_levels(V.tolist(),level=ie).copy()
#                 
#                 X.columns = W
#                 
##                 X.columns.levels[ie] = V
                 
             return X
    def ijk(self,V):
        IJK = to_ijk(self.nx,self.ny,self.nz,V)
        return IJK                

def get_by_keywords(ecl_summary, keywd):

    """ Return a data frame with the values of keywd; Usage: 
        >>> wopr = ecllib.get_by_keywords(ecl_summary,'WOPR') """

    if keywd.startswith('W'):
        w = ecl_summary.get_by_WellKey(keywd)
        return w
    elif keywd.startswith('F'):
        f = ecl_summary.get_by_FieldKey(keywd)
        return f
    elif keywd.startswith('G'):
        g = ecl_summary.get_by_GroupKey(keywd)
        return g
    elif keywd.startswith('B'):
        b = ecl_summary.get_by_BGridKey(keywd)
        return b
    else:
        print( '%s key not recognized' % keywd)
        return []
def resample(df,dates):
    t=(df.index-dates[0]).total_seconds()
    tt = (dates-dates[0]).total_seconds()
    f=scipy.interpolate.interp1d(t,df.iloc[:,:],axis=0)
    d = f(tt)
    dfloc = pd.DataFrame(data=d,columns=df.columns,index=dates)
    return dfloc

def avg_rate(dft,dates=None,return_iterpolated=False):
    """
    compute calendar (average) rates for cumulative in a dataframe dft
    cumulative can be interpolated if suitable dates are provided
    
    """
#    if (isinstance(dates,np.ndarray)):
    if ( dates is not None):
        t=(dft.index-dates[0]).total_seconds()
        tt = (dates-dates[0]).total_seconds()
        f=scipy.interpolate.interp1d(t,dft.iloc[:,:],axis=0)
        d = f(tt)
        dfloc = pd.DataFrame(data=d,columns=dft.columns,index=dates)
    else:
        dfloc=dft.copy()
       
    t = dfloc.index    
    dt=t[1:]-t[:-1]
    dt=dt.total_seconds()
    dt = np.array(dt)/24./3600.
    df1 = dfloc.diff().fillna(0.)
    nr = dt.shape[0]
    x = np.zeros((nr+1,dfloc.shape[1]))
    x[1:,:] = df1.iloc[1:,:].values/dt[:,np.newaxis]
    df1.loc[1:,:] = x[1:,:]
    
    
    return df1

def func_name():
    import traceback
    return traceback.extract_stack(None, 2)[0][2]

def units(df):
    try:
        lev = df.columns.levels    
        for s in lev:
            if s.name =='UNITS':
                return s[0]
                break
    except:
        try:
            lev = df.columns[0]
            return lev
        except:
            return None
    return None
def to_ijk(nx,ny,nz,Vector):
    ijk = np.zeros((len(Vector),3),dtype=np.int)
    l =-1;
    for i,n in enumerate(Vector):
        l=l+1
        ie = n
        ll = int(ie/(nx*ny))
        m = int(ie%(nx*ny))
        if m>0:
            ll = ll +1
        k = ll
        
        ie = ie - (k-1)*nx*ny
        
        ll = int(ie/(nx))
        m = int(ie%(nx))
        if m>0:
            ll = ll +1
        j=ll
        
        ie = ie - nx*(j-1)
        
        i = ie

        ijk[l,:] = i,j,k
    return ijk
    
class EgridFile(EclipseFile):
    def isegrid(self):
        if self.extension == 'EGRID':
         print('EGRID')
    def pack_data(self):
        if (len(self.Content) ==0  ):
            self.ReadContent()
            
        keys =[]
        values =[]
        for s in self.Content:
            keys.append(s[0])
            values.append(s[1])
            
        for (key, value) in zip(keys, values):
            key=key.strip()
            self.__dict__[key] = value
        nx,ny,nz = self.GRIDHEAD[1:4]
        self.nx  = nx
        self.ny = ny
        self.nz = nz
#                         
        self.Content =[]
    def to_grdecl(self,filename):
        f = open(filename,'w')
        f.write('COORD\n')
        
        x= np.reshape(self.COORD,(np.int(((self.nx+1)*(self.ny+1))),6))
        np.savetxt(f,x,fmt='%g %g %g %g %g %g')
        f.write('/\n\n')
        f.write('ZCORN\n')
        x= np.reshape(self.ZCORN,(self.nx*self.ny*self.nz*4,2))
        np.savetxt(f,x,fmt=' %g %g')
        f.write('/\n\n')
        f.write('ACTNUM\n')
        np.savetxt(f,self.ACTNUM,fmt='%d')
        f.write('/\n\n')
        f.close()
        
        
    def write_grdecl(self):
        pass

        

        
        
        

                                       
                
                    


        
            
        
        
    



        
