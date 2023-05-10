# -*- coding: utf-8 -*-

import datetime
from typing import Any, List, Optional, Sequence, Tuple, Union

import heliosat
import numba
import numpy as np
import os
from scipy.signal import detrend, welch
######
import matplotlib.dates as mdates
from matplotlib.dates import  DateFormatter
import datetime
from datetime import timedelta

from sunpy.coordinates import frames, get_horizons_coord
from sunpy.time import parse_time

import cdflib
import pickle

import logging

logger = logging.getLogger(__name__)


class FittingData(object):
    
    """
    Class(object) to handle the data used for fitting.
    Sets the following properties for self:
        length                  length of list of observers
        observers               list of observers
        reference_frame         reference frame to work in
    
    Arguments:
        observers               list of observers
        reference_frame         reference frame to work in
        
    Returns:
        None
        
    Functions:
        add_noise
        generate_noise
        generate_data
        sumstat
    """
    
    data_dt: List[np.ndarray]
    data_b: List[np.ndarray]
    data_o: List[np.ndarray]
    data_m: List[np.ndarray]
    data_l: List[int]

    psd_dt: List[np.ndarray]
    psd_fft: List[np.ndarray]

    length: int
    noise_model: str
    observers: list
    reference_frame: str
    sampling_freq: int

    def __init__(self, observers: list, reference_frame: str) -> None:
        self.observers = observers
        self.reference_frame = reference_frame
        self.length = len(self.observers)

    def add_noise(self, profiles: np.ndarray) -> np.ndarray:
        if self.noise_model == "psd":
            _offset = 0
            for o in range(self.length):
                dt = self.psd_dt[o]
                fft = self.psd_fft[o]
                dtl = self.data_l[o]

                sampling_fac = np.sqrt(self.sampling_freq)

                ensemble_size = len(profiles[0])

                null_flt = profiles[1 + _offset : _offset + (dtl + 2) - 1, :, 0] != 0

                # generate noise for each component
                for c in range(3):
                    noise = np.real(
                        np.fft.ifft(
                            np.fft.fft(
                                np.random.normal(
                                    0, 1, size=(ensemble_size, len(fft))
                                ).astype(np.float32)
                            )
                            * fft
                        )
                        / sampling_fac
                    ).T
                    profiles[1 + _offset : _offset + (dtl + 2) - 1, :, c][
                        null_flt
                    ] += noise[dt][null_flt]

                _offset += dtl + 2
                
        elif self.noise_model == "gaussian":
            _offset = 0
            for o in range(self.length):
                dtl = self.data_l[o]
                sampling_fac = np.sqrt(self.sampling_freq)

                ensemble_size = len(profiles[0])

                null_flt = profiles[1 + _offset : _offset + (dtl + 2) - 1, :, 0] != 0

                # generate noise for each component
                for c in range(3):
                    noise = (
                        np.random.normal(0, 1, size=(ensemble_size, dtl))
                        .astype(np.float32)
                        .T
                    )

                    profiles[1 + _offset : _offset + (dtl + 2) - 1, :, c][
                        null_flt
                    ] += noise[null_flt]

                _offset += dtl + 2
        else:
            raise NotImplementedError

        return profiles

    def generate_noise(
        self, noise_model: str = "psd", sampling_freq: int = 300, **kwargs: Any
    ) -> None:
        
        """
        Generates noise according to the noise model.
        Sets the following properties for self:
            psd_dt                altered time axis for power spectrum
            psd_fft               power spectrum
            sampling_freq         sampling frequency
            noise_model           model used to calculate noise

        Arguments:
            noise_model    "psd"     model to use for generating noise (e.g. power spectrum distribution)
            sampling_freq  300       sampling frequency of data

        Returns:
            None
        """
        
        self.psd_dt = []
        self.psd_fft = []
        self.sampling_freq = sampling_freq

        self.noise_model = noise_model

        if noise_model == "psd":
            # get data for each observer
            for o in range(self.length):
                observer, dt, dt_s, dt_e, _ = self.observers[o]

                observer_obj = getattr(heliosat, observer)()

                _, data = observer_obj.get(
                    [dt_s, dt_e],
                    "mag",
                    reference_frame=self.reference_frame,
                    sampling_freq=sampling_freq,
                    cached=True,
                    as_endpoints=True,
                )

                data[np.isnan(data)] = 0 #set all nan values to 0

                # fF, fS = power_spectral_density(dt, data, format_for_fft=True)
                fF, fS = mag_fft(dt, data, sampling_freq=sampling_freq) # computes the mean power spectrum distribution


                kdt = (len(fS) - 1) / (dt[-1].timestamp() - dt[0].timestamp())
                fT = np.array(
                    [int((_.timestamp() - dt[0].timestamp()) * kdt) for _ in dt]
                )

                self.psd_dt.append(fT) # appends the altered time axis
                self.psd_fft.append(fS) # appends the power spectrum 
                
        elif noise_model == "gaussian":
            pass
        else:
            raise NotImplementedError

    def generate_data(self, time_offset: Union[int, Sequence], **kwargs: Any) -> None:
        
        """
        Generates data for each observer at the given times. 
        Sets the following properties for self:
            data_dt      all needed timesteps [dt_s, dt, dt_e]
            data_b       magnetic field data for data_dt
            data_o       trajectory of observers
            data_m       mask for data_b with 1 for each point except first and last
            data_l       length of data

        Arguments:
            time_offset  shift timeseries for observer   
            **kwargs     Any

        Returns:
            None
        """
        
        self.data_dt = []
        self.data_b = []
        self.data_o = []
        self.data_m = []
        self.data_l = []
        
        # Each observer is treated separately

        for o in range(self.length):
            
            # The values of the observer are unpacked
            
            observer, dt, dt_s, dt_e, dt_shift = self.observers[o]
            
            # The reference points are corrected by time_offset

            instrument = kwargs.get("instrument", "mag")

            if hasattr(time_offset, "__len__"):
                dt_s -= datetime.timedelta(hours=time_offset[o])
                dt_e += datetime.timedelta(hours=time_offset[o])
            else:
                dt_s -= datetime.timedelta(hours=time_offset)
                dt_e += datetime.timedelta(hours=time_offset)
            
            # The observer object is created

            observer_obj = getattr(heliosat, observer)()
            
            # The according magnetic field data 
            # for the fitting points is obtained

            _, data = observer_obj.get(
                dt,
                instrument,
                reference_frame=self.reference_frame,
                cached=True,
                **kwargs
            )
            # dt are fitting points, dt_all is with start and end time
            dt_all = [dt_s] + dt + [dt_e] # all time points
            #print(data)

            trajectory = observer_obj.trajectory(
                dt_all, reference_frame=self.reference_frame
            ) # returns the spacecraft trajectory
            # an array containing the data plus one additional 
            # zero each at the beginning and the end is created
            
            b_all = np.zeros((len(data) + 2, 3))
            b_all[1:-1] = data
            
            # the mask is created, a list containing 1] for each 
            # data point and 0 for the first and last entry
            
            mask = [1] * len(b_all)
            mask[0] = 0
            mask[-1] = 0

            if dt_shift:
                self.data_dt.extend([_ + dt_shift for _ in dt_all])
            else:
                self.data_dt.extend(dt_all)
            self.data_b.extend(b_all)
            self.data_o.extend(trajectory)
            self.data_m.extend(mask)
            self.data_l.append(len(data))

    def sumstat(
        self, values: np.ndarray, stype: str = "norm_rmse", use_mask: bool = True
    ) -> np.ndarray:
        
        """
        Returns the summary statistic comparing given values to the data object.

        Arguments:
            values                   fitted values to compare with the data  
            stype      "norm_rmse"   method to use for the summary statistic
            use_mask   True          mask the data

        Returns:
            sumstat                  Summary statistic for each observer
        """
        
        if use_mask:
            return sumstat(
                values,
                self.data_b,
                stype,
                mask=self.data_m,
                data_l=self.data_l,
                length=self.length,
            )
        else:
            return sumstat(
                values, self.data_b, stype, data_l=self.data_l, length=self.length
            )


def sumstat(
    values: np.ndarray, reference: np.ndarray, stype: str = "norm_rmse", **kwargs: Any
) -> np.ndarray:
    """
    Returns the summary statistic comparing given values to reference.
    
    Arguments:
        values                   fitted values
        reference                data
        stype      "norm_rmse"   method to use
        **kwargs   Any
        
    Returns:
        rmse_all.T
    """
        
    if stype == "norm_rmse":
        data_l = np.array(kwargs.pop("data_l")) # length of data        
        length = kwargs.pop("length") # length of observer list
        mask = kwargs.pop("mask", None)

        varr = np.array(values) # array of values

        rmse_all = np.zeros((length, varr.shape[1])) #stores the rmse for all observers in one array. Gets initialized with zeros


        _dl = 0
        
        # iterate through all observers

        for i in range(length):
            slc = slice(_dl, _dl + data_l[i] + 2) #create slice object
            # for _dl = 0: slice(0,length of data for obs1+2)
            
            values_i = varr[slc]# value array of slice
            # values only for current observer
            
            reference_i = np.array(reference)[slc]
            # reference only for current observer

            normfac = np.mean(np.sqrt(np.sum(reference_i**2, axis=1)))
            #normfactor is created for current reference 

            if mask is not None:
                mask_i = np.array(mask)[slc]
                # mask for current observer is used
            else:
                mask_i = None
            
            # rmse is calculated for the current observer 
            # and added to rmse_all    

            rmse_all[i] = rmse(values_i, reference_i, mask=mask_i) / normfac

            _dl += data_l[i] + 2 # dl is changed to obtain the correct data for each observer

        return rmse_all.T
    else:
        raise NotImplementedError


@numba.njit
def rmse(
    values: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    rmse = np.zeros(len(values[0]))
    
    """
    Returns the rmse of values to reference.

    Arguments:
        values       fitted values
        reference    data
        mask

    Returns:
        rmse         root mean squared error
    """ 

    if mask is not None:
        for i in range(len(reference)):
            # compute the rmse for each value
            _error_rmse(values[i], reference[i], mask[i], rmse)
            
        #computes the mean for the full array
        rmse = np.sqrt(rmse / len(values))

        mask_arr = np.copy(rmse)
        
        #check if magnetic field at reference points is 0
        for i in range(len(reference)):
            _error_mask(values[i], mask[i], mask_arr)

        return mask_arr
    else:
        for i in range(len(reference)):
            # compute the rmse for each value
            _error_rmse(values[i], reference[i], 1, rmse)
            
        #computes the mean for the full array
        rmse = np.sqrt(rmse / len(values))

        return rmse


@numba.njit
def _error_mask(values_t: np.ndarray, mask: np.ndarray, rmse: np.ndarray) -> None:
    """
    Sets the rmse to infinity if reference points have nonzero magnetic field.
    """
    for i in numba.prange(len(values_t)):
        _v = np.sum(values_t[i] ** 2)
        if (_v > 0 and mask == 0) or (_v == 0 and mask != 0):
            rmse[i] = np.inf


@numba.njit
def _error_rmse(
    values_t: np.ndarray, ref_t: np.ndarray, mask: np.ndarray, rmse: np.ndarray
) -> None:
    """
    Returns the rmse of values to reference.
    """
    for i in numba.prange(len(values_t)):
        if mask == 1:
            rmse[i] += np.sum((values_t[i] - ref_t) ** 2)


def mag_fft(
    dt: Sequence[datetime.datetime], bdt: np.ndarray, sampling_freq: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean power spectrum distribution from a magnetic field measurements over all three vector components.
    Note: Assumes that P(k) is the same for all three vector components.
    """
    n_s = int(((dt[-1] - dt[0]).total_seconds() / 3600) - 1)
    n_perseg = np.min([len(bdt), 256])

    p_bX = detrend(bdt[:, 0], type="linear", bp=n_s)
    p_bY = detrend(bdt[:, 1], type="linear", bp=n_s)
    p_bZ = detrend(bdt[:, 2], type="linear", bp=n_s)

    _, wX = welch(p_bX, fs=1 / sampling_freq, nperseg=n_perseg)
    _, wY = welch(p_bY, fs=1 / sampling_freq, nperseg=n_perseg)
    wF, wZ = welch(p_bZ, fs=1 / sampling_freq, nperseg=n_perseg)

    wS = (wX + wY + wZ) / 3

    # convert P(k) into suitable form for fft
    fF = np.fft.fftfreq(len(p_bX), d=sampling_freq)
    fS = np.zeros((len(fF)))

    for i in range(len(fF)):
        k = np.abs(fF[i])
        fS[i] = np.sqrt(wS[np.argmin(np.abs(k - wF))])

    return fF, fS




############## CUSTOM DATA

class custom_observer(object):
    
    """Handles custom data and sets the following attributes for self:
            data         full custom dataset

        Arguments:
            data_path    where to find the data
            kwargs       any

        Returns:
            None
        """
    
    def __init__(self, data_path:str, **kwargs: Any) -> None:
        
        try:
            file = pickle.load(open('py3dcore/custom_data/'+ data_path, 'rb'))
            self.data = file
            #self.sphere2cart()
        except:
            logger.info("Did not find %s, creating pickle file from cdf", data_path)
            #try:
            createpicklefiles(self,data_path)
            file = pickle.load(open('py3dcore/custom_data/'+ data_path, 'rb'))
            self.data = file
        
        
    def sphere2cart(self):

        self.data['x'] = self.data['r'] * np.cos(np.deg2rad(self.data['lon'])) * np.cos(np.deg2rad(self.data['lat']))
        #print(self.data['x'])
        self.data['y'] = self.data['r'] * np.sin(np.deg2rad(self.data['lon'] )) * np.cos( np.deg2rad(self.data['lat'] ))
        self.data['z'] = self.data['r'] * np.sin(np.deg2rad( self.data['lat'] ))

        
    def get(self, dtp: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], data_key: str, **kwargs: Any) -> np.ndarray:
        
        sampling_freq = kwargs.pop("sampling_freq", 60)
        
        if kwargs.pop("as_endpoints", False):
            _ = np.linspace(dtp[0].timestamp(), dtp[-1].timestamp(), int((dtp[-1].timestamp() - dtp[0].timestamp()) // sampling_freq))  
            dtp = [datetime.datetime.fromtimestamp(_, datetime.timezone.utc) for _ in _]
            
        dat = []
        tt = [x.replace(tzinfo=None,second=0, microsecond=0) for x in dtp]
        
        ii = [np.where(self.data['time']==x)[0][0] for x in tt if np.where(self.data['time']==x)[0].size > 0]
        
        for t in ii:
            res = [self.data[com][t] for com in ['bx','by','bz']]
            dat.append((res))
            
        return np.array(dtp), np.array(dat)

    
    def trajectory(self, dtp: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], **kwargs: Any) -> np.ndarray:
        
        tra = []
        tt = [x.replace(tzinfo=None,second=0, microsecond=0) for x in dtp]

        ii = [np.where(self.data['time']==x)[0][0] for x in tt if np.where(self.data['time']==x)[0].size > 0]
        
        for t in ii:
            res = [self.data[com][t] for com in ['x','y','z']]
            tra.append(res)
            
        return np.array(tra)
    
def cdftopickle(magpath, swapath, sc):
    
    '''
    creating a pickle file from cdf
    
    magpath        path to directory with magnetic field data
    swapath        path to directory with solar wind data (not needed for 3DCORE)
    sc             spacecraft
    
    '''
    
    
    if sc == 'solo':
        fullname = 'solar orbiter'
    if sc == 'psp':
        fullname = 'Parker Solar Probe'
    if sc == 'wind':
        fullname = 'Wind'    
    
    timep = np.zeros(0,dtype=[('time',object)])
    den = np.zeros(0)
    temp = np.zeros(0)
    vr = np.zeros(0)
    vt = np.zeros(0)
    vn = np.zeros(0)
        
    if os.path.exists(swapath):
        ll_path = swapath
    
        files = os.listdir(ll_path)
        files.sort()
        llfiles = [os.path.join(ll_path, f) for f in files if f.endswith('.cdf')]
    

        for i in np.arange(0,len(llfiles)):
            p1 = cdflib.CDF(llfiles[i])

            den1 = p1.varget('N')
            speed1 = p1.varget('V_RTN')
            temp1 = p1.varget('T')

            vr1 = speed1[:, 0]
            vt1 = speed1[:, 1]
            vn1 = speed1[:, 2]

            vr = np.append(vr1, vr)
            vt = np.append(vt1, vt)
            vn = np.append(vn1, vn)


            temp = np.append(temp1, temp)
            den = np.append(den1, den)


            time1 = p1.varget('EPOCH')
            t1 = parse_time(cdflib.cdfastropy.convert_to_astropy(time1, format=None)).datetime
            timep = np.append(timep, t1)

            temp = temp*(1.602176634*1e-19) / (1.38064852*1e-23) # from ev to K 

    ll_path = magpath

    files = os.listdir(ll_path)
    files.sort()
    llfiles = [os.path.join(ll_path, f) for f in files if f.endswith('.cdf')]
    
    br1 = np.zeros(0)
    bt1 = np.zeros(0)
    bn1 = np.zeros(0)
    time1 = np.zeros(0,dtype=[('time',object)])
    
    for i in np.arange(0,len(llfiles)):
        m1 = cdflib.CDF(llfiles[i])
        
        if sc == 'solo':
            print('solo cdf')
            b = m1.varget('B_RTN')
            time = m1.varget('EPOCH')
        
        elif sc == 'psp':
            print('psp cdf')
            b = m1.varget('psp_fld_l2_mag_RTN_1min')
            time = m1.varget('epoch_mag_RTN_1min')
        
        elif sc =='wind':
            print('wind cdf')
            b = m1.varget('BRTN')
            time = m1.varget('Epoch')
                
        else:
            print('No data from cdf file')
            
        br = b[:, 0]
        bt = b[:, 1]
        bn = b[:, 2]

        br1 = np.append(br1, br)
        bt1 = np.append(bt1, bt)
        bn1 = np.append(bn1, bn)

        #try:
        #    time = m1.varget('EPOCH')
        #except:
        #    time = m1.varget('epoch_mag_RTN_1min')
            
        t1 = parse_time(cdflib.cdfastropy.convert_to_astropy(time, format=None)).datetime
        time1 = np.append(time1,t1)
        
    starttime = time1[0].replace(hour = 0, minute = 0, second=0, microsecond=0)
    endtime = time1[-1].replace(hour = 0, minute = 0, second=0, microsecond=0)
    
    time_int = []
    while starttime < endtime:
        time_int.append(starttime)
        starttime += timedelta(minutes=1)


    time_int_mat = mdates.date2num(time_int)
    time1_mat = mdates.date2num(time1)
    timep_mat = mdates.date2num(timep)  
    
    if os.path.exists(swapath):
        ll = np.zeros(np.size(time_int),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),('r', float),('lat', float),('lon', float),('x', float),('y', float),('z', float),('vx', float),('vy', float),('vz', float),('vt', float),('tp', float),('np', float) ] )
    else:
        ll = np.zeros(np.size(time_int),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),('r', float),('lat', float),('lon', float),('x', float),('y', float),('z', float)] )
        

    ll = ll.view(np.recarray)  
    
    # replace all unreasonable large and small values with nan
    thresh = 1e6
    br1[br1 > thresh] = np.nan
    br1[br1 < -thresh] = np.nan
    bt1[bt1 > thresh] = np.nan
    bt1[bt1 < -thresh] = np.nan
    bn1[bn1 > thresh] = np.nan
    bn1[bn1 < -thresh] = np.nan

    # interpolate between non nan values
    ll.time = time_int
    ll.bx = np.interp(time_int_mat, time1_mat[~np.isnan(br1)], br1[~np.isnan(br1)])
    ll.by = np.interp(time_int_mat, time1_mat[~np.isnan(bt1)], bt1[~np.isnan(bt1)])
    ll.bz = np.interp(time_int_mat, time1_mat[~np.isnan(bn1)], bn1[~np.isnan(bn1)])
    ll.bt = np.sqrt(ll.bx**2 + ll.by**2 + ll.bz**2)
    
    if os.path.exists(swapath):
        ll.np = np.interp(time_int_mat, timep_mat, den)
        ll.tp = np.interp(time_int_mat, timep_mat, temp) 
        ll.vx = np.interp(time_int_mat, timep_mat, vr)
        ll.vy = np.interp(time_int_mat, timep_mat, vt)
        ll.vz = np.interp(time_int_mat, timep_mat, vn)
        ll.vt = np.sqrt(ll.vx**2 + ll.vy**2 + ll.vz**2)

    if sc == 'solo':
        # Solar Orbiter position with sunpy
        coord = get_horizons_coord(sc, time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)

    elif sc == 'psp':    
        # PSP position with sunpy
        coord = get_horizons_coord(sc, time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)

    elif sc == 'wind':
        # Wind position with sunpy
        
        # use EM-L1 as proxy for the Wind spacecraft position
        # this is valid from 2004 on, where Wind was placed in orbit around L1
        coord = get_horizons_coord('EM-L1', time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)
     
    else:
        print('no spacecraft position')
        
    return ll
    
    
def sphere2cart(ll):
        
    ll.x = ll.r * np.cos(np.deg2rad(ll.lon)) * np.cos(np.deg2rad(ll.lat))
    ll.y = ll.r * np.sin(np.deg2rad(ll.lon)) * np.cos(np.deg2rad(ll.lat))
    ll.z = ll.r * np.sin(np.deg2rad(ll.lat))
    
    return ll
        
def createpicklefiles(self, data_path):
    name = data_path.split('.')[0]
    sc = name.split('_')[0]
    ev = name.split('_')[1]

    magpath = 'py3dcore/custom_data/' + sc +'_mag_'+ ev
    swapath = 'py3dcore/custom_data/' + sc +'_swa_'+ ev

    ll = cdftopickle(magpath, swapath, sc)

    filename= sc +'_'+ ev + '.p'

    pickle.dump(ll, open('py3dcore/custom_data/' + filename, "wb"))
    logger.info("Created pickle file from cdf: %s", filename)
                
        
class CustomData(FittingData):
    """
    Class(object) to handle custom data used for fitting.
    Sets the following properties for self:
        length                  length of list of observers
        observers               list of observers
        reference_frame         reference frame to work in
    
    Arguments:
        observers               list of observers
        reference_frame         reference frame to work in        
        data_path               where to find the dataset
        
    Returns:
        None
        
    Functions:
        add_noise
        generate_noise
        generate_data
        sumstat
    """
    
    data_dt: List[np.ndarray]
    data_b: List[np.ndarray]
    data_o: List[np.ndarray]
    data_m: List[np.ndarray]
    data_l: List[int]

    psd_dt: List[np.ndarray]
    psd_fft: List[np.ndarray]    

    length: int
    noise_model: str
    observers: list
    reference_frame: str
    sampling_freq: int

    def __init__(self, observers: list, reference_frame: str) -> None:
        
        FittingData.__init__(self, observers, reference_frame)
    
    def generate_data(self, time_offset: Union[int, Sequence], **kwargs: Any) -> None:
        
        """
        Generates data for each observer at the given times. 
        Sets the following properties for self:
            data_dt      all needed timesteps [dt_s, dt, dt_e]
            data_b       magnetic field data for data_dt
            data_o       trajectory of observers
            data_m       mask for data_b with 1 for each point except first and last
            data_l       length of data

        Arguments:
            time_offset  shift timeseries for observer   
            **kwargs     Any

        Returns:
            None
        """
        
        self.data_dt = []
        self.data_b = []
        self.data_o = []
        self.data_m = []
        self.data_l = []
        
        # Each observer is treated separately

        for o in range(self.length):
            
            # The values of the observer are unpacked
            
            
            observer, dt, dt_s, dt_e, dt_shift, self.data_path = self.observers[o]
            
            logger.info("Using custom datafile: %s", self.data_path)
            
            # The reference points are corrected by time_offset

            if hasattr(time_offset, "__len__"):
                dt_s -= datetime.timedelta(hours=time_offset[o])  # type: ignore
                dt_e += datetime.timedelta(hours=time_offset[o])  # type: ignore
            else:
                dt_s -= datetime.timedelta(hours=time_offset)  # type: ignore
                dt_e += datetime.timedelta(hours=time_offset)  # type: ignore
            # The observer object is created
                        
            observer_obj = custom_observer(self.data_path)
            
            # The according magnetic field data 
            # for the fitting points is obtained
            
            _, data = observer_obj.get(dt, "mag", reference_frame=self.reference_frame, use_cache=True, **kwargs)
            
            dt_all = [dt_s] + dt + [dt_e] # all time points
            trajectory = observer_obj.trajectory(dt_all, reference_frame=self.reference_frame) # returns the spacecraft trajectory
            # an array containing the data plus one additional 
            # zero each at the beginning and the end is created
            
            b_all = np.zeros((len(data) + 2, 3))
            b_all[1:-1] = data
            
            # the mask is created, a list containing 1] for each 
            # data point and 0 for the first and last entry
            mask = [1] * len(b_all)
            mask[0] = 0
            mask[-1] = 0

            if dt_shift:
                self.data_dt.extend([_ + dt_shift for _ in dt_all])
            else:
                self.data_dt.extend(dt_all)
            self.data_b.extend(b_all)
            self.data_o.extend(trajectory)
            self.data_m.extend(mask)
            self.data_l.append(len(data))

    def generate_noise(self, noise_model: str = "psd", sampling_freq: int = 300, **kwargs: Any) -> None:
        

        """
        Generates noise according to the noise model.
        Sets the following properties for self:
            psd_dt                altered time axis for power spectrum
            psd_fft               power spectrum
            sampling_freq         sampling frequency
            noise_model           model used to calculate noise

        Arguments:
            noise_model    "psd"     model to use for generating noise (e.g. power spectrum distribution)
            sampling_freq  300       sampling frequency of data

        Returns:
            None
        """
            
        self.psd_dt = []
        self.psd_fft = []
        self.sampling_freq = sampling_freq

        self.noise_model = noise_model

        if noise_model == "psd":
        # get data for each observer
            for o in range(self.length):
                observer, dt, dt_s, dt_e, _, self.data_path = self.observers[o]

                 # The observer object is created
                        
                observer_obj = custom_observer(self.data_path)
            
                # The according magnetic field data 
                # for the fitting points is obtained
            
                _, data = observer_obj.get([dt_s, dt_e], "mag", reference_frame=self.reference_frame, sampling_freq=sampling_freq, use_cache=True, as_endpoints=True)
                
                data[np.isnan(data)] = 0 #set all nan values to 0

                fF, fS = mag_fft(dt, data, sampling_freq=sampling_freq) # computes the mean power spectrum distribution

                kdt = (len(fS) - 1) / (dt[-1].timestamp() - dt[0].timestamp()) 
                fT = np.array([int((_.timestamp() - dt[0].timestamp()) * kdt) for _ in dt]) 

                self.psd_dt.append(fT) # appends the altered time axis
                self.psd_fft.append(fS)
                # appends the power spectrum 
        else:
            raise NotImplementedError