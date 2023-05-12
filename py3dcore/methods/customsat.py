


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
        if data_path == 'realtime':
            file = loadrealtime()
            self.data = file
        else:
            
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

    

        
def loadrealtime():
    
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y%m%d_%H%M")

    filename= 'dscvr_realtime_'+nowstr+'.p'
    
    request_mag=urllib.request.urlopen('https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json')
    file_mag = request_mag.read()
    data_mag = json.loads(file_mag)
    noaa_mag_gsm = pds.DataFrame(data_mag[1:], columns=['timestamp', 'b_x', 'b_y', 'b_z', 'lon_gsm', 'lat_gsm', 'b_tot'])

    noaa_mag_gsm['timestamp'] = pds.to_datetime(noaa_mag_gsm['timestamp'])
    noaa_mag_gsm['b_x'] = noaa_mag_gsm['b_x'].astype('float')
    noaa_mag_gsm['b_y'] = noaa_mag_gsm['b_y'].astype('float')
    noaa_mag_gsm['b_z'] = noaa_mag_gsm['b_z'].astype('float')
    noaa_mag_gsm['b_tot'] = noaa_mag_gsm['b_tot'].astype('float')
    print(noaa_mag_gsm)

    logger.info("Created pickle file from realtime data: %s", filename)
    pickle.dump(noaa_mag_gsm, open('py3dcore/custom_data/' + filename, "wb"))
    
    raise NotImplementedError('ATTENTION: Still need to finish dscvr data import! So far only mag field components are loaded and no information about positions exists!')
    
    return noaa_mag_gsm