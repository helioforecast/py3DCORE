import numpy as np
import pandas as pd
import datetime
import itertools


#input datetime to return T1, T2 and T3 based on Hapgood 1992
#http://www.igpp.ucla.edu/public/vassilis/ESS261/Lecture03/Hapgood_sdarticle.pdf
def get_transformation_matrices(timestamp):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(timestamp)
    jd=ts.to_julian_date()
    mjd=float(int(jd-2400000.5)) #use modified julian date    
    T0=(mjd-51544.5)/36525.0
    UT=ts.hour + ts.minute / 60. + ts.second / 3600. #time in UT in hours    
    #define position of geomagnetic pole in GEO coordinates
    pgeo=78.8+4.283*((mjd-46066)/365.25)*0.01 #in degrees
    lgeo=289.1-1.413*((mjd-46066)/365.25)*0.01 #in degrees
    #GEO vector
    Qg=[np.cos(pgeo*np.pi/180)*np.cos(lgeo*np.pi/180), np.cos(pgeo*np.pi/180)*np.sin(lgeo*np.pi/180), np.sin(pgeo*np.pi/180)]
    #now move to equation at the end of the section, which goes back to equations 2 and 4:
    #CREATE T1; T0, UT is known from above
    zeta=(100.461+36000.770*T0+15.04107*UT)*np.pi/180
    ################### theta und z
    T1=np.matrix([[np.cos(zeta), np.sin(zeta),  0], [-np.sin(zeta) , np.cos(zeta) , 0], [0,  0,  1]]) #angle for transpose
    LAMBDA=280.460+36000.772*T0+0.04107*UT
    M=357.528+35999.050*T0+0.04107*UT
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180
    #CREATE T2, LAMBDA, M, lt2 known from above
    ##################### lamdbda und Z
    t2z=np.matrix([[np.cos(lt2), np.sin(lt2),  0], [-np.sin(lt2) , np.cos(lt2) , 0], [0,  0,  1]])
    et2=(23.439-0.013*T0)*np.pi/180
    ###################### epsilon und x
    t2x=np.matrix([[1,0,0],[0,np.cos(et2), np.sin(et2)], [0, -np.sin(et2), np.cos(et2)]])
    T2=np.dot(t2z,t2x)  #equation 4 in Hapgood 1992
    #matrix multiplications   
    T2T1t=np.dot(T2,np.matrix.transpose(T1))
    ################
    Qe=np.dot(T2T1t,Qg) #Q=T2*T1^-1*Qq
    psigsm=np.arctan(Qe.item(1)/Qe.item(2)) #arctan(ye/ze) in between -pi/2 to +pi/2
    T3=np.matrix([[1,0,0],[0,np.cos(-psigsm), np.sin(-psigsm)], [0, -np.sin(-psigsm), np.cos(-psigsm)]])
    return T1, T2, T3


def GSE_to_GSM(df):
    B_GSM = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_transformation_matrices(df['timestamp'].iloc[0])
        B_GSE_i = np.matrix([[df['b_x'].iloc[i]],[df['b_y'].iloc[i]],[df['b_z'].iloc[i]]]) 
        B_GSM_i = np.dot(T3,B_GSE_i)
        B_GSM_i_list = B_GSM_i.tolist()
        flat_B_GSM_i = list(itertools.chain(*B_GSM_i_list))
        B_GSM.append(flat_B_GSM_i)
    df_transformed = pd.DataFrame(B_GSM, columns=['b_x', 'b_y', 'b_z'])
    df_transformed['b_tot'] = np.linalg.norm(df_transformed[['b_x', 'b_y', 'b_z']], axis=1)
    df_transformed['timestamp'] = df['timestamp']
    return df_transformed


def GSM_to_GSE(df):
    B_GSE = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_transformation_matrices(df['timestamp'].iloc[0])
        T3_inv = np.linalg.inv(T3)
        B_GSM_i = np.matrix([[df['b_x'].iloc[i]],[df['b_y'].iloc[i]],[df['b_z'].iloc[i]]]) 
        B_GSE_i = np.dot(T3_inv,B_GSM_i)
        B_GSE_i_list = B_GSE_i.tolist()
        flat_B_GSE_i = list(itertools.chain(*B_GSE_i_list))
        B_GSE.append(flat_B_GSE_i)
    df_transformed = pd.DataFrame(B_GSE, columns=['b_x', 'b_y', 'b_z'])
    df_transformed['b_tot'] = np.linalg.norm(df_transformed[['b_x', 'b_y', 'b_z']], axis=1)
    df_transformed['timestamp'] = df['timestamp']
    return df_transformed


def GSE_to_RTN_approx(df):
    df_transformed = pd.DataFrame()
    df_transformed['timestamp'] = df['timestamp']
    df_transformed['b_tot'] = df['b_tot']
    df_transformed['b_x'] = -df['b_x']
    df_transformed['b_y'] = -df['b_y']
    df_transformed['b_z'] = df['b_z']
    return df_transformed

    
def GSM_to_RTN_approx(df):
    df_gse = GSM_to_GSE(df)
    df_transformed = GSE_to_RTN_approx(df_gse)
    return df_transformed


def convert_GSE_to_HEEQ(sc):
    '''
    for Wind magnetic field components: convert GSE to HEE to HAE to HEEQ
    '''
       
    sc=copy.deepcopy(sc_in)

    print('conversion GSE to HEEQ start')                              
    jd=np.zeros(len(sc))
    mjd=np.zeros(len(sc))
        

    for i in np.arange(0,len(sc)):

        jd[i]=parse_time(sc.time[i]).jd
        mjd[i]=float(int(jd[i]-2400000.5)) #use modified julian date    
        
        #GSE to HEE
        #Hapgood 1992 rotation by 180 degrees, or simply change sign in bx by    
        #rotangle=np.radians(180)
        #c, s = np.cos(rotangle), np.sin(rotangle)
        #T1 = np.array(((c,s, 0), (-s, c, 0), (0, 0, 1)))
        #[bx_hee,by_hee,bz_hee]=T1[sc.bx[i],sc.by[i],sc.bz[i]]        
        b_hee=[-sc.bx[i],-sc.by[i],sc.bz[i]]
        
        #HEE to HAE        
        
        #define T00 and UT
        T00=(mjd[i]-51544.5)/36525.0          
        dobj=sc.time[i]
        UT=dobj.hour + dobj.minute / 60. + dobj.second / 3600. #time in UT in hours   

        #lambda_sun in Hapgood, equation 5, here in rad
        M=np.radians(357.528+35999.050*T00+0.04107*UT)
        LAMBDA=280.460+36000.772*T00+0.04107*UT        
        lambda_sun=np.radians( (LAMBDA+(1.915-0.0048*T00)*np.sin(M)+0.020*np.sin(2*M)) )
        
        #S-1 Matrix equation 12 hapgood 1992, change sign in lambda angle for inversion HEE to HAE instead of HAE to HEE
        c, s = np.cos(-(lambda_sun+np.radians(180))), np.sin(-(lambda_sun+np.radians(180)))
        Sm1 = np.array(((c,s, 0), (-s, c, 0), (0, 0, 1)))
        b_hae=np.dot(Sm1,b_hee)

        #HAE to HEEQ
        
        iota=np.radians(7.25)
        omega=np.radians((73.6667+0.013958*((mjd[i]+3242)/365.25)))                      
        theta=np.arctan(np.cos(iota)*np.tan(lambda_sun-omega))                       
                      
    
        #quadrant of theta must be opposite lambda_sun minus omega; Hapgood 1992 end of section 5   
        #get lambda-omega angle in degree mod 360 and theta in degrees
        lambda_omega_deg=np.mod(np.degrees(lambda_sun)-np.degrees(omega),360)
        theta_node_deg=np.degrees(theta)


        ##if the 2 angles are close to similar, so in the same quadrant, then theta_node = theta_node +pi           
        if np.logical_or(abs(lambda_omega_deg-theta_node_deg) < 1, abs(lambda_omega_deg-360-theta_node_deg) < 1): theta=theta+np.pi                                                                                                          
        
        #rotation around Z by theta
        c, s = np.cos(theta), np.sin(theta)
        S2_1 = np.array(((c,s, 0), (-s, c, 0), (0, 0, 1)))

        #rotation around X by iota  
        iota=np.radians(7.25)
        c, s = np.cos(iota), np.sin(iota)
        S2_2 = np.array(( (1,0,0), (0,c, s), (0, -s, c)) )
                
        #rotation around Z by Omega  
        c, s = np.cos(omega), np.sin(omega)
        S2_3 = np.array( ((c,s, 0), (-s, c, 0), (0, 0, 1)) )
        
        #matrix multiplication to go from HAE to HEEQ components                
        [bx_heeq,by_heeq,bz_heeq]=np.dot(  np.dot(   np.dot(S2_1,S2_2),S2_3), b_hae) 
        
        sc.bx[i]=bx_heeq
        sc.by[i]=by_heeq
        sc.bz[i]=bz_heeq

    
    print('conversion GSE to HEEQ done')                                
    return sc