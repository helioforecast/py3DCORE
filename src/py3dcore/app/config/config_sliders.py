'''
Configuration for the sliders
'''

############################################################
# Geometrical Models

sliders_3dcore = {
    'Inclination': {'Standard': {'min': 0., 'max': 360., 'step': 1., 'def': 0.},
                    'unit': '[deg.]',
                    'variablename': 'inc',
                    'variablename_double': 'inc_double'
                   },
    'Diameter 1 AU': {'Standard': {'min': 0.05, 'max': 0.35, 'step': 0.01, 'def': 0.2},
                      'unit': '[AU]',
                      'variablename': 'dia',
                      'variablename_double': 'dia_double'
                     },
    'Aspect Ratio': {'Standard': {'min': 1., 'max': 6., 'step': 0.1, 'def':3.},
                     'unit': '',
                     'variablename': 'asp',
                     'variablename_double': 'asp_double'
                    },
    'Launch Radius': {'Standard': {'min' : 5, 'max': 100 , 'step': 1, 'def':20},
                      'unit': '[R_Sun]',
                      'variablename': 'l_rad',
                      'variablename_double': 'l_rad_double'
                     },
    'Launch Velocity': {'Standard': {'min': 400, 'max': 1200, 'step': 10, 'def':800},
                        'unit': '[km/s]',
                        'variablename': 'l_vel',
                        'variablename_double': 'l_vel_double'
                       },
    'Expansion Rate': {'Standard': {'min': 0.3 , 'max': 2., 'step':0.01 , 'def':1.14},
                       'unit': '',
                       'variablename': 'exp_rat',
                       'variablename_double': 'exp_rat_double',
                      },
    'Background Drag': {'Standard': {'min': 0.2, 'max': 3., 'step': 0.01, 'def':1.},
                        'unit': '',
                        'variablename': 'b_drag',
                        'variablename_double': 'b_drag_double',
                       },
    'Background Velocity': {'Standard': {'min': 100, 'max': 700, 'step': 10, 'def':500},
                            'unit': '[km/s]',
                            'variablename': 'bg_vel',
                            'variablename_double': 'bg_vel_double'
                           },
    }
    


sliders_dict = {'3DCORE': sliders_3dcore}

mag_sliders_3dcore = {
    'T_Factor': {'Standard': {'min': -250, 'max': 250, 'step': 1, 'def':100},
                 'unit': '',
                 'variablename': 't_fac',
                 'variablename_double': 't_fac_double'
                },
    'Magnetic Decay Rate': {'Standard': {'min': 1., 'max': 2., 'step': 0.01, 'def':1.64},
                            'unit': '',
                            'variablename': 'mag_dec',
                            'variablename_double': 'mag_dec_double'
                           },
    'Magnetic Field Strength 1 AU': {'Standard': {'min': 5, 'max': 50, 'step': 1, 'def':25},
                                     'unit': '[nT]',
                                     'variablename': 'mag_strength',
                                     'variablename_double': 'mag_strength_double'
                                    },
    }
    
mag_sliders_dict = {'3DCORE': mag_sliders_3dcore}