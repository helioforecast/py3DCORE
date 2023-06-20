'''
 A dictionary of the selected imagers
 How to use this:

'''
import astropy.units as u
from sunpy.net import attrs as a

imager_dict = dict.fromkeys(['LC2'], [(a.Instrument.lasco, a.Detector.c2),
                                      {'dimensions': (1024*u.pixel, 1024*u.pixel), 'polar': 'Clear', 'superpixel': 2}, 'LASCO-C2'])
imager_dict.update(dict.fromkeys(['LC3'], [(a.Instrument.lasco, a.Detector.c3),
                                           {'dimensions': (1024*u.pixel, 1024*u.pixel), 'polar': 'Clear', 'superpixel': 2}, 'LASCO-C3']))
imager_dict.update(dict.fromkeys(['AIA'], [(a.Instrument.aia, a.Wavelength(19.3 * u.nm), a.Sample(1*u.minute)),
                                           {'dimensions': (4096*u.pixel, 4096*u.pixel), 'superpixel': 8, 'exposure': 1.90}, 'SDO-AIA']))
imager_dict.update(dict.fromkeys(['COR2A'], [(a.Source('STEREO_A'), a.Detector.cor2),
                                             {'dimensions': (2048*u.pixel, 2048*u.pixel), 'polar': 1001, 'superpixel': 4}, 'STA-COR2']))
imager_dict.update(dict.fromkeys(['COR2B'], [(a.Source('STEREO_B'), a.Detector.cor2),
                                             {'dimensions': (2048*u.pixel, 2048*u.pixel), 'polar': 1001, 'superpixel': 4}, 'STB-COR2']))
imager_dict.update(dict.fromkeys(['EUVIA'], [(a.Source('STEREO_A'), a.Detector.euvi, a.Wavelength(19.5 * u.nm)),
                                             {'dimensions': (2048*u.pixel, 2048*u.pixel), 'superpixel': 4}, 'STA-EUVI']))
imager_dict.update(dict.fromkeys(['EUVIB'], [(a.Source('STEREO_B'), a.Detector.euvi, a.Wavelength(19.5 * u.nm)),
                                             {'dimensions': (2048*u.pixel, 2048*u.pixel), 'superpixel': 4}, 'STB-EUVI']))
imager_dict.update(dict.fromkeys(['COR1A'], [(a.Source('STEREO_A'), a.Detector.cor1),
                                             {'dimensions': (512*u.pixel, 512*u.pixel)}, 'STA-COR1']))  # 'polar':0
imager_dict.update(dict.fromkeys(['COR1B'], [(a.Source('STEREO_B'), a.Detector.cor1),
                                             {'dimensions': (512*u.pixel, 512*u.pixel)}, 'STB-COR1']))  # 'polar':0
imager_dict.update(dict.fromkeys(['HI1A'], [(a.Source('STEREO_A'), a.Detector.hi1),
                                            {'dimensions': (1024*u.pixel, 1024*u.pixel), 'superpixel': 2}, 'STA-HI1']))
imager_dict.update(dict.fromkeys(['HI1B'], [(a.Source('STEREO_B'), a.Detector.hi1),
                                            {'dimensions': (1024*u.pixel, 1024*u.pixel), 'superpixel': 2}, 'STB-HI1']))
imager_dict.update(dict.fromkeys(['HI2A'], [(a.Source('STEREO_A'), a.Detector.hi2),
                                            {'dimensions': (1024*u.pixel, 1024*u.pixel), 'superpixel': 2}, 'STA-HI2']))
imager_dict.update(dict.fromkeys(['HI2B'], [(a.Source('STEREO_B'), a.Detector.hi2),
                                            {'dimensions': (1024*u.pixel, 1024*u.pixel)}, 'STB-HI2']))




insitu_dict = ["BepiColombo",
               "DSCOVR",
               "PSP",
               "SolarOrbiter",
               "STEREO A",
               "STEREO B",
               "Wind"]

'''
Configuration for the seleced bodies.
'''

bodies_dict = {'STEREO B': ['-235', 'b'],
               'STEREO A': ['-234', 'red'],
               'SolarOrbiter': ['-144', 'dodgerblue'],
               'PSP': ['-96', 'purple'],
               'BepiColombo': ['-121', 'orange'],
               'DSCOVR': ['-78', 'grey'],
               'Wind': ['-8', 'darkgrey']
              }

planets_dict = {'Sun': ['10', 'yellow'],
                'Mercury': ['199', 'darkturquoise'],
               'Venus': ['299', 'darkorchid'],
               'Earth': ['399', 'green'],
               'Mars': ['499', 'maroon'],
               'Jupiter': ['599', 'navy'],
               }