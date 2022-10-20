import numpy as np
"""
This file contains localization functions Gaspari-Cohn and Bolin-Wallin for 
single length scale and multiscale localization.
"""

def gc_leq1(x):
    loc_weights = -.25 * x**5 + .5 * x**4 + .625 * x**3 - (5/3) * x**2 + 1
    return loc_weights

def gc_leq2(x):
    loc_weights = (1/12) * x**5 - .5 * x**4 + .625 * x**3 + (5/3) * x**2 - 5 * x + 4 - 2/(3*x)
    return loc_weights

def gaspari_cohn_univariate(distance, localization_half_width):
    """ Fifth-order piecewise rational localization function from Gaspari & Cohn (1999)
    
    INPUT:
    distance - one or more distances where we will calculate localization weights
    localization_half_width = localization radius / 2
    
    OUTPUT:
    localization weights
    """
    x = np.abs(distance)/localization_half_width
    # cases
    leq1 = x<=1
    leq2 = np.logical_and(x>1, x<=2)
    # define functions
    f1 = lambda a : -.25 * a**5 + .5 * a**4 + .625 * a**3 - (5/3) * a**2 + 1
    f2 = lambda a : (1/12) * a**5 - .5 * a**4 + .625 * a**3 + (5/3) * a**2 - 5 * a + 4 - 2/(3*a)
    # piecewise function
    loc_weights = np.piecewise(x, [leq1, leq2], [f1, f2])
    return loc_weights

def gaspari_cohn_cross(distance, localization_half_width1, localization_half_width2):
    """ Cross-localization function from Stanley, Grooms and Kleiber (2021)
    
    INPUT:
    distance - distance where we will calculate cross-localization weights
    localization_half_width1 = (1/2) * localization radius for process 1
    localization_half_width2 = (1/2) * localization radius for process 2
    
    OUTPUT:
    cross-localization weights
    """
    c_min = min(localization_half_width1, localization_half_width2)
    c_max = max(localization_half_width1, localization_half_width2)
    kappa = np.sqrt(c_max/c_min)
    distance = np.abs(distance)
    x = distance/(kappa*c_min)
    localization_weights = np.zeros(x.shape)
    # Two cases: kappa >= sqrt(2) and kappa < sqrt(2)
    # case 1: kappa >= sqrt(2)
    if kappa >= np.sqrt(2):
        # four cases
        leq_min = distance <= c_min
        leq_dif = np.logical_and(c_min < distance, distance <= c_max-c_min)
        leq_max = np.logical_and(c_max - c_min < distance, distance <= c_max )
        leq_sum = np.logical_and(c_max < distance, distance <= c_min + c_max )
        # evaluation of fifth order piecewise rational function
        localization_weights[leq_min] = -(1/6)*x[leq_min]**5 + (1/2)*(1/kappa)*x[leq_min]**4 -(5/3)*(kappa**-3)*x[leq_min]**2 + (5/2)*(kappa**-3) - (3/2)*(kappa**-5)
        localization_weights[leq_dif] = -(5/2)*(kappa**-4)*x[leq_dif] - ((1/3)*(kappa**-6))/x[leq_dif] + (5/2)*(kappa**-3)
        localization_weights[leq_max] = ( -(1/12)*x[leq_max]**5 + (1/4)*(kappa-1/kappa)*x[leq_max]**4 + (5/8)*x[leq_max]**3 
            - (5/6)*(kappa**3-kappa**-3)*x[leq_max]**2 +(5/4)*(kappa**4-kappa**2-kappa**-2-kappa**-4)*x[leq_max] 
            + (5/12)/x[leq_max] - (3/8)*(kappa**4+kappa**-4)/x[leq_max] + (1/6)*(kappa**6-kappa**-6)/x[leq_max] 
            + (5/4)*(kappa**3+kappa**-3) - (3/4)*(kappa**5-kappa**-5) )
        localization_weights[leq_sum] = ( (1/12)*x[leq_sum]**5 - (1/4)*(kappa+1/kappa)*x[leq_sum]**4 + (5/8)*x[leq_sum]**3 
            + (5/6)*(kappa**3+kappa**-3)*x[leq_sum]**2 - (5/4)*(kappa**4+kappa**2+kappa**-2+kappa**-4)*x[leq_sum] 
            + (5/12)/x[leq_sum] - (3/8)*(kappa**4+kappa**-4)/x[leq_sum] - (1/6)*(kappa**6+kappa**-6)/x[leq_sum] 
            + (5/4)*(kappa**3+kappa**-3) + (3/4)*(kappa**5+kappa**-5) )
    # case 2: kappa < sqrt(2)
    else:
        # four cases
        leq_dif = distance <= c_max-c_min
        leq_min = np.logical_and(c_max-c_min < distance, distance <= c_min)
        leq_max = np.logical_and(c_min < distance, distance <= c_max )
        leq_sum = np.logical_and(c_max < distance, distance <= c_min + c_max )
        # evaluation of fifth order piecewise rational function
        localization_weights[leq_dif] = -(1/6)*x[leq_dif]**5 + (1/2)*(1/kappa)*x[leq_dif]**4 -(5/3)*(1/kappa**3)*x[leq_dif]**2 + (5/2)*(1/kappa**3) - (3/2)*(1/kappa**5)
        localization_weights[leq_min] = ( -(1/4)*x[leq_min]**5 + (1/4)*(kappa+1/kappa)*x[leq_min]**4 + (5/8)*x[leq_min]**3 - (5/6)*(kappa**3+1/kappa**3)*x[leq_min]**2 
            + (5/4)*(kappa**4-kappa**2-1/kappa**2+1/kappa**4)*x[leq_min] + (1/6)*(kappa**6+1/kappa**6)/x[leq_min] - (3/8)*(kappa**4+1/kappa**4)/x[leq_min] 
            + (5/12)/x[leq_min] - (3/4)*(kappa**5+1/kappa**5) + (5/4)*(kappa**3+1/kappa**3) )
        localization_weights[leq_max] = ( -(1/12)*x[leq_max]**5 + (1/4)*(kappa-1/kappa)*x[leq_max]**4 + (5/8)*x[leq_max]**3 - (5/6)*(kappa**3-1/kappa**3)*x[leq_max]**2 
            + (5/4)*(kappa**4-kappa**2-1/kappa**2-1/kappa**4)*x[leq_max] + (5/12)/x[leq_max] - (3/8)*(kappa**4+1/kappa**4)/x[leq_max]  
            + (1/6)*(kappa**6-1/kappa**6)/x[leq_max] + (5/4)*(kappa**3+1/kappa**3) - (3/4)*(kappa**5-1/kappa**5) )
        localization_weights[leq_sum] = ( (1/12)*x[leq_sum]**5 - (1/4)*(kappa+1/kappa)*x[leq_sum]**4 + (5/8)*x[leq_sum]**3 + (5/6)*(kappa**3+1/kappa**3)*x[leq_sum]**2 
            - (5/4)*(kappa**4+kappa**2+1/kappa**2+1/kappa**4)*x[leq_sum] + (5/12)/x[leq_sum] - (3/8)*(kappa**4+1/kappa**4)/x[leq_sum]  
            - (1/6)*(kappa**6+1/kappa**6)/x[leq_sum] + (5/4)*(kappa**3+1/kappa**3) + (3/4)*(kappa**5+1/kappa**5) )
    return localization_weights

def calculate_volume_spherical_cap(r, x):
    ''' This function calculates the volume of a spherical cap with 
    triangular height x of a sphere with radius r
    '''
    V = (np.pi/3) * (r - x)**2 * (2*r + x)
    return V

def calculate_intersection_volume(d, R, r):
    ''' This function calculates the volume of the intersection of two spheres,
    with radii r and R with centers separated by a distance d
    '''
    V1 = calculate_volume_spherical_cap(R, (d**2 + R**2 - r**2)/(2*d))
    V2 = calculate_volume_spherical_cap(r, (d**2 + r**2 - R**2)/(2*d))
    V =  V1 + V2
    return V

def bolin_wallin_cross(distance, localization_half_width1, localization_half_width2):
    ''' This function calculates the normalized volume of intersection of two spheres with
    centers d units apart, where localization half widths 1 & 2 are the radii of the two spheres
    The normalization is chosen so that the normalized volume of two colocated spheres 
    with the same radius is 1. 
    
    Function defined in Bolin & Wallin (2016), Section 2.1
    Note that there is an typo in Bolin and Wallin, which is corrected here.
    B & W write pi * r^3/2 when they mean pi * r^3.
    '''
    c_min = min(localization_half_width1, localization_half_width2)
    c_max = max(localization_half_width1, localization_half_width2)
    distance = np.abs(distance)
    # two cases
    leq_dif = distance <= c_max-c_min # one sphere contained in the other
    leq_sum = np.logical_and(c_max - c_min < distance, distance <= c_max + c_min ) # lens intersection
    # calculate volume
    volume = np.zeros(distance.shape)
    volume[leq_dif] = (4/3) * np.pi * c_min**3
    volume[leq_sum] = calculate_intersection_volume(distance[leq_sum], c_max, c_min)
    # calculate normalization factors
    normalization_factor = (4/3) * np.pi * np.sqrt(c_min**3 * c_max**3)
    # localization weight is the normalized volume
    localization_weights = volume / normalization_factor
    return localization_weights

def bolin_wallin_univariate(distance, localization_half_width):
    ''' This function calculates the normalized volume of intersection of two spheres with
    centers d units apart, the radii of the two spheres are the same
    The normalization is chosen so that the normalized volume of two colocated spheres is 1. 
    
    Function defined in Bolin & Wallin (2016), Section 2.1
    Note that there is an typo in Bolin and Wallin, which is corrected here.
    B & W write pi * r^3/2 when they mean pi * r^3.
    '''
    # non-zero localization weights are less than 2 * loc. half width
    distance = np.abs(distance)
    leq_sum = (distance <= 2 * localization_half_width ) # lens intersection
    # calculate volume
    volume = np.zeros(distance.shape)
    volume[leq_sum] = 2*calculate_volume_spherical_cap(localization_half_width, distance[leq_sum]/2)
    # calculate normalization factors
    normalization_factor = (4/3) * np.pi * localization_half_width**3
    # localization weight is the normalized volume
    localization_weights = volume / normalization_factor
    return localization_weights

def demonstrate():
    ''' For demonstration purposes only'''
    # Gaspari-Cohn localization with distance=1 and localization radius=2 is computed as follows:
    loc_gc = gaspari_cohn_univariate(distance=[1], localization_half_width=1)
    print('Gaspari-Cohn localization at distance=1 and localization radius=2 is: '+str(loc_gc))
    
    # If localization radius of temperature=2 and localization radius of humidity=1 and distance=1
    # then the GC cross localization is computed as follows
    loc_gc_cross = gaspari_cohn_cross(distance=[1], localization_half_width1=1, localization_half_width2=0.5)
    print('If localization radius of temperature=2 and localization radius of humidity=1 and distance=1 then the GC QT cross localization is: '+str(loc_gc_cross))

    # Now we do the same computations with Bolin-Wallin
    loc_bw = bolin_wallin_univariate(distance=[1], localization_half_width=1)
    print('Bolin-Wallin localization at distance=1 and localization radius=2 is: '+str(loc_bw))

    loc_bw_cross = bolin_wallin_cross(distance=[1], localization_half_width1=1, localization_half_width2=0.5)
    print('If localization radius of temperature=2 and localization radius of humidity=1 and distance=1 then the BW QT cross localization is: '+str(loc_bw_cross))

    # At shorter distances GC-cross is larger than BW-cross.
    loc_gc_cross0 = gaspari_cohn_cross(distance=[0], localization_half_width1=1, localization_half_width2=0.5)
    loc_bw_cross0 = bolin_wallin_cross(distance=[0], localization_half_width1=1, localization_half_width2=0.5)
    print('With distance=0 and localization radius of temp=2 and loc rad of humidity=1 GC-cross, '+str(loc_gc_cross0)+', is larger than BW-cross, '+str(loc_bw_cross0)+'.')

if __name__ == "__main__":
    demonstrate()





