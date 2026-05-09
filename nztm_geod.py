# =============================================================================
# nztm_geod.py
# =============================================================================
#
# PURPOSE
# -------
# Implements coordinate transformations between the New Zealand Transverse
# Mercator (NZTM) projection and geographic latitude/longitude.
#
# DESCRIPTION
# -----------
# Provides routines to convert NZTM eastings/northings (metres) to geodetic
# latitude/longitude (radians), and vice versa.
#
# NZTM PARAMETERS
# ---------------
#   Projection:        Transverse Mercator
#   Longitude origin:  173.0 degrees
#   Latitude origin:   0.0 degrees
#   Scale factor:      0.9996
#   False easting:     1,600,000 m
#   False northing:    10,000,000 m
#
# PROVENANCE
# ----------
# The underlying algorithm is derived from legacy C code published by
# Toitū Te Whenua Land Information New Zealand (LINZ):
#
#   https://www.linz.govt.nz/products-services/geodetic/geodetic-software-and-downloads
#
# The original source was non-Python and has been substantially refactored,
# modified, and restructured for use in this package.
#
# This Python implementation and all modifications are by Phillippe Bruneau.
#
# LICENSING
# ---------
# Original source terms/conditions (LINZ website):
#
#   "Download and use of these software applications is taken to be acceptance
#    of the following conditions. Toitū Te Whenua Land Information New Zealand
#    does not offer any support for this software. The software is provided
#    'as is' and without warranty of any kind. Toitū Te Whenua will not be liable
#    for loss of any kind related to the download, installation and use of the
#    software. The software is provided for free."
#
# This Python implementation is an independent port of the underlying algorithm
# and is distributed under the MIT License, consistent with the pypsa_nza_data
# package.
#
# REFERENCES
# ----------
#   LINZ Concord (online calculator):
#   https://www.geodesy.linz.govt.nz/concord/
#
# HISTORY
# -------
# Created: 2024-06-20
# Author:  Phillippe Bruneau
#
# =============================================================================

import numpy as np
# import matplotlib.pyplot as plt
from dataclasses import dataclass
# import pyproj as pj


# Define the parameters for the International Ellipsoid
# used for the NZGD2000 datum (and hence for NZTM)

@dataclass
class TransverseMercator():
    meridian: float = 173.0             # Central meridian (degrees)
    cm:       float = 0.0               # Central meridian (radians)
    k_0:      float = 0.9996            # Central meridian scale factor (k_0)
    phi_0:    float = 0.0               # Origin latitude (phi_0)
    E_0:      float = 1600000.0         # False easting (m)
    N_0:      float = 10000000.0        # False northing (m)
    a:        float = 6378137.0         # Semi-major axis (m)- ref ellipsoid
    b:        float = 0.0               # Semi-minor axis (m)- ref ellipsoid
    f_inv:    float = 298.257222101     # Inverse flattening
    f:        float = 0.0               # Flattening
    e2:       float = 0.0               # Eccentricity squared
    e_p2:     float = 0.0               # Second eccentricity
    e4:       float = 0.0               # Eccentricity to fourth power
    e6:       float = 0.0               # Eccentricity  to sixth power
    n:        float = 0.0               # Third flattening
    m_0:      float = 0.0               #


# Define general constants
PI = np.pi
TWOPI = 2*PI


# Initiallize the TM structure
def init_tm_proj(tm):
    # Convert primary meridian to radians
    tm.cm = np.deg2rad(tm.meridian)

    # Flattening and reciprocal (inverse) flattening
    if (tm.f_inv != 0.0):
        tm.f = 1.0/tm.f_inv
    else:
        tm.f = 0.0

    # Semi-minor axis
    tm.b = tm.a*(1 - tm.f)

    # Third flattening
    tm.n = (tm.a - tm.b) / (tm.a + tm.b)

    # Eccentricity
    tm.e2 = 2.0*tm.f - tm.f**2
    tm.e4 = tm.e2*tm.e2
    tm.e6 = tm.e4*tm.e2

    # Second eccentricity
    tm.e_p2 = tm.e2 / (1.0 - tm.e2)

    # Meridional arc
    tm.om = meridian_distance(tm, tm.cm)

    # Ellipsoid constant, 'G'
    n = tm.n
    n2 = n*n
    n4 = n2*n2
    tm.G = tm.a*(1 - n)*(1 - n2)*(1 + 9*n2/4 + 225*n4/64) * (np.pi/180)

    return tm


# ****************************************************************************
# *                                                                          *
# *  meridian_distance                                                       *
# *                                                                          *
# *  Returns the length of meridional arc (Helmert formula),                 *
# *  Method based on Redfearn's formulation as expressed in GDA technical    *
# *  manual at http://www.anzlic.org.au/icsm/gdatm/index.html                *
# *                                                                          *
# *  Parameters are                                                          *
# *    tm  : projection class instance                                       *
# *    phi : latitude (radians)                                              *
# *                                                                          *
# *  Return value                                                            *
# #     m : meridional arc length in metres                                  *
# *                                                                          *
# ****************************************************************************

def meridian_distance(tm, phi):
    a = tm.a

    e2 = tm.e2
    e4 = tm.e4
    e6 = tm.e6

    A0 = 1 - (e2/4.0) - (3.0*e4/64.0) - (5.0*e6/256.0)
    A2 = (3.0/8.0) * (e2 + e4/4.0 + 15.0*e6/128.0)
    A4 = (15.0/256.0) * (e4 + 3.0*e6/4.0)
    A6 = 35.0*e6/3072.0

    return a*(A0*phi - A2*np.sin(2*phi) + A4*np.sin(4*phi) - A6*np.sin(6*phi))


# ****************************************************************************
# *                                                                          *
# *   foot_point_lat                                                         *
# *                                                                          *
# *   Calculates the foot-point latitude from the meridional arc             *
# *   Method based on Redfearn's formulation as expressed in GDA technical   *
# *   manual at http://www.anzlic.org.au/icsm/gdatm/index.html               *
# *                                                                          *
# *   The foot-point latitude is the latitude for which the meridian         *
# *   disrance equals the true Northing divided by the central scale-factor. *
# *                                                                          *
# *   Takes parameters                                                       *
# *      tm definition for scale factor                                      *
# *                                                                          *
# *                                                                          *
# *                                                                          *
# *      meridional arc (metres)                                             *
# *                                                                          *
# *   Returns the foot point latitude (radians)                              *
# *                                                                          *
# ****************************************************************************

def foot_point_lat(n, sigma):

    n2 = n*n
    n3 = n2 * n
    n4 = n2 * n2

    k1 = (3*n/2 - 27*n3/32)*np.sin(2*sigma)
    k2 = (21*n2/16 - 55*n4/32)*np.sin(4*sigma)
    k3 = (151*n3/96) * np.sin(6*sigma)
    k4 = (1097*n4/512) * np.sin(8*sigma)
    phi_p = sigma + k1 + k2 + k3 + k4

    # print('phi_p')
    # print(phi_p)
    return phi_p


def curvature(tm, phi):

    # -------------------------------------------------------------------------
    #
    #  curvature
    #     Computes the radii of curvature of the meridian (rho) in the prime
    #     vertical.
    #
    #     Inputs:
    #       e2 - eccentricity squared
    #       phi - latiude (radians)
    #
    #     Returns:
    #       rho - (m)
    #       nu  - (m)
    # -------------------------------------------------------------------------

    r1 = tm.a * (1 - tm.e2)
    r2 = 1 - tm.e2 * (np.sin(phi))**2
    r3 = r2**1.5
    rho = r1/r3

    nu = tm.a / np.sqrt(1 - tm.e2 * (np.sin(phi))**2)

    # psi = nu/rho
    # r2 = rho*nu*tm.k_0**2

    return rho, nu


def northing_conv(t, x, E_p, k_0, rho, phi_p, psi):

    # -------------------------------------------------------------------------
    #
    #  northing_conv
    #  Generates the equivalent geographical latitude (decimal degrees) given
    #  an NZTM specification and the appropriate parameters derived therefrom.
    #
    #  'northing_conv' is called from 'tm_geod' which computes the required
    #  input parameters.
    #
    #  Returns:
    #      phi: latitude in radians
    #
    # -------------------------------------------------------------------------

    # Constants
    t2 = t*t
    t4 = t2*t2
    t6 = t4*t2
    C0 = (t * E_p) / (k_0 * rho)

    # Term 1
    k1 = C0 * x * 0.5

    # Term 2
    C2 = -4.0*(psi**2) + 9.0*(1 - t2)*psi + 12.0*t2
    k2 = C0 * (x**3) * C2 / 24

    # Term 3
    C3 = (
        8*(11 - 24*t2)*psi**4
        - 12 * (21 - 71*t2) * psi**3
        + 15 * (15 - 98*t2 + 15*t4) * psi**2
        + 180 * (5*t2 - 3*t4) * psi
        + 360.0*t4
    )
    k3 = C0 * (x**5/720) * C3

    # Term 4
    C4 = 1385 + 3633*t2 + 4095*t4 + 1575*t6
    k4 = C0 * C4 * (x**7) / 40320

    phi = phi_p - k1 + k2 - k3 + k4
    # print('PHI = ', phi)
    return phi


def easting_conv(t, x, lam_0, sec, psi):

    # -------------------------------------------------------------------------
    #
    #  Converts the NZTM easting coordinate (m) to the equivalent geographical
    #  longitude (decimal degrees) coordinate.
    #
    # -------------------------------------------------------------------------

    # Constants
    t2 = t*t
    t4 = t2*t2
    t6 = t4*t2

    # Term 1
    C1 = x * sec
    k1 = C1

    # Term 2
    C2 = (sec * x**3) / 6
    k2 = C2 * (psi + 2*t2)

    # Term 3
    C3 = (sec * x**5) / 120
    k3 = C3 * (
        - 4 * (1 - 6*t2) * psi**3
        + (9 - 68*t2) * psi**2
        + (72*t2) * psi
        + 24*t4
    )

    # Term 4
    C4 = (sec * x**7) / 5040
    k4 = C4 * (61.0 + 662*t2 + 1320*t4 + 720.0*t6)

    lam = lam_0 + k1 - k2 + k3 - k4

    return lam


# ****************************************************************************
# *                                                                          *
# *   tm_geod                                                                *
# *                                                                          *
# *   Converts from Tranverse Mercator northings and eatings to gepgraphic   *
#     latitude and longitude.                                                *
# *   Method based on Redfearn's formulation.                                *
# *                                                                          *
# *   Takes parameters                                                       *
# *      input easting (metres)                                              *
# *      input northing (metres)                                             *
# *      output latitude (radians)                                           *
# *      output longitude (radians)                                          *
# *                                                                          *
# ****************************************************************************

def tm_geod(tm, E, N):

    N_p = N - tm.N_0  # *** v
    E_p = E - tm.E_0  # *** v

    m_p = tm.m_0 + N_p / tm.k_0  # *** v
    sigma = (m_p * np.pi) / (180*tm.G)  # *** v
    phi_p = foot_point_lat(tm.n, sigma)  # Foot-point latitude

    rho_p, nu_p = curvature(tm, phi_p)  # curvature - evaluated at phi_p
    psi = nu_p / rho_p

    t = np.tan(phi_p)
    x = E_p / (tm.k_0 * nu_p)
    sec = 1/np.cos(phi_p)

    # Latitude (Northing conversion)
    lat = northing_conv(t, x, E_p, tm.k_0, rho_p, phi_p, psi)

    # Longitude (Easting conversion)
    long = easting_conv(t, x, tm.cm, sec, psi)

    return lat, long


# ****************************************************************************
# *                                                                          *
# *   geod_tm                                                                *
# *                                                                          *
# *   Convert latitude and longitude to Transverse Mercator.                 *
# *   Method based on Redfearn's formulation as outlined in the LINZ         *
# *   web pages.                                                             *
# *                                                                          *
# *   Takes parameters                                                       *
# *      input latitude (radians)                                            *
# *      input longitude (radians)                                           *
# *      output easting  (metres)                                            *
# *      output northing (metres)                                            *
# *                                                                          *
# ****************************************************************************

def geod_tm(tm, ln, lt, ce, cn):

    fn = tm.falsen
    fe = tm.falsee
    sf = tm.scalef
    e2 = tm.e2
    a = tm.a
    cm = tm.cm
    om = tm.om
    utom = tm.utom

    dlon = ln - cm
    while (dlon > PI):
        dlon -= TWOPI
    while (dlon < -PI):
        dlon += TWOPI

    m = meridian_distance(tm, lt)

    slt = np.sin(lt)

    eslt = (1.0-e2*slt*slt)
    eta = a/np.sqrt(eslt)
    rho = eta * (1.0-e2) / eslt
    psi = eta/rho

    clt = np.cos(lt)
    w = dlon

    wc = clt*w
    wc2 = wc*wc

    t = slt/clt
    t2 = t*t
    t4 = t2*t2
    t6 = t2*t4

    trm1 = (psi-t2)/6.0

    trm2 = (((4.0*(1.0-6.0*t2)*psi
              + (1.0+8.0*t2))*psi
             - 2.0*t2)*psi+t4)/120.0
    trm3 = (61 - 479.0*t2 + 179.0*t4 - t6)/5040.0

    gce = (sf*eta*dlon*clt)*(((trm3*wc2+trm2)*wc2+trm1)*wc2+1.0)
    ce = gce/utom+fe

    trm1 = 1.0/2.0
    trm2 = ((4.0*psi+1)*psi-t2)/24.0

    trm3 = ((((8.0*(11.0-24.0*t2)*psi
            - 28.0*(1.0-6.0*t2))*psi
            + (1.0-32.0*t2))*psi
            - 2.0 * t2)*psi
            + t4)/720.0

    trm4 = (1385.0-3111.0*t2+543.0*t4-t6)/40320.0

    gcn = (eta*t)*((((trm4*wc2+trm3)*wc2+trm2)*wc2+trm1)*wc2)
    cn = (gcn + m - om)*sf/utom + fn

    return ce, cn


if __name__ == '__main__':

    # Define transverse mercator class instance
    tm = TransverseMercator()

    # Initialise the projection variable
    init_tm_proj(tm)

    # NZTM coordinates of Christchurch Cathedral are (E, N) = 1570634, 5180148
    # 43.5314° S, 172.6375° E
    # ce = 1570634
    # cn = 5180148
    # christchurch international airport
    # 1560557	5179492

    # Albury, SI
    # 44.2303° S, 170.8740° E from web search
    # Converted from geodetic to NZTM using :LINZ Online converter at
    # https://www.geodesy.linz.govt.nz/concord/ to give:
    ce = [1430210.221, 1695406.077]
    cn = [5100349.129, 5669677.696]
    # 5669677.696
    # Results from my implementation - above NZTM to geodetic
    lat, long = tm_geod(tm, ce, cn)
    lat = np.rad2deg(lat)
    long = np.rad2deg(long)

    # lat
    # Out[24]: -44.2303069471741
    # long
    # Out[25]: 170.874009613773
    #
    # Identical to 4 decimal places

    # print("Input NZTM e,n:  %12.3lf %12.3lf\n",e,n)
    # print("Output Lat/Long: %12.6lf %12.6lf\n",lt*rad2deg,ln*rad2deg)
    # print("Output NZTM e,n: %12.3lf %12.3lf\n",e1,n1)
    # print("Difference:      %12.3lf %12.3lf\n\n",e1-e,n1-n)

    #  -36.85255676, 174.76225110 .Auckland
