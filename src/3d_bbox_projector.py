from synthetic_data_generator import *
import math as m


# calculates horizontal interpolation limits g_h_1 & g_h_2
# Note that origin is given by [y_0, x_0]. See documentaiton for further info.
def horizontal_interpolation_limits(origin=[-5.735179668849011e-09, 1.711045726422736],
                                    angle=60):
    m_h_1 = m.tan(m.radians((180-angle)/2))
    m_h_2 = -m_h_1
    b_h_1 = origin[1] - m_h_1*origin[0]
    b_h_2 = origin[1] - m_h_2*origin[0]

    return m_h_1, m_h_2, b_h_1, b_h_2


# calculates vertical interpolation limits g_v_1 & g_v_2
# Note that origin is given by [x_0, z_0]. See documentaiton for further info.
def vertical_interpolation_limits(origin=[1.711045726422736, 0.9431449279047172],
                                    angle=38):
    m_v_1 = m.tan(angle/2)
    m_v_2 = -m_v_1
    b_v_1 = origin[1] - m_v_1*origin[0]
    b_v_2 = origin[1] - m_v_2*origin[0]

    return m_v_1, m_v_2, b_v_1, b_v_2


# mapes lidar points onto image on the x-axis
def map_lidar_point_onto_image_x(point, y_1_h=1920, y_1_l=0):
    h_limits = horizontal_interpolation_limits()
    y_2_m = point[1]
    x_2_m = point[0]
    y_2_h = (x_2_m-h_limits[2])/h_limits[0]
    y_2_l = (x_2_m-h_limits[3])/h_limits[1]
    # linear interpolation
    x_bild = y_1_l + ((y_2_m-y_2_l)/(y_2_h-y_2_l)) * (y_1_h-y_1_l)

    return x_bild/2

x = map_lidar_point_onto_image_x([6.81953126, 3.83764032])
print(x)