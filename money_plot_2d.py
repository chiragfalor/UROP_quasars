# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from matplotlib.widgets import Slider

import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib
import random
import pickle

number_of_linalgerr = 0
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})

#plt.rc('axes.spines', **{'bottom':False, 'left':False, 'right':False, 'top':False})
plt.axis('off')
np.random.seed(0)
random.seed(2)

class Conic():
  def __init__(self):
    pass
  def distance(self, c1, c2):
    '''
    input : 2 tuples (coordinates)
    output : real
    returns cartesian distance between 2 coordinates
    '''
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)



class Circle(Conic):
    def __init__(self, c1, c2, c3):
        '''
        initialize a circle in cartesian coordinates where c1, c2, c3 are tuples of coordinates in the form of (x_i, y_i)
        '''
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.circle_matrix = np.array([[c1[0]**2 + c1[1]**2, c1[0], c1[1], 1],
                                    [c2[0]**2 + c2[1]**2, c2[0], c2[1], 1],
                                    [c3[0]**2 + c3[1]**2, c3[0], c3[1], 1]])
        
        self.center = self.findCenter(c1, c2, c3)
        self.radius = self.distance(self.center, c1)
        self.coefficients = np.array([1, -2*self.center[0], -2*self.center[1], self.center[0]**2 + self.center[1]**2 - self.radius**2]) # coefficients are as A(x^2 + y^2) + Bx + Cy + D = 0
        self.equation = "x^2 + y^2 + " + str(self.coefficients[1]) + "x + " + str(self.coefficients[2]) + "y + " + str(self.coefficients[3])

    def findCenter(self, c1, c2, c3):
        '''
        input : 3 tuples (coordinates)
        returns : 1 tuple (coordinate)
        find center of the circle passing through the given 3 coordinates
        '''
        x_c = 1/2*np.linalg.det(np.delete(self.circle_matrix, 1, 1))/np.linalg.det(np.delete(self.circle_matrix, 0, 1))
        y_c = -1/2*np.linalg.det(np.delete(self.circle_matrix, 2, 1))/np.linalg.det(np.delete(self.circle_matrix, 0, 1))
        return (x_c, y_c)

    def plot(self):
        x = np.linspace(-9, 9, 400)
        y = np.linspace(-5, 5, 400)
        x, y = np.meshgrid(x, y)
        a, b, c, d, e, f = 1, 0, 1, self.coefficients[1], self.coefficients[2], self.coefficients[3]
        def axes():
            plt.axhline(0, alpha=.1)
            plt.axvline(0, alpha=.1)
        axes()
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='k')
        plt.plot(self.c1[0], self.c1[1], 'r+')
        plt.plot(self.c2[0], self.c2[1], 'r+')
        plt.plot(self.c3[0], self.c3[1], 'r+')
        plt.show()



class RectangularHyperbola(Conic):
    def __init__(self, c1, c2, c3, c4):
        '''
        initialize a rectangular hyperbola in cartesian coordinates where c1, c2, c3, c4 are tuples of coordinates in the form of (x_i, y_i)
        '''
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        equation_matrix = np.array([self.equation_maker_hyperbola(c1), self.equation_maker_hyperbola(c2), self.equation_maker_hyperbola(c3), self.equation_maker_hyperbola(c4)])
        equation_solutions = np.array([1,1,1,1])
        '''
        Assumes that equation of rectangular hyperbola is of the form A(x^2 - y^2) + Bxy + Cx + Dy = 1 and solves the 4 equations got by substituting the 4 coordinates
        '''
        self.coefficients = np.linalg.solve(equation_matrix, equation_solutions)
        self.center = self.findCenter()
        self.equation = str(self.coefficients[0]) + "(x^2 - y^2) + " + str(self.coefficients[1]) + "xy + " + str(self.coefficients[2]) + "x + " + str(self.coefficients[3]) + "y = 1"

    def findCenter(self):
        '''
        returns the center of the current hyperbola by partial differentiation of the equation wrt x and y separately and solving the linear equation
        '''
        A, B, C, D = self.coefficients
        x_c = -(B*D + 2*A*C)/(4*(A**2) + B**2)
        y_c = -(B*C - 2*A*D)/(4*(A**2) + B**2)
        return (x_c, y_c)

    def equation_maker_hyperbola(self, c1):
        '''
        input: tuple (Coordinate)
        output: from tuple(x,y), it returns (x**2-y**2, x*y, x, y)
        '''
        x = c1[0]
        y = c1[1]
        return (x**2-y**2, x*y, x, y)

    def plot(self):
        x = np.linspace(-3, 3, 400)
        y = np.linspace(-3, 3, 400)
        x, y = np.meshgrid(x, y)
        a, b, c, d, e, f = self.coefficients[0], self.coefficients[1], -self.coefficients[0], self.coefficients[2], self.coefficients[3], -1
        def axes():
            plt.axhline(0, alpha=.1)
            plt.axvline(0, alpha=.1)
        axes()
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='k')
        plt.plot(self.c1[0], self.c1[1], 'r+')
        plt.plot(self.c2[0], self.c2[1], 'r+')
        plt.plot(self.c3[0], self.c3[1], 'r+')
        plt.plot(self.c4[0], self.c4[1], 'r+')
        plt.show()


class Quasar(Conic):
  def __init__(self, c1, c2, c3, c4 = None):
    self.c1 = c1
    self.c2 = c2
    self.c3 = c3
    if not c4:
      self.c4 = self.quasar_determiner(c1, c2, c3)
    else:
      self.c4 = c4
      self.quasar_circle = Circle(c1, c2, c3)
      self.quasar_hyperbola = RectangularHyperbola(c1, c2, c3, c4)
      self.quasar_hyperbola_center = self.quasar_hyperbola.center

    self.normalize_configuration()

    self.sort_configuration()

    self.configuration_angles = np.arctan2(self.quasar_norm_array[:, 1], self.quasar_norm_array[:, 0])

    self.psi = self.psi_calculator()
    
    self.quasar_rotator(-self.psi)

    self.quasar_hyperbola = RectangularHyperbola(self.c1, self.c2, self.c3, self.c4)

    self.theta_23 = self.angle_difference(self.configuration_angles[2], self.configuration_angles[1])

    self.ratio = self.configuration_angles[1]/(self.angle_difference(np.pi/2, self.configuration_angles[2]))

    #self.configurationInvariant = self.Calculate_configuration_invariant(self.configuration_angles)

    #self.separationInvariant = self.separation_invariant(self.quasar_norm_array)

    #self.diff_WW_KK = self.WW_invariant_calculator(self.configuration_angles)

    self.causticity = self.calculate_causticity()

    self.new_causticity = self.calculate_new_causticity()

    self.astroidal_angle = self.calculate_astroidal_angle()

    #self.property_checker()
    
    self.quasar_rotator(-np.pi/2)



  
  def quasar_determiner(self, c1, c2, c3):
    '''
    input: 3 tuples(coordinates) of quasar
    returns: tuple, the 4th coordinate assuming vanishingly elliptic potential
    '''
    self.quasar_circle = Circle(c1, c2, c3)
    self.quasar_hyperbola = RectangularHyperbola(c1, c2, c3, self.quasar_circle.center)
    self.quasar_hyperbola_center = self.quasar_hyperbola.center
    x_quasar = 2*(self.quasar_hyperbola.center[0] + self.quasar_circle.center[0]) - (c1[0] + c2[0] + c3[0])
    y_quasar = 2*(self.quasar_hyperbola.center[1] + self.quasar_circle.center[1]) - (c1[1] + c2[1] + c3[1])
    return (x_quasar, y_quasar)
    
  def normalize_configuration(self):
    C = self.quasar_circle
    npcenter = np.array(C.center)
    quasar_array = np.array([self.c1, self.c2, self.c3, self.c4])
    quasar_norm_array = (quasar_array - npcenter)/C.radius
    self.quasar_hyperbola_center = (self.quasar_hyperbola_center - npcenter)/C.radius
    self.quasar_norm_array = quasar_norm_array
    self.c1 = quasar_norm_array[0]
    self.c2 = quasar_norm_array[1]
    self.c3 = quasar_norm_array[2]
    self.c4 = quasar_norm_array[3]
    self.quasar_circle = Circle(self.c1, self.c2, self.c3)
    return quasar_norm_array
    

  def angle_calculator(self, c1):
    return np.arctan2(c1[1], c1[0])

  def sort_configuration(self):
    '''
    sorts the 4 points of quasar. It assigns image 2 and 3 to the points on the arm of hyperbola which does not contain the circle center and assigns 1, 4 to the other 2.
    '''
    def side_determiner(v, c1):
      '''
      Input : vector and a point
      draws a line perpendicular to the given vector
      returns: +1 if the point is on the side of line in which vector point, 0 if point is on the line, and -1 if point is on the side of origin
      '''
      c1 = np.array(c1)
      v = np.array(v)
      return np.sign(np.dot(c1, v) - np.dot(v,v))
    self.quasar_norm_array = np.array(sorted(self.quasar_norm_array, key=self.angle_calculator))
    is_image_23 = []
    for c in self.quasar_norm_array:
      is_image_23.append(side_determiner(self.quasar_hyperbola_center, c))
    if sum(is_image_23) != 0:
      raise Exception("More than 2 quasar points lie on 1 branch of hyperbola")
      #plt.plot(self.c1, self.c2, self.c3, self.c4)
      #self.quasar_hyperbola.plot()
    def rotate_list(list_to_rotate):
      return np.roll(list_to_rotate, 1, axis=0)
    while (is_image_23[0] != 1) or (is_image_23[1] != 1):
      is_image_23 = rotate_list(is_image_23)
      self.quasar_norm_array = rotate_list(self.quasar_norm_array)
    self.c2 = self.quasar_norm_array[0]
    self.c3 = self.quasar_norm_array[1]
    self.c1 = self.quasar_norm_array[2]
    self.c4 = self.quasar_norm_array[3]
    self.quasar_norm_array = np.array([self.c1, self.c2, self.c3, self.c4])
    return self.quasar_norm_array

  def configuration_angles_to_coordinates(self, configuration_angles):
    '''
    sets the cartesian coordinates of the image as per the configuration angles given
    '''
    self.quasar_norm_array = np.array([np.cos(configuration_angles), np.sin(configuration_angles)]).T
    self.c1 = self.quasar_norm_array[0]
    self.c2 = self.quasar_norm_array[1]
    self.c3 = self.quasar_norm_array[2]
    self.c4 = self.quasar_norm_array[3]

  def quasar_rotator(self, phi):
    '''
    rotates the quasar image coordinates by an angle of phi clockwise
    '''
    def angle_sum(phi, theta):
      sum = phi + theta
      while sum>np.pi or sum<-np.pi:
        if sum>np.pi:
          sum -= 2*np.pi
        elif sum < -np.pi:
          sum += 2*np.pi
      return sum
    new_configurational_angles = np.zeros(4)
    for i in range(4):
      new_configurational_angles[i] = angle_sum(self.configuration_angles[i], phi)
    self.configuration_angles = new_configurational_angles
    self.configuration_angles_to_coordinates(new_configurational_angles)

  def angle_difference(self, a1, a2):
      positive_difference = abs(a1 - a2)
      if positive_difference > np.pi:
        return 2*np.pi - positive_difference
      else:
        return positive_difference

  def psi_calculator(self):
    psi = np.sum(self.configuration_angles)/4 - np.pi/4
    theta_2, theta_3 = self.configuration_angles[1], self.configuration_angles[2]
    while self.angle_difference(psi, theta_2) > np.pi/2 or self.angle_difference(psi, theta_3) > np.pi/2  or (self.angle_difference(psi, theta_2) - self.angle_difference(psi, theta_3) >0):
      psi = psi + np.pi/2
      if psi > np.pi:
        psi -= 2*np.pi
    #r = np.random.randint(4)
    #psi += r*np.pi/2
    return psi

  def saddle_min_switcher(self):
    '''
    Changes the saddle to min and min to saddle
    '''
    self.c1 = self.quasar_norm_array[3]
    self.c2 = self.quasar_norm_array[2]
    self.c3 = self.quasar_norm_array[1]
    self.c4 = self.quasar_norm_array[4]
    self.quasar_norm_array = np.array([self.c1, self.c2, self.c3, self.c4])
    return self.quasar_norm_array

  def calculate_causticity(self):
    '''
    theta = configuration_angles
    theta_23 = self.angle_difference(theta[1], theta[2])
    '''
    x_sum = np.sum(self.quasar_norm_array[:,0])
    y_sum = np.sum(self.quasar_norm_array[:,1])
    causticity = (np.cbrt((x_sum/2)**(2)) + np.cbrt((y_sum/2)**(2)))**(3/2)
    #print('Causticity', x_sum, y_sum, causticity)
    return causticity

  def calculate_new_causticity(self):
    t = self.configuration_angles
    theta_23 = self.angle_difference(t[1], t[2])
    new_causticity = (((np.cos(t[1])*np.cos(t[2])*np.cos((t[1]+t[2])/2))**(2/3)+(np.sin(t[1])*np.sin(t[2])*np.sin((t[1]+t[2])/2))**(2/3))**(1.5))/np.cos(theta_23/2)
    #new_causticity = 2*np.sqrt((np.cos(t[1])*np.cos(t[2])*np.cos((t[1]+t[2])/2))**2+(np.sin(t[1])*np.sin(t[2])*np.sin((t[1]+t[2])/2))**2)/np.cos(theta_23/2)
    return new_causticity





  def calculate_astroidal_angle(self):
    x_sum = np.sum(self.quasar_norm_array[:,0])
    y_sum = np.sum(self.quasar_norm_array[:,1])
    alpha = np.arctan2(y_sum,x_sum)
    alpha_new = np.arctan(np.cbrt(np.tan(self.configuration_angles[1])*np.tan(self.configuration_angles[2])*np.tan((self.configuration_angles[1] + self.configuration_angles[2])/2)))
    '''
    if self.configuration_angles[1]>np.pi/2:
      alpha_new += np.pi
    elif self.configuration_angles[1]<-np.pi/2:
      alpha_new -= np.pi
    '''
    return np.pi/2 - alpha_new

  def property_checker(self):
    epsilon = 0.01
    theta = self.configuration_angles
    '''
    property 1
    tan(theta_2)*tan(theta_3)*tan((theta_2+theta_3)/2) = tan alpha = (sum sin theta)/(sum cos theta)
    '''
    a = np.arctan(np.tan(self.configuration_angles[1])*np.tan(self.configuration_angles[2])*np.tan((self.configuration_angles[1] + self.configuration_angles[2])/2))
    b = np.arctan(np.sum(np.sin(theta))/np.sum(np.cos(theta)))
    if (abs(a-b) > epsilon):
      self.__str__()
      raise Exception("property l lost")
    '''
    property 2
    sum sin theta = 2 sin theta_2 sin theta_3 sin ((theta_2+theta_3)/2)/cos(theta_23/2)
    '''
    a = np.sum(np.sin(theta))
    b = 2*np.sin(theta[1])*np.sin(theta[2])*np.sin((theta[1]+theta[2])/2)/np.cos((theta[1]- theta[2])/2)
    if abs(a-b) > epsilon:
      self.__str__()
      raise Exception("property 2 lost")
    


  def plot(self):
    radius_ratio = 1
    xs,ys = self.quasar_norm_array[:,0]*radius_ratio, self.quasar_norm_array[:,1]*radius_ratio
    cc = plt.Circle((0,0), radius_ratio , alpha=0.1)
    # plot the points
    plt.scatter(xs,ys,c='#d62728', marker = 'o')
    plt.gca().set_aspect('equal')
    plt.gca().add_artist( cc )
    # zip joins x and y coordinates in pairs
    j=0
    for x,y in zip(xs,ys):
        j += 1
        label = j
        if j == 1:
          offset = (-8, -5)
        elif j == 3:
          offset = (0,5)
        elif j==2:
          offset = (8,-3)
        elif j==4:
          offset = (0, -15)
        else:
          raise(Exception("Why you hurt me in this way :("))
        plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=offset, # distance from text to points (x,y)
                    ha='center',
                    size = 10)
    plt.show()

  def __str__(self):
    print("Configuration Angles", self.configuration_angles)
    print("Causticity", self.causticity)
    print("Astroidal angle", self.astroidal_angle)
    print("Quasar Coordinates", self.quasar_norm_array)
    self.plot()
    return "plotted"

  def __repr__(self):
    return str(self.configuration_angles)
    


def plot_astroid(r, gamma = 0, origin=(0,0)):
  theta = np.linspace(0, np.pi/2, 2000)
  
  radius = r
  if origin==(0,0):
    axis_x = np.linspace(0, r/(1-gamma), 2000)
    axis_y = np.linspace(0, r/(1+gamma), 2000)
    a = radius*(np.cos(theta))**3/(1-gamma)
    b = radius*(np.sin(theta))**3/(1+gamma)
    plt.plot(axis_x, np.zeros_like(axis_x), linestyle=(0,(5,5)), color='k',zorder=-100)
    plt.plot(np.zeros_like(axis_y) , axis_y, linestyle=(0,(5,5)), color='k', zorder=-100)
    plt.plot(a,b, zorder=-100)
  else:
    axis_x = np.linspace(origin[0], origin[0]-r/(1-gamma), 2000)
    axis_y = np.linspace(origin[1], origin[1]-r/(1+gamma), 2000)
    a = -radius*(np.cos(theta))**3/(1-gamma) + origin[0]
    b = -radius*(np.sin(theta))**3/(1+gamma) + origin[1]
    plt.plot(axis_x, origin[1]*np.ones_like(axis_x), linestyle=(0,(5,5)), color='k',zorder=-100)
    plt.plot(np.ones_like(axis_y)*origin[0] , axis_y, linestyle=(0,(5,5)), color='k', zorder=-100)
    plt.plot(a,b,zorder=-100)
  '''
  x = np.linspace(0.1, r, 4000)
  y = (r**(2/3) - x**(2/3))**3/2
  plt.plot(x,y)
  plt.plot(y,x)
  '''


def Quasar_random_shear_list(shear):
  '''
  returns list of nicely spaced quasars over astroid with non-zero shear
  '''
  global number_of_linalgerr
  N = 200000
  random.seed(3)
  max_causticity = 0.9
  plotting_dictionary = {(max_causticity, 0):0, (max_causticity, 1):0, (max_causticity, np.pi/4):0, (max_causticity, np.pi/2-1):0,  (max_causticity, np.pi/2):0, (3*max_causticity/4, 0):0, (3*max_causticity/4, np.pi/2):0,  (max_causticity/2,0):0, (max_causticity/2, np.pi/4):0, (max_causticity/2, np.pi/2):0, (max_causticity/4, 0):0, (max_causticity/4, np.pi/2):0, (0, 0):0}
  error_ratio = (2/np.pi)**2
  Quasar_list =[]
  def rounding_error(a):
      return abs(a-round(a))
  for i in range(N):
      del_1 = random.uniform(0, 10*np.pi)
      del_2 = random.uniform(0, 10*np.pi)
      a1 = del_1
      a2 = del_1 + del_2
      c1 = (2,0)
      c2 = (1+np.cos(a1), np.sin(a1))
      c3 = (1+np.cos(a2), np.sin(a2))
      try:
          Q = Quasar(c1, c2, c3)
          Quasar_tuple = (Q, Q.new_causticity, Q.astroidal_angle)
          #print(Q.theta_23, Q.ratio)
          zeta =  Q.new_causticity
          alpha = Q.astroidal_angle
          for t in plotting_dictionary.keys():
            err = (zeta-t[0])**2 + error_ratio*(alpha-t[1])**2
            if t == (0,0):
              err = (zeta-t[0])**2
            if plotting_dictionary[t] == 0:
                Quasar_list.append(Quasar_tuple)
                ind = len(Quasar_list) - 1
                plotting_dictionary[t] = (Q, ind, err)
                break
            else:
                ind =  plotting_dictionary[t][1]
                old_Q_err = plotting_dictionary[t][2]
                if old_Q_err > err:
                    Quasar_list[ind] = Quasar_tuple
                    plotting_dictionary[t] = (Q, ind, err)
                    break
      except np.linalg.LinAlgError:
          print("linalg err")
          number_of_linalgerr = number_of_linalgerr + 1
      except KeyError:
          print("theta_23 or ratio out of bounds")
          #print(t)
      #except:
       #    print("bad err")
        #   raise(Exception("solve it"))
  with open("quasars_shear1_3_list_new.txt", "wb") as fp:   #Pickling
    pickle.dump(Quasar_list, fp)
  #print(Quasar_list)
  return Quasar_list




def plot_quasars_shear_try2(Quasar_list, shear):
    i = 0
    #L = len(Quasar_list)
    plot_astroid(1, shear)
    #plot_astroid(max_causticity/2, shear)
    def p23(x):
      return (np.cbrt(x))**2
    for Quasar_tuple in Quasar_list:
        i += 1
        Q, causticity, astroidal_angle = Quasar_tuple
        radius_ratio = 1/24
        xs,ys = Q.quasar_norm_array[:,0]*radius_ratio/(1+shear) + causticity*(np.cos(astroidal_angle))**3/(1-shear) , Q.quasar_norm_array[:,1]*radius_ratio/(1-shear) + causticity*(np.sin(astroidal_angle))**3/(1+shear)
        cc = matplotlib.patches.Ellipse((causticity*(np.cos(astroidal_angle))**3/(1-shear) ,causticity*(np.sin(astroidal_angle))**3/(1+shear)), 2*radius_ratio/(1+shear) , 2*radius_ratio/(1-shear), alpha=1, facecolor = 'aquamarine', zorder = 1)
        # plot the points
        plt.scatter(xs,ys,c='#d62728', marker = 'o', s=msize, zorder=2)
        plt.gca().set_aspect('equal')
        plt.gca().add_artist(cc)
        # zip joins x and y coordinates in pairs
        '''
        j=0
        for x,y in zip(xs,ys):
            j += 1
            label = j
            if j == 1:
              offset = (-8, -3)
            elif j == 3:
              offset = (5,3)
            elif j==2:
              offset = (8,-5)
            elif j==4:
              offset = (0, -15)
            else:
              raise(Exception("Why you hurt me in this way :("))
            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=offset, # distance from text to points (x,y)
                        ha='center',
                        size = 14)
          
    ax1 = plt.gca()
    ax1.set_xlim(0, 1)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.autoscale()
    #ax1.set_title("Max Causticity = "+str(max_causticity))
          '''
    plt.savefig("test.svg", format="svg")
    plt.savefig('foo.pdf')




def plot_quasars_shear_1_3_try(Quasar_list, shear):
    i = 0
    #L = len(Quasar_list)
    origin =[1.6, 0.95]
    plot_astroid(1, shear, origin)
    #plot_astroid(max_causticity/2, shear)
    def p23(x):
      return (np.cbrt(x))**2
    for Quasar_tuple in Quasar_list:
        i += 1
        Q, causticity, astroidal_angle = Quasar_tuple
        radius_ratio = 1/25
        xs,ys = origin[0] - (Q.quasar_norm_array[:,0]*radius_ratio/(1+shear) + causticity*(np.cos(astroidal_angle))**3/(1-shear)) ,origin[1] - (Q.quasar_norm_array[:,1]*radius_ratio/(1-shear) + causticity*(np.sin(astroidal_angle))**3/(1+shear))
        cc = matplotlib.patches.Ellipse((origin[0] - causticity*(np.cos(astroidal_angle))**3/(1-shear) ,origin[1] - causticity*(np.sin(astroidal_angle))**3/(1+shear)), 2*radius_ratio/(1+shear) , 2*radius_ratio/(1-shear), alpha=1, facecolor = 'aquamarine', zorder = 1)
        # plot the points
        plt.scatter(xs,ys,c='#d62728', marker = 'o', s=msize, zorder =2)
        plt.gca().set_aspect('equal')
        plt.gca().add_artist( cc )
        # zip joins x and y coordinates in pairs
        '''
        j=0
        for x,y in zip(xs,ys):
            j += 1
            label = j
            if j == 1:
              offset = (-8, -3)
            elif j == 3:
              offset = (5,3)
            elif j==2:
              offset = (8,-5)
            elif j==4:
              offset = (0, -15)
            else:
              raise(Exception("Why you hurt me in this way :("))
            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=offset, # distance from text to points (x,y)
                        ha='center',
                        size = 14)
          
    ax1 = plt.gca()
    ax1.set_xlim(0, 1)
        '''
    ax1 = plt.gca()
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.autoscale()
    #ax1.set_title("Max Causticity = "+str(max_causticity))

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig("test.svg", format="svg")
    plt.savefig('foo.pdf')
    plt.show()


#Quasar_list = Quasar_random_shear_list(1/3)

#Quasar_list = all_Quasars_random()

#Quasar_list = Quasar_random_grid_plot_test()

#Quasar_list = Quasar_random_ca_plot()

#Quasar_list = Quasar_random_ca_plot_four_sided()

#Quasar_list = Quasar_random_shear_plot()
msize = 100
#plt.axis([0, 2, 0, 1]) 
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 500
fig_size[1] = 500

plt.rcParams["figure.figsize"] = fig_size

with open("quasars_shear0_list_new.txt", "rb") as fp:   # Unpickling
  new_Quasar_list = pickle.load(fp)



plot_quasars_shear_try2(new_Quasar_list, 0)


with open("quasars_shear1_3_list_new.txt", "rb") as fp:   # Unpickling
  new_Quasar_list = pickle.load(fp)

plot_quasars_shear_1_3_try(new_Quasar_list, 1/3)

plt.show()
'''
with open("plot_quasars_shear.txt", "rb") as fp:   # Unpickling
  new_Quasar_list = pickle.load(fp)
'''
#Plot_quasars_shear(new_Quasar_list)
'''
with open("test.txt", "rb") as fp:   # Unpickling
  new_Quasar_list = pickle.load(fp)
#Plot_Quasars(new_Quasar_list)
#plt.show()

with open("causticity.txt", "rb") as fp:   # Unpickling
  new_Quasar_list = pickle.load(fp)
#plot_causticity_astroidal_angle(new_Quasar_list)
'''
print('number of linalg err=',number_of_linalgerr)



#Quasar_test()
