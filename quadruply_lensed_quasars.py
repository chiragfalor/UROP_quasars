from matplotlib.markers import MarkerStyle
from copy import deepcopy
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib
import random
import pickle
from mpl_toolkits.axes_grid.inset_locator import inset_axes


#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})
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
  def coordinate_difference(self, c1, c2):
    return [c1[0]-c2[0], c1[1]-c2[1]]



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
        # plt.plot(self.c1[0], self.c1[1], 'r+')
        # plt.plot(self.c2[0], self.c2[1], 'r+')
        # plt.plot(self.c3[0], self.c3[1], 'r+')
        # plt.show()

class AlignedEllipse(Conic):
    def __init__(self, c1, c2, c3, c4, iscentered=False):
        '''
        initialize a rectangular hyperbola in cartesian coordinates where c1, c2, c3, c4 are tuples of coordinates in the form of (x_i, y_i)
        '''
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        equation_matrix = np.array([self.equation_maker_ellipse(c1), self.equation_maker_ellipse(c2), self.equation_maker_ellipse(c3), self.equation_maker_ellipse(c4)])
        equation_solutions = np.array([1,1,1,1])
        '''
        Assumes that equation of ellipse is of the form Ax^2 + By^2 + Cx + Dy = 1 and solves the 4 equations got by substituting the 4 coordinates
        '''
        self.coefficients = np.linalg.solve(equation_matrix, equation_solutions)
        self.center = self.findCenter_ellipse()
        self.equation = str(self.coefficients[0]) + "x^2 + " + str(self.coefficients[1]) + "y^2 + " + str(self.coefficients[2]) + "x + " + str(self.coefficients[3]) + "y = 1"
        self.centered_ellipse_coordinates = [self.coordinate_difference(c1, self.center),self.coordinate_difference(c2, self.center),self.coordinate_difference(c3, self.center),self.coordinate_difference(c4, self.center)]
        if iscentered:
          if (self.center[0])**2 + (self.center[1])**2>0.000001:
            raise(Exception("Ellipse is not centered"))
          if self.coefficients[0]<0 or self.coefficients[1]<0:
            raise(Exception("This is not ellipse but hyperbola or imaginary figure"))
          self.a = 1/np.sqrt(self.coefficients[0])
          self.b = 1/np.sqrt(self.coefficients[1])
          self.parametric_angles = [self.coordinate_to_parametric_angle(c1), self.coordinate_to_parametric_angle(c2), self.coordinate_to_parametric_angle(c3), self.coordinate_to_parametric_angle(c4)]
        else:
          centered_ellipse = AlignedEllipse(self.centered_ellipse_coordinates[0], self.centered_ellipse_coordinates[1],self.centered_ellipse_coordinates[2],self.centered_ellipse_coordinates[3], True)
          self.a = centered_ellipse.a
          self.b = centered_ellipse.b
          self.parametric_angles = centered_ellipse.parametric_angles
        self.normalized_coordinates = list(map(self.coordinate_normalizer, self.centered_ellipse_coordinates))
          
    def coordinate_normalizer(self,c1):
      nc = (c1[0]/self.a, c1[1]/self.b)
      return nc

    def coordinate_to_parametric_angle(self, c1):
      nc = (c1[0]/self.a, c1[1]/self.b)
      return np.arctan2(nc[1], nc[0])
      

    def findCenter_ellipse(self):
        '''
        returns the center of the current hyperbola by partial differentiation of the equation wrt x and y separately and solving the linear equation
        '''
        A, B, C, D = self.coefficients
        x_c = -C/(2*A)
        y_c = -D/(2*B)
        return (x_c, y_c)

    def equation_maker_ellipse(self, c1):
        '''
        input: tuple (Coordinate)
        output: from tuple(x,y), it returns (x**2-y**2, x*y, x, y)
        '''
        x = c1[0]
        y = c1[1]
        return (x**2, y**2, x, y)

    def plot(self):
        x = np.linspace(-2, 2, 400)
        y = np.linspace(-2, 2, 400)
        x, y = np.meshgrid(x, y)
        a, b, c, d, e, f = self.coefficients[0], 0, self.coefficients[1], self.coefficients[2], self.coefficients[3], -1
        def axes():
            plt.axhline(0, alpha=.1)
            plt.axvline(0, alpha=.1)
        axes()
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='k', linewidths=2)
        #plt.scatter(0, 0, c='c', marker = 'o', zorder = 10, s=100)
        # plt.plot(self.c1[0], self.c1[1], 'r+')
        # plt.plot(self.c2[0], self.c2[1], 'r+')
        # plt.plot(self.c3[0], self.c3[1], 'r+')
        # plt.plot(self.c4[0], self.c4[1], 'r+')
        #plt.show()

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

    def plot(self, points=False):
        x = np.linspace(-2, 2, 4000)
        y = np.linspace(-2, 2, 4000)
        x, y = np.meshgrid(x, y)
        a, b, c, d, e, f = self.coefficients[0], self.coefficients[1], -self.coefficients[0], self.coefficients[2], self.coefficients[3], -1
        def axes():
            plt.axhline(0, alpha=.1)
            plt.axvline(0, alpha=.1)
        #axes()
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='k', linewidths = 7)
        msize = 300
        if points:
            plt.scatter(self.c1[0], self.c1[1], c='#d62728', marker = 'o', zorder = 10, s=msize)
            plt.scatter(self.c2[0], self.c2[1], c='#d62728', marker = 'o', zorder = 10, s=msize)
            plt.scatter(self.c3[0], self.c3[1], c='#d62728', marker = 'o', zorder = 10, s=msize)
            plt.scatter(self.c4[0], self.c4[1], c='#d62728', marker = 'o', zorder = 10, s=msize)
    def plot_branch1(self, color = 'k'):
        x = np.linspace(self.center[0], 2, 4000)
        y = np.linspace(self.center[1], 2, 4000)
        x, y = np.meshgrid(x, y)
        a, b, c, d, e, f = self.coefficients[0], self.coefficients[1], -self.coefficients[0], self.coefficients[2], self.coefficients[3], -1
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors=color, linewidths = 7)
    def plot_branch2(self, color = 'k'):
        x = np.linspace(-2, self.center[0], 4000)
        y = np.linspace(-2, self.center[1], 4000)
        x, y = np.meshgrid(x, y)
        a, b, c, d, e, f = self.coefficients[0], self.coefficients[1], -self.coefficients[0], self.coefficients[2], self.coefficients[3], -1
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors=color, linewidths = 7)
            



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

    self.quasar_hyperbola_center = self.quasar_hyperbola.center

    self.theta_23 = self.angle_difference(self.configuration_angles[2], self.configuration_angles[1])

    self.ratio = self.configuration_angles[1]/(self.angle_difference(np.pi/2, self.configuration_angles[2]))

    #self.configurationInvariant = self.Calculate_configuration_invariant(self.configuration_angles)

    #self.separationInvariant = self.separation_invariant(self.quasar_norm_array)

    #self.diff_WW_KK = self.WW_invariant_calculator(self.configuration_angles)

    self.causticity = self.calculate_causticity()

    self.new_causticity = self.calculate_new_causticity()

    self.astroidal_angle = self.calculate_astroidal_angle()

    self.property_checker()

    t2 = self.configuration_angles[1]
    t3 = self.configuration_angles[2]
    self.fake_astroidal_angle = np.arctan((np.tan(t2)*np.tan(t3)*np.tan((t2+t3)/2)))



  
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
    self.quasar_hyperbola = RectangularHyperbola(self.c1, self.c2, self.c3, self.c4)
    self.quasar_hyperbola_center = self.quasar_hyperbola.center

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


  def Calculate_configuration_invariant(self, configuration_angles):
    phi = configuration_angles
    I_c = np.cos((phi[0] + phi[1] - phi[2] - phi[3])/2) + np.cos((phi[0] + phi[2] - phi[1] - phi[3])/2) + np.cos((phi[0] + phi[3] - phi[1] - phi[2])/2)
    return I_c

  def separation_invariant(self, quasar_norm_array):
    c1, c2, c3, c4 = quasar_norm_array
    d_12 = self.distance(c1, c2)
    d_13 = self.distance(c1, c3)
    d_14 = self.distance(c1, c4)
    d_23 = self.distance(c2, c3)
    d_24 = self.distance(c2, c4)
    d_34 = self.distance(c3, c4)
    I_s = (d_14**2 + d_23**2 - d_13**2 - d_24**2)/(d_13*d_24 - d_14*d_23) + (d_14**2 + d_23**2 - d_12**2 - d_34**2)/(d_12*d_34 + d_14*d_23) + (d_12**2 + d_34**2 - d_13**2 - d_24**2)/(d_12*d_34 - d_13*d_24)  
    return I_s

  def WW_invariant_calculator(self, configuration_angles):
    theta = configuration_angles
    theta_23 = self.angle_difference(theta[1], theta[2])
    theta_12 = self.angle_difference(theta[0], theta[1])
    theta_34 = self.angle_difference(theta[2], theta[3])
    #print(theta_23, theta_12, theta_34)
    WW_theta_23 = -5.792 + 1.783*theta_12 + 0.1648*theta_12**2 - 0.04591*theta_12**3 - 0.0001486*theta_12**4 + 1.784*theta_34 - 0.7275*theta_34*theta_12 + 0.0549*theta_34*theta_12**2 + 0.01487*theta_34*theta_12**3 + 0.1643*theta_34**2 + 0.05493*theta_34**2*theta_12 - 0.03429*theta_34**2*theta_12**2 - 0.04579*theta_34**3 + 0.01487*theta_34**3*theta_12 - 0.0001593*theta_34**4
    inv = theta_23  -  WW_theta_23
    return inv
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
    new_causticity = 2*np.sqrt((np.cos(t[1])*np.cos(t[2])*np.cos((t[1]+t[2])/2))**2+(np.sin(t[1])*np.sin(t[2])*np.sin((t[1]+t[2])/2))**2)/np.cos(theta_23/2)
    return new_causticity





  def calculate_astroidal_angle(self):
    x_sum = np.sum(self.quasar_norm_array[:,0])
    y_sum = np.sum(self.quasar_norm_array[:,1])
    alpha = np.arctan2(np.cbrt(y_sum),np.cbrt(x_sum))
    alpha_new = np.arctan(np.cbrt(np.tan(self.configuration_angles[1])*np.tan(self.configuration_angles[2])*np.tan((self.configuration_angles[1] + self.configuration_angles[2])/2)))
    if self.configuration_angles[1]>np.pi/2:
      alpha_new += np.pi
    elif self.configuration_angles[1]<-np.pi/2:
      alpha_new -= np.pi
    return alpha_new

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
    '''
    property 3
    center of hyperbola = centroid of plotted configurations
    '''
    x_sum = np.sum(self.quasar_norm_array[:,0])
    y_sum = np.sum(self.quasar_norm_array[:,1])
    if abs((x_sum/2 - self.quasar_hyperbola_center[0])**2 + (y_sum/2 - self.quasar_hyperbola_center[1])**2)>epsilon:
      self.__str__()
      raise Exception("property 3 lost")
    


  def plot(self):
    radius_ratio = 1
    xs,ys = self.quasar_norm_array[:,0]*radius_ratio, self.quasar_norm_array[:,1]*radius_ratio
    cc = plt.Circle((0,0), radius_ratio , alpha=0.1)
    # plot the points
    plt.scatter(xs,ys,c='#d62728', marker = 'o', s = 100)
    plt.gca().set_aspect('equal')
    plt.gca().add_artist( cc )
    # zip joins x and y coordinates in pairs
    j=0

  def plot2(self, marker_shape):
      radius_ratio = 1
      xs,ys = self.quasar_norm_array[:,0]*radius_ratio, self.quasar_norm_array[:,1]*radius_ratio
      e = AlignedEllipse((xs[0], ys[0]),(xs[1], ys[1]),(xs[2], ys[2]),(xs[3], ys[3]))
      e.plot()
      # plot the points
      if marker_shape=='o':
        plt.scatter(xs,ys,c='r', marker = marker_shape, zorder = 3, s=1000)
      else:
        plt.scatter(xs,ys,c='#d62728', marker = marker_shape, zorder = 2, s=160)
      plt.gca().set_aspect('equal')
      # zip joins x and y coordinates in pairs
      j=0
      #plt.show()

  def __str__(self):
    print("Configuration Angles", self.configuration_angles)
    print("Causticity", self.causticity)
    print("Astroidal angle", self.astroidal_angle)
    print("Quasar Coordinates", self.quasar_norm_array)
    self.plot()
    return "plotted"

  def __repr__(self):
    return str(self.configuration_angles)
    


def WynneSchechterConstructionPlot():
  with open("causticity.txt", "rb") as fp:   # Unpickling
    new_Quasar_list = pickle.load(fp)
  Q = new_Quasar_list[0][0]
  angles = Q.configuration_angles
  a = 1.5
  b= 1
  c1 = (a*np.cos(angles[0]), b*np.sin(angles[0]))
  c2 = (a*np.cos(angles[1]), b*np.sin(angles[1]))
  c3 = (a*np.cos(angles[2]), b*np.sin(angles[2]))
  c4 = (a*np.cos(angles[3]), b*np.sin(angles[3])) 
  ell = AlignedEllipse(c1, c2, c3, c4)
  hyp = RectangularHyperbola(c1, c2, c3, c4)
  plt.axis('off')
  plt.gca().set_aspect('equal')
  # hyp.plot()
  # plt.savefig('hyperbola.pdf', bbox_inches = 'tight',pad_inches = 0)
  # plt.show()
  ell.plot()
  plt.axis('off')
  plt.gca().set_aspect('equal')
  #Q.quasar_hyperbola.plot()
  plt.savefig('ellipse.pdf', bbox_inches = 'tight',pad_inches = 0)
  plt.show()
  # fig_size= [8,8]
  # plt.rcParams["figure.figsize"] = fig_size
  # ell.plot()
  # hyp.plot(True)
  # plt.axis('off')
  # plt.gca().set_aspect('equal')
  # plt.savefig('combined_hyp_ellipse.pdf', bbox_inches = 'tight',pad_inches = 0)
  # plt.show()


def WynneSchechter_ACP_construction():
  with open("quasar_ca_2.txt", "rb") as fp:   # Unpickling
    result = pickle.load(fp)

  with open("quasar_ca_3.txt", "rb") as fp:   # Unpickling
    new_result = pickle.load(fp)
  #plot_causticity_astroidal_angle(new_Quasar_list)
  
  fig_size= [8,8]
  plt.rcParams["figure.figsize"] = fig_size
  print(result)
  ax = plt.gca()
  ax.set_xlim(-1.3, 1.3)
  ax.set_ylim(-1.3, 1.3)
  plt.scatter(0,0, s=800, marker='*', c= 'orange', zorder=20)
  plt.axis('off')


  def plot_special_quasar(i):
    c1, c2, c3, c4 = new_result[i][0].quasar_norm_array
    Q = Quasar(c1, c2, c3)
    Q.plot2('o')
    Q.quasar_hyperbola.plot_branch1(color='b')
    Q.quasar_hyperbola.plot_branch2(color='g')

  # result[0][0].plot2('o')
  # result[0][0].quasar_hyperbola.plot_branch1(color='b')
  # result[0][0].quasar_hyperbola.plot_branch2(color='g')
  # plt.savefig('wynne_schechter_construction_mouse_ears.pdf', bbox_inches = 'tight',pad_inches = 0)

  # result[1][0].plot2('o')
  # result[1][0].quasar_hyperbola.plot_branch1(color='b')
  # result[1][0].quasar_hyperbola.plot_branch2(color='g')
  # plt.savefig('wynne_schechter_construction_square.pdf', bbox_inches = 'tight',pad_inches = 0)

  # result[2][0].plot2('o')
  # result[2][0].quasar_hyperbola.plot_branch1(color='b')
  # result[2][0].quasar_hyperbola.plot_branch2(color='g')
  # plot_special_quasar(4)
  # plt.savefig('wynne_schechter_construction_kite.pdf', bbox_inches = 'tight',pad_inches = 0)

  # result[3][0].plot2('o')
  # result[3][0].quasar_hyperbola.plot()

  plt.show()

#WynneSchechter_ACP_construction()

def Eccentric_anomaly(gamma = 1/2, theta=np.pi/3):
  fig_size= [6,6]
  plt.rcParams["figure.figsize"] = fig_size
  E = AlignedEllipse((1+gamma,0), (0, 1-gamma), (0, gamma-1), (-1-gamma, 0))
  E.plot()
  plt.scatter((1+gamma)*np.cos(theta), (1-gamma)*np.sin(theta), s=100, zorder = 5)
  C = Circle((1,0), (0,1), (-1, 0))
  cc = plt.Circle((0,0), 1 , alpha=0.1)
  #C.plot()
  plt.scatter(np.cos(theta), np.sin(theta), s=100, zorder = 5)
  ax = plt.gca()
  ax.add_artist(cc)
  ax.annotate("$1+\gamma $",
            xy=(1+gamma, 0), xycoords='data',
            xytext=(0.5, -0.05), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = 10
            )
  ax.annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(0.5, 0), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = 10
            )
  ax.annotate("$1-\gamma $",
            xy=(0, 1-gamma), xycoords='data',
            xytext=(-0.15, 0.2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = 10
            )
  ax.annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(0, 0.2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = 10
            )
  ax.annotate(r'$((1+\gamma) \cos \theta , (1-\gamma) \sin \theta )$', xy = ((1+gamma)*np.cos(theta)-0.1, (1-gamma)*np.sin(theta)+0.1), size = 10)
  ax.annotate(r'$( \cos \theta , \sin \theta )$', xy = (np.cos(theta), np.sin(theta)+0.1), size = 10)
  

  ax.set_xlim(-2, 2)
  ax.set_ylim(-1.2, 1.2)
  ax.set_aspect('equal')
  plt.axis('off')
  plt.savefig('eccentric_anomaly.pdf', bbox_inches = 'tight',pad_inches = 0)
  plt.show()

Eccentric_anomaly()

