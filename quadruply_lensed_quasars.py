import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib
import random
import pickle

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
    alpha = np.arctan2(y_sum,x_sum)
    alpha_new = np.arctan(np.tan(self.configuration_angles[1])*np.tan(self.configuration_angles[2])*np.tan((self.configuration_angles[1] + self.configuration_angles[2])/2))
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
    for x,y in zip(xs,ys):
        j += 1
        label = j
        if j == 1:
          offset = (-15, -5)
        elif j == 3:
          offset = (0,7)
        elif j==2:
          offset = (15,-6)
        elif j==4:
          offset = (0, -25)
        else:
          raise(Exception("Why you hurt me in this way :("))
        plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=offset, # distance from text to points (x,y)
                    ha='center',
                    size = 20)

  def __str__(self):
    print("Configuration Angles", self.configuration_angles)
    print("Causticity", self.causticity)
    print("Astroidal angle", self.astroidal_angle)
    print("Quasar Coordinates", self.quasar_norm_array)
    self.plot()
    return "plotted"

  def __repr__(self):
    return str(self.configuration_angles)
    





def Quasar_test():
    Quasar_list = []
    #Quasar 1
    c1 = (-0.4687227116946,-0.1394554969222)
    c2 = (-0.2645759507525,0.4191047269303)
    c3 = (0.4685837614644,0.2194239159365)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 2
    c1 = (-0.37363,-0.09213)
    c2 = (0.45774,0.47197)
    c3 = (0.27179,-0.29896)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 3
    c1 = (-0.3314531487923,-0.478088981895)
    c2 = (-0.3378506885721,0.3276875521858)
    c3 = (0.3677931950746,0.2397434370077)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 4
    c1 = (0.2307217097772,-0.6029347861578)
    c2 = (-0.0402895474632,-0.2970055432608)
    c3 = (-0.214509849381,0.2959060883864)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 5
    c1 = (-0.1920684745075,-0.2062297547996)
    c2 = (-0.2103383237915,0.1559403721496)
    c3 = (0.694766090173,-0.2686633488896)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 6
    c1 = (-0.1067352528558,-0.2047789838924)
    c2 = (0.8401975696417,0.0689982256173)
    c3 = (0.4967102553992,0.5968091209585)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    i = 0
    for Q in Quasar_list:
        i += 1
        print("configurational angles", Q.configuration_angles)
        print("Quasar norm array", Q.quasar_norm_array)
        print("circle equation", Q.quasar_circle.equation)
        print("Quasar_coordinates", Q.quasar_norm_array)
        print("sum_X=", Q.quasar_norm_array[:,0].sum())
        print("sum_Y=", Q.quasar_norm_array[:,1].sum())
        print("separation Invariant",i, Q.separationInvariant)
        print("configuration(KK) Invariant",i,Q.configurationInvariant)
        print("diff_WW_KK",i,Q.diff_WW_KK)
        print('theta_23', Q.theta_23)
        print('ratio', Q.ratio)
        print('astroidal difference', Q.astroidal_angle - Q.fake_astroidal_angle)

def nongrid_Quasar_plot():   
    Quasar_list = []
    #Quasar 1
    c1 = (-0.4687227116946,-0.1394554969222)
    c2 = (-0.2645759507525,0.4191047269303)
    c3 = (0.4685837614644,0.2194239159365)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 2
    c1 = (-0.37363,-0.09213)
    c2 = (0.45774,0.47197)
    c3 = (0.27179,-0.29896)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 3
    c1 = (-0.3314531487923,-0.478088981895)
    c2 = (-0.3378506885721,0.3276875521858)
    c3 = (0.3677931950746,0.2397434370077)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 4
    c1 = (0.2307217097772,-0.6029347861578)
    c2 = (-0.0402895474632,-0.2970055432608)
    c3 = (-0.214509849381,0.2959060883864)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 5
    c1 = (-0.1920684745075,-0.2062297547996)
    c2 = (-0.2103383237915,0.1559403721496)
    c3 = (0.694766090173,-0.2686633488896)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    #Quasar 6
    c1 = (-0.1067352528558,-0.2047789838924)
    c2 = (0.8401975696417,0.0689982256173)
    c3 = (0.4967102553992,0.5968091209585)
    Q = Quasar(c1, c2, c3)
    Quasar_list.append(Q)
    i = 0 
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    for Q in Quasar_list:
        i += 1
        if i<4:
            xs,ys = Q.quasar_norm_array[:,0] + 3*i, Q.quasar_norm_array[:,1]
            cc = plt.Circle(( 3*i , 0 ), 1 , alpha=0.1) 
        if i>3:
            xs,ys = Q.quasar_norm_array[:,0] + 3*i - 9, Q.quasar_norm_array[:,1] +3
            cc = plt.Circle(( 3*i -9 , 3 ), 1 , alpha=0.1) 
        # plot the points
        plt.scatter(xs,ys)
        plt.gca().set_aspect('equal')
        plt.gca().add_artist( cc ) 
        # zip joins x and y coordinates in pairs
        j=0
        for x,y in zip(xs,ys):
            j += 1
            label = j
            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center')

def Hyperbola_test():
    H = RectangularHyperbola((4-1,1/4-1),(-2,-2),(-1/2,1), (4,0.2-1))
    print(H.center)
    print(H.equation)
    H.plot()

def Circle_test():
    C = Circle((-2,-2),(-1/2,1), (4,0.2-1))
    print(C.center)
    print(C.equation)
    C.plot()    

def Quasar_grid_plot_test():
    del_1 = np.linspace(0.01, np.pi/2, num=7, endpoint=False)
    del_2 = np.linspace(0.01, np.pi, num=7, endpoint = False)
    Quasar_list = []
    for i in del_1:
        for j in del_2:
            a1 = i
            a2 = i+j
            c1 = (2,0)
            c2 = (1+np.cos(a1), np.sin(a1))
            c3 = (1+np.cos(a2), np.sin(a2))
            try:
                Q = Quasar(c1, c2, c3)
                Quasar_tuple = (Q, Q.theta_23, Q.ratio)
                print(Q.theta_23, Q.ratio)
                if Q.ratio < 2:
                    Quasar_list.append(Quasar_tuple)
            except np.linalg.LinAlgError:
                print("linalg err")
            except:
                print("bad err")

    # print(Q.quasar_norm_array)
    i = 0
    #Quasar_list = Quasar_list[:1]
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 30
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    L = len(Quasar_list)
    for Quasar_tuple in Quasar_list:
        i += 1
        Q, theta_23, ratio = Quasar_tuple
        N = int(np.sqrt(L))
        xs,ys = Q.quasar_norm_array[:,0] + 20*theta_23, Q.quasar_norm_array[:,1] + 10*ratio
        cc = plt.Circle(( 20*theta_23 , 10*ratio ), 1 , alpha=0.1) 
        # plot the points
        plt.scatter(xs,ys)
        plt.gca().set_aspect('equal')
        plt.gca().add_artist( cc ) 
        # zip joins x and y coordinates in pairs
        j=0
        for x,y in zip(xs,ys):
            j += 1
            label = j
            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,5), # distance from text to points (x,y)
                        ha='center')

def Quasar_random_grid_plot_test():
    N = 20000
    random.seed(3)
    Quasar_list = []
    theta_23_divisions = 3
    ratio_divisions = 2
    plotting_dictionary = dict.fromkeys((np.ndindex((theta_23_divisions + 1,ratio_divisions + 1))),0 )
    ratio_max = 1
    ratio_min = 0
    theta_23_max = np.pi/2
    theta_23_min = 0
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
            Quasar_tuple = (Q, Q.theta_23, Q.ratio)
            #print(Q.theta_23, Q.ratio)
            theta_23 =  Q.theta_23
            ratio = Q.ratio
            a = round(theta_23_divisions*(theta_23 -  theta_23_min)/(theta_23_max - theta_23_min))
            b = round(ratio_divisions*(ratio-ratio_min)/(ratio_max-ratio_min))
            if plotting_dictionary[(a,b)] == 0:
                Quasar_list.append(Quasar_tuple)
                ind = len(Quasar_list) - 1
                plotting_dictionary[(a,b)] = (Q, ind)
            else:
                ind =  plotting_dictionary[(a,b)][1]
                old_Q_theta_23, old_Q_ratio = Quasar_list[ind][1], Quasar_list[ind][2]
                if (rounding_error(theta_23_divisions*(theta_23 -  theta_23_min)/(theta_23_max - theta_23_min)) + rounding_error(ratio_divisions*(ratio-ratio_min)/(ratio_max-ratio_min))) < (rounding_error(theta_23_divisions*(old_Q_theta_23 -  theta_23_min)/(theta_23_max - theta_23_min)) + rounding_error(ratio_divisions*(old_Q_ratio-ratio_min)/(ratio_max-ratio_min))):
                    Quasar_list[ind] = Quasar_tuple
                    plotting_dictionary[(a,b)] = (Q, ind)
        except np.linalg.LinAlgError:
            print("linalg err")
        except KeyError:
            print("theta_23 or ratio out of bounds")
        #except:
         #   print("bad err")
    with open("test.txt", "wb") as fp:   #Pickling
      pickle.dump(Quasar_list, fp)
    return Quasar_list

def Plot_Quasars(Quasar_list):
    i = 0
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    #L = len(Quasar_list)
    for Quasar_tuple in Quasar_list:
        i += 1
        Q, theta_23, ratio = Quasar_tuple
        radius_ratio = 1/6
        xs,ys = Q.quasar_norm_array[:,0]*radius_ratio + theta_23, Q.quasar_norm_array[:,1]*radius_ratio + ratio
        cc = plt.Circle((theta_23 , ratio ), radius_ratio , alpha=0.1)
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
              offset = (8, -3)
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
                        size = 14)
    ax1 = plt.gca()
    ax1.set_ylabel('Ratio')
    ax1.set_xlabel('$θ_{23}$ (radians)')
    #plt.show()
theta_23_err =0
def all_Quasars_random():
    global theta_23_err
    N = 200000
    random.seed(3)
    Quasar_list = []
    #theta_23_divisions = 3
    #ratio_divisions = 2
    #plotting_dictionary = dict.fromkeys((np.ndindex((theta_23_divisions + 1,ratio_divisions + 1))),0 )
    ratio_max = 1
    ratio_min = 0
    theta_23_max = np.pi/2
    theta_23_min = 0
    #theta_23_array, ratio_array, causticity_array, astroidal_angle_array = [], [], [], []
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
            if Q.theta_23<theta_23_max and Q.theta_23> theta_23_min and Q.ratio<ratio_max and Q.ratio>ratio_min:
              pass
            else:
              raise(KeyError)
            Quasar_tuple = (Q, Q.theta_23, Q.ratio, Q.causticity, Q.astroidal_angle)
            '''
            theta_23_array.append(Q.theta_23)
            ratio_array.append(Q.ratio)
            causticity_array.append(Q.causticity)
            astroidal_angle_array.append(Q.astroidal_angle)
            #print(Q.theta_23, Q.ratio)
            '''
            Quasar_list.append(Quasar_tuple)
        except np.linalg.LinAlgError:
            print("linalg err")
        except KeyError:
            print("theta_23 or ratio out of bounds")
            theta_23_err += 1
        #except:
         #   print("bad err")
    Quasar_list = np.array(Quasar_list)
    with open("causticity_new.txt", "wb") as fp:   #Pickling
      pickle.dump(Quasar_list, fp)
    return Quasar_list

def plot_astroid(r, gamma = 0):
  theta = np.linspace(0, np.pi/2, 2000)

  radius = r

  a = (1+gamma)*radius*(np.cos(theta))**3
  b = (1-gamma)*radius*(np.sin(theta))**3
  plt.plot(a,b)
  '''
  x = np.linspace(0.1, r, 4000)
  y = (r**(2/3) - x**(2/3))**3/2  
  plt.plot(x,y)
  plt.plot(y,x)
  '''

def contour_plot_causticity_astroidal_angle(Quasar_list):
  plt.rcParams.update({'font.size': 22})
  theta_23_array = Quasar_list[:,1]
  ratio_array = Quasar_list[:, 2]
  causticity_array = Quasar_list[:, 3]
  astroidal_angle_array = Quasar_list[:,4]
  x = list(theta_23_array)
  y = list(ratio_array)
  astroidal = True
  if astroidal:
    z=[]
    for i in astroidal_angle_array:
      z.append(np.arctan(np.cbrt(np.tan(i))))
  else:
    z = list(causticity_array)
  ax2 = plt.gca()
  fig = plt.gcf()
  ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
  cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
  fig.colorbar(cntr2, ax=ax2)
  ax2.set_ylabel('Ratio')
  ax2.set_xlabel('$θ_{23}$ (radians)')
  if astroidal:
    ax2.set_title("Contour plot of astroidal angle")
  else:
    ax2.set_title("Contour plot of causticity")
  plt.show()

def Quasar_random_ca_plot():
  N = 50000
  random.seed(2)
  Quasar_list = []
  max_causticity = 0.95
  alpha_divisions = 5
  plot_astroid(max_causticity)
  plot_astroid(max_causticity/2)
  plotting_dictionary = {(max_causticity, 0):0, (max_causticity, np.pi/2 - 0.24):0, (max_causticity, np.pi/4):0, (max_causticity, 0.24):0, (max_causticity, np.pi/2):0, (3*max_causticity/4, 0):0, (3*max_causticity/4, np.pi/2):0,  (max_causticity/2,0):0, (max_causticity/2, np.pi/4):0, (max_causticity/2, np.pi/2):0, (max_causticity/4, 0):0, (max_causticity/4, np.pi/2):0, (0, 0):0}
  error_ratio = (np.pi/2)**2
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
          Quasar_tuple = (Q, Q.causticity, Q.astroidal_angle)
          #print(Q.theta_23, Q.ratio)
          zeta =  Q.causticity
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
      except KeyError:
          print("theta_23 or ratio out of bounds")
          print(t)
      except:
           print("bad err")
  with open("new_plot_ca.txt", "wb") as fp:   #Pickling
    pickle.dump(Quasar_list, fp)
  return Quasar_list

def Quasar_random_ca_plot_four_sided():
  N = 100000
  Quasar_list = []
  max_causticity = 1
  alpha_divisions = 5
  #plot_astroid(max_causticity)
  #plot_astroid(max_causticity/2)
  plotting_dictionary = {(max_causticity, 0):0, (max_causticity, np.pi/2 - 0.54):0, (max_causticity, np.pi/4):0, (max_causticity, 0.54):0, (max_causticity, np.pi/2):0, (max_causticity, np.pi - 0.54):0, (max_causticity, 3*np.pi/4):0, (max_causticity, np.pi/2 + 0.54):0, (max_causticity, np.pi):0, \
                                                 (max_causticity, -np.pi/2 + 0.54):0, (max_causticity, -np.pi/4):0, (max_causticity, -0.54):0, (max_causticity, -np.pi/2):0, (max_causticity, -np.pi + 0.54):0, (max_causticity, -3*np.pi/4):0, (max_causticity, -np.pi/2-0.54):0, \
                          (3*max_causticity/4, 0):0, (3*max_causticity/4, np.pi/2):0, (3*max_causticity/4, -np.pi/2):0, (3*max_causticity/4, np.pi):0, \
                          (max_causticity/2,0):0, (max_causticity/2, np.pi/4):0, (max_causticity/2, 3*np.pi/4):0,(max_causticity/2,np.pi):0,(max_causticity/2,np.pi/2):0, (max_causticity/2, -np.pi/4):0, (max_causticity/2, -np.pi/2):0, (max_causticity/2, -3*np.pi/4):0, \
                          (max_causticity/4, 0):0, (max_causticity/4, np.pi/2):0,(max_causticity/4, np.pi):0, (max_causticity/4, -np.pi/2):0,\
                          (0, 0):0}
  error_ratio = (np.pi/2)**2
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
          Quasar_tuple = (Q, Q.causticity, Q.astroidal_angle)
          #print(Q.theta_23, Q.ratio, Q.astroidal_angle, Q.configuration_angles[1])
          zeta =  Q.causticity
          alpha = Q.astroidal_angle
          for t in plotting_dictionary.keys():
            err = (zeta-t[0])**2 + error_ratio*(alpha-t[1])**2
            if err != err:
              print(alpha)
              break
            if t == (0,0):
              err = (zeta-t[0])**2
            if plotting_dictionary[t] == 0:
                Quasar_list.append(Quasar_tuple)
                ind = len(Quasar_list) - 1
                plotting_dictionary[t] = (Q, ind, err)
                #print(err)
                break
            else:
                ind =  plotting_dictionary[t][1]
                old_Q_err = plotting_dictionary[t][2]
                if old_Q_err > err:
                    Quasar_list[ind] = Quasar_tuple
                    plotting_dictionary[t] = (Q, ind, err)
                    #print(err)
                    #print(old_Q_err)
                    break
      except np.linalg.LinAlgError:
          print("linalg err")
      except KeyError:
          print("theta_23 or ratio out of bounds")
      except:
           print("bad err")
  with open("new_plot_ca.txt", "wb") as fp:   #Pickling
    pickle.dump(Quasar_list, fp)
  print(plotting_dictionary[(1, -np.pi/4)])
  return Quasar_list

def Plot_Quasars_ca(Quasar_list):
    i = 0
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 15
    plt.rcParams["figure.figsize"] = fig_size
    #L = len(Quasar_list)
    max_causticity = 1
    plot_astroid(max_causticity)
    plot_astroid(max_causticity/2)
    def p23(x):
      return (np.cbrt(x))**2
    for Quasar_tuple in Quasar_list:
        i += 1
        Q, causticity, astroidal_angle = Quasar_tuple
        radius_ratio = 1/20
        xs,ys = Q.quasar_norm_array[:,0]*radius_ratio + causticity*np.cos(astroidal_angle)/(p23(np.sin(astroidal_angle))+p23(np.cos(astroidal_angle)))**1.5 , Q.quasar_norm_array[:,1]*radius_ratio + causticity*np.sin(astroidal_angle)/(p23(np.sin(astroidal_angle))+p23(np.cos(astroidal_angle)))**1.5 
        cc = plt.Circle((causticity*np.cos(astroidal_angle)/(p23(np.sin(astroidal_angle))+p23(np.cos(astroidal_angle)))**1.5 ,causticity*np.sin(astroidal_angle)/(p23(np.sin(astroidal_angle))+p23(np.cos(astroidal_angle)))**1.5 ), radius_ratio , alpha=0.1)
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
    ax1.set_title("Max Causticity = "+str(max_causticity))
    plt.savefig('foo.pdf')
    plt.show()

def Quasar_random_shear_plot():
  '''
  plot of quasars over astroid with non-zero shear
  '''
  N = 5000
  random.seed(2)
  shear = 1/3
  Quasar_list = []
  max_causticity = 0.95
  alpha_divisions = 5
  #plot_astroid(max_causticity)
  #plot_astroid(max_causticity/2)
  plotting_dictionary = {(max_causticity, 0):0, (max_causticity, np.pi/2 - 0.34):0, (max_causticity, np.pi/4):0, (max_causticity, 0.34):0, (3/4*max_causticity, 0.54):0, (3/4*max_causticity, np.pi/2-0.54):0,  (max_causticity, np.pi/2):0, (3*max_causticity/4, 0):0, (3*max_causticity/4, np.pi/2):0,  (max_causticity/2,0):0, (max_causticity/2, np.pi/4):0, (max_causticity/2, np.pi/2):0, (max_causticity/4, 0):0, (max_causticity/4, np.pi/2):0, (0, 0):0}
  error_ratio = (2/np.pi)**2
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
      except KeyError:
          print("theta_23 or ratio out of bounds")
          #print(t)
      #except:
       #    print("bad err")
        #   raise(Exception("solve it"))
  with open("plot_quasars_shear.txt", "wb") as fp:   #Pickling
    pickle.dump(Quasar_list, fp)
  print(Quasar_list)
  return Quasar_list


def Plot_quasars_shear(Quasar_list):
    i = 0
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 15
    plt.rcParams["figure.figsize"] = fig_size
    #L = len(Quasar_list)
    max_causticity = 0.95
    plot_astroid(max_causticity, 1/3)
    plot_astroid(max_causticity/2, 1/3)
    for Quasar_tuple in Quasar_list:
        i += 1
        Q, new_causticity, astroidal_angle = Quasar_tuple
        radius_ratio = 1/20
        xs,ys = Q.quasar_norm_array[:,0]*radius_ratio + new_causticity*np.cos(astroidal_angle), Q.quasar_norm_array[:,1]*radius_ratio + new_causticity*(np.sin(astroidal_angle))
        cc = plt.Circle((new_causticity*(np.cos(astroidal_angle)) ,new_causticity*(np.sin(astroidal_angle)) ), radius_ratio , alpha=0.1)
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
              offset = (8, -3)
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
                        size = 14)
    ax1 = plt.gca()
    ax1.set_title("Max Causticity = "+str(max_causticity))
    plt.savefig('foo.pdf')
    plt.show()

def plot_quasars_shear_try2(Quasar_list):
    i = 0
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 15
    plt.rcParams["figure.figsize"] = fig_size
    #L = len(Quasar_list)
    max_causticity = 0.95
    plot_astroid(max_causticity)
    plot_astroid(max_causticity/2)
    def p23(x):
      return (np.cbrt(x))**2
    for Quasar_tuple in Quasar_list:
        i += 1
        Q, causticity, astroidal_angle = Quasar_tuple
        shear = -1/3
        radius_ratio = 1/20
        xs,ys = Q.quasar_norm_array[:,0]*radius_ratio/(1+shear) + causticity*np.cos(astroidal_angle)/(p23(np.sin(astroidal_angle))+p23(np.cos(astroidal_angle)))**1.5 , Q.quasar_norm_array[:,1]*radius_ratio/(1-shear) + causticity*np.sin(astroidal_angle)/(p23(np.sin(astroidal_angle))+p23(np.cos(astroidal_angle)))**1.5 
        cc = matplotlib.patches.Ellipse((causticity*np.cos(astroidal_angle)/(p23(np.sin(astroidal_angle))+p23(np.cos(astroidal_angle)))**1.5 ,causticity*np.sin(astroidal_angle)/(p23(np.sin(astroidal_angle))+p23(np.cos(astroidal_angle)))**1.5 ), 2*radius_ratio/(1+shear) , 2*radius_ratio/(1-shear), alpha=0.1)
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
    ax1.set_title("Max Causticity = "+str(max_causticity))
    plt.savefig('foo.pdf')
    plt.show()


def Quasar_random_shear_plot_try2():
  N = 5000
  random.seed(2)
  Quasar_list = []
  shear = 1/3
  max_causticity = 0.95
  alpha_divisions = 5
  plot_astroid(max_causticity)
  plot_astroid(max_causticity/2)
  plotting_dictionary = {(max_causticity, 0):0, (max_causticity, np.pi/2 - 0.24):0, (max_causticity, np.pi/4):0, (max_causticity, 0.24):0, (max_causticity, np.pi/2):0, (3*max_causticity/4, 0):0, (3*max_causticity/4, np.pi/2):0,  (max_causticity/2,0):0, (max_causticity/2, np.pi/4):0, (max_causticity/2, np.pi/2):0, (max_causticity/4, 0):0, (max_causticity/4, np.pi/2):0, (0, 0):0}
  error_ratio = (np.pi/2)**2
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
          Quasar_tuple = (Q, Q.causticity, Q.astroidal_angle)
          #print(Q.theta_23, Q.ratio)
          zeta =  Q.causticity
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
      except KeyError:
          print("theta_23 or ratio out of bounds")
          print(t)
      except:
           print("bad err")
  with open("new_plot_ca.txt", "wb") as fp:   #Pickling
    pickle.dump(Quasar_list, fp)
  return Quasar_list


#Quasar_list = all_Quasars_random()

#Quasar_list = Quasar_random_grid_plot_test()

#Quasar_list = Quasar_random_ca_plot()

#Quasar_list = Quasar_random_ca_plot_four_sided()

#Quasar_list = Quasar_random_shear_plot()

with open("new_plot_ca.txt", "rb") as fp:   # Unpickling
  new_Quasar_list = pickle.load(fp)
new_Quasar_list[2][0].plot()
plt.axis('off')
ax = plt.gca()
ax.set_xlim(-1.05, 1)
ax.set_ylim(-1.05,1)
plt.savefig('Labeling_conv.pdf')
plt.show()
#plot_quasars_shear_try2(new_Quasar_list)

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
'''

'''
with open("causticity.txt", "rb") as fp:   # Unpickling
  new_Quasar_list = pickle.load(fp)
#plot_causticity_astroidal_angle(new_Quasar_list)
'''
'''
#all_Quasars_random()
with open("causticity_new.txt", "rb") as fp:   # Unpickling
  new_Quasar_list = pickle.load(fp)
contour_plot_causticity_astroidal_angle(new_Quasar_list)
#plt.savefig("astroidal angle contour plot.pdf")
print("Number of theta_23 errors is", theta_23_err)
'''
#Quasar_test()
