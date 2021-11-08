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
            


def Plot_Eccentric_anomaly_fig(gamma = -1/4, theta=1):
  fig_size= [6,6]
  plt.rcParams["figure.figsize"] = fig_size
  E = AlignedEllipse((1/(1+gamma),0), (0, 1/(1-gamma)), (0, -1/(1-gamma)), (-1/(1+gamma), 0))
  E.plot()
  plt.scatter(np.cos(theta)/(1+gamma), np.sin(theta)/(1-gamma), s=100, zorder = 5)
  C = Circle((1,0), (0,1), (-1, 0))
  cc = plt.Circle((0,0), 1 , alpha=1, color='lightblue')
  plt.gca().add_patch(cc)
  cnew = plt.Circle((0,0), 1/(1+gamma) , alpha=0.2, color='#d62728', zorder = -10)
  plt.gca().add_patch(cnew)
  #C.plot()
  plt.scatter(np.cos(theta), np.sin(theta), s=100, zorder = 5)
  plt.scatter(np.cos(theta)/(1+gamma), np.sin(theta)/(1+gamma), s=100, zorder = 5)
  ax = plt.gca()
  txtsize = 16
  ax.annotate(r"$\frac{1}{1+\gamma} $",
            xy=(1/(1+gamma), 0), xycoords='data',
            xytext=(0.6, -0.05), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = txtsize
            )
  ax.annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(0.6, 0), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = txtsize
            )
  ax.annotate(r"$\frac{1}{1-\gamma} $",
            xy=(0, 1/(1-gamma)), xycoords='data',
            xytext=(-0.15, 0.3), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = txtsize
            )
  ax.annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(0, 0.25), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = txtsize
            )
  ax.annotate(r'$\left( \frac{\cos \eta}{1+\gamma} , \frac{\sin \eta}{1-\gamma} \right)$', xy = (np.cos(theta)/(1+gamma)+0.1, np.sin(theta)/(1-gamma)), size = txtsize)
  ax.annotate(r'$(\cos \theta, \sin \theta)= $'+'\n \t'+ r'$( \cos \eta , \sin \eta )$', xy = (np.cos(theta)-1.2, np.sin(theta)+0.1), size = txtsize-4)
  # draw a line passing through the origin at an angle of theta
  ax.annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(np.cos(theta)/(1+gamma), np.sin(theta)/(1+gamma)), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            connectionstyle="arc3"), size = txtsize
            )
  # draw a vertical line passing through point on circle at angle theta
  ax.annotate("",
            xytext=(np.cos(theta)/(1+gamma), np.sin(theta)/(1+gamma)), xycoords='data',
            xy=(np.cos(theta)/(1+gamma), np.sin(theta)/(1-gamma)), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), size = txtsize
            )
  # curved arc or radius 0.5 near origin which spans theta
  ac = matplotlib.patches.Arc((0,0), 0.5, 0.5, theta1 = 0, theta2 = theta*180/np.pi, linewidth = 2, zorder = 20)
  ax.add_patch(ac)
  # label the arc
  ax.annotate(r'$\theta$', xy = (0.1, 0), xytext = (0.25, 0.1), size = txtsize)
  ax.set_xlim(-1.8, 2)
  ax.set_ylim(-1.55, 1.55)
  ax.set_aspect('equal')
  plt.axis('off')
  plt.savefig('eccentric_anomaly.pdf', bbox_inches = 'tight',pad_inches = 0)
  plt.show()

Plot_Eccentric_anomaly_fig(gamma = -1/3, theta=0.9)

