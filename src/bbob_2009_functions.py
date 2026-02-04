import numpy as np
import math

# BBOB 2009 TEST SUITE FUNCTIONS

def c1(x):
    if x > 0:
        result = 10
    else:
        result = 5.5
    return result
    
def c2(x):
    if x > 0:
        result = 7.9
    else:
        result = 3.1
    return result
    
def x_hat(x):
    if x != 0:
        result = np.log10(np.abs(x))
    else:
        result = 0
    return result
    
def ill_conditioning(x):
    result = np.sign(x)*np.exp(x_hat(x)+0.049*(math.sin(c1(x)*x_hat(x))+math.sin(c2(x)*x_hat(x))))
    return result

# # sphere function function 1 (SOLVED, D=20) DONE

class function_1:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)
    
    def rosenbrock_fitness(self, coor_list, radius=0):
        result = 0
        for i in range(self.rosenbrock_degree):
            result += (coor_list[i]-self.x_opt_list[i])**2
        result -= radius

        return result 

# ellipsoidal function function 2 (SOLVED, D=20)

class function_2:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)
    
    def rosenbrock_fitness(self, coor_list):
        result = 0
        for i in range(self.rosenbrock_degree):
            result += np.power(10,6*((i)/(self.rosenbrock_degree-1)))*np.power(ill_conditioning(coor_list[i]-self.x_opt_list[i]),2)
        return result

# modified rastrigin function: function 3 (SOLVED, D=20)

class function_13:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)

    def rosenbrock_fitness(self, coor_list):

        def delta_entry(index, alpha=10.0):
            result = np.power(alpha, 0.5*((index)/(self.rosenbrock_degree-1)))
            return result

        def t_asy(x, index, beta=0.2):
            if x > 0:
                result = np.power(x, 1+beta*(index/(self.rosenbrock_degree-1))*np.power(x,0.5))
            else:
                result = x
            return result

        first_term = 0
        second_term = 0
        for i in range(self.rosenbrock_degree):
            first_term += math.cos(2*math.pi*delta_entry(i)*t_asy(ill_conditioning(coor_list[i]-self.x_opt_list[i]), i)) # cos term
            second_term += np.power(delta_entry(i)*t_asy(ill_conditioning(coor_list[i]-self.x_opt_list[i]), i), 2)

        result = 10*(self.rosenbrock_degree - first_term) + second_term

        return result

# Buche-Rastrigin Function function 4 (SOLVED, D=20)

class function_14:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)

    def rosenbrock_fitness(self, coor_list):

        def s_i_first(z_i, i):
            if i % 2 == 0:
                result = 10*np.power(10,0.5*(i/(self.rosenbrock_degree-1)))
            else:
                result = np.power(10,0.5*(i/(self.rosenbrock_degree-1)))
            return result

        def z_i_first(x, i):
            s_i = s_i_first(x, i)
            z_i = s_i+ill_conditioning(x)
            return z_i

        def z_i_second(x, i):
            z_i = z_i_first(x, i)
            if (i % 2 == 0) and (z_i > 0):
                s_i = 10*np.power(10,0.5*(i/(self.rosenbrock_degree-1)))
            else:
                s_i = np.power(10,0.5*(i/(self.rosenbrock_degree-1)))
            z_i_second = s_i*ill_conditioning(x)

            return z_i_second

        def f_pen(x):
            result = np.power(np.max([0, np.abs(x)-5]), 2)
            return result

        first_term = 0
        second_term = 0
        third_term = 0
        for i in range(self.rosenbrock_degree):
            first_term += math.cos(2*math.pi*z_i_second(coor_list[i]-self.x_opt_list[i], i)) # cos term
            second_term += np.power(z_i_second(coor_list[i]-self.x_opt_list[i], i),2)
            third_term += f_pen(coor_list[i])

        result = 10 * (self.rosenbrock_degree - first_term) + second_term + 100*third_term 

        return result
    
# Linear slope: function 5 (SOLVED, D=20) DONE

class function_3:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        random_vec = np.random.randint(0,2,self.rosenbrock_degree)
        random_vec[random_vec == 0] = -1
        self.x_opt_list = 5 * random_vec
    
    def rosenbrock_fitness(self, coor_list):

        def sign_func(x, i):
            result = np.sign(x)*np.power(10, i/(self.rosenbrock_degree-1))
            return result

        def z_func(x, x_opt):
            if x_opt*x < 25:
                z = x
            else:
                z = x_opt
            return z

        def f_pen(x):
            result = np.power(np.max([0, np.abs(x)-5]), 2)
            return result

        result = 0
        pen_term = 0
        for i in range(self.rosenbrock_degree):
            result += 5*np.abs(sign_func(self.x_opt_list[i], i)) - sign_func(self.x_opt_list[i], i)*z_func(coor_list[i], self.x_opt_list[i])
            pen_term += f_pen(coor_list[i])

        result += pen_term

        return result

# attractive sector function: function 6 (SOLVED, D=20) DONE

class function_4:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)
        self.A_1 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.A_2 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_1, _ = np.linalg.qr(self.A_1)
        self.Q_2, _ = np.linalg.qr(self.A_2)

    def rosenbrock_fitness(self, coor_list):

        def delta_matrix(alpha=10.0):
            result = np.diag([np.power(alpha, 0.5*((index)/(self.rosenbrock_degree-1))) for index in range(self.rosenbrock_degree)])
            return result

        def s_func(x, z):
            if z*x > 0:
                result = 10**2
            else:
                result = 1
            return result

        z = np.matmul(self.Q_2, np.matmul(delta_matrix(), np.matmul(self.Q_1, (coor_list-self.x_opt_list))))

        result = 0
        for index in range(self.rosenbrock_degree):
            result += np.power(s_func(self.x_opt_list[index], z[index])*z[index], 2)
        result = ill_conditioning(np.power(result, 0.9))

        return result

# step ellipsoidal function: function 7 

class function_5:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)
        self.A_1 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.A_2 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_1, _ = np.linalg.qr(self.A_1)
        self.Q_2, _ = np.linalg.qr(self.A_2)

    def rosenbrock_fitness(self, coor_list):

        def delta_matrix(alpha=10.0):
            result = np.diag([np.power(alpha, 0.5*((index)/(self.rosenbrock_degree-1))) for index in range(self.rosenbrock_degree)])
            return result

        def z_tilde_func(z):
            result_array = np.zeros(z.shape[0])
            result_array[np.abs(z) < 0.5] = (np.floor(z+0.5))[np.abs(z) < 0.5]
            result_array[np.abs(z) >= 0.5] = (np.floor(10*z+0.5)/10)[np.abs(z) >= 0.5]

            return result_array

        def f_pen(x):
            result = np.power(np.max([0, np.abs(x)-5]), 2)
            return result

        z_hat = np.matmul(delta_matrix(), np.matmul(self.Q_1, (coor_list-self.x_opt_list)))
        z_tilde = z_tilde_func(z_hat)
        z = np.matmul(self.Q_2, z_tilde)

        first_term = 0
        pen_term = 0
        for index in range(self.rosenbrock_degree):
            first_term += np.power(10, 2*(index/(self.rosenbrock_degree-1)))*np.power(z[index], 2)
            pen_term += f_pen(coor_list[index])

        result = 0.1*np.max([np.abs(z_hat[0])/np.power(10,4), first_term])+pen_term

        return result

# rosenbrock function function 8 (SOLVED, D=20) DONE

class function_6:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-3, high=3, size=self.rosenbrock_degree)
    
    def rosenbrock_fitness(self, coor_list, f_opt=0):
        result = 0

        def z_func(x, x_opt):
            z = np.max([1, (np.sqrt(self.rosenbrock_degree)/8)])*(x-x_opt)+1
            return z

        for i in range(self.rosenbrock_degree-1):
            result +=  100*(z_func(coor_list[i], self.x_opt_list[i])**2-z_func(coor_list[i+1], self.x_opt_list[i+1]))**2+(z_func(coor_list[i],  self.x_opt_list[i])-1)**2 

        return result + f_opt

# rosenbrock function rotated: function 9 (SOLVED, D=20) DONE

class function_7:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-3, high=3, size=self.rosenbrock_degree)
        self.A_1 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_1, _ = np.linalg.qr(self.A_1)

    def rosenbrock_fitness(self, coor_list, f_opt=0):

        z = np.max([1,(np.sqrt(self.rosenbrock_degree)/8)])*np.matmul(self.Q_1, coor_list) + np.ones(self.rosenbrock_degree)/2

        result = 0
        for i in range(self.rosenbrock_degree-1):
            result +=  100*(z[i]**2-z[i+1])**2+(z[i]-1)**2 

        return result + f_opt

# ellipsoidal rotated: function 10 

class function_8:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)
        self.A_1 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_1, _ = np.linalg.qr(self.A_1)

    def rosenbrock_fitness(self, coor_list):

        z = np.matmul(self.Q_1, (coor_list-self.x_opt_list))

        result = 0
        for i in range(self.rosenbrock_degree):
            result += np.power(10,6*((i)/(self.rosenbrock_degree-1)))*np.power(ill_conditioning(z[i]),2)

        return result

# discus function: function 11 (D=20)

class function_9:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)
        self.A_1 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_1, _ = np.linalg.qr(self.A_1)

    def rosenbrock_fitness(self, coor_list):

        z = np.matmul(self.Q_1, (coor_list-self.x_opt_list))

        first_term = 0
        for i in range(1, self.rosenbrock_degree):
            first_term += np.power(ill_conditioning(z[i]),2)
        result = np.power(10,6)*z[0]**2+first_term

        return result

# bent cigar: function 12 (D=20) DONE

class function_10:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-3, high=3, size=self.rosenbrock_degree)
        self.A_1 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_1, _ = np.linalg.qr(self.A_1)

    def rosenbrock_fitness(self, coor_list):

        def t_asy(x, beta=0.5):
            array = np.zeros(self.rosenbrock_degree)
            for index in range(self.rosenbrock_degree):
                if x[index] > 0:
                    result = np.power(x[index], 1+beta*(index/(self.rosenbrock_degree-1))*np.power(x[index],0.5))
                else:
                    result = x[index]
                array[index] = result
            return array

        z = np.matmul(self.Q_1, t_asy(np.matmul(self.Q_1, (coor_list-self.x_opt_list))))
    
        first_term = 0
        for i in range(1, self.rosenbrock_degree):
            first_term += np.power(z[i], 2)
        result = z[0]**2+np.power(10,6)*first_term

        return result

# sharp ridge: function 13 (D=20) DONE

class function_11:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)
        self.A_1 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_1, _ = np.linalg.qr(self.A_1)
        self.A_2 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_2, _ = np.linalg.qr(self.A_2)

    def rosenbrock_fitness(self, coor_list):


        def delta_matrix(alpha=10.0):
            result = np.diag([np.power(alpha, 0.5*((index)/(self.rosenbrock_degree-1))) for index in range(self.rosenbrock_degree)])
            return result

        z = np.matmul(self.Q_2, np.matmul(delta_matrix(), np.matmul(self.Q_1, (coor_list-self.x_opt_list))))

        second_term = 0
        for index in range(1, self.rosenbrock_degree):
            second_term += np.power(z[index], 2)
        second_term = 100 * np.sqrt(second_term)

        result = np.power(z[0], 2) + second_term

        return result

# different powers: function 14 (D=20) DONE!!

class function_12:
    def __init__(self, random_seed, rosenbrock_degree):
        self.random_seed = random_seed
        self.rosenbrock_degree = rosenbrock_degree
        np.random.seed(random_seed)
        self.x_opt_list = np.random.uniform(low=-5, high=5, size=self.rosenbrock_degree)
        self.A_1 = np.random.randn(self.rosenbrock_degree, self.rosenbrock_degree)
        self.Q_1, _ = np.linalg.qr(self.A_1)

    def rosenbrock_fitness(self, coor_list):

        z = np.matmul(self.Q_1, (coor_list-self.x_opt_list))

        result = 0
        for index in range(self.rosenbrock_degree):
            result += np.power(np.abs(z[index]), (2+4*(index/(self.rosenbrock_degree-1))))

        result = np.sqrt(result)

        return result
