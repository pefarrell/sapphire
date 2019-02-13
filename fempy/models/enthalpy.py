""" An enthalpy model class for melting and solidification """
import firedrake as fe
import fempy.unsteady_model
from matplotlib import pyplot as plt

    
class Model(fempy.unsteady_model.Model):
    
    def __init__(self):
        
        self.stefan_number = fe.Constant(1.)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.smoothing = fe.Constant(1./32.)
        
        super().__init__()
        
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = 4)
        
    def porosity(self, T):
        
        T_L = self.liquidus_temperature
        
        s = self.smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - T_L)/s))
    
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        T = self.solution
        
        T_n = self.initial_values
        
        Delta_t = self.timestep_size
        
        T_t = (T - T_n)/Delta_t
        
        phil = self.porosity
        
        phil_t = (phil(T) - phil(T_n))/Delta_t
        
        self.time_discrete_terms = T_t, phil_t
    
    def init_weak_form_residual(self):
        
        T = self.solution
        
        Ste = self.stefan_number
        
        T_t, phil_t = self.time_discrete_terms
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*(T_t + 1./Ste*phil_t) +\
            dot(grad(v), grad(T))
        
    def init_solver(self, solver_parameters = {"ksp_type": "cg"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)
            
    def plot(self):
    
        V = self.function_space
        
        T = self.solution
        
        phil = fe.interpolate(self.porosity(T), V)
        
        timestr = str(self.time.__float__())
        
        for f, label, filename in zip(
                (T, phil),
                ("T", "\\phi_l"),
                ("T", "phil")):
            
            fe.plot(f)
            
            plt.axis("square")
        
            plt.xlabel(r"$x$")

            plt.ylabel(r"$y$")

            plt.title(r"$" + label + 
                ", t = " + timestr + "$")
            
            self.output_directory_path.mkdir(
                parents = True, exist_ok = True)
        
            filepath = self.output_directory_path.joinpath(filename + 
                "_t" + timestr.replace(".", "p")).with_suffix(".png")
            
            print("Writing plot to " + str(filepath))
            
            plt.savefig(str(filepath))
            
            plt.close()
    