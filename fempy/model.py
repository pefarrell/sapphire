""" An abstract class on which to base finite element models """
import firedrake as fe
import pathlib
import matplotlib.pyplot as plt


class Model(object):
    """ A class on which to base finite element models. """
    def __init__(self, quadrature_degree, spatial_order):
        
        self.quadrature_degree = quadrature_degree
        
        self.spatial_order = spatial_order
        
        self.init_mesh()
        
        self.init_element()
        
        self.init_function_space()
        
        self.init_solutions()
        
        self.init_integration_measure()
        
        self.init_weak_form_residual()
        
        self.init_dirichlet_boundary_conditions()
        
        self.init_problem()
        
        self.init_solver()
        
        self.quiet = False
        
        self.output_directory_path = pathlib.Path("output/")
        
        self.snes_iteration_counter = 0
        
    def init_mesh(self):
        """ Redefine this to set `self.mesh` to a `fe.Mesh`.
        """
        assert(False)
    
    def init_element(self):
        """ Redefine this to set `self.element` 
        to a  `fe.FiniteElement` or `fe.MixedElement`.
        """
        assert(False)
        
    def init_weak_form_residual(self):
        """ Redefine this to set `self.weak_form_residual` 
        to a `fe.NonlinearVariationalForm`.
        """
        assert(False)
    
    def init_function_space(self):
    
        self.function_space = fe.FunctionSpace(self.mesh, self.element)
    
    def init_solutions(self):
    
        self.solutions = [fe.Function(self.function_space)]
        
        self.solution = self.solutions[0]
    
    def init_dirichlet_boundary_conditions(self):
        """ Optionallay redefine this 
        to set `self.dirichlet_boundary_conditions`
        to a tuple of `fe.DirichletBC` """
        self.dirichlet_boundary_conditions = None
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = self.quadrature_degree)
        
    def init_problem(self):
    
        r = self.weak_form_residual*self.integration_measure
        
        u = self.solution
        
        self.problem = fe.NonlinearVariationalProblem(
            r, u, self.dirichlet_boundary_conditions, fe.derivative(r, u))
        
    def init_solver(self, solver_parameters = {
            "snes_type": "newtonls",
            "snes_monitor": True,
            "ksp_type": "preonly", 
            "pc_type": "lu", 
            "mat_type": "aij",
            "pc_factor_mat_solver_type": "mumps"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)
    
    def assign_parameters(self, parameters):
    
        for key, value in parameters.items():
        
            attribute = getattr(self, key)
            
            if type(attribute) is type(fe.Constant(0.)):
            
                attribute.assign(value)
                
            else:
            
                setattr(self, key, value)
    
    def solve(self):
    
        self.solver.solve()
        
        self.snes_iteration_counter += self.solver.snes.getIterationNumber()
    
    def unit_vectors(self):
        
        dim = self.mesh.geometric_dimension()
        
        return tuple([fe.unit_vector(i, dim) for i in range(dim)])
        
    def plot(self):
        
        for i, f in enumerate(self.solution.split()):
            
            fe.plot(f)
            
            plt.axis("square")
            
            plt.title(r"$w_" + str(i) + "$")
            
            self.output_directory_path.mkdir(
                parents = True, exist_ok = True)
        
            filepath = self.output_directory_path.joinpath(
                "solution_" + str(i)).with_suffix(".png")
            
            print("Writing plot to " + str(filepath))
            
            plt.savefig(str(filepath))
            
            plt.close()
        