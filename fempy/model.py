""" Provides the base class for all models """
import warnings
import pathlib
import firedrake as fe
import fempy.time_discretization
import fempy.output


def solve(
        variational_form_residual,
        solution,
        dirichlet_bcs = None,
        solver_parameters = None):
    """ Solve the problem defined by the form and boundary conditions,
    first by constructing the problem and solver on demand.
    
    Parameters
    ----------
    variational_form_residual : fe.Form
    
    solution : fe.Function
    
    dirichlet_bcs : fe.DirichletBC or list/tuple of them, optional
    
    solver_parameters : dict, optional
    
    Examples
    --------
    a) Solve a nonlinear diffusion problem.
    >>> import firedrake as fe
    >>> from fempy.model import solve
    >>> mesh = fe.UnitIntervalMesh(2)
    >>> P1 = fe.FiniteElement("P", mesh.ufl_cell(), 1)
    >>> V = fe.FunctionSpace(mesh, P1)
    >>> u = fe.Function(V)
    >>> v = fe.TestFunction(V)
    >>> bc = fe.DirichletBC(V, 0., "on_boundary")
    >>> alpha = 1 + u/10.
    >>> x = fe.SpatialCoordinate(mesh)[0]
    >>> div, grad, dot, sin, pi = fe.div, fe.grad, fe.dot, fe.sin, fe.pi
    >>> dx = fe.dx
    >>> s = 10.*sin(pi*x)
    >>> F = (-dot(grad(v), alpha*grad(u)) - v*s)*dx
    >>> u, its = solve(F, u, bc)
    >>> print("Solved in {0} iterations.".format(its))
    >>> print("u = {0}".format(u.vector().array()))
    u = [ 0.         -1.07046388  0.        ]
    """
    problem = fe.NonlinearVariationalProblem(
        F = variational_form_residual,
        u = solution,
        bcs = dirichlet_bcs,
        J = fe.derivative(variational_form_residual, solution))
        
    solver = fe.NonlinearVariationalSolver(
        problem = problem,
        solver_parameters = solver_parameters)
        
    solver.solve()
    
    return solution, solver.snes.getIterationNumber()
    
    
class Model(object):
    """ A PDE-based model for time-dependent simulations,
    discretized in space with mixed finite elements 
    and in time with finite differences.
    """
    def __init__(self, 
            mesh, 
            element, 
            variational_form_residual,
            dirichlet_bcs,
            initial_values,
            quadrature_degree = None,
            time_dependent = True,
            time_stencil_size = 2):
        
        self.mesh = mesh
        
        self.element = element
        
        self.function_space = fe.FunctionSpace(mesh, element)
        
        self.quadrature_degree = quadrature_degree
        
        self.solutions = [fe.Function(self.function_space) 
            for i in range(time_stencil_size)]
            
        self.solution = self.solutions[0]
        
        if time_dependent:
            
            assert(time_stencil_size > 1)
            
            self.time = fe.Constant(0.)
            
            self.timestep_size = fe.Constant(1.)
            
            self.time_tolerance = 1.e-8
            
        else:
        
            warn()
        
            time_stencil_size = 1
            
        self.initial_values = initial_values(model = self)
        
        for solution in self.solutions:
        
            solution.assign(self.initial_values)
        
        self.variational_form_residual = variational_form_residual(
                model = self,
                solution = self.solution)
                
        self.dirichlet_bcs = \
            dirichlet_bcs(model = self)
        
        self.output_directory_path = pathlib.Path("output/")
        
        self.snes_iteration_count = 0
        
    def solve(self, *args, **kwargs):
        
        self.solution, snes_iteration_count = solve(*args,
            variational_form_residual = self.variational_form_residual,
            solution = self.solution,
            dirichlet_bcs = \
                self.dirichlet_bcs,
            **kwargs)
           
        self.snes_iteration_count += snes_iteration_count
        
        return self.solution, self.snes_iteration_count
    
    def push_back_solutions(self):
        
        for i in range(len(self.solutions[1:])):
        
            self.solutions[-(i + 1)].assign(
                self.solutions[-(i + 2)])
                
        return self.solutions
        
    def run(self,
            endtime,
            solve = None,
            report = False,
            postprocess = None,
            write_solution = False,
            plot = None):
        
        if solve is None:
        
            solve = self.solve
        
        self.output_directory_path.mkdir(
            parents = True, exist_ok = True)
        
        if write_solution:
        
            solution_filepath = self.\
                output_directory_path.joinpath("solution").with_suffix(".pvd")
        
            solution_file = fe.File(str(solution_filepath))
            
        if report:
            
            fempy.output.report(
                self, postprocess = postprocess, write_header = True)
        
        if write_solution:
        
            write_solution(solution_file)
        
        if plot:
            
            plot(self)
            
        while self.time.__float__() < (
                endtime - self.time_tolerance):
            
            self.time.assign(self.time + self.timestep_size)
            
            self.solution, self.snes_iteration_count = solve()
            
            if report:
            
                fempy.output.report(
                    self, postprocess = postprocess, write_header = False)
                
            if write_solution:
        
                fempy.output.write_solution(self, solution_file)
                
            if plot:
            
                plot(self, self.solution)
            
            self.solutions = self.push_back_solutions()
            
            print("Solved at time t = {0}".format(self.time.__float__()))
                
        return self.solutions, self.time, self.snes_iteration_count
        
    def assign_parameters(self, parameters):
    
        for key, value in parameters.items():
        
            attribute = getattr(self, key)
            
            if type(attribute) is type(fe.Constant(0.)):
            
                attribute.assign(value)
                
            else:
            
                setattr(self, key, value)
                
        return self
        
def unit_vectors(mesh):
    
    dim = mesh.geometric_dimension()
    
    return tuple([fe.unit_vector(i, dim) for i in range(dim)])
    
    
def time_discrete_terms(solutions, timestep_size):
    
    time_discrete_terms = [
        fempy.time_discretization.bdf(
            [fe.split(solutions[n])[i] for n in range(len(solutions))],
            timestep_size = timestep_size)
        for i in range(len(fe.split(solutions[0])))]
        
    if len(time_discrete_terms) == 1:
    
        time_discrete_terms = time_discrete_terms[0]
        
    else:
    
        time_discrete_terms = time_discrete_terms

    return time_discrete_terms
    