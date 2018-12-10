import firedrake as fe
import fempy.models.enthalpy_phasechange


class Model(fempy.models.enthalpy_phasechange.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        super().__init__()
        
        self.stefan_number.assign(0.045)
        
        self.liquidus_temperature.assign(0.)

    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def init_initial_values(self):
        
        self.initial_values = fe.interpolate(
            fe.Expression(
                self.cold_wall_temperature.__float__(),
                element = self.element),
            self.function_space)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(V, self.hot_wall_temperature, 1),
            fe.DirichletBC(V, self.cold_wall_temperature, 2)]

    def run_and_plot(self, endtime):
        
        output_prefix = self.output_prefix
        
        while self.time.__float__() < (endtime - self.time_tolerance):
        
            self.run(endtime = self.time.__float__() + self.timestep_size.__float__())
            
            self.output_prefix = output_prefix + "t" + str(self.time.__float__()) + "_"
            
            self.plot(save = True, show = False)
    
if __name__ == "__main__":
    
    model = Model(meshsize = 128)
    
    model.timestep_size.assign(1.)
    
    model.phase_interface_smoothing.assign(1./256.)
    
    model.run_and_plot(endtime = 70.)
    