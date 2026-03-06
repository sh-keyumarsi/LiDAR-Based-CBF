import matplotlib.pyplot as plt
import matplotlib.animation as animation

PYSIM = True
#PYSIM = False # for experiment or running via ROS

if PYSIM:
    from scenarios.sim2D_obstacle_GP import FeedbackInformation, Controller, ControlOutput, SimSetup, SimulationCanvas
else:
    from .scenarios.sim2D_obstacle_GP import FeedbackInformation, Controller, ControlOutput, SimSetup, SimulationCanvas

class Simulate():
    def __init__(self):
        # Initialize components
        self.environment = SimulationCanvas() # Always the first to call, define main setup
        self.controller_block = Controller() 
        # Initialize messages to pass between blocks
        self.feedback_information = FeedbackInformation() 
        self.control_input = ControlOutput() 

    # MAIN LOOP CONTROLLER & VISUALIZATION
    def loop_sequence(self, i = 0):
        # Showing Time Stamp
        if (i > 0) and (i % round(1/SimSetup.Ts) == 0):
            t = round(i*SimSetup.Ts)
            if t < SimSetup.tmax: print('simulating t = {}s.'.format(t))

        # Compute control input and advance simulation
        self.controller_block.compute_control( self.feedback_information, self.control_input )
        self.environment.update_simulation( self.control_input, self.feedback_information )
        

def main():
    print("start")    
    # Ìnitialize Simulation
    sim = Simulate()

    # Step through simulation
    ani = animation.FuncAnimation(sim.environment.fig, sim.loop_sequence, save_count=\
                                  round(SimSetup.tmax / SimSetup.Ts)+ 1, interval = SimSetup.Ts*1000)
    if SimSetup.save_animate: # default not showing animation
        print('saving animation ...')
        ani.save(SimSetup.sim_fname_output, fps=round(1/SimSetup.Ts))
        #ani.save(r'animation.mp4', writer=animation.FFMpegWriter(fps=round(1/dt)))
    else:
        plt.show()

    print('Finish')
    # TODO: logger and profiling

if __name__ == '__main__':
    main()