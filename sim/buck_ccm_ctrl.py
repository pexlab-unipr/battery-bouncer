import numpy as np
import control as ct
import matplotlib.pyplot as plt

# Converter parameters
fsw = 100e3
Tsw = 1/fsw
Tend = 200*Tsw
Tsim = Tsw/20
Lx = 100e-6
Cx = 1e-6
Ro = 30
Go = 1/Ro
Duty0 = 0.7
Vi0 = 12
Io0 = 1
Vo0 = Duty0*Vi0
IL0 = Io0 + Vo0/Ro
VC0 = Vo0

# Control parameters: PI from duty to vC
Kp = 1e-2
Ki = 1e3
# Control parameters: PI from duty to iL
Kpi = 0
Kii = 1e4
# Control parameters: PI from controlled iL to vC
Kpv = 1
Kiv = 0

# Control design
#  * We develop the control algorithm by assuming all quantities as constants,
#    manipulating duty to control the voltage.
#  * To have a linear state-space representation, setting duty and output current
#    as inputs, and Vi0 as a system parameter.
A, B, C, D = np.array([[0, -1/Lx], [1/Cx, -Go/Cx]]), np.array([[Vi0/Lx, 0], [0, -1/Cx]]), np.identity(2), np.zeros((2, 2))
buck_ccm = ct.ss(A, B, C, D,
                 inputs=['d', 'io'], 
                 outputs=['iL', 'vC'], 
                 states=['iL', 'vC'],
                 name='buck_ccm')
ts = np.linspace(0, Tend, round(Tend/Tsim), endpoint=False)

# Visualize open-loop system response with given input values
out_step = ct.forced_response(buck_ccm, ts, np.array([Duty0, Io0])[:,np.newaxis] @ np.ones_like(ts)[np.newaxis,:])
out_step.plot()
plt.show(block=False)

# Create a PI transfer function
ctrl = ct.tf([Kp, Ki], [1, 0], name='ctrl')

# Connect the PI at the input of the system, neglect unused inputs and outputs
loop_gain = ct.interconnect(
    [ctrl, buck_ccm],
    connections=[['buck_ccm.d', 'ctrl.y[0]']],
    inplist=['ctrl.u[0]'],
    outlist=['buck_ccm.vC'])

# Closed in unitary, negative feedback the loop gain
closed_loop = ct.feedback(loop_gain)

# Visualize the three bode responses together
ct.bode_plot([buck_ccm[1, 0], loop_gain, closed_loop])
plt.show(block=False)

# Plot the closed-loop step response over time (SISO system now)
closed_step = ct.step_response(closed_loop)
closed_step.plot()
plt.show(block=False)

# Pole placement, three strategies (for now, tried only one!)
# In all of them, discard the active load current, since it is an input
# which cannot be effectively controlled.
p = [-1e4, -1e5]
K = ct.place(A, B[:,0], p) # B reduced to avoid considering output current input
#Ka = ct.place_acker(buck_ccm[:, 0].A, buck_ccm[:, 0].B, p) # only SISO, bugfix soon?
#Kv = ct.place_varga(buck_ccm[:, 0].A, buck_ccm[:, 0].B, p) # needs slycot
print(f'K = {K}\n')

# Augmented system (pole placement + reference following)
p = [-2e4, -4e4, -1e5]
Csel = np.array([[0, 1]])
Aaug = np.block([[A, np.zeros((2,1))], [-Csel, np.zeros((1,1))]])
Baug = np.block([[B[:,0].reshape(-1, 1)],[-D[1,0]]])
K = ct.place(
    Aaug, Baug, p)
#Ka = ct.place_acker(buck_ccm[:, 0].A, buck_ccm[:, 0].B, p) # only SISO, bugfix soon?
#Kv = ct.place_varga(buck_ccm[:, 0].A, buck_ccm[:, 0].B, p) # needs slycot
print(f'K = {K}\n')

# Evaluate the closed-loop system with pole placement
# First, use the matrix approach
buck_ccm_cl = ct.ss(Aaug - Baug*K, np.array([0, 0, 1]).reshape(3,1), np.array([0, 1, 0]).reshape(1,3), 0)
ct.time_response_plot(buck_ccm_cl.step_response())
plt.show(block=False)
ct.bode_plot(buck_ccm_cl)
plt.show(block=False)

# LQR controller on the AC component of the system
# The idea is to follow a set-point given as steady-state and without the integral
# Disturbances are assumed slower than state transient, so their AC is null
# As a result, only duty input available (OK for LQR design)
A = np.array([[0, -1/Lx], [1/Cx, -Go/Cx]])
B = np.array([[Vi0/Lx], [0]])
C = np.identity(2)
D = np.zeros((2, 1))
Q = np.diag([1, 1])
R = 100
buck_ccm_duty_ac = ct.ss(A, B, C, D)
K, S, E = ct.lqr(A, B, Q, R)
print(K)
print(S)
print(E)

# LQI ???
Q = np.diag([1, 1e-3, 1e6])
K, S, E = ct.lqr(Aaug, Baug, Q, R)
print(K)
print(S)
print(E)

print("Ciao!")