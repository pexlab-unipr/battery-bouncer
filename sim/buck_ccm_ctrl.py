import numpy as np
import control as ct
import matplotlib.pyplot as plt

# Converter parameters
fsw = 100e3
Tsw = 1/fsw
Tend = 50*Tsw
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

# Control design
#    * We develop the control algorithm by assuming all quantities as constants,
#      manipulating duty to control the voltage.
#    * To have a linear state-space representation, setting duty and output current
#      as inputs, and Vi0 as a system parameter (even if, in reality, it is an input).
A = np.array([[0, -1/Lx], [1/Cx, -Go/Cx]])
B = np.array([[Vi0/Lx, 0], [0, -1/Cx]])
C = np.identity(2)
D = np.zeros((2, 2))
buck_ccm = ct.ss(
    A, B, C, D,
    inputs=['d', 'io'], 
    outputs=['iL', 'vC'], 
    states=['iL', 'vC'],
    name='buck_ccm')
ts = np.linspace(0, Tend, round(Tend/Tsim), endpoint=False)

# Visualize open-loop system response with given input values
out_step = ct.forced_response(buck_ccm, ts, np.array([Duty0, Io0])[:,np.newaxis] @ np.ones_like(ts)[np.newaxis,:])
out_step.plot()
plt.show(block=False)

# PI controller with Bode design
#    * Create a PI transfer function
#    * Connect the PI at the input of the system, neglecting unused inputs and outputs
#      thus reducing it to its SISO form (duty to capacitor voltage)
#    * Close the loop with unitary gain
ctrl = ct.tf(
    [Kp, Ki], [1, 0], 
    inputs='err',
    outputs='d',
    name='ctrl_pi')
buck_ccm_d_vC = buck_ccm[1,0]
loop_gain = ct.ss(ctrl*buck_ccm_d_vC)
closed_loop = ct.feedback(loop_gain)

# Visualize the three bode responses together
ct.bode_plot([buck_ccm_d_vC, loop_gain, closed_loop])
plt.show(block=False)

# Plot the closed-loop step response over time (SISO system now)
closed_step = ct.step_response(closed_loop)
closed_step.plot()
plt.show(block=False)

# Pole placement controller design
#    * Discard the active load current, since it is an input that cannot be effectively controlled.
#    * Barely nonsense, only stabilizing, no possibility to follow a reference!
#    * Target poles chosen for bandwidth
buck_ccm_d = buck_ccm[:,0]
p = [-1e4, -1e5]
K = ct.place(buck_ccm_d.A, buck_ccm_d.B, p) # B reduced to avoid considering output current input
#Ka = ct.place_acker(buck_ccm[:, 0].A, buck_ccm[:, 0].B, p) # only SISO, bugfix soon?
#Kv = ct.place_varga(buck_ccm[:, 0].A, buck_ccm[:, 0].B, p) # needs slycot
print(f'K pp    = {K}')

# Augmented system (pole placement + reference following)
p = [-2e4, -4e4, -1e5]
Csel = np.array([[0, 1]])
Aaug = np.block([[A, np.zeros((2,1))], [-Csel, np.zeros((1,1))]])
Baug = np.block([[B[:,0].reshape(-1, 1)],[-D[1,0]]])
K = ct.place(Aaug, Baug, p)
print(f'K ppi   = {K}')

# Evaluate the closed-loop system with pole placement
# First, use the matrix approach
buck_ccm_cl = ct.ss(Aaug - Baug*K, np.array([0, 0, 1]).reshape(3,1), np.array([0, 1, 0]).reshape(1,3), 0)
ct.time_response_plot(buck_ccm_cl.step_response())
plt.show(block=False)
ct.bode_plot(buck_ccm_cl)
plt.show(block=True)

# LQR controller on the AC component of the system
#    * The set-point is followed as steady-state value feed forward, without the integral.
#    * Disturbances are assumed slower than state transient, so their AC value is null;
#      as a result, only duty input is available (OK for LQR design, since output current
#      is not manipulated).
#    * Since with constant Vi0 the system is linear, its large-signal and small-signal
#      descriptions coincide (same ABCD matrices).
A = np.array([[0, -1/Lx], [1/Cx, -Go/Cx]])
B = np.array([[Vi0/Lx], [0]])
C = np.identity(2)
D = np.zeros((2, 1))
Q = np.diag([1, 1])
R = 100
buck_ccm_duty_ac = ct.ss(A, B, C, D)
K, S, E = ct.lqr(A, B, Q, R)
print(f'K lqrac = {K}')

# LQI controller
#    * Does not need any feed-forward term
#    * How to properly weight system state energy and reference tracking error is unknown
Q = np.diag([1, 1e-3, 1e6])
R = 100
K, S, E = ct.lqr(Aaug, Baug, Q, R)
print(f'K lqi   = {K}')

print("Ciao!")