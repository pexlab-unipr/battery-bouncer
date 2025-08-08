# Control of boost converter in continuous conduction mode (CCM)

import numpy as np
import control as ct
import matplotlib.pyplot as plt

# Parameters
Vi0 = 8 # [V] Input voltage
Vout0 = 24 # [V] Output voltage
Iout0 = 1.5 # [A] Output current
L = 15e-6  # [H] Input inductance
C = 50e-6  # [F] Output capacitance
Ro = 1e3 # [ohm] Ballast (load) resistance
Kp = 1e-3 # [V] Proportional gain for the PID controller
Ki = 10 # [V/s] Integral gain for the PID controller

# Derived parameters
Go = 1/Ro
# delta = 1 - duty is the complementary duty cycle
Delta0 = Vi0 / Vout0  # Complementary duty cycle in steady state
Iin0 = (Go * Vout0 + Iout0) / Delta0 # Steady state input current

# State-space representation of the boost converter in CCM
# State variables: iL (inductor current), vC (capacitor voltage)
A = np.array([
    [0, -Delta0/L],
    [Delta0/C, -Go/C]])
B = np.array([
    [1/L, 0, -Vout0/L],
    [0, -1/C, Iin0/C]])
C = np.eye(2)  # Output matrix, identity for both states
D = np.zeros_like(B)
sys = ct.ss(
    A, B, C, D, name='boost_ccm',
    states=['iL', 'vC'], inputs=['vi', 'io', 'delta'], outputs=['iin', 'vout'])
sys_tf = ct.tf(sys, name='boost_ccm_tf')
dtov = -sys_tf[1,2]  # Transfer function from duty to output voltage
pid = ct.tf([Kp, Ki], [1, 0], name='pid_controller')  # PID controller
# Lead-lag controller for the boost converter
w0 = Vi0**2/L/Vout0/(Go*Vout0 + Iout0)
llz = w0/30
llp = w0*30
ll = ct.tf([1/llz, 1], [1/llp, 1], name='lead_lag_controller')  # Lead-lag controller
ctrl = ll*pid
loop = ctrl * dtov
closed_loop = ct.feedback(loop, 1)  # Closed-loop transfer function

# Results
print(dtov)
print("Poles:", ct.poles(dtov))
print("Zeros:", ct.zeros(dtov))

fres = [
    ct.frequency_response(dtov), 
    ct.frequency_response(loop),
    ct.frequency_response(closed_loop)]
tres = [
    ct.step_response(dtov),
    ct.step_response(loop),
    ct.step_response(closed_loop)]
plt.figure()
ct.bode_plot(fres, initial_phase=0)
plt.show()
plt.figure()
ct.time_response_plot(tres[2])
plt.show()

print("Ciao!")
