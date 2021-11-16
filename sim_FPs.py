from scipy.integrate import odeint

t_half_cell = 20
lambda_dilution = 0.693/t_half_cell
k = 1

t = np.linspace(0, 100, 50)
f = lambda y: k - lambda_dilution*y
res = odeint()
