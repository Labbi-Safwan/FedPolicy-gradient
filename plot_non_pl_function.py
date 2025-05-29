import numpy as np
import matplotlib.pyplot as plt

# Define the functions
temperature = 1
p_c1 = 0
q_c1 = 1
q_c2 = 0.01
p_c2 = 0.99
gamma = 0.999

def H(x):
    return -x *np.log(x) - (1-x)*np.log(1-x)

def s(theta):
    return 1/(1+ np.exp(-theta))

def f(x, p,q):
    return p  +(q-p)*x

def J1(theta):
    nominateur1 = temperature * H(s(theta)) + gamma * f(s(theta), p_c1, q_c1) * (1+ temperature*np.log(2)  + gamma* temperature * np.log(2))
    denominateur1 = 1 - gamma * (1- f(s(theta),p_c1, q_c1))
    return (nominateur1/denominateur1)
def J2(theta):
    nominateur2 = temperature * H(s(theta)) + gamma * f(s(theta), p_c2, q_c2) * (1+ temperature*np.log(2)  + gamma* temperature * np.log(2))
    denominateur2 = 1 - gamma * (1- f(s(theta),p_c2, q_c2))
    return(nominateur2/denominateur2)
def J(theta):
    return((J1(theta) +J2(theta))/2)

# Generate x values
x = np.linspace(-15,15, 1000)

# Calculate y values
y1 = J1(x)
y2 = J2(x)
y3 = J(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=r'$J_1(\theta) $', color='blue')
plt.plot(x, y2, label=r'$J_2(\theta)$', color='green')
plt.plot(x, y3, label=r'$J(\theta)$', color='red')

# Add labels, legend, and grid
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'$\theta$', fontsize=24)
plt.ylabel(r'$J(\theta)$', fontsize=24)
plt.legend(fontsize=14, loc =5)
plt.grid(True)


# Save the plot to a PDF
plt.savefig("functions_plot.pdf")

# Show the plot
plt.show()