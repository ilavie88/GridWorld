import os
import numpy as np
import matplotlib.pyplot as plt


# Define the path to the main directory
main_directory = '/home/ilavie/local_PI/experiments/local_finetuning_exp/20dist/mazes'
# main_directory = '/home/ilavie/local_PI/experiments/local_finetuning_exp/demo_run/mazes'


iterations = []
vs0 = []
radiuses = []
# Iterate over each subdirectory in the main directory
for subdir in os.listdir(main_directory):
    subdirectory_path = os.path.join(main_directory, subdir)

    # Check if the item is a directory
    if os.path.isdir(subdirectory_path):

        # Load iterations_and_radius.txt
        iterations_and_radius_file = os.path.join(subdirectory_path, 'iterations_and_radius.txt')
        if os.path.exists(iterations_and_radius_file):
            iterations_and_radius_data = np.loadtxt(iterations_and_radius_file)
            n = len(iterations_and_radius_data)
            print(f'Iterations and radius data for {subdir}:')
            curr_iterations = np.zeros(n)
            curr_radius = np.zeros(n)
            for i, data in enumerate(iterations_and_radius_data):
                curr_iterations[i] = data[0]
                curr_radius[i] = data[1]
        else:
            print(f'iterations_and_radius.txt not found in {subdir}.')

        # Load s0_Value_function_and_radius.txt
        s0_value_function_and_radius_file = os.path.join(subdirectory_path, 's0_Value_function_and_radius.txt')
        if os.path.exists(s0_value_function_and_radius_file):
            s0_value_function_and_radius_data = np.loadtxt(s0_value_function_and_radius_file)
            print(f's0 Value function and radius data for {subdir}:')
            curr_vs0 = np.zeros(n)
            for i, data in enumerate(s0_value_function_and_radius_data):
                curr_vs0[i] = data[0]
        else:
            print(f's0_Value_function_and_radius.txt not found in {subdir}.')
        iterations.append(curr_iterations)
        vs0.append(curr_vs0)
        radiuses.append(curr_radius)
iterations_stack = np.stack(iterations)
iteration_mean = np.mean(iterations_stack, axis=0)
iteration_std = np.std(iterations_stack, axis=0)
vs0_stack = np.stack(vs0)
vs0_mean = np.mean(vs0, axis=0)
vs0_std = np.std(vs0, axis=0)
radiuses_stack = np.stack(radiuses)
radiuses_mean = np.mean(radiuses, axis=0)

np.savetxt(os.path.join(main_directory, 'experiment_mean_iter_vs0_radius.txt'),
           np.column_stack((iteration_mean, iteration_std ,vs0_mean, vs0_std, radiuses_mean)))

# Clear the current plot
plt.clf()  # or plt.cla()

# Generate plot
# plt.errorbar(radiuses_mean, vs0_mean, yerr=vs0_std, fmt='-o', label='Data with Error Bars')
plt.plot(radiuses_mean, vs0_mean)

low_error = vs0_mean - vs0_std
low_error[low_error < 0] = 0
up_error = vs0_mean + vs0_std
up_error[up_error > 1] = 1
plt.fill_between(radiuses_mean, low_error, up_error, color='gray', alpha=0.3)


plt.xlabel('Neighborhood radius')
plt.ylabel('Normalized $V[s0]$')
plt.title('Normalized $V[s0]$ as a Function of Radius')
plt.grid(True)

# Save plot as image file
plt.savefig(os.path.join(main_directory, 'value_function_plot.png'))

# Show plot (optional)
plt.show()