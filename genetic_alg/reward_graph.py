import glob as glob
import os
import matplotlib.pyplot as plt
import argparse

# Your list of numbers
numbers = [1, 2, 3, 4, 5]
def draw_graph(reward, generations, output = 'out/genetic_alg/evolved_nn/weights_ez/eval'):
    # Create a simple plot
    plt.plot(reward)

    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Rewards')
    plt.title('Reward over Generations')
    
    os.makedirs(output, exist_ok=True)
    # Save the plot as an image (e.g., PNG)
    plt.savefig(f'{output}/metrics_eval.png')

if '__name__' == '__name__':

    parser = argparse.ArgumentParser(description="Draw graph of mean reward across generations ")
    parser.add_argument("weight_folder", default= 'out/genetic_alg/evolved_nn/weights_ez', help="Path to the weights folder.")
    
    args = parser.parse_args()
    weight_folder = args.weight_folder

    files = glob.glob(f'{weight_folder}/*pkl')
    gens = [int(os.path.basename(path).split('_')[1]) for path in files]
    rew = [float(os.path.basename(path).split('_')[-1].split('.pkl')[0]) for path in files]

    rew = [x for _, x in sorted(zip(gens, rew))]
    gens = [x for _, x in sorted(zip(gens, gens))]

    gens[1:] = [x+1 for x in gens[1:]]
    print(gens)
    print(rew)

    draw_graph(reward=rew, generations=gens, output=os.path.join(weight_folder,'eval'))