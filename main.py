import random
import json
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt

# problem = tsplib95.load('ALL_TSP/bayg29.tsp')
problem = tsplib95.load('ALL_TSP/att48.tsp')
# problem = tsplib95.load('ALL_TSP/berlin52.tsp')
# problem = tsplib95.load('ALL_TSP/burma14.tsp')
# problem = tsplib95.load('ALL_TSP/gr24.tsp')
# problem = tsplib95.load('ALL_TSP/dantzig42.tsp')
# problem = tsplib95.load('ALL_TSP/brg180.tsp')
# problem = tsplib95.load('ALL_TSP/eil51.tsp')


class TSPGASolver:
    """
    Class to solve the TSP using Genetic Algorithm

    Attributes:
            problem (tsplib95.models.StandardProblem): The TSP problem instance
            generations (int): Number of generations for which the algorithm will run
            population_size (int): Population size for each generation
            no_of_cities (int): Number of cities in the problem
            mutation_rate (float): Mutation rate for the algorithm
            best_tour (list): Best tour found so far initialized to the MST preorder starting from node 1
            best_fitness (float): Best fitness found so far initialized to infinity
            population (list): Population list for each generation
            node_coordinates (dict): Dictionary containing the (x,y) coordinates of each node
            optimal_solution (float): Optimal solution for the problem
            fig (matplotlib.figure.Figure): Figure object for the plot
            ax1 (matplotlib.axes.Axes): Axes object for the plot
            ax2 (matplotlib.axes.Axes): Axes object for the plot
            fitness_data (list): List containing the fitness data for each generation
    """

    def __init__(self, problem, generations, population_size, mutation_rate) -> None:
        """
        Initialize the solver with the problem instance and other parameters

        Args:
            problem (tsplib95.models.StandardProblem): The TSP problem instance
            generations (int): Number of generations for which the algorithm will run
            population_size (int): Population size for each generation
            mutation_rate (float): Mutation rate for the algorithm
        """

        self.problem = problem
        self.generations = generations
        self.population_size = population_size
        self.no_of_cities = self.problem.dimension
        self.mutation_rate = mutation_rate
        self.best_tour = self.get_mst_preorder(next(self.problem.get_nodes()))
        self.best_fitness = float('inf')
        self.population = []
        self.node_coordinates = {}
        self.nodes = list(self.problem.get_nodes())

        # Load optimal solution from TSPLIB_solutions.json
        with open('TSPLIB_solutions.json', encoding='utf-8') as f:
            data = json.load(f)
            self.optimal_solution = data[self.problem.name]

        # Fetch node coordinates from the problem instance depending on whether display_data is available or not
        if self.problem.display_data:
            self.node_coordinates = self.problem.display_data
        else:
            self.node_coordinates = self.problem.node_coords

        # Initialize population with MST preorder tours
        for _ in range(self.population_size):
            self.population.append(self.get_mst_preorder(
                random.randint(self.nodes[0], self.nodes[-1])))

        # Initialize population randomly
        # for _ in range(self.population_size):
        #     p = [i for i in self.problem.get_nodes()]
        #     self.population.append(self.shuffle(p, 0, len(p)-1))

        # Initialize figure for plotting the tour and fitness
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        # List to store the fitness data for each generation (for plotting)
        self.fitness_data = []

    def shuffle(self, arr, start, end):
        for i in range(end, start, -1):
            random_index = random.randint(start, end-1)
            arr[random_index], arr[i] = arr[i], arr[random_index]
        return arr

    def get_mst_preorder(self, source):
        """
        Get the preorder traversal of the MST starting from the given source node

        Args:
            source (int): Source node for the MST

        Returns:
            list: Preorder traversal of the MST starting from the given source node
        """

        mst_edges = list(nx.minimum_spanning_edges(
            self.problem.get_graph(), algorithm='prim', data=False))
        mst = nx.Graph(mst_edges)

        # Perform a depth-first traversal to get MST nodes in preorder
        preorder_nodes = list(nx.dfs_preorder_nodes(mst, source))

        return preorder_nodes + [source]

    def evaluate_fitness(self, tours):
        """
        Evaluate the fitness of the given tours and sort them in descending order of fitness

        Args:
            tours (list): List of tours to evaluate fitness for

        Returns:
            list: List of fitnesses and tours sorted in descending order of fitness
        """

        fitnesses = []

        for ind, fitness in enumerate(self.problem.trace_tours(tours)):
            fitnesses.append([1/fitness, tours[ind]])

        return sorted(fitnesses, reverse=True)

    def mutate(self, tour, mutation_rate=0.2):
        """
        Mutate the given tour with the given mutation rate

        Args:
            tour (list): Tour to mutate
            mutation_rate (float, optional): Mutation rate. Defaults to 0.1.

        Returns:
            list: Mutated tour
        """
        # USING REVERSE SEQUENCE MUTATION (RSM)
        # (RSM IS MORE EFFICIENT ACCORDING TO RESEARCH PAPERS: https://arxiv.org/pdf/1203.3099.pdf)

        if random.random() > mutation_rate:
            return tour

        i = random.randint(0, self.no_of_cities-2)
        j = random.randint(i+1, self.no_of_cities-1)

        for _ in range((j-i+1)//2):

            tour[i], tour[j] = tour[j], tour[i]
            i += 1
            j -= 1

        return tour

    def crossover(self, parent1, parent2):
        """
        Perform crossover between the given parents

        Args:
            parent1 (list): First parent
            parent2 (list): Second parent

        Returns:
            list: Child tour
        """

        # USING ORDER CROSSOVER (OX1)
        # select a random subset from parent1 and copy it to child
        # then fill the rest of the child with the remaining numbers from parent2
        child = [-1 for _ in range(self.no_of_cities)]
        start = random.randint(0, self.no_of_cities-1)
        end = random.randint(start, self.no_of_cities-1)
        child[start:end+1] = parent1[start:end+1]
        i = 0

        for j in range(self.no_of_cities):
            if child[j] == -1:
                while parent2[i] in child:
                    i += 1
                child[j] = parent2[i]
                i += 1
        return child

    def plot_tour(self, tour):
        """
        Plot the given tour on the figure

        Args:
            tour (list): Tour to plot
        """

        edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
        # Connecting the last node to the first to form a closed loop
        edges.append((tour[-1], tour[0]))

        G = nx.Graph()
        G.add_nodes_from(self.node_coordinates)
        G.add_edges_from(edges)

        # Plot on ax1
        self.ax1.clear()
        self.ax1.set_title("Best Tour Found")
        pos = self.node_coordinates

        nx.draw(G, pos, ax=self.ax1, with_labels=True,
                node_size=100, font_size=8)

        # Draw plots
        self.fig.canvas.draw()
        plt.pause(0.01)

    def plot_fitness(self, fitness_data):
        """
        Plot the fitness data on the figure

        Args:
            fitness_data (list): List of fitness data
        """

        difference_percentage = abs(
            self.best_fitness - self.optimal_solution)/self.optimal_solution*100
        self.ax2.clear()
        self.ax2.plot(fitness_data, label='Current Fitness')
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Fitness')
        self.ax2.set_title('Fitness vs Generation')
        self.ax2.text(
            0.5, 0.7, f"Best Fitness = {self.best_fitness}", transform=self.ax2.transAxes)
        self.ax2.text(
            0.5, 0.65, f"Optimal Solution = {self.optimal_solution}", transform=self.ax2.transAxes)
        self.ax2.text(
            0.5, 0.60, f"Difference % = {round(difference_percentage, 2)}", transform=self.ax2.transAxes)
        self.ax2.axhline(y=self.optimal_solution, color='r',
                         linestyle='-', label='Optimal Solution')
        self.ax2.legend()
        # Draw plots
        self.fig.canvas.draw()
        plt.pause(0.01)

    def run(self):
        """
        Run the genetic algorithm for the given number of generations
        """

        for generation in range(self.generations):
            fitnesses = self.evaluate_fitness(self.population)

            next_gen = []
            mutated_pop = []
            individual_tour_fitness = []
            for f in fitnesses:
                individual_tour_fitness.append(f[0])

            for i in range(self.population_size//4):
                next_gen.append(fitnesses[i][1])

            # Using Roulette Wheel Selection
            for _ in range(self.population_size//2):
                parent1 = random.choices(
                    fitnesses, weights=individual_tour_fitness)[0][1]
                parent2 = random.choices(
                    fitnesses, weights=individual_tour_fitness)[0][1]
                child = self.crossover(parent1, parent2)
                next_gen.append(child)

            # perform mutation (quarter of the population mutated)
            for tour in next_gen[:self.population_size//4]:
                mutated_pop.append(self.mutate(tour))
                # next_gen.append(tour)

            next_gen.extend(mutated_pop)
            next_gen_fitness = self.evaluate_fitness(next_gen)

            # Update best tour and best fitness
            if self.problem.trace_tours([next_gen_fitness[0][1]])[0] < self.best_fitness:
                self.best_tour = next_gen_fitness[0][1]
                self.best_fitness = self.problem.trace_tours(
                    [next_gen_fitness[0][1]])[0]

            # Update population
            self.population = [tour for fitness,
                               tour in next_gen_fitness[:self.population_size]]
            print(
f"BEST FITNESS FOR GENERATION {generation} = {self.best_fitness}")

            # Some instances of TSPLIB are not depictable (They have no data regarding their coordinates)
            if self.problem.is_depictable():
                self.plot_tour(self.best_tour)

            # Plot fitness data
            self.fitness_data.append(self.best_fitness)
            self.plot_fitness(self.fitness_data)
            difference_percentage = abs(self.best_fitness - self.optimal_solution)/self.optimal_solution*100
            if difference_percentage == 0:
                print(f"Found the optimal solution in {generation} generations")
                break


tsp = TSPGASolver(problem, generations=35000,
                  population_size=700, mutation_rate=0.20)
tsp.run()
input("Enter anything to exit")
