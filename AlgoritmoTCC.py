from math import sqrt
from random import shuffle, randint
import random
import argparse
import os
import psutil
import time
import copy
import numpy.random as npr


def same_column_indexes(problem_grid, i, j, N, itself=True):

    sub_grid_column = i % N
    cell_column = j % N

    for a in range(sub_grid_column, len(problem_grid), N):
        for b in range(cell_column, len(problem_grid), N):
            if (a, b) == (i, j) and not itself:
                continue

            yield (a, b)


def same_row_indexes(problem_grid, i, j, N, itself=True):

    sub_grid_row = int(i / N)
    cell_row = int(j / N)

    for a in range(sub_grid_row * N, sub_grid_row * N + N):
        for b in range(cell_row * N, cell_row * N + N):
            if (a, b) == (i, j) and not itself:
                continue

            yield (a, b)


def get_cells_from_indexes(grid, indexes):

    for a, b in indexes:
        yield grid[a][b]


def solve(problem_grid, population_size=1000, selection_rate=0.5, max_generations_count=1000, mutation_rate=0.05):

    N = int(sqrt(len(problem_grid)))

    def empty_grid(elem_generator=None):

        return [
            [
                (None if elem_generator is None else elem_generator(i, j))
                for j in range(len(problem_grid))
            ] for i in range(len(problem_grid))
        ]

    def deep_copy_grid(grid):

        return empty_grid(lambda i, j: grid[i][j])
    problem_grid = deep_copy_grid(problem_grid)

    def same_sub_grid_indexes(i, j, itself=True):

        for k in range(len(problem_grid)):
            if k == j and not itself:
                continue

            yield (i, k)

    def fill_predetermined_cells():

        track_grid = empty_grid(lambda *args: [val for val in range(1, len(problem_grid) + 1)])
        
        def pencil_mark(i, j):
            for a, b in same_sub_grid_indexes(i, j, itself=False):
                try:
                    track_grid[a][b].remove(problem_grid[i][j])
                except (ValueError, AttributeError) as e:
                    pass

            for a, b in same_row_indexes(problem_grid, i, j, N, itself=False):
                try:
                    track_grid[a][b].remove(problem_grid[i][j])
                except (ValueError, AttributeError) as e:
                    pass

            for a, b in same_column_indexes(problem_grid, i, j, N, itself=False):
                try:
                    track_grid[a][b].remove(problem_grid[i][j])
                except (ValueError, AttributeError) as e:
                    pass

        for i in range(len(problem_grid)):
            for j in range(len(problem_grid)):
                if problem_grid[i][j] is not None:
                    pencil_mark(i, j)

        while True:
            anything_changed = False

            for i in range(len(problem_grid)):
                for j in range(len(problem_grid)):
                    if track_grid[i][j] is None:
                        continue

                    if len(track_grid[i][j]) == 0:
                        raise Exception('The puzzle is not solvable.')
                    elif len(track_grid[i][j]) == 1:
                        problem_grid[i][j] = track_grid[i][j][0]
                        pencil_mark(i, j)

                        track_grid[i][j] = None

                        anything_changed = True

            if not anything_changed:
                break

        return problem_grid

    def generate_initial_population():
        
        candidates = []
        for k in range(population_size):
            candidate = empty_grid()
            for i in range(len(problem_grid)):
                shuffled_sub_grid = [n for n in range(1, len(problem_grid) + 1)]
                shuffle(shuffled_sub_grid)

                for j in range(len(problem_grid)):
                    if problem_grid[i][j] is not None:
                        candidate[i][j] = problem_grid[i][j]
                        
                        shuffled_sub_grid.remove(problem_grid[i][j])
                for j in range(len(problem_grid)):
                    if candidate[i][j] is None:
                        candidate[i][j] = shuffled_sub_grid.pop()
            candidates.append(candidate)
        return candidates

    def fitness(grid):
        row_duplicates_count = 0
        for i in range (len(grid)):
            row_duplicates_count += len(grid[i]) - len(set(grid[i]))
        columns = []
        for a, b in same_column_indexes(problem_grid, 0, 0, N):
            row = list(get_cells_from_indexes(grid, same_row_indexes(problem_grid, a, b, N)))
            columns.append(row)
            #print(len(row) - len(set(row)))
            row_duplicates_count += len(row) - len(set(row))
        columns = list(map(list, zip(*columns)))
        
        for i in range (len(columns)):
            row_duplicates_count += len(columns[i]) - len(set(columns[i]))
        
        return row_duplicates_count
    
    def probability(prob):
        
        sum_probs = sum(map(float,prob))
        probability = [e/sum_probs for e in prob]
        
        return probability
        
    def selection(candidates):

        index_fitness = []
        for i in range(len(candidates)):
            index_fitness.append(tuple([i, fitness(candidates[i])]))

        index_fitness.sort(key=lambda elem: elem[1])

        selected_part = index_fitness[0: int(len(index_fitness))]
        indexes   = [e[0] for e in selected_part]
        fitnesses = [e[1] for e in selected_part]
        sum_fitness = sum(e[1] for e in selected_part)
        
        if sum_fitness == 0:
            return [candidates[i] for i in indexes], selected_part[0][1], 0
        
        try:
            chance = [
                (1/(e[1]))
                /sum_fitness for e in selected_part]
        except ZeroDivisionError:
            chance = [
                (1/(1))
                /sum_fitness for e in selected_part]
        
        prob = probability(chance)
        
        return [candidates[i] for i in indexes], selected_part[0][1], prob
    

    
    fill_predetermined_cells()

    population = generate_initial_population()
    best_fitness = None
    
    for generation in range(max_generations_count):
        population, best_fitness, prob = selection(population)
        
        if generation == max_generations_count - 1 or fitness(population[0]) == 0:
            break
        
        new_population = []
        count = 0
        
        # CROSSOVER
        
        while True:
            solution_1, solution_2 = None, None
            if(len(population) <= 1):
                break
            
            try:
                num = npr.choice(len(population),p= prob)
                
                solution_1 = population.pop(num)
                prob.pop(num)
                prob = probability(prob)
                
            except IndexError:
                break

            try:
                num = npr.choice(len(population),p= prob)
                solution_2 = population.pop()
                prob.pop(num)
                prob = probability(prob)
                
            except IndexError:
                break
            
            luck_number = random.random()
            if(luck_number >= (1 - selection_rate)):
                cross_point = randint(0, len(problem_grid) - 2)
                temp_sub_grid = solution_1[cross_point]
                solution_1[cross_point] = solution_2[cross_point + 1]
                solution_2[cross_point + 1] = temp_sub_grid
            
            new_population.append(solution_1)
            new_population.append(solution_2)
            
        # MUTATION
        
        for candidate in new_population[0:len(new_population)]:
            for sub_grid_index in candidate[0:len(candidate)]:
                for i, val in enumerate(sub_grid_index):
                    luck_number = random.random()
                    if(luck_number <= mutation_rate):
                        for j, val2 in enumerate(sub_grid_index):
                            tmp = copy.copy(candidate)
                            if(i == j):
                                continue
                            fittmp = fitness(tmp)
                            sub_grid_index[i], sub_grid_index[j] = sub_grid_index[j], sub_grid_index[i]
                            fitcand = fitness(candidate)
                            if(fittmp < fitcand):
                                sub_grid_index[i], sub_grid_index[j] = sub_grid_index[j], sub_grid_index[i]
        population.extend(new_population)

        
    return generation, population[0], best_fitness


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input file that contains Sudoku's problem.", default="Facil.txt")
    parser.add_argument("-o", "--output-file", help="Output file to store problem's solution.", type=str, default="output.csv")
    parser.add_argument("-p", "--population-size", type=int, default=4)
    parser.add_argument("-s", "--selection_rate", type=float, default=0.3)
    parser.add_argument("-m", "--max-generations-count", type=int, default=100)
    parser.add_argument("-u", "--mutation-rate", type=float, default=1)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    num = 1
    contador = 1

    start_time = time.time()
    try:
        with open(args.file, "r") as input_file:
            file_content = input_file.read()
            file_lines = file_content.split('\n')
            problem_grid = [[] for i in range(len(file_lines))]
            sqrt_n = int(sqrt(len(file_lines)))
            for j in range(len(file_lines)):
                line_values = [(int(value) if value != '-' else None) for value in
                            file_lines[j].split(' ')]
                for i in range(len(line_values)):
                    problem_grid[
                        int(i / sqrt_n) +
                        int(j / sqrt_n) * sqrt_n
                        ].append(line_values[i])
            try:
                generation, solution, best_fitness = solve(
                    problem_grid,
                    population_size=args.population_size,
                    selection_rate=args.selection_rate,
                    max_generations_count=args.max_generations_count,
                    mutation_rate=args.mutation_rate
                )
                
                print('Gerações utilizadas: ')
                print(generation)
                test = "Resultado encontrado \n"
                for a, b in same_column_indexes(solution, 0, 0, sqrt_n):
                    row = list(get_cells_from_indexes(solution, same_row_indexes(solution, a, b, sqrt_n)))
                
                    test += " ".join([str(elem) for elem in row]) + '\n'
                    
                print(test[:-1])
                
                cpu_usage = (psutil.cpu_percent() / 100)
                memory_usage = (psutil.virtual_memory()[3] / (1024 * 1024 * 1024))
                elapsed_time = (time.time() - start_time)
                # POPULAÇÃO, TAXA DE SELEÇÃO, TAXA DE MUTAÇÃO, MELHOR FITNESS, USO DE CPU, USO DE MEMÓRIA, TEMPO USADO
                output_str = str(contador) + ";" +str(args.population_size) + ";" + str(args.selection_rate) + ";" + str(args.mutation_rate) + ";" + str(best_fitness) + ";" + str(cpu_usage) + ";" + str(memory_usage) + ";" + str(elapsed_time) +  ";" + str(generation) + ";" + 'H1' + "\n"
                output_str = output_str
                num += 1
                contador += 1
    
                if args.output_file:
                    with open(args.output_file, "a") as output_file:
                        output_file.write(output_str)
    
                if not args.quiet:
                    print(output_str[:-1])
    
            except Exception as e: 
                #print(e)
                cpu_usage = (psutil.cpu_percent() / 100)
                memory_usage = (psutil.virtual_memory()[3] / (1024 * 1024 * 1024))
                elapsed_time = (time.time() - start_time)
                # POPULAÇÃO, TAXA DE SELEÇÃO, TAXA DE MUTAÇÃO, MELHOR FITNESS, USO DE CPU, USO DE MEMÓRIA, TEMPO USADO
                output_str = str(contador) + ";" +str(args.population_size) + ";" + str(args.selection_rate) + ";" + str(args.mutation_rate) + ";" + '-1' + ";" + str(cpu_usage) + ";" + str(memory_usage) + ";" + str(elapsed_time) + str(generation) + ";" + ";" + 'H1' + "\n"
                output_str = output_str
                num += 1
                contador += 1
    
                if args.output_file:
                    with open(args.output_file, "a") as output_file:
                        output_file.write(output_str)
                print(str(contador) + ";" +str(args.population_size) + ";" + str(args.selection_rate) + ";" + str(args.mutation_rate) + ";" + '-1' + ";" + str(cpu_usage) + ";" + str(memory_usage) + ";" + str(elapsed_time) + ";" + str(generation) + ";" + 'H1' + "\n")
    
    except FileNotFoundError:
        exit("Input file not found.")
