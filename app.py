from __future__ import annotations

import math
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class Package:
    id: int
    x: float
    y: float
    weight: int
    priority: int


@dataclass
class ProblemData:
    cooling_rate: float
    capacities: List[int]
    packages: List[Package]
    depot: Tuple[float, float] = (0.0, 0.0)


@dataclass
class Metrics:
    distance: float
    priority_penalty: float
    capacity_violations: int
    cost: float


# ============================================================
# Parsing / Validation
# ============================================================

def parse_input_file(path: str) -> ProblemData:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 3:
        raise ValueError(
            'Input file must contain at least 3 non-empty lines:\n'
            '1) cooling rate\n2) vehicle capacities\n3+) package lines: x y weight priority'
        )

    try:
        cooling_rate = float(lines[0])
    except ValueError as exc:
        raise ValueError('First line must be a floating-point cooling rate, e.g. 0.95') from exc

    if not (0.80 <= cooling_rate < 1.0):
        raise ValueError('Cooling rate should usually be between 0.80 and 0.999.')

    try:
        capacities = [int(x) for x in lines[1].split()]
    except ValueError as exc:
        raise ValueError('Second line must contain integer vehicle capacities.') from exc

    if not capacities:
        raise ValueError('At least one vehicle capacity is required.')
    if any(c <= 0 for c in capacities):
        raise ValueError('Vehicle capacities must all be positive integers.')

    packages: List[Package] = []
    for i, line in enumerate(lines[2:]):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f'Package line {i + 3} is invalid. Expected format: x y weight priority'
            )
        try:
            x, y = float(parts[0]), float(parts[1])
            weight, priority = int(parts[2]), int(parts[3])
        except ValueError as exc:
            raise ValueError(
                f'Package line {i + 3} must contain numbers in format: x y weight priority'
            ) from exc

        if weight <= 0:
            raise ValueError(f'Package {i} has non-positive weight.')
        if priority < 0:
            raise ValueError(f'Package {i} has negative priority.')
        if weight > max(capacities):
            raise ValueError(
                f'Package {i} weighs {weight}, which exceeds every vehicle capacity.'
            )

        packages.append(Package(i, x, y, weight, priority))

    return ProblemData(cooling_rate=cooling_rate, capacities=capacities, packages=packages)


# ============================================================
# Shared Evaluation Helpers
# ============================================================

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def package_map(packages: List[Package]) -> Dict[int, Package]:
    return {p.id: p for p in packages}


def split_into_capacity_trips(route: List[int], capacity: int, pkg_map: Dict[int, Package]) -> List[List[int]]:
    """
    Split a vehicle route into feasible trips. Whenever the next package does not fit
    in the remaining capacity, the vehicle returns to the depot and starts a new trip.
    """
    trips: List[List[int]] = []
    current_trip: List[int] = []
    load = 0

    for pid in route:
        w = pkg_map[pid].weight
        if current_trip and load + w > capacity:
            trips.append(current_trip)
            current_trip = [pid]
            load = w
        else:
            current_trip.append(pid)
            load += w

    if current_trip:
        trips.append(current_trip)

    return trips


def route_distance(route: List[int], capacity: int, pkg_map: Dict[int, Package], depot: Tuple[float, float]) -> float:
    total = 0.0
    for trip in split_into_capacity_trips(route, capacity, pkg_map):
        current = depot
        for pid in trip:
            p = pkg_map[pid]
            nxt = (p.x, p.y)
            total += euclidean(current, nxt)
            current = nxt
        total += euclidean(current, depot)
    return total


def route_priority_penalty(route: List[int], capacity: int, pkg_map: Dict[int, Package]) -> float:
    penalty = 0.0
    delivery_rank = 1
    for trip in split_into_capacity_trips(route, capacity, pkg_map):
        for pid in trip:
            # Higher priority packages should appear earlier, so later delivery costs more.
            penalty += delivery_rank * pkg_map[pid].priority
            delivery_rank += 1
    return penalty


def evaluate_solution(
    solution: List[List[int]],
    capacities: List[int],
    packages: List[Package],
    depot: Tuple[float, float] = (0.0, 0.0),
    priority_weight: float = 0.35,
) -> Metrics:
    pkg_map = package_map(packages)
    total_distance = 0.0
    total_penalty = 0.0
    violations = 0

    seen = set()
    for v, route in enumerate(solution):
        for pid in route:
            if pid in seen:
                violations += 1
            seen.add(pid)
        total_distance += route_distance(route, capacities[v], pkg_map, depot)
        total_penalty += route_priority_penalty(route, capacities[v], pkg_map)

    if len(seen) != len(packages):
        violations += abs(len(packages) - len(seen))

    hard_penalty = 10000.0 * violations
    cost = total_distance + priority_weight * total_penalty + hard_penalty
    return Metrics(
        distance=total_distance,
        priority_penalty=total_penalty,
        capacity_violations=violations,
        cost=cost,
    )


# ============================================================
# GA Representation
# ============================================================

@dataclass
class GAIndividual:
    assignment: List[int]      # vehicle index for each package id
    sequence_key: List[float]  # ordering key for each package id


def decode_individual(ind: GAIndividual, capacities: List[int], packages: List[Package]) -> List[List[int]]:
    routes = [[] for _ in capacities]
    priority_lookup = {p.id: p.priority for p in packages}

    for pid in range(len(packages)):
        routes[ind.assignment[pid]].append(pid)

    for route in routes:
        route.sort(key=lambda pid: (ind.sequence_key[pid], -priority_lookup[pid]))

    return routes


def random_individual(packages: List[Package], capacities: List[int]) -> GAIndividual:
    assignment = []
    sequence_key = []
    max_cap = max(capacities)
    num_vehicles = len(capacities)

    for p in packages:
        valid = [v for v, cap in enumerate(capacities) if p.weight <= cap]
        if not valid:
            valid = list(range(num_vehicles))
        assignment.append(random.choice(valid))
        sequence_key.append(random.random())

    return GAIndividual(assignment, sequence_key)


def clone_individual(ind: GAIndividual) -> GAIndividual:
    return GAIndividual(ind.assignment[:], ind.sequence_key[:])


def tournament_select(population: List[GAIndividual], scores: List[float], k: int = 4) -> GAIndividual:
    candidates = random.sample(range(len(population)), k=min(k, len(population)))
    best_idx = min(candidates, key=lambda idx: scores[idx])
    return clone_individual(population[best_idx])


def crossover_ga(a: GAIndividual, b: GAIndividual) -> Tuple[GAIndividual, GAIndividual]:
    n = len(a.assignment)
    if n < 2:
        return clone_individual(a), clone_individual(b)

    cut1 = random.randint(0, n - 1)
    cut2 = random.randint(cut1, n - 1)

    child1_assign = a.assignment[:]
    child2_assign = b.assignment[:]
    child1_keys = a.sequence_key[:]
    child2_keys = b.sequence_key[:]

    child1_assign[cut1:cut2 + 1] = b.assignment[cut1:cut2 + 1]
    child2_assign[cut1:cut2 + 1] = a.assignment[cut1:cut2 + 1]
    child1_keys[cut1:cut2 + 1] = b.sequence_key[cut1:cut2 + 1]
    child2_keys[cut1:cut2 + 1] = a.sequence_key[cut1:cut2 + 1]

    return GAIndividual(child1_assign, child1_keys), GAIndividual(child2_assign, child2_keys)


def mutate_ga(ind: GAIndividual, packages: List[Package], capacities: List[int], mutation_rate: float) -> GAIndividual:
    num_vehicles = len(capacities)
    for pid, p in enumerate(packages):
        if random.random() < mutation_rate:
            valid = [v for v, cap in enumerate(capacities) if p.weight <= cap]
            ind.assignment[pid] = random.choice(valid)
        if random.random() < mutation_rate:
            ind.sequence_key[pid] = min(1.0, max(0.0, ind.sequence_key[pid] + random.uniform(-0.25, 0.25)))
    return ind


def repair_ga(ind: GAIndividual, packages: List[Package], capacities: List[int]) -> GAIndividual:
    for pid, p in enumerate(packages):
        if p.weight > capacities[ind.assignment[pid]]:
            valid = [v for v, cap in enumerate(capacities) if p.weight <= cap]
            ind.assignment[pid] = random.choice(valid)
    return ind


def local_improve_routes(solution: List[List[int]], capacities: List[int], packages: List[Package], depot=(0.0, 0.0)) -> List[List[int]]:
    improved = [route[:] for route in solution]
    pkg_map = package_map(packages)

    # Simple nearest-neighbor reorder per vehicle, biased by priority.
    for v, route in enumerate(improved):
        if len(route) <= 2:
            continue
        remaining = set(route)
        ordered: List[int] = []
        current = depot
        while remaining:
            nxt = min(
                remaining,
                key=lambda pid: euclidean(current, (pkg_map[pid].x, pkg_map[pid].y)) - 0.12 * pkg_map[pid].priority,
            )
            ordered.append(nxt)
            current = (pkg_map[nxt].x, pkg_map[nxt].y)
            remaining.remove(nxt)
        improved[v] = ordered
    return improved


def genetic_algorithm(
    packages: List[Package],
    capacities: List[int],
    depot=(0.0, 0.0),
    population_size: int = 80,
    generations: int = 250,
    mutation_rate: float = 0.08,
    priority_weight: float = 0.35,
    elite_count: int = 4,
    seed: Optional[int] = None,
    log_callback=None,
):
    if seed is not None:
        random.seed(seed)

    population = [random_individual(packages, capacities) for _ in range(population_size)]

    best_solution = None
    best_metrics = None
    history = []

    for gen in range(generations):
        decoded = [decode_individual(ind, capacities, packages) for ind in population]
        decoded = [local_improve_routes(sol, capacities, packages, depot) for sol in decoded]
        metrics_list = [evaluate_solution(sol, capacities, packages, depot, priority_weight) for sol in decoded]
        scores = [m.cost for m in metrics_list]

        ranked = sorted(range(len(population)), key=lambda i: scores[i])
        elite = [clone_individual(population[i]) for i in ranked[:elite_count]]

        if best_metrics is None or scores[ranked[0]] < best_metrics.cost:
            best_solution = [route[:] for route in decoded[ranked[0]]]
            best_metrics = metrics_list[ranked[0]]

        history.append(best_metrics.cost)
        if log_callback and gen % 10 == 0:
            log_callback(f'GA generation {gen:>3}: best cost = {best_metrics.cost:.2f}')

        new_population = elite[:]
        while len(new_population) < population_size:
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            c1, c2 = crossover_ga(p1, p2)
            c1 = repair_ga(mutate_ga(c1, packages, capacities, mutation_rate), packages, capacities)
            c2 = repair_ga(mutate_ga(c2, packages, capacities, mutation_rate), packages, capacities)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population
        mutation_rate = max(0.02, mutation_rate * 0.995)

    return best_solution, best_metrics, history


# ============================================================
# SA Representation
# ============================================================

def greedy_initial_solution(packages: List[Package], capacities: List[int]) -> List[List[int]]:
    # Priority desc first, then heavier packages.
    ordered = sorted(packages, key=lambda p: (-p.priority, -p.weight, p.id))
    routes = [[] for _ in capacities]
    current_trip_loads = [0 for _ in capacities]

    for p in ordered:
        # Try vehicle with best immediate fit.
        candidates = []
        for v, cap in enumerate(capacities):
            if current_trip_loads[v] + p.weight <= cap:
                remaining = cap - (current_trip_loads[v] + p.weight)
                candidates.append((remaining, -p.priority, v))
        if candidates:
            _, _, best_v = min(candidates)
            routes[best_v].append(p.id)
            current_trip_loads[best_v] += p.weight
        else:
            # Start a new trip on the vehicle that can carry it with the smallest capacity waste.
            feasible = [(cap - p.weight, v) for v, cap in enumerate(capacities) if p.weight <= cap]
            _, best_v = min(feasible)
            routes[best_v].append(p.id)
            current_trip_loads[best_v] = p.weight

    return routes


def random_neighbor(solution: List[List[int]], capacities: List[int], packages: List[Package]) -> List[List[int]]:
    pkg_lookup = package_map(packages)
    neighbor = [route[:] for route in solution]
    move_type = random.choice(['swap_between', 'move_between', 'reinsert_same', 'reverse_segment'])

    non_empty = [i for i, route in enumerate(neighbor) if route]
    if not non_empty:
        return neighbor

    if move_type == 'swap_between' and len(non_empty) >= 2:
        a, b = random.sample(non_empty, 2)
        ia = random.randrange(len(neighbor[a]))
        ib = random.randrange(len(neighbor[b]))
        neighbor[a][ia], neighbor[b][ib] = neighbor[b][ib], neighbor[a][ia]
        return neighbor

    if move_type == 'move_between' and len(non_empty) >= 1 and len(neighbor) >= 2:
        src = random.choice(non_empty)
        dst_candidates = [i for i in range(len(neighbor)) if i != src]
        if dst_candidates:
            dst = random.choice(dst_candidates)
            idx = random.randrange(len(neighbor[src]))
            pid = neighbor[src].pop(idx)
            insert_at = random.randint(0, len(neighbor[dst]))
            neighbor[dst].insert(insert_at, pid)
            return neighbor

    if move_type == 'reinsert_same':
        v = random.choice(non_empty)
        if len(neighbor[v]) >= 2:
            i = random.randrange(len(neighbor[v]))
            pid = neighbor[v].pop(i)
            j = random.randint(0, len(neighbor[v]))
            neighbor[v].insert(j, pid)
        return neighbor

    if move_type == 'reverse_segment':
        v = random.choice(non_empty)
        if len(neighbor[v]) >= 3:
            i, j = sorted(random.sample(range(len(neighbor[v])), 2))
            neighbor[v][i:j + 1] = list(reversed(neighbor[v][i:j + 1]))
        return neighbor

    return neighbor


def simulated_annealing(
    packages: List[Package],
    capacities: List[int],
    depot=(0.0, 0.0),
    cooling_rate: float = 0.95,
    initial_temperature: float = 300.0,
    min_temperature: float = 0.5,
    steps_per_temp: int = 120,
    priority_weight: float = 0.35,
    seed: Optional[int] = None,
    log_callback=None,
):
    if seed is not None:
        random.seed(seed)

    current = greedy_initial_solution(packages, capacities)
    current = local_improve_routes(current, capacities, packages, depot)
    current_metrics = evaluate_solution(current, capacities, packages, depot, priority_weight)

    best = [route[:] for route in current]
    best_metrics = current_metrics
    history = [best_metrics.cost]

    T = initial_temperature
    temp_step = 0

    while T > min_temperature:
        for _ in range(steps_per_temp):
            neighbor = random_neighbor(current, capacities, packages)
            neighbor = local_improve_routes(neighbor, capacities, packages, depot)
            neighbor_metrics = evaluate_solution(neighbor, capacities, packages, depot, priority_weight)
            delta = neighbor_metrics.cost - current_metrics.cost

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                current = neighbor
                current_metrics = neighbor_metrics

                if current_metrics.cost < best_metrics.cost:
                    best = [route[:] for route in current]
                    best_metrics = current_metrics

        history.append(best_metrics.cost)
        if log_callback and temp_step % 5 == 0:
            log_callback(f'SA temp step {temp_step:>3}: T = {T:>7.3f}, best cost = {best_metrics.cost:.2f}')
        T *= cooling_rate
        temp_step += 1

    return best, best_metrics, history


# ============================================================
# UI Application
# ============================================================

class DeliveryOptimizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Advanced Delivery Route Optimizer')
        self.geometry('1400x860')
        self.minsize(1200, 760)

        self.problem: Optional[ProblemData] = None
        self.last_solution: Optional[List[List[int]]] = None
        self.last_metrics: Optional[Metrics] = None
        self.history: List[float] = []

        self._build_style()
        self._build_layout()

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('SubHeader.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Metric.TLabel', font=('Consolas', 11))

    def _build_layout(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill='both', expand=True)

        left = ttk.Frame(root)
        left.pack(side='left', fill='y', padx=(0, 10))

        right = ttk.Frame(root)
        right.pack(side='left', fill='both', expand=True)

        self._build_controls(left)
        self._build_right_panel(right)

    def _build_controls(self, parent):
        file_frame = ttk.LabelFrame(parent, text='Project Input', padding=10)
        file_frame.pack(fill='x', pady=(0, 10))

        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var, width=40).pack(fill='x', pady=(0, 6))
        ttk.Button(file_frame, text='Browse Input File', command=self.load_file).pack(fill='x')

        algo_frame = ttk.LabelFrame(parent, text='Algorithm Settings', padding=10)
        algo_frame.pack(fill='x', pady=(0, 10))

        self.algorithm_var = tk.StringVar(value='compare')
        ttk.Label(algo_frame, text='Mode').pack(anchor='w')
        ttk.Combobox(
            algo_frame,
            textvariable=self.algorithm_var,
            state='readonly',
            values=['compare', 'genetic algorithm', 'simulated annealing'],
        ).pack(fill='x', pady=(0, 8))

        self.priority_weight_var = tk.DoubleVar(value=0.35)
        ttk.Label(algo_frame, text='Priority weight').pack(anchor='w')
        ttk.Scale(algo_frame, from_=0.0, to=2.0, variable=self.priority_weight_var, orient='horizontal').pack(fill='x')
        self.priority_weight_label = ttk.Label(algo_frame, text='0.35')
        self.priority_weight_label.pack(anchor='e', pady=(0, 8))
        self.priority_weight_var.trace_add('write', lambda *_: self.priority_weight_label.config(text=f'{self.priority_weight_var.get():.2f}'))

        ga_frame = ttk.LabelFrame(parent, text='GA Parameters', padding=10)
        ga_frame.pack(fill='x', pady=(0, 10))
        self.ga_population_var = tk.IntVar(value=80)
        self.ga_generations_var = tk.IntVar(value=250)
        self.ga_mutation_var = tk.DoubleVar(value=0.08)
        self._labeled_entry(ga_frame, 'Population size', self.ga_population_var)
        self._labeled_entry(ga_frame, 'Generations', self.ga_generations_var)
        self._labeled_entry(ga_frame, 'Mutation rate', self.ga_mutation_var)

        sa_frame = ttk.LabelFrame(parent, text='SA Parameters', padding=10)
        sa_frame.pack(fill='x', pady=(0, 10))
        self.sa_temp_var = tk.DoubleVar(value=300.0)
        self.sa_steps_var = tk.IntVar(value=120)
        self.sa_cooling_var = tk.DoubleVar(value=0.95)
        self._labeled_entry(sa_frame, 'Initial temperature', self.sa_temp_var)
        self._labeled_entry(sa_frame, 'Steps / temperature', self.sa_steps_var)
        self._labeled_entry(sa_frame, 'Cooling rate', self.sa_cooling_var)

        action_frame = ttk.Frame(parent)
        action_frame.pack(fill='x', pady=(0, 10))
        ttk.Button(action_frame, text='Run Optimizer', command=self.run_optimizer).pack(fill='x', pady=(0, 5))
        ttk.Button(action_frame, text='Clear Log', command=lambda: self.log_text.delete('1.0', 'end')).pack(fill='x', pady=(0, 5))
        ttk.Button(action_frame, text='Export Results', command=self.export_results).pack(fill='x')

        summary_frame = ttk.LabelFrame(parent, text='Quick Summary', padding=10)
        summary_frame.pack(fill='both', expand=True)

        self.summary_vars = {
            'algorithm': tk.StringVar(value='—'),
            'distance': tk.StringVar(value='—'),
            'penalty': tk.StringVar(value='—'),
            'violations': tk.StringVar(value='—'),
            'cost': tk.StringVar(value='—'),
        }
        for key, label in [
            ('algorithm', 'Selected result'),
            ('distance', 'Total distance'),
            ('penalty', 'Priority penalty'),
            ('violations', 'Assignment violations'),
            ('cost', 'Total cost'),
        ]:
            row = ttk.Frame(summary_frame)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=f'{label}:', style='SubHeader.TLabel').pack(side='left')
            ttk.Label(row, textvariable=self.summary_vars[key], style='Metric.TLabel').pack(side='right')

    def _labeled_entry(self, parent, label: str, variable):
        ttk.Label(parent, text=label).pack(anchor='w')
        ttk.Entry(parent, textvariable=variable).pack(fill='x', pady=(0, 6))

    def _build_right_panel(self, parent):
        ttk.Label(parent, text='Advanced Delivery Route Optimizer', style='Header.TLabel').pack(anchor='w', pady=(0, 8))

        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)

        result_tab = ttk.Frame(notebook, padding=8)
        plot_tab = ttk.Frame(notebook, padding=8)
        convergence_tab = ttk.Frame(notebook, padding=8)
        log_tab = ttk.Frame(notebook, padding=8)

        notebook.add(result_tab, text='Results Table')
        notebook.add(plot_tab, text='Route Plot')
        notebook.add(convergence_tab, text='Convergence')
        notebook.add(log_tab, text='Execution Log')

        self.results_tree = ttk.Treeview(
            result_tab,
            columns=('vehicle', 'packages', 'trip_count', 'route_distance'),
            show='headings',
            height=18,
        )
        for col, text, width in [
            ('vehicle', 'Vehicle', 90),
            ('packages', 'Ordered packages', 650),
            ('trip_count', 'Trips', 80),
            ('route_distance', 'Distance', 120),
        ]:
            self.results_tree.heading(col, text=text)
            self.results_tree.column(col, width=width, anchor='center')
        self.results_tree.pack(fill='both', expand=True)

        self.route_fig = Figure(figsize=(8, 5), dpi=100)
        self.route_ax = self.route_fig.add_subplot(111)
        self.route_canvas = FigureCanvasTkAgg(self.route_fig, master=plot_tab)
        self.route_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.conv_fig = Figure(figsize=(8, 5), dpi=100)
        self.conv_ax = self.conv_fig.add_subplot(111)
        self.conv_canvas = FigureCanvasTkAgg(self.conv_fig, master=convergence_tab)
        self.conv_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.log_text = tk.Text(log_tab, wrap='word', font=('Consolas', 10))
        self.log_text.pack(fill='both', expand=True)

    def log(self, message: str):
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')
        self.update_idletasks()

    def load_file(self):
        path = filedialog.askopenfilename(
            title='Select input file',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
        )
        if not path:
            return
        self.file_var.set(path)
        try:
            self.problem = parse_input_file(path)
            self.log(f'Loaded file: {path}')
            self.log(f'Packages: {len(self.problem.packages)}, Vehicles: {len(self.problem.capacities)}')
            messagebox.showinfo('Success', 'Input file loaded and validated successfully.')
        except Exception as exc:
            self.problem = None
            messagebox.showerror('Input Error', str(exc))

    def run_optimizer(self):
        if not self.file_var.get():
            messagebox.showwarning('Missing file', 'Please load an input file first.')
            return

        try:
            self.problem = parse_input_file(self.file_var.get())
        except Exception as exc:
            messagebox.showerror('Input Error', str(exc))
            return

        mode = self.algorithm_var.get().strip().lower()
        priority_weight = float(self.priority_weight_var.get())
        self.log('=' * 70)
        self.log(f'Run mode: {mode}')

        try:
            ga_result = sa_result = None

            if mode in ('compare', 'genetic algorithm'):
                self.log('Starting Genetic Algorithm ...')
                ga_solution, ga_metrics, ga_history = genetic_algorithm(
                    self.problem.packages,
                    self.problem.capacities,
                    depot=self.problem.depot,
                    population_size=int(self.ga_population_var.get()),
                    generations=int(self.ga_generations_var.get()),
                    mutation_rate=float(self.ga_mutation_var.get()),
                    priority_weight=priority_weight,
                    log_callback=self.log,
                )
                ga_result = ('Genetic Algorithm', ga_solution, ga_metrics, ga_history)
                self.log(f'GA finished. Best cost = {ga_metrics.cost:.2f}')

            if mode in ('compare', 'simulated annealing'):
                self.log('Starting Simulated Annealing ...')
                sa_solution, sa_metrics, sa_history = simulated_annealing(
                    self.problem.packages,
                    self.problem.capacities,
                    depot=self.problem.depot,
                    cooling_rate=float(self.sa_cooling_var.get()),
                    initial_temperature=float(self.sa_temp_var.get()),
                    steps_per_temp=int(self.sa_steps_var.get()),
                    priority_weight=priority_weight,
                    log_callback=self.log,
                )
                sa_result = ('Simulated Annealing', sa_solution, sa_metrics, sa_history)
                self.log(f'SA finished. Best cost = {sa_metrics.cost:.2f}')

            chosen = None
            if mode == 'compare':
                candidates = [r for r in (ga_result, sa_result) if r is not None]
                chosen = min(candidates, key=lambda item: item[2].cost)
                self.log(f'Compare mode selected: {chosen[0]} won with cost {chosen[2].cost:.2f}')
            else:
                chosen = ga_result or sa_result

            if chosen is None:
                raise RuntimeError('No algorithm result was produced.')

            algo_name, solution, metrics, history = chosen
            self.last_solution = solution
            self.last_metrics = metrics
            self.history = history

            self._update_summary(algo_name, metrics)
            self._populate_results_tree(solution)
            self._draw_routes(solution, algo_name)
            self._draw_convergence(history, algo_name)
            messagebox.showinfo('Done', f'{algo_name} completed successfully.')

        except Exception as exc:
            messagebox.showerror('Runtime Error', str(exc))
            self.log(f'ERROR: {exc}')

    def _update_summary(self, algorithm_name: str, metrics: Metrics):
        self.summary_vars['algorithm'].set(algorithm_name)
        self.summary_vars['distance'].set(f'{metrics.distance:.2f}')
        self.summary_vars['penalty'].set(f'{metrics.priority_penalty:.2f}')
        self.summary_vars['violations'].set(str(metrics.capacity_violations))
        self.summary_vars['cost'].set(f'{metrics.cost:.2f}')

    def _populate_results_tree(self, solution: List[List[int]]):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        pkg_map = package_map(self.problem.packages)
        for v, route in enumerate(solution):
            trips = split_into_capacity_trips(route, self.problem.capacities[v], pkg_map)
            dist = route_distance(route, self.problem.capacities[v], pkg_map, self.problem.depot)
            display_route = ' | '.join(str(pid) for pid in route) if route else '—'
            self.results_tree.insert('', 'end', values=(f'Vehicle {v + 1}', display_route, len(trips), f'{dist:.2f}'))

    def _draw_routes(self, solution: List[List[int]], algorithm_name: str):
        self.route_ax.clear()
        pkg_map = package_map(self.problem.packages)
        depot = self.problem.depot
        marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

        self.route_ax.scatter([depot[0]], [depot[1]], marker='*', s=220, label='Depot')

        for v, route in enumerate(solution):
            if not route:
                continue
            trips = split_into_capacity_trips(route, self.problem.capacities[v], pkg_map)
            for trip_index, trip in enumerate(trips):
                xs = [depot[0]]
                ys = [depot[1]]
                for pid in trip:
                    p = pkg_map[pid]
                    xs.append(p.x)
                    ys.append(p.y)
                xs.append(depot[0])
                ys.append(depot[1])
                self.route_ax.plot(xs, ys, marker=marker_cycle[v % len(marker_cycle)], label=f'V{v + 1} Trip {trip_index + 1}')

                for pid in trip:
                    p = pkg_map[pid]
                    self.route_ax.annotate(f'{pid}\nP{p.priority}', (p.x, p.y), fontsize=8)

        self.route_ax.set_title(f'Optimized Routes - {algorithm_name}')
        self.route_ax.set_xlabel('X')
        self.route_ax.set_ylabel('Y')
        self.route_ax.grid(True)
        self.route_ax.legend(loc='best', fontsize=8)
        self.route_canvas.draw()

    def _draw_convergence(self, history: List[float], algorithm_name: str):
        self.conv_ax.clear()
        if history:
            self.conv_ax.plot(range(1, len(history) + 1), history)
        self.conv_ax.set_title(f'Convergence - {algorithm_name}')
        self.conv_ax.set_xlabel('Iteration / Temperature Step')
        self.conv_ax.set_ylabel('Best Cost')
        self.conv_ax.grid(True)
        self.conv_canvas.draw()

    def export_results(self):
        if self.last_solution is None or self.last_metrics is None or self.problem is None:
            messagebox.showwarning('No results', 'Run the optimizer first.')
            return

        path = filedialog.asksaveasfilename(
            title='Export results',
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt')],
        )
        if not path:
            return

        pkg_map = package_map(self.problem.packages)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('Advanced Delivery Route Optimizer Results\n')
            f.write('=' * 50 + '\n')
            f.write(f'Total distance: {self.last_metrics.distance:.2f}\n')
            f.write(f'Priority penalty: {self.last_metrics.priority_penalty:.2f}\n')
            f.write(f'Violations: {self.last_metrics.capacity_violations}\n')
            f.write(f'Total cost: {self.last_metrics.cost:.2f}\n\n')
            for v, route in enumerate(self.last_solution):
                trips = split_into_capacity_trips(route, self.problem.capacities[v], pkg_map)
                f.write(f'Vehicle {v + 1}: {route}\n')
                f.write(f'  Trips: {trips}\n')
                f.write(
                    f'  Distance: {route_distance(route, self.problem.capacities[v], pkg_map, self.problem.depot):.2f}\n\n'
                )

        self.log(f'Results exported to {path}')
        messagebox.showinfo('Exported', 'Results exported successfully.')


if __name__ == '__main__':
    app = DeliveryOptimizerApp()
    app.mainloop()
