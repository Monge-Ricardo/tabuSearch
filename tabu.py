import networkx as nx
import random
import copy
from collections import deque

def build_campus_graph():
    """
    Construye la topología del campus con edificios, tramos críticos
    (escaleras, puertas) y puntos de encuentro.
    """
    G = nx.DiGraph()

    # Capacidad (personas/min) y tiempo base (minutos)
    edges = [
        # Bloque A (Aulas y laboratorios)
        ('Bloque_A', 'Escalera_A', 60, 1.2),
        ('Escalera_A', 'Puerta_A', 40, 1.0),      # Cuello de botella crítico
        ('Puerta_A', 'Cancha_Futbol', 120, 2.0),
        ('Puerta_A', 'Punto_Central', 80, 1.8),

        # Bloque B (Administrativo y aulas)
        ('Bloque_B', 'Escalera_B', 50, 1.0),
        ('Escalera_B', 'Puerta_B', 35, 1.2),      # Cuello de botella
        ('Puerta_B', 'Punto_Central', 90, 1.5),
        ('Puerta_B', 'Plaza_Cívica', 70, 2.2),

        # Bloque C (Biblioteca y auditorio)
        ('Bloque_C', 'Escalera_C', 70, 1.1),
        ('Escalera_C', 'Puerta_C', 45, 1.0),
        ('Puerta_C', 'Plaza_Cívica', 100, 1.4),
        ('Puerta_C', 'Cancha_Futbol', 60, 2.5),

        # Conexiones intermedias entre patios
        ('Punto_Central', 'Cancha_Futbol', 150, 1.5),
        ('Punto_Central', 'Plaza_Cívica', 150, 1.5),
    ]

    for u, v, cap, base_time in edges:
        G.add_edge(u, v, capacity=cap, base_time=base_time)

    return G

# Población a evacuar por origen
SOURCE_POPULATION = {
    'Bloque_A': 1200,
    'Bloque_B': 800,
    'Bloque_C': 1500
}

SAFE_ZONES = {'Cancha_Futbol', 'Plaza_Cívica'}

def get_candidate_paths(G, sources, targets, max_paths=4):
    """Genera las K rutas más cortas sin congestión para cada origen."""
    candidate_paths = {}
    for s in sources:
        candidate_paths[s] = []
        for t in targets:
            if nx.has_path(G, s, t):
                paths = list(nx.shortest_simple_paths(G, s, t, weight='base_time'))
                candidate_paths[s].extend(paths[:max_paths])
    return candidate_paths

def evaluate_solution(solution, G, populations, alpha=2.0, beta=2.0):
    """
    Calcula el tiempo total de despeje y las penalizaciones por saturación
    usando una función tipo Bureau of Public Roads (BPR).
    """
    edge_loads = {e: 0 for e in G.edges()}

    # 1. Asignar carga a cada arista según las rutas activas
    for source, path in solution.items():
        people = populations[source]
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            edge_loads[edge] += people

    # 2. Calcular tiempo efectivo de cada arista según congestión
    edge_effective_times = {}
    total_bottleneck_penalty = 0.0

    for edge in G.edges():
        u, v = edge
        cap = G[u][v]['capacity']
        base_t = G[u][v]['base_time']
        load = edge_loads[edge]

        # Penalización si la carga supera la tasa de evacuación estimada
        # Supongamos ventana de evacuación objetivo de 5 minutos
        flow_rate = load / 5.0
        congestion_ratio = flow_rate / cap

        effective_time = base_t * (1.0 + alpha * (congestion_ratio ** beta))
        edge_effective_times[edge] = effective_time

        if congestion_ratio > 1.0:
            total_bottleneck_penalty += (congestion_ratio - 1.0) * 100.0

    # 3. Calcular tiempo de evacuación por bloque y el makespan
    arrival_times = []
    for source, path in solution.items():
        path_time = sum(edge_effective_times[(path[i], path[i+1])] for i in range(len(path) - 1))
        arrival_times.append(path_time)

    makespan = max(arrival_times)
    cost = makespan + total_bottleneck_penalty

    return cost, makespan, edge_loads

def tabu_search_evacuation(G, sources, targets, candidate_paths, max_iter=80, tabu_size=10):
    # Generar solución inicial voraz (primera ruta de menor distancia)
    current_solution = {s: candidate_paths[s][0] for s in sources}
    best_solution = copy.deepcopy(current_solution)

    best_cost, best_makespan, _ = evaluate_solution(best_solution, G, SOURCE_POPULATION)
    current_cost = best_cost

    # Memoria Tabú: registra movimientos (nodo_origen, indice_ruta)
    tabu_list = deque(maxlen=tabu_size)

    for iteration in range(max_iter):
        neighborhood = []

        # Generar vecindario: cambiar la ruta de un solo bloque a otra alternativa
        for source in sources:
            for idx, alt_path in enumerate(candidate_paths[source]):
                if alt_path != current_solution[source]:
                    move = (source, idx)
                    neighbor_sol = copy.deepcopy(current_solution)
                    neighbor_sol[source] = alt_path
                    cost, makespan, loads = evaluate_solution(neighbor_sol, G, SOURCE_POPULATION)
                    neighborhood.append((move, neighbor_sol, cost, makespan))

        if not neighborhood:
            break

        # Ordenar vecinos de menor a mayor costo
        neighborhood.sort(key=lambda x: x[2])

        # Selección con Lista Tabú y Criterio de Aspiración
        selected_move = None
        selected_sol = None
        selected_cost = float('inf')
        selected_makespan = float('inf')

        for move, cand_sol, cand_cost, cand_makespan in neighborhood:
            # Criterio de aspiración: si mejora la mejor marca histórica, se ignora el tabú
            if cand_cost < best_cost:
                selected_move = move
                selected_sol = cand_sol
                selected_cost = cand_cost
                selected_makespan = cand_makespan
                break
            # Movimiento no tabú
            elif move not in tabu_list:
                selected_move = move
                selected_sol = cand_sol
                selected_cost = cand_cost
                selected_makespan = cand_makespan
                break

        if selected_sol is None:
            # Si todos son tabú y ninguno aspira, tomar el mejor disponible
            selected_move, selected_sol, selected_cost, selected_makespan = neighborhood[0]

        # Actualizar estado actual
        current_solution = selected_sol
        current_cost = selected_cost
        tabu_list.append(selected_move)

        # Actualizar óptimo global
        if current_cost < best_cost:
            best_cost = current_cost
            best_makespan = selected_makespan
            best_solution = copy.deepcopy(current_solution)

    return best_solution, best_cost, best_makespan

if __name__ == '__main__':
    campus = build_campus_graph()
    sources = list(SOURCE_POPULATION.keys())

    candidates = get_candidate_paths(campus, sources, SAFE_ZONES)

    print("Iniciando optimización por Búsqueda Tabú...")
    best_routes, best_score, best_time = tabu_search_evacuation(
        campus, sources, SAFE_ZONES, candidates, max_iter=60, tabu_size=8
    )

    print("\n--- Plan Óptimo de Evacuación Encontrado ---")
    for src, path in best_routes.items():
        print(f"[{src} - {SOURCE_POPULATION[src]} pers.] -> Ruta: {' -> '.join(path)}")
    print(f"\nTiempo estimado de despeje total: {best_time:.2f} minutos")
    print(f"Puntaje de función objetivo (costo + congestión): {best_score:.2f}")