"""
Pipeline de Optimización de Evacuación Masiva - ESPE Sangolquí
Metaheurística: Búsqueda Tabú con Memoria y Criterio de Aspiración
Fuente Topológica: export.geojson (Red vial peatonal del campus ESPE)
"""

import json
import math
import copy
from collections import deque
import networkx as nx

def haversine_distance_meters(p1, p2):
    """Calcula distancia euclidiana proyectada en metros para latitud ~ -0.31°."""
    lon1, lat1 = p1
    lon2, lat2 = p2
    dx = (lon2 - lon1) * 111320 * math.cos(math.radians(-0.31))
    dy = (lat2 - lat1) * 110574
    return math.hypot(dx, dy)

def build_graph_from_geojson(geojson_path):
    """
    Construye el grafo topológico bidireccional NetworkX a partir de export.geojson,
    restringiéndose a vías internas transitables (footway, service, path, track).
    Asigna capacidades (personas/minuto) y velocidad peatonal de evacuación (1.2 m/s).
    """
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.Graph()

    for feat in data['features']:
        geom = feat.get('geometry', {})
        props = feat.get('properties', {})
        hw = props.get('highway')
        name = props.get('name', 'Camino Interno')

        if geom.get('type') == 'LineString' and hw in ['footway', 'service', 'path', 'track']:
            coords = geom['coordinates']
            for i in range(len(coords) - 1):
                p1 = (round(coords[i][0], 6), round(coords[i][1], 6))
                p2 = (round(coords[i+1][0], 6), round(coords[i+1][1], 6))
                dist_m = haversine_distance_meters(p1, p2)

                # Capacidad nominal según tipo de vía
                if hw == 'service':
                    cap = 150.0   # Vía amplia vehicular/peatonal mixta
                elif hw == 'footway':
                    cap = 80.0    # Acera/sendero peatonal
                else:
                    cap = 50.0    # Sendero o área verde más estrecho

                # Tiempo base en minutos (Velocidad media de marcha en evacuación = 1.2 m/s -> 72 m/min)
                base_time_min = dist_m / 72.0 if dist_m > 0 else 0.01

                G.add_edge(
                    p1, p2,
                    length=dist_m,
                    capacity=cap,
                    base_time=base_time_min,
                    highway=hw,
                    name=name
                )

    # Filtrar al componente conectado principal del campus para asegurar alcanzabilidad
    comps = list(nx.connected_components(G))
    main_component = max(comps, key=len)
    campus_graph = G.subgraph(main_component).copy()
    return campus_graph

def find_nearest_node(graph, target_lon, target_lat):
    """Encuentra el nodo del grafo más cercano a las coordenadas geográficas dadas."""
    best_node = None
    min_dist = float('inf')
    for node in graph.nodes():
        d = haversine_distance_meters(node, (target_lon, target_lat))
        if d < min_dist:
            min_dist = d
            best_node = node
    return best_node, min_dist

def get_candidate_paths(G, sources, targets, k_paths=4):
    """Genera las K rutas más cortas sin congestión para cada edificio origen hacia las zonas seguras."""
    candidates = {}
    for s_name, s_node in sources.items():
        paths_list = []
        for t_name, t_node in targets.items():
            if nx.has_path(G, s_node, t_node):
                generator = nx.shortest_simple_paths(G, s_node, t_node, weight='base_time')
                count = 0
                for p in generator:
                    paths_list.append({
                        'target_name': t_name,
                        'path': p
                    })
                    count += 1
                    if count >= k_paths:
                        break
        # Ordenar por tiempo base libre de congestión
        paths_list.sort(key=lambda item: sum(
            G[item['path'][i]][item['path'][i+1]]['base_time'] for i in range(len(item['path']) - 1)
        ))
        candidates[s_name] = paths_list[:k_paths * len(targets)]
    return candidates

def evaluate_solution(solution, G, populations, alpha=2.0, beta=2.0, target_evac_window_min=10.0):
    """
    Función de evaluación de tráfico peatonal basada en el modelo Bureau of Public Roads (BPR).
    Calcula:
      1. Carga por arista.
      2. Factor de congestión: Flow / Capacidad.
      3. Penalización por saturación y cuello de botella.
      4. Makespan: tiempo máximo que le toma al último grupo llegar a zona segura.
    """
    edge_loads = {tuple(sorted(e)): 0 for e in G.edges()}

    # Acumular carga de personas
    for s_name, choice in solution.items():
        path = choice['path']
        people = populations[s_name]
        for i in range(len(path) - 1):
            e = tuple(sorted((path[i], path[i+1])))
            if e in edge_loads:
                edge_loads[e] += people

    # Calcular tiempos efectivos con congestión
    edge_effective_times = {}
    total_bottleneck_penalty = 0.0

    for u, v, data in G.edges(data=True):
        e = tuple(sorted((u, v)))
        cap = data['capacity']
        base_t = data['base_time']
        load = edge_loads.get(e, 0)

        # Flujo por minuto estimado
        flow_rate = load / target_evac_window_min
        congestion_ratio = flow_rate / cap

        effective_time = base_t * (1.0 + alpha * (congestion_ratio ** beta))
        edge_effective_times[e] = effective_time

        # Penalización si se sobrepasa la capacidad crítica
        if congestion_ratio > 1.0:
            total_bottleneck_penalty += ((congestion_ratio - 1.0) ** 2) * 200.0

    # Tiempo de evacuación por origen
    times_by_source = {}
    for s_name, choice in solution.items():
        path = choice['path']
        t_total = sum(edge_effective_times[tuple(sorted((path[i], path[i+1])))] for i in range(len(path) - 1))
        times_by_source[s_name] = t_total

    makespan = max(times_by_source.values()) if times_by_source else 0.0
    total_cost = makespan + total_bottleneck_penalty
    return total_cost, makespan, times_by_source, total_bottleneck_penalty

def tabu_search_evacuation(G, sources, targets, candidate_paths, populations, max_iter=100, tabu_size=12):
    """
    Algoritmo Metaheurístico de Búsqueda Tabú para asignación óptima de rutas de evacuación.
    """
    # 1. Solución Inicial Voraz: cada bloque toma su camino más corto estático
    current_solution = {s: candidate_paths[s][0] for s in sources}
    best_solution = copy.deepcopy(current_solution)

    best_cost, best_makespan, best_times, best_penalty = evaluate_solution(
        best_solution, G, populations
    )
    current_cost = best_cost

    # Memoria Tabú: registra tuplas (edificio, id_ruta_candidata)
    tabu_list = deque(maxlen=tabu_size)

    history = []
    history.append((0, best_cost, best_makespan))

    for iteration in range(1, max_iter + 1):
        neighborhood = []

        # Generar vecindario 1-opt: cambiar la ruta de un edificio a una alternativa
        for s_name in sources:
            for idx, cand in enumerate(candidate_paths[s_name]):
                if cand['path'] != current_solution[s_name]['path']:
                    move = (s_name, idx)
                    neighbor_sol = copy.deepcopy(current_solution)
                    neighbor_sol[s_name] = cand

                    cost, makespan, times_src, penalty = evaluate_solution(
                        neighbor_sol, G, populations
                    )
                    neighborhood.append((move, neighbor_sol, cost, makespan, times_src, penalty))

        if not neighborhood:
            break

        # Ordenar vecindario por costo de menor a mayor
        neighborhood.sort(key=lambda x: x[2])

        selected = None
        for cand_move, cand_sol, cand_cost, cand_makespan, cand_times, cand_penalty in neighborhood:
            # Criterio de aspiración: si mejora el óptimo global, ignorar tabú
            if cand_cost < best_cost:
                selected = (cand_move, cand_sol, cand_cost, cand_makespan, cand_times, cand_penalty)
                break
            # Movimiento estándar no prohibido
            elif cand_move not in tabu_list:
                selected = (cand_move, cand_sol, cand_cost, cand_makespan, cand_times, cand_penalty)
                break

        # Si todos los movimientos válidos están en tabú, tomar el mejor disponible
        if selected is None:
            selected = neighborhood[0]

        move, current_solution, current_cost, current_makespan, cur_times, cur_pen = selected
        tabu_list.append(move)

        # Actualización de la mejor solución global encontrada
        if current_cost < best_cost:
            best_cost = current_cost
            best_makespan = current_makespan
            best_solution = copy.deepcopy(current_solution)
            best_times = cur_times
            best_penalty = cur_pen

        history.append((iteration, best_cost, best_makespan))

    return best_solution, best_cost, best_makespan, best_times, best_penalty, history

def run_evacuation_pipeline(geojson_file='export.geojson'):
    print("=" * 70)
    print(" PIPELINE DE EVACUACIÓN MULTI-AGENTE (ESPE SANGOLQUÍ)")
    print(" Algoritmo: Búsqueda Tabú con Topología Geoespacial Real")
    print("=" * 70)

    # 1. Cargar y construir grafo desde export.geojson
    print(f"\n[1/4] Parseando capas geográficas desde '{geojson_file}'...")
    campus_graph = build_graph_from_geojson(geojson_file)
    print(f"      -> Grafo peatonal construido con éxito:")
    print(f"         • Vértices (intersecciones): {campus_graph.number_of_nodes()}")
    print(f"         • Tramos peatonales (aristas): {campus_graph.number_of_edges()}")

    # 2. Definir orígenes (Edificios con población) mapeados a nodos reales del GeoJSON
    # Coordenadas geográficas extraídas del relevamiento OSM del campus
    source_coords = {
        'Bloque_Central_Aulas': (-78.445219, -0.314549),
        'Edificio_Aulas_Norte': (-78.444886, -0.313121),
        'Complejo_Gimnasio_Coliseo': (-78.443850, -0.316722),
    }

    # Zonas Seguras de Evacuación (Puntos de Encuentro)
    target_coords = {
        'Zona_Segura_Garita_Principal': (-78.441983, -0.315000),
        'Zona_Segura_Parqueaderos_Sur': (-78.443215, -0.317934),
    }

    source_nodes = {}
    for s_name, (lon, lat) in source_coords.items():
        node, dist = find_nearest_node(campus_graph, lon, lat)
        source_nodes[s_name] = node

    target_nodes = {}
    for t_name, (lon, lat) in target_coords.items():
        node, dist = find_nearest_node(campus_graph, lon, lat)
        target_nodes[t_name] = node

    # Censos de población simultánea estimada
    populations = {
        'Bloque_Central_Aulas': 1800,
        'Edificio_Aulas_Norte': 1200,
        'Complejo_Gimnasio_Coliseo': 900
    }
    total_people = sum(populations.values())

    print(f"\n[2/4] Población Total a Evacuar: {total_people} personas")
    for s_name, count in populations.items():
        print(f"      • {s_name}: {count} estudiantes/docentes")

    print(f"\n[3/4] Generando rutas candidatas K-Shortest Paths...")
    candidate_paths = get_candidate_paths(campus_graph, source_nodes, target_nodes, k_paths=4)

    # Evaluar la asignación ingenua/voraz (Dijkstra independiente por edificio)
    naive_solution = {s: candidate_paths[s][0] for s in source_nodes}
    n_cost, n_makespan, n_times, n_pen = evaluate_solution(naive_solution, campus_graph, populations)

    # 3. Ejecutar Metaheurística Búsqueda Tabú
    print(f"\n[4/4] Ejecutando Optimización Metaheurística Tabú...")
    best_sol, best_cost, best_makespan, best_times, best_pen, hist = tabu_search_evacuation(
        campus_graph, source_nodes, target_nodes, candidate_paths, populations, max_iter=60, tabu_size=8
    )

    # 4. Mostrar Resultados Comparativos
    print("\n" + "=" * 70)
    print(" RESULTADOS COMPARATIVOS: ALGORITMO VORAZ vs BÚSQUEDA TABÚ")
    print("=" * 70)
    print(f"{'Métrica':<35} | {'Voraz (Sin Metaheurística)':<25} | {'Búsqueda Tabú (Óptimo)':<22}")
    print("-" * 88)
    print(f"{'Tiempo de Despeje Total (Makespan)':<35} | {n_makespan:>18.2f} min | {best_makespan:>15.2f} min")
    print(f"{'Penalización Cuellos de Botella':<35} | {n_pen:>18.2f} pts | {best_pen:>15.2f} pts")
    print(f"{'Costo Total de la Función Objetivo':<35} | {n_cost:>18.2f} pts | {best_cost:>15.2f} pts")
    
    mejora = ((n_cost - best_cost) / n_cost) * 100 if n_cost > 0 else 0
    print(f"\n>>> MEJORA EN LA EFICIENCIA DE EVACUACIÓN: {mejora:.1f}%")

    print("\n" + "-" * 70)
    print(" PLAN MAESTRO DE RUTAS ASIGNADAS POR EDIFICIO")
    print("-" * 70)
    for s_name, choice in best_sol.items():
        path = choice['path']
        tgt = choice['target_name']
        t_despeje = best_times[s_name]
        dist_m = sum(campus_graph[path[i]][path[i+1]]['length'] for i in range(len(path) - 1))
        
        print(f"\n[ORIGEN]: {s_name} ({populations[s_name]} personas)")
        print(f"  -> Destino asignado : {tgt}")
        print(f"  -> Distancia a pie  : {dist_m:.1f} metros ({len(path)} waypoints)")
        print(f"  -> Tiempo efectivo  : {t_despeje:.2f} minutos")

    print("\n" + "=" * 70)
    print(" PIPELINE FINALIZADO CORRECTAMENTE.")
    print("=" * 70)

if __name__ == '__main__':
    run_evacuation_pipeline('export.geojson')
