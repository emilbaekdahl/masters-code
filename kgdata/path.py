import networkx as nx


def relation_paths(data, head, tail, min_length=1, max_length=3):
    graph = nx.MultiDiGraph(zip(data["head"], data["tail"], data["relation"]))

    rel_paths = []

    paths = (
        path
        for path in nx.all_simple_edge_paths(graph, head, tail, cutoff=max_length)
        if len(path) >= min_length
    )

    for path in paths:
        rel_path = [relation for _head, _tail, relation in path]

        if rel_path not in rel_paths:
            rel_paths.append(rel_path)

    return rel_paths
