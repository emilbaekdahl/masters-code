def rel_dists(data):
    return (
        data.melt(id_vars="relation", value_name="entity")
        .value_counts(["entity", "relation"])
        .unstack(fill_value=0)
    )
