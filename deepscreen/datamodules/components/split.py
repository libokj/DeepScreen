# TODO: some functions are borrowed from tdc.utils.split; need to be rewritten later

def cold_start(df, frac, entities, fold_seed=123):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        entities (Union[str, List[str]]): either a single "cold" entity or a list of
            "cold" entities on which the split is done

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e].drop_duplicates().sample(
            frac=test_frac, replace=False, random_state=fold_seed
        ).values for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            'No test samples found. Try another seed, increasing the test frac or a '
            'less stringent splitting strategy.'
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e].drop_duplicates().sample(
            frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed
        ).values for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            'No validation samples found. Try another seed, increasing the test frac '
            'or a less stringent splitting strategy.'
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {'train': train.reset_index(drop=True),
            'valid': val.reset_index(drop=True),
            'test': test.reset_index(drop=True)}
