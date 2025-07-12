import ir_datasets

def load_dataset(dataset_name):
    dataset = ir_datasets.load(dataset_name)
    return list(dataset.docs_iter())