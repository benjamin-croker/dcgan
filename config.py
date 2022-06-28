# TODO: move to json config
DATASET_PARAMS =  {
    "dataset_dir": "data/",
    "image_size": 64
}
MODEL_PARAMS = {
    "n_channels": 3,
    "n_latent": 100,
    "n_feat_map_gen": 64,
    "n_feat_map_dis": 64
}
OPTIMIZER_PARAMS = {
    "batch_size": 128,
    "workers": 2,
    "n_epochs": 5,
    "learning_rate": 0.0002,
    "beta1": 0.5,
    "device": 'cuda',
    "seed": 42
}