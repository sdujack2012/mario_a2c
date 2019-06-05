

# traning parameters
input_shape = (80, 100, 4)
sample_size = 32
epoch = 1
max_episodes = 2000
training_before_update_target = 5000
max_steps = 1200
experiences_before_training = 6000
esplison = 0.7
esplison_decay = 0.0001
min_esplison = 0.3
render = True
load_model = True
episodes_before_training = 10
n_env = 4
n_steps = 200
max_grad_norm = 0.5
ent_coef = 0.001
vf_coef = 0.5
GAMMA = 0.99
LAMBDA = 0.95
lr = 0.000003 
skip_frames = 4

# reward gamma

# prioritized memory replay
e = 0.00001
a = 0.7
beta = 0.4
beta_increment_per_sampling = 0.00001
capacity = 80000
max_priority = 1.0

# state
top_cutoff = 32
bottom_cutoff = 210
frame_size = (80, 100)
stack_size = 4