

# traning parameters
sample_size = 32
epoch = 10
max_episodes = 2000
training_before_update_target = 5000
max_steps = 1200
experiences_before_training = 6000
esplison = 0.7
esplison_decay = 0.0001
min_esplison = 0.3
render = True
load_model = False
episodes_before_training = 10
n_env = 4
n_steps = 128
max_grad_norm = 0.5
ent_coef = 0.01
vf_coef = 0.5
GAMMA = 0.99
LAMBDA = 0.95
lr = 0.00005
skip_frames = 4
clip_range = 0.2
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
frame_size = (84, 84)
stack_size = 4
input_shape = (84, 84, stack_size)
