import jax
import jax.numpy as jnp
import numpy as np


##representation a single neuron as a jnp array
def create_neurons(num_neurons=1):

    neuron = jnp.zeros((num_neurons, 2), dtype=jnp.float32) #v, on/off for spike
    spike_time = [[] for _ in range(num_neurons)]
    return neuron, spike_time 


##calculate the change in membrane potential and spike state of a neuron
def step(neuron, input_current, miu, dt, threshold, v_reset):

    v, spike = neuron
    ##calculate the change in membrane potential
    dv = miu*(input_current - v)*dt
    v_new = v + dv
    ##calculate the spike state, if the membrane potential is above the threshold, the neuron spikes 
    ##change spike state to 1 and reset the membrane potential to v_reset
    spike = (v_new >= threshold)
    v_new = jnp.where(spike, v_reset, v_new)
    return jnp.array([v_new, spike], dtype=jnp.float32)

##vectorize the step function
vectorized_step = jax.vmap(step, in_axes=(0, 0, None, None, None, None))

##network of 3 layers of neurons
def simulate_network(neuron_in, neuron_hid, neuron_out, spikes_in, spikes_hid, spikes_out, weight_in_hid, weight_hid_out, input_current, miu, dt, duration, threshold, v_reset, optional_input=0.0):
    num_steps = int(duration / dt)
    time = 0.0
    for _ in range(num_steps):
        time += dt
        ##update input neuron
        neuron_in = vectorized_step(neuron_in, input_current, miu, dt, threshold, v_reset)
        for i, spike in enumerate(neuron_in[:, 1]):
            if float(spike) > 0.0:
                spikes_in[i].append(time)

        ##update hidden neuron
        synapse_input_from_neuron = jnp.dot(weight_in_hid, neuron_in[:, 1])
        total_input = synapse_input_from_neuron + optional_input
        neuron_hid = vectorized_step(neuron_hid, total_input, miu, dt, threshold, v_reset)
        for i, spike in enumerate(neuron_hid[:, 1]):
            if float(spike) > 0.0:
                spikes_hid[i].append(time)
        
        ##update output neuron
        synapse_input_from_hidden_neuron = jnp.dot(weight_hid_out, neuron_hid[:, 1])
        neuron_out = vectorized_step(neuron_out, synapse_input_from_hidden_neuron, miu, dt, threshold, v_reset)
        for i, spike in enumerate(neuron_out[:, 1]):
            if float(spike) > 0.0:
                spikes_out[i].append(time)

    return neuron_in, neuron_hid, neuron_out, spikes_in, spikes_hid, spikes_out




def make_weight_matrix(num_input, num_output):
    ##return a random weight matrix
    ##only positive values
    return jax.random.uniform(jax.random.PRNGKey(42), (num_output, num_input), minval=0.0, maxval=1.0, dtype=jnp.float32)

#make weight matrix from array
def make_weight_matrix_from_array(neuron_array):
    toReturn = []
    for i in range(len(neuron_array)-1):
        toReturn.append(make_weight_matrix(neuron_array[i], neuron_array[i+1]))
    return toReturn

"""""" ##Example of using the code

#4 neuron in input layer, 3 neuron in hidden layer, 2 neuron in output layer
# weight_in_hid = make_weight_matrix(4, 3)
# weight_hid_out = make_weight_matrix(3, 2)
# #input neuron layer
# neuron_in, spikes_in = create_neurons(4)
# #hidden neuron layer
# neuron_hid, spikes_hid = create_neurons(3)
# #output neuron layer
# neuron_out, spikes_out = create_neurons(2)

# #input current
# input_current = jnp.array([1.5, 1.7, 2.0, 1.8], dtype=jnp.float32)

# #simulation parameters
# miu = 0.1
# dt = 0.1
# duration = 30
# threshold = 1.0
# v_reset = 0.0
# optional_input = 0.0

# neuron_in, neuron_hid, neuron_out, spikes_in, spikes_hid, spikes_out = simulate_network(neuron_in, neuron_hid, neuron_out, spikes_in, spikes_hid, spikes_out, weight_in_hid, weight_hid_out, input_current, miu, dt, duration, threshold, v_reset, optional_input)

# print(spikes_in)
# print(spikes_hid)
# print(spikes_out)

#index 0 is first layer, index 1 is second layer, index 2 is third layer...
#[index] = number of neurons in that layer

""""""



##################################################################################
#Prototype of the code
neurons = [4, 3, 2]
weights = make_weight_matrix_from_array(neurons)
def simulate_network2(neuron_layers, weights, input_current, miu, dt, duration, threshold, v_reset, optional_input=0.0):
    num_layers = len(neuron_layers)
    layers = []
    spikes = []
    last_layer_spike = []
    first_layer_spike = [[] for _ in range(neuron_layers[0])]
    for i in range(num_layers):
        layer, spike = create_neurons(neuron_layers[i])
        layers.append(layer)
        spikes.append(spike)
    
    num_steps = int(duration / dt)
    time = 0.0
    for _ in range(num_steps):
        time += dt
        ##update input neuron
        layers[0] = vectorized_step(layers[0], input_current, miu, dt, threshold, v_reset)
        for i, spike in enumerate(layers[0][:, 1]):
            if float(spike) > 0.0:
                first_layer_spike[i].append(time)
        for i in range(1, num_layers):
            synapse_input_from_neuron = jnp.dot(weights[i-1], layers[i-1][:, 1])
            total_input = synapse_input_from_neuron + optional_input
            layers[i] = vectorized_step(layers[i], total_input, miu, dt, threshold, v_reset)
            
            #record the spike of the output layer
            if i == num_layers-1:
                for j, spike in enumerate(layers[i][:, 1]):
                    if float(spike) > 0.0:
                        last_layer_spike.append(time)
    return layers, last_layer_spike, first_layer_spike


#input current
input_current = jnp.array([1.5, 1.7, 2.0, 1.8], dtype=jnp.float32)

#simulation parameters
miu = 0.1
dt = 0.1
duration = 30
threshold = 1.0
v_reset = 0.0
optional_input = 0.0

##test 1st way of simulating network
weight_in_hid = weights[0]
weight_hid_out = weights[1]
#input neuron layer
neuron_in, spikes_in = create_neurons(4)
#hidden neuron layer
neuron_hid, spikes_hid = create_neurons(3)
#output neuron layer
neuron_out, spikes_out = create_neurons(2)

neuron_in, neuron_hid, neuron_out, spikes_in, spikes_hid, spikes_out = simulate_network(neuron_in, neuron_hid, neuron_out, spikes_in, spikes_hid, spikes_out, weight_in_hid, weight_hid_out, input_current, miu, dt, duration, threshold, v_reset, optional_input)
print(spikes_in)
print("----------------")

#test 2nd way of simulating network
layers, last_layer_spike, first_layer_spike = simulate_network2(neurons, weights, input_current, miu, dt, duration, threshold, v_reset, optional_input)
print(first_layer_spike)