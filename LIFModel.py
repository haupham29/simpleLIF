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



#4 neuron in input layer, 3 neuron in hidden layer, 2 neuron in output layer
weight_in_hid = make_weight_matrix(4, 3)
weight_hid_out = make_weight_matrix(3, 2)
#input neuron layer
neuron_in, spikes_in = create_neurons(4)
#hidden neuron layer
neuron_hid, spikes_hid = create_neurons(3)
#output neuron layer
neuron_out, spikes_out = create_neurons(2)

#input current
input_current = jnp.array([1.5, 1.7, 2.0, 1.8], dtype=jnp.float32)

#simulation parameters
miu = 0.1
dt = 0.1
duration = 30
threshold = 1.0
v_reset = 0.0
optional_input = 0.0

neuron_in, neuron_hid, neuron_out, spikes_in, spikes_hid, spikes_out = simulate_network(neuron_in, neuron_hid, neuron_out, spikes_in, spikes_hid, spikes_out, weight_in_hid, weight_hid_out, input_current, miu, dt, duration, threshold, v_reset, optional_input)

print(spikes_in)
print(spikes_hid)
print(spikes_out)


