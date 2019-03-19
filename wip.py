import numpy as np
from torch import nn
from torch import tensor
import torch


# mxnet pick :https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.pick
# cntk gather : https://cntk.ai/pythondocs/cntk.ops.html?highlight=load_model#cntk.ops.gather
# chainer selct : http://docs.chainer.org/en/stable/reference/generated/chainer.functions.select_item.html#chainer.functions.select_item


class minval_constraint():
    def __init__(self, minval=0.0001):
        self.minvalval = minval
        self.minval = torch.constant(minval)

    def get_config(self):
        return ({"minimal value": self.minvalval})

    def __call__(self, w):
        return (torch.max(w, self.minval))


class maxval_constraint():
    def __init__(self, maxval=4.0):
        self.maxvalval = maxval
        self.maxval = torch.constant(maxval)

    def get_config(self):
        return ({"maximal value": self.maxvalval})

    def __call__(self, w):
        return (torch.min(w, self.maxval))


class minmax_constraint():
    def __init__(self, minval=0.0001, maxval=0.0001):
        self.maxvalval = maxval
        self.maxval = torch.constant(maxval)
        self.minvalval = minval
        self.minval = torch.constant(minval))

    def get_config(self):
        return ({"maximal value": self.maxvalval, "minimal value": self.minvalval})

    def __call__(self, w):
        return (torch.max(torch.min(w, self.maxval), self.minval))


class dendriter(nn.Module):
    def __init__(self, units, dendrite_size, activation=None, function: int = 0,
                 one_permutation: bool = False, idx=-2,
                 weight_twice=True, custom_dendrites=None, dendrite_conf='normal',
                 dendrite_shift=1,  # sequences
                 bias: bool = True, uniqueW=False, trainable=True, activity_regularizer=None,
                 #W_init=torch.glorot_normal_initializer(), B_init=torch.glorot_normal_initializer(),
                 #W_reg=None, B_reg=None,
                 W_constrain=minval_constraint(minval=0.00005), B_constrain=None, version=1, **kwargs):
        """
        size=number of cells/nodes
        dendrite_size=size of connections for each dendrite
        dendrite_mode=normal: all inputs used once, sparse: some input not used, overlap: duplicate input for some dendrites
        dendrite_shift: number of input to ignore or duplicate depending on the mode
        bigger_dendrite= if the number of input is uneven, will make the last dendrite bigger or smaller
        function= use simple addition or average,only addition is supported for now (0)
        one_permitation=use only one connection scheme for every cells or not
        idx=dimention of the input to use to make segments

        #extra:
        theorical backprop/weight update
        wi=0.001#input weight
        wdi=1.0#dendrite weight
        wi=max((wi(1+changeW),0.001))
        wdi=wdi*(1+changeW)
        #changeW indicate the error?
        changeW=max_learn_r*exp(-abs(delta)/scaler)*sign(delta)#bigger value = smaller change, max = 0.05, min =-0.05

        """
        super(dendriter, self).__init__()
        self.units = units
        self.modes = ['normal', 'sparse', 'overlap']
        try:
            self.function = [torch.unsorted_segment_sum, torch.unsorted_segment_mean][function % 2]
            # cannot import unsorted_segment_mean, with torch1.7 on my setup, but worth trying
        except (ImportError, AttributeError):
            self.function = torch.unsorted_segment_sum
        if isinstance(dendrite_conf, int):
            assert dendrite_conf <= 2 and dendrite_conf >= 0
            dendrite_conf = self.modes[dendrite_conf]

        else:
            assert dendrite_conf in self.modes
        if dendrite_conf == self.modes[1] or dendrite_conf == self.modes[2]:
            raise Exception("tensorflow only allows for normal mode currently")

        self.dendrite_size = dendrite_size
        self.one_perm = one_permutation
        self.idx = idx
        self.dendrite_mode = dendrite_conf
        # only normal
        self.dendrite_shift = dendrite_shift
        self.weight_twice = weight_twice
        self.use_bias = bias
        self.uniqueW = uniqueW
        self.dendrites = custom_dendrites
        self.Weight_initializer = W_init
        if B_init is None:
            self.Bias_initializer = torch.initializers.ones
        else:
            self.Bias_initializer = B_init
        self.Weight_regularizer = W_reg
        self.Weight_constraint = W_constrain
        self.Bias_regularizer = B_reg
        self.Bias_constraint = B_constrain

        self.activation = activation
        self.version = version
        self.built=False

    def segmenter(self, ):
        """
        must work on the node (units) and not the data itself?
        makes list with permuted index
        connections= size of input sequence
        permutations= number of lists
        connection_size=size of tuples in the list
        """
        # if self.dendrite_mode!=self.modes[0]:
        #    altered_vals=[]
        full_conn_storage = []
        connections_list = []

        for i in range(self.units):  # for each unit, make connections
            con_pre_tup = np.random.permutation(self.connections).tolist()
            if len(full_conn_storage) >= self.connections:
                full_conn_storage.clear()
            while con_pre_tup in connections_list:  # full_conn_storage:
                if len(connections_list) > (self.connections ** 2):
                    break
                else:
                    np.random.shuffle(con_pre_tup)
            connections_list.append(con_pre_tup)
            # full_conn_storage.append(copy.deepcopy(con_pre_tup))

        del full_conn_storage
        tuples = []
        if self.dendrite_mode == self.modes[1]:  # sparse
            connections = self.connections - self.dendrite_shift

        elif self.dendrite_mode == self.modes[2]:  # overlap:
            connections = self.connections + self.dendrite_shift
        else:
            connections = self.connections

        groups = (connections // self.dendrite_size)
        if not (len(connections_list[0]) == self.dendrite_size * groups):
            raise Exception(
                'Size of input is not equal to the number of connections, partial connections are not yet supported')
        for perm in connections_list:  # turn list to tuples
            temp = [perm[self.dendrite_size * i: self.dendrite_size * (i + 1)] for i in range(groups)]
            if len(perm) > self.dendrite_size * groups:
                tlist = list(perm[self.dendrite_size * groups:])
                # numpy does not support nan in int array
                temp.append(tlist + [np.nan for i in range(self.dendrite_size - len(tlist))])

        self.seql = len(tuples[0])
        if self.version == 2:
            self.num_id = self.seql * len(tuples)
        else:
            self.num_id = self.seql

        return (tuples)

    def build(self, input_shape):
        print("building")

        self.input_shapes = input_shape.shape.as_list()
        self.len_input = len(self.input_shapes)
        self.connections = self.input_shapes[-1]
        if self.dendrite_mode == self.modes[1]:  # sparse
            self.connections -= self.dendrite_shift
        elif self.dendrite_mode == self.modes[2]:  # overlap:
            self.connections += self.dendrite_shift
        if self.dendrites is None:
            self.dendrites = self.segmenter()  # list of dendrites per neuron
        if self.version == 4:
            self.dendrites = torch.constant(self.dendrites)
        self.pre_dendrites = self.connections * self.units  # neurons*previous_layer_neurons
        if self.version == 1:
            dwshape = [self.seql, self.units]
        else:
            dwshape = [self.seql, self.units]
            # dwshape=[self.units,self.seql,*[1 for _ in range(self.len_input-1)]]
        # self.num_dendrites=self.pre_dendrites/self.dendrite_size
        # if self.bigger_dendrite:
        #    self.num_dendrites=math.floor(self.num_dendrites)
        # else:
        #    self.num_dendrites=math.ceil(self.num_dendrites)

        # input_shape = tensor_shape.TensorShape(input_shape)
        if self.version == 2:
            if len(self.input_shapes) > 2:
                part_inshape = (*self.input_shapes[1:-1], -1)
            else:
                part_inshape = (-1,)
            self.debuildshape = (self.units * self.connections, *part_inshape)
            self.deseqshape = (self.units * self.connections,)
            self.rebuildshape = (self.units, self.seql, *part_inshape)
        print('line228')
        if self.weight_twice:
            """if self.uniqueW==2:#useless since all input are there once, could also work with sparse
                print([self.dendrite_size,self.seql, self.units])
                self.kernel=self.add_variable('Weight',shape=[*[1 for _ in range(self.len_input-1)],self.dendrite_size,self.seql, self.units],
                                    initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                    constraint=self.Weight_constraint,dtype=self.dtype,
                                    trainable=True)"""
            if self.uniqueW:
                self.kernel = nn.Parameter(torch.randn(*[1 for _ in range(self.len_input - 1)], self.input_shapes[-1],self.units))

            else:
                self.kernel = nn.Parameter(torch.randn(1, self.units))
        print('line246')
        self.dendriticW = nn.Parameter (torch.randn(dwshape))
        print("added dendw")
        if self.use_bias:
            if self.weight_twice:
                if self.uniqueW:
                    self.bias = nn.Parameter(torch.randn(self.input_shapes[-1], self.units))
                else:
                    self.bias = nn.Parameter(torch.randn(self.units))
            if self.uniqueW:
                self.dendriticB = nn.Parameter(torch.randn(self.seql, self.units))
            else:
                self.dendriticB = nn.Parameter(self.units)
        self.input_spec = torch.layers.InputSpec(min_ndim=2, max_ndim=3, axes={-1: self.connections})
        print("supered")
        self.built = True
        print('builded')

    def get_connections(self, ):
        "return the connections for replication"
        return (self.dendrites)



    def forward(self, inputs):
        if not(self.built):
            self.build(inputs.shape)
        # inputs = torch.ops.convert_to_tensor(inputs, dtype=self.dtype)
        if not (inputs.dtype == self.dendriticW.dtype):
            print("casting")
            inputs = torch.cast(inputs, dtype=self.dendriticW.dtype)
        print('input shape', inputs.shape)
        if self.weight_twice:
            # each dendrit COULD have unique weight for each input, meaning Wshape=[input,dendrite,units]
            output = inputs.unsqueeze( 0)
            if self.uniqueW:
                output = torch.multiply(output, self.kernel)
            else:
                output = torch.dot(output, self.kernel, (-1, 0))
            print(output.shape, 'first weighting')
            if self.use_bias:
                output = torch.transpose(output + self.bias)
            print(output.shape, 'bias1')

        else:
            output = torch.transpose(inputs)  # units,x,batch

            # loopv1
            # condition= lambda inn,hold: torch.less(inn,self.input_shapes[-1])
            # looper=lambda inn,hold : [torch.add(inn,incr), torch.unsorted_segment_sum(output[inn],self.dendrites[inn],self.seql)]
            # output=torch.while_loop(condition,looper,[ix,hold])
            # output=torch.stack(output[1])
        if self.version == 3:
            if self.weight_twice:
                output = torch.unstack(output)
                output = torch.stack(
                    [torch.unsorted_segment_sum(data, self.dendrites[i], self.seql) for i, data in enumerate(output)])
            else:
                output = torch.stack([torch.unsorted_segment_sum(output, seq, self.seql) for seq in self.dendrites])
        if self.version == 2:
            if self.weight_twice:
                print(self.debuildshape)
                output = torch.reshape(output, self.debuildshape)
                output = torch.unsorted_segment_sum(output, torch.reshape(self.dendrites, self.deseqshape), self.num_id)
                output = torch.reshape(output, self.rebuildshape)
                print(torch.transpose(output).shape, self.rebuildshape)
            else:
                print(self.debuildshape)
                output = torch.mm(torch.expand_dims(output, 0),
                                     torch.ones((self.units, *[1 for _ in range(self.len_input)]), dtype=output.dtype))
                output = torch.reshape(torch.transpose(output), self.debuildshape)
                output = torch.unsorted_segment_sum(output, torch.reshape(self.dendrites, self.deseqshape), self.num_id)
                output = torch.reshape(output, self.rebuildshape)
                print(torch.transpose(output).shape, self.rebuildshape)
        if self.version == 1:
            output = self.function(output, self.dendrites, self.num_id, )
            # too much squashing
            print(output.shape, 'unsorted shape')
            output = torch.tensordot(torch.transpose(output), self.dendriticW, (-1, 0))
        else:
            print(output.shape, self.dendriticW.shape)
            output = torch.mm(torch.transpose(output),
                                 self.dendriticW)  # perfect since it's elementwise and not dot product
        print(output.shape, '2w shape')
        if self.use_bias:
            output = torch.nn.bias_add(output, self.dendriticB)
        print(output.shape, '2b shape')
        output = torch.sum(output, -2)  # sum the dendrites
        if self.activation is not None:
            return self.activation(output)
        print('outshap is {}'.format(output.shape))
        # print('GOOD OUTPUT SHAPE') if output.shape==(*self.input_shapes[:-1],self.units) else print("BAD OUTPUT SHAPE")
        return (output)


# FORWARD
"""
input->Basalcomp->(soma*W_neuron_inter)->basal_inter->soma_inter
soma-(soma_inter*W_inter_neuron) ->apical 
(soma*W_neuron_neuron)->basal(new_neuron)"""
# BACKWARDpg6
"""Loss(apical)->soma(change it, lead to change in incoming weight for basal comp)
#v2:
soma_up + apical-soma inter->soma ->soma_up (error)?
"""
"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

def _GuidedReluGrad(op, grad):
    return torch.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), torch.zeros(grad.get_shape()))

    with torch.Session() as sess:
        g = torch.get_default_graph()
        x = torch.constant([10., 2.])
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            y = torch.nn.relu(x)
            z = torch.reduce_sum(-y ** 2)
        torch.initialize_all_variables().run()

        print x.eval(), y.eval(), z.eval(), torch.gradients(z, x)[0].eval()# > [ 10.   2.] [ 10.   2.] -104.0 [ 0.  0.]
"""

if __name__=='__main__':
    size = 78
    test_tens = np.random.rand(3, size)
    test_data = torch.from_numpy(test_tens)
    layt=dendriter(6,3)
    outs=layt.forward(test_data)
    print(outs)
