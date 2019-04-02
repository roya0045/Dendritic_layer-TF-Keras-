from torch import nn
from torch.nn import init as finit
from torch import tensor
from torch.optim import Adam
import torch
import numpy as np


# mxnet pick :https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.pick
# cntk gather : https://cntk.ai/pythondocs/cntk.ops.html?highlight=load_model#cntk.ops.gather
# chainer selct : http://docs.chainer.org/en/stable/reference/generated/chainer.functions.select_item.html#chainer.functions.select_item


class minval_constraint():
    def __init__(self, minval=0.0001):
        self.minvalval = minval
        self.minval = minval

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
        self.minval = torch.constant(minval)

    def get_config(self):
        return ({"maximal value": self.maxvalval, "minimal value": self.minvalval})

    def __call__(self, w):
        return (torch.max(torch.min(w, self.maxval), self.minval))


class dendriter(nn.Module):
    def __init__(self, units, dendrite_size,inputshape, activation=None, function: int = 0,
                 one_permutation: bool = False, idx=-2, oper=torch.cumsum,
                 weight_twice=True, custom_dendrites=None, dendrite_conf='normal',
                 dendrite_shift=1,  # sequences
                 bias: bool = True, uniqueW=False, trainable=True, activity_regularizer=None,
                 # W_init=torch.glorot_normal_initializer(), B_init=torch.glorot_normal_initializer(),
                 # W_reg=None, B_reg=None,
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
        if isinstance(dendrite_conf, int):
            assert dendrite_conf <= 2 and dendrite_conf >= 0
            dendrite_conf = self.modes[dendrite_conf]

        else:
            assert dendrite_conf in self.modes
        if dendrite_conf == self.modes[1] or dendrite_conf == self.modes[2]:
            raise Exception("tensorflow only allows for normal mode currently")
        self.oper = oper
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
        self.kernel,self.bias,self.dendriticW,self.dendriticB=None,None,None,None
        '''
        self.Weight_initializer = W_init
        if B_init is None:
            self.Bias_initializer = torch.initializers.ones
        else:
            self.Bias_initializer = B_init
        self.Weight_regularizer = W_reg
        self.Bias_regularizer = B_reg
        self.Weight_constraint = W_constrain
        self.Bias_constraint = B_constrain
        '''
        self.activation = activation
        self.version = version
        self.params = list()
        self.build(inputshape)


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
            tuples.append(temp)
        self.seql = len(tuples[0])
        if self.version == 2:
            self.num_id = self.seql * len(tuples)
        else:
            self.num_id = self.seql
        self.dendrites = torch.tensor(tuples, dtype=torch.long)
        self.dendrites = torch.transpose(self.dendrites, 0, len(self.dendrites.shape) - 1)

    def build(self, input_shape, dtype=torch.float64):
        print("building")

        self.input_shapes = input_shape
        self.len_input = len(self.input_shapes)
        self.connections = self.input_shapes[-1]
        if self.dendrite_mode == self.modes[1]:  # sparse
            self.connections -= self.dendrite_shift
        elif self.dendrite_mode == self.modes[2]:  # overlap:
            self.connections += self.dendrite_shift
        if self.dendrites is None:
            self.segmenter()  # list of dendrites per neuron
        if self.version == 4:
            self.dendrites = torch.constant(self.dendrites)
        self.pre_dendrites = self.connections * self.units  # neurons*previous_layer_neurons
        if self.version != 1:
            dwshape = [self.units, self.seql]
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
                kernel = torch.empty(*[1 for _ in range(self.len_input - 1)], self.input_shapes[-1], self.units, dtype=dtype)

            else:
                kernel = torch.empty(1, self.units, dtype=dtype)
            finit.kaiming_normal(kernel)
            self.kernel = nn.Parameter(kernel)
            self.register_parameter('kernel', self.kernel)
            self.params.append(self.kernel)
        print('line246')
        dw = torch.empty(*dwshape,dtype=dtype)
        finit.kaiming_normal(dw)
        self.dendriticW = nn.Parameter(dw)
        self.params.append(self.dendriticW)
        print(self.dendriticW)
        print("added dendw")
        if self.use_bias:
            if self.weight_twice:
                if self.uniqueW:
                    b = torch.empty(self.input_shapes[-1], self.units, dtype=dtype)
                else:
                    b = torch.empty(1,self.units, dtype=dtype)
                try:
                    finit.kaiming_normal_(b)
                except:
                    finit.xavier_normal_(b)
                self.bias = nn.Parameter(b)
                self.register_parameter('Bias', self.bias)
                self.params.append(self.bias)
            if self.uniqueW:
                db = torch.empty(self.seql, self.units, dtype=dtype)
            else:
                db = torch.empty(1,self.units, dtype=dtype)
            finit.kaiming_normal_(db)
            self.dendriticB = nn.Parameter(db)
            self.params.append(self.dendriticB)
            self.register_parameter('dendritic_B', self.dendriticB)
        print("supered")
        #self.register_parameter('dentritic_W', self.dendriticW)
        self.built = True
        print('builded')

    def get_connections(self, ):
        "return the connections for replication"
        return (self.dendrites)

    def dendritic_op(self, input_data):
        if input_data.shape != self.dendrites.shape and False:
            input_data = torch.transpose(input_data, 0, len(input_data.shape) - 1)
        gathered = torch.gather(input_data, 1, self.dendrites)
        print(gathered.shape)
        return (self.oper(gathered, dim=0))

    def forward(self, inputs):
        #if not (self.built):
        #   self.build(inputs.shape)
        # inputs = torch.ops.convert_to_tensor(inputs, dtype=self.dtype)
        print(inputs.dtype, self.kernel.dtype, self.dendriticW.dtype)
        # if not (inputs.dtype == self.dendriticW.dtype):
        #    print("casting")
        #    inputs = torch.cast(inputs, dtype=self.dendriticW.dtype)
        print('input shape', inputs.shape)

        if self.weight_twice:
            # each dendrit COULD have unique weight for each input, meaning Wshape=[input,dendrite,units]
            output = inputs.unsqueeze(-1)
            print(output.dtype)
            print(output.shape, self.kernel.shape)
            if self.uniqueW:
                output = torch.multiply(output, self.kernel)
            else:
                output = torch.tensordot(output, self.kernel, dims=([-1, ], [0, ]))
            print(output.shape, 'first weighting')
            print(self.bias.shape)
            if self.use_bias:
                output += self.bias  # torch.transpose(output + self.bias)
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
                output = torch.matmul(torch.expand_dims(output, 0),
                                      torch.ones((self.units, *[1 for _ in range(self.len_input)]), dtype=output.dtype))
                output = torch.reshape(torch.transpose(output), self.debuildshape)
                output = torch.unsorted_segment_sum(output, torch.reshape(self.dendrites, self.deseqshape), self.num_id)
                output = torch.reshape(output, self.rebuildshape)
                # print(torch.transpose(output).shape, self.rebuildshape)
        if self.version == 1:
            output = self.dendritic_op(output, )
            # too much squashing
            print(output.shape, 'unsorted shape', self.dendriticW.shape,output.dtype,self.dendriticW.dtype)
            output = torch.mul(output, self.dendriticW)  # matmul
            # output = torch.tensordot(output, self.dendriticW,)# dims=([[-1,],[0,]]))
        else:
            print(output.shape, self.dendriticW.shape)
            output = torch.mul(output, self.dendriticW)  # perfect since it's elementwise and not dot product
        print(output.shape, '2w shape', self.dendriticB.shape)
        if self.use_bias:
            output += self.dendriticB
        print(output.shape, '2b shape')
        output = torch.sum(output, -2)  # sum the dendrites
        if self.activation is not None:
            return self.activation(output)
        print('outshap is {}'.format(output.shape))
        # print('GOOD OUTPUT SHAPE') if output.shape==(*self.input_shapes[:-1],self.units) else print("BAD OUTPUT SHAPE")
        return (output)


if __name__ == '__main__':
    size = 78
    test_tens = np.random.rand(3, size)
    test_data = torch.from_numpy(test_tens)
    layt = dendriter(13, 3,test_data.shape)
    outs = layt.forward(test_data)
    print(outs)
    print([f for f in layt.parameters(recurse=False)])
    module = dendriter(8, 4,test_data.shape)
    net = module.to('cpu')
    loss = nn.SmoothL1Loss()
    print(len([f for f in net.parameters()]),'params')
    print(dir(module))
    print(dir(net))
    print(net.params,module.params)
    optim = Adam(net.parameters(), lr=0.2, betas=(0.5, 0.999))
    corrections = torch.ones(3,13)
    OGW = list()
    for param in net.parameters():
        OGW.append(param)
    output = net(test_data)
    net.zero_grad()
    outloss = loss(output, corrections)
    outloss.backward()
    optim.step()
    OPW = list()
    for param in net.parameters():
        OPW.append(param)
    diff = list()
    for i, v in enumerate(OPW):
        diff.append(v - OGW[i])
    print(diff)
