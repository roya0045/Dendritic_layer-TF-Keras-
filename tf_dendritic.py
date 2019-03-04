
import numpy as np
import tensorflow as tf
import copy
from keras.backend import floatx
from keras.engine.topology import Layer
from keras.activations import get

#mxnet pick :https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.pick
#cntk gather : https://cntk.ai/pythondocs/cntk.ops.html?highlight=load_model#cntk.ops.gather
#chainer selct : http://docs.chainer.org/en/stable/reference/generated/chainer.functions.select_item.html#chainer.functions.select_item


        
class minval_constraint():
    def __init__(self,minval=0.0001):
        self.minvalval=minval
        self.minval=tf.constant(minval,dtype=floatx())
    def get_config(self):
        return({"minimal value":self.minvalval})
    def __call__(self,w):
        return(tf.maximum(w, self.minval))
class maxval_constraint():
    def __init__(self,maxval=4.0):
        self.maxvalval=maxval
        self.maxval=tf.constant(maxval,dtype=floatx())
    def get_config(self):
        return({"maximal value":self.maxvalval})
    def __call__(self,w):
        return(tf.minimum(w, self.maxval))
class minmax_constraint():
    def __init__(self,minval=0.0001,maxval=0.0001):
        self.maxvalval=maxval
        self.maxval=tf.constant(maxval,dtype=floatx())
        self.minvalval=minval
        self.minval=tf.constant(minval,dtype=floatx())
    def get_config(self):
        return({"maximal value":self.maxvalval,"minimal value":self.minvalval})
    def __call__(self,w):
        return(tf.maximum(tf.minimum(w, self.maxval),self.minval))

class dendriter(Layer):
    def __init__(self,units,dendrite_size,bigger_dendrite=False,activation=None,function:int=0,one_permutation:bool=False,idx=-2,
                 weight_twice=True,custom_dendrites=None,dendrite_conf='normal',
                 dendrite_shift=1,#sequences
                 bias:bool=True,uniqueW=False,trainable=True,activity_regularizer=None,
                 W_init=tf.glorot_normal_initializer(),B_init=tf.glorot_normal_initializer(),
                 W_reg=None,B_reg=None,
                 W_constrain=minval_constraint(minval=0.00005),B_constrain=None,version=1,**kwargs):
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
        super(dendriter, self).__init__(trainable=trainable, 
                                #activity_regularizer=activity_regularizer,
                                **kwargs)
        self.units=units
        self.modes=['normal','sparse','overlap']
        try:
            self.function=[tf.unsorted_segment_sum,tf.unsorted_segment_mean][function%2]
            #cannot import unsorted_segment_mean, with tf1.7 on my setup, but worth trying
        except (ImportError,AttributeError):
            self.function=tf.unsorted_segment_sum
        if isinstance(dendrite_conf, int):
            assert dendrite_conf <=2 and dendrite_conf >=0
            dendrite_conf=self.modes[dendrite_conf]
            
        else:
            assert dendrite_conf in self.modes
        if dendrite_conf==self.modes[1] or dendrite_conf==self.modes[2]:
            raise Exception("tensorflow only allows for normal mode currently")
            
        self.dendrite_size=dendrite_size
        self.bigger_dendrite=bigger_dendrite
        self.one_perm=one_permutation
        self.idx=idx
        self.dendrite_mode=dendrite_conf
        #only normal 
        self.dendrite_shift=dendrite_shift
        self.weight_twice=weight_twice
        self.use_bias=bias
        self.uniqueW=uniqueW
        self.dendrites=custom_dendrites
        self.Weight_initializer=W_init
        if B_init is None:
            self.Bias_initializer=tf.initializers.ones
        else:
            self.Bias_initializer=B_init
        self.Weight_regularizer=W_reg
        self.Weight_constraint=W_constrain
        self.Bias_regularizer=B_reg
        self.Bias_constraint=B_constrain
        if isinstance(activation, str):
            self.activation=get(activation)
        else:
            self.activation=activation
        self.version=version

    def segmenter(self,):
        """
        must work on the node (units) and not the data itself?
        makes list with permuted index
        connections= size of input sequence
        permutations= number of lists
        connection_size=size of tuples in the list
        """
        #if self.dendrite_mode!=self.modes[0]:
        #    altered_vals=[]
        full_conn_storage=[]
        connections_list=[]
        
        for i in range(self.units):#for each unit, make connections
            con_pre_tup=np.random.permutation(self.connections).tolist()
            if len(full_conn_storage)>=self.connections:
                full_conn_storage.clear()
            while con_pre_tup in connections_list:#full_conn_storage:
                if len(connections_list)>(self.connections**2):
                    break
                else:
                    np.random.shuffle(con_pre_tup)
            connections_list.append(con_pre_tup)
            #full_conn_storage.append(copy.deepcopy(con_pre_tup))
            """
            Could be made to work by isolating elements with an extra index and removing/slicing them from the dendrites
            if self.dendrite_mode==self.modes[1]:#sparse
                if len(altered_vals)>=self.connections:
                    altered_vals.clear()
                popval=np.random.randint(self.connections)
                while con_pre_tup[popval] in altered_vals:
                    popval=np.random.randint(self.connections)
                altered_vals.append((con_pre_tup.pop(popval),))
            There are no known workaround for overlapping since each element can only have 1 index, so duplication would entail manual slicing and addition
            elif self.dendrite_mode==self.modes[2]:#overlap
                if len(altered_vals)>=self.connections:
                    position=np.random.randint(self.connections)
                    inval=np.random.randint(self.connections)
                    while inval in altered_vals:#make sure you get unique value
                        inval=np.random.randint(self.connections)
                    while inval in con_pre_tup[position-self.dendrite_size:position+self.dendrite_size]:#ensure no proximal duplicates
                        position=np.random.randint(self.connections)
                    altered_vals.append(inval)
                else:
                    altered_vals.clear()"""
        del full_conn_storage
        tuples=[]
        if self.dendrite_mode==self.modes[1]:#sparse
            connections=self.connections-self.dendrite_shift
            
        elif self.dendrite_mode==self.modes[2]:#overlap:
            connections=self.connections+self.dendrite_shift
        else:
            connections=self.connections
        bigger=False if (connections%self.dendrite_size)==0 else self.bigger_dendrite
        groups = (connections//self.dendrite_size)
        if bigger:
            groups-=1
        for perm in connections_list:#turn list to tuples
            temp=[perm[self.dendrite_size*i:self.dendrite_size*(i+1)] for i in range(groups)]
            if len(perm)>self.dendrite_size*groups:
                temp.append(list(perm[self.dendrite_size*groups:]))
            #print(self.dendrite_size,groups,perm)
            #print(perm[self.dendrite_size*groups:])
            #print(temp)
            tuples.append(temp)
        self.seql=len(tuples[0])
        if self.version==2:
            self.num_id=self.seql*len(tuples)
        else:
            self.num_id=self.seql
        output=np.empty((self.units,connections,),dtype=int)
        for iseq,sequence in enumerate(tuples):
            for value,indexes in enumerate(sequence):
                for index in indexes:
                    if self.version==2:
                        output[iseq,index]=value+iseq*self.seql
                    else:
                        output[iseq,index]=value
        return(output)
    
    def build(self,input_shape):
        print("building")

        self.input_shapes=input_shape.shape.as_list()
        self.len_input=len(self.input_shapes)
        self.connections=self.input_shapes[-1]
        if self.dendrite_mode==self.modes[1]:#sparse
            self.connections-=self.dendrite_shift
        elif self.dendrite_mode==self.modes[2]:#overlap:
            self.connections+=self.dendrite_shift
        if self.dendrites is None:
            self.dendrites=self.segmenter()#list of dendrites per neuron
        if self.version==4:
            self.dendrites=tf.constant(self.dendrites)
        self.pre_dendrites=self.connections*self.units#neurons*previous_layer_neurons
        if self.version==1:
            dwshape=[self.seql,self.units]
        else:
            dwshape=[self.seql,self.units]
            #dwshape=[self.units,self.seql,*[1 for _ in range(self.len_input-1)]]
        #self.num_dendrites=self.pre_dendrites/self.dendrite_size
        #if self.bigger_dendrite:
        #    self.num_dendrites=math.floor(self.num_dendrites)
        #else:
        #    self.num_dendrites=math.ceil(self.num_dendrites)
        
        #input_shape = tensor_shape.TensorShape(input_shape)
        if self.version==2:
            if len(self.input_shapes)>2:
                part_inshape=(*self.input_shapes[1:-1],-1)
            else:
                part_inshape=(-1,)
            self.debuildshape=(self.units*self.connections,*part_inshape)
            self.deseqshape=(self.units*self.connections,)
            self.rebuildshape=(self.units,self.seql,*part_inshape)
        print('line228')
        if self.weight_twice:
            """if self.uniqueW==2:#useless since all input are there once, could also work with sparse
                print([self.dendrite_size,self.seql, self.units])
                self.kernel=self.add_variable('Weight',shape=[*[1 for _ in range(self.len_input-1)],self.dendrite_size,self.seql, self.units],
                                    initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                    constraint=self.Weight_constraint,dtype=self.dtype,
                                    trainable=True)"""
            if self.uniqueW:
                self.kernel=self.add_weight('Weight',shape=[*[1 for _ in range(self.len_input-1)],self.input_shapes[-1], self.units],
                                    initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                    constraint=self.Weight_constraint,#dtype=self.dtype,
                                    trainable=True)

            else:
                self.kernel=self.add_weight('Weight',shape=[1, self.units],
                                    initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                    constraint=self.Weight_constraint,#dtype=self.dtype,
                                    trainable=True)
        print('line246')
        self.dendriticW=self.add_weight('dendriticWeight',shape=dwshape,
                                initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                constraint=self.Weight_constraint,#dtype=self.dtype,
                                trainable=True)
        print("added dendw")
        if self.use_bias:
            if self.weight_twice:
                if self.uniqueW:
                    self.bias=self.add_weight('bias',shape=[self.input_shapes[-1],self.units,],
                                    initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                    constraint=self.Bias_constraint,#dtype=self.dtype,
                                    trainable=True)
                else:
                    self.bias=self.add_weight('bias',shape=[self.units,],
                                    initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                    constraint=self.Bias_constraint,# dtype=self.dtype,
                                    trainable=True)
            if self.uniqueW:
                self.dendriticB=self.add_weight('dendriticBias',shape=[self.seql,self.units,],
                                initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                constraint=self.Bias_constraint,#dtype=self.dtype,
                                trainable=True)
            else:
                self.dendriticB=self.add_weight('dendriticB',shape=[self.units,],
                                initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                constraint=self.Bias_constraint,#dtype=self.dtype,
                                trainable=True)
        self.input_spec = tf.layers.InputSpec(min_ndim=2,max_ndim=3,axes={-1: self.connections})

        super(dendriter, self).build(input_shape) 
        print("supered")
        self.built=True
        print('builded')
        
        
    def get_connections(self,):
        "return the connections for replication"
        return(self.dendrites)

    def call(self, inputs):
        return(self.__call__(inputs))

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)#tf.shape(input_shape)#
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
            
        print('outputshape is {}'.format(input_shape[:-1].concatenate(self.units)))
        return input_shape[:-1].concatenate(self.units)


    def __call__(self,inputs):
        try:
            assert not(self.dendriticW.dtype is None)
        except:
            self.build(inputs)#.shape)
        #inputs = tf.ops.convert_to_tensor(inputs, dtype=self.dtype)
        if not(inputs.dtype==self.dendriticW.dtype):
            print("casting")
            inputs=tf.cast(inputs,dtype=self.dendriticW.dtype)
        print('input shape',inputs.shape)
        if self.weight_twice:
            #each dendrit COULD have unique weight for each input, meaning Wshape=[input,dendrite,units]
            output=tf.expand_dims(inputs,-1)
            if self.uniqueW:
                output=tf.multiply(output, self.kernel)
            else:
                output=tf.tensordot(output,self.kernel,(-1,0))
            print(output.shape,'first weighting')
            if self.use_bias:
                output=tf.transpose(tf.nn.bias_add(output,self.bias))
            print(output.shape,'bias1')
            
        else:
            output=tf.transpose(inputs)#units,x,batch

        if self.version==4:#https://stackoverflow.com/questions/37441140/how-to-use-tf-while-loop-in-tensorflow
            #loopv2
            #output=tf.unstack(output)
            ix=tf.constant(0,dtype=tf.int32)
            holder=tf.TensorArray(output.dtype,size=self.units)
            incr=tf.constant(1,dtype=tf.int32)
            
            def test_condition(i, *args):
                return(tf.less( i , self.units))
            
            def iteration(i, outputs_):
                print(i,'I')
                if self.weight_twice:
                    intermid= tf.unsorted_segment_sum(output[i],self.dendrites[i],self.seql)
                else:
                    outputs_ = outputs_.write(i,intermid)
                print(intermid.shape,'while loop data')
                outputs_ = outputs_.write(i,intermid)
                return( tf.add(i,incr), outputs_)

            i, output = tf.while_loop(test_condition, iteration,[ix,holder])
            output=output.stack()
            print(output.shape,'stacked')
            #loopv1
            #condition= lambda inn,hold: tf.less(inn,self.input_shapes[-1])
            #looper=lambda inn,hold : [tf.add(inn,incr), tf.unsorted_segment_sum(output[inn],self.dendrites[inn],self.seql)]
            #output=tf.while_loop(condition,looper,[ix,hold])
            #output=tf.stack(output[1])
        if self.version==3:
            if self.weight_twice:
                output=tf.unstack(output)
                output=tf.stack([tf.unsorted_segment_sum(data, self.dendrites[i],self.seql) for i,data in enumerate(output)])
            else:
                output=tf.stack([tf.unsorted_segment_sum(output, seq,self.seql) for seq in self.dendrites])
        if self.version==2:
            if self.weight_twice:
                print(self.debuildshape)
                output=tf.reshape(output,self.debuildshape)
                output=tf.unsorted_segment_sum(output,tf.reshape(self.dendrites,self.deseqshape),self.num_id)
                output=tf.reshape(output,self.rebuildshape)
                print(tf.transpose(output).shape,self.rebuildshape)
            else:
                print(self.debuildshape)
                output=tf.multiply(tf.expand_dims(output,0),tf.ones((self.units,*[1 for _ in range(self.len_input)]),dtype=output.dtype))
                output=tf.reshape(tf.transpose(output),self.debuildshape)
                output=tf.unsorted_segment_sum(output,tf.reshape(self.dendrites,self.deseqshape),self.num_id)
                output=tf.reshape(output,self.rebuildshape)
                print(tf.transpose(output).shape,self.rebuildshape)
        if self.version==1:
            output=self.function(output, self.dendrites, self.num_id,)
        #too much squashing
            print(output.shape,'unsorted shape')
            output=tf.tensordot(tf.transpose(output),self.dendriticW,(-1,0))
        else:
            print(output.shape,self.dendriticW.shape)
            output=tf.multiply(tf.transpose(output),self.dendriticW)#perfect since it's elementwise and not dot product
        print(output.shape,'2w shape')
        if self.use_bias:
            output=tf.nn.bias_add(output, self.dendriticB)
        print(output.shape,'2b shape')
        output=tf.reduce_sum(output,-2)#sum the dendrites
        if self.activation is not None:
            return self.activation(output)  
        print('outshap is {}'.format(output.shape))
        #print('GOOD OUTPUT SHAPE') if output.shape==(*self.input_shapes[:-1],self.units) else print("BAD OUTPUT SHAPE")
        return(output)


#FORWARD
"""
input->Basalcomp->(soma*W_neuron_inter)->basal_inter->soma_inter
soma-(soma_inter*W_inter_neuron) ->apical 
(soma*W_neuron_neuron)->basal(new_neuron)"""
#BACKWARDpg6
"""Loss(apical)->soma(change it, lead to change in incoming weight for basal comp)
#v2:
soma_up + apical-soma inter->soma ->soma_up (error)?
"""
"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

def _GuidedReluGrad(op, grad):
    return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

    with tf.Session() as sess:
        g = tf.get_default_graph()
        x = tf.constant([10., 2.])
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            y = tf.nn.relu(x)
            z = tf.reduce_sum(-y ** 2)
        tf.initialize_all_variables().run()

        print x.eval(), y.eval(), z.eval(), tf.gradients(z, x)[0].eval()# > [ 10.   2.] [ 10.   2.] -104.0 [ 0.  0.]
"""
    
    
