
import numpy as np
import tensorflow as tf

class dendriter(tf.layers.Layer):
    def __init__(self,units,dendrite_size,bigger_dendrite=False,function:int=0,one_permutation:bool=False,idx=-2,
                 bias:bool=True,uniqueW=False,uniqueB=False,many_weights=True,trainable=True,activity_regularizer=None,
                 W_init="glorot_normal_initializer",B_init="glorot_normal_initializer",
                 W_reg=None,B_reg=None,
                 W_constrain=None,B_constrain=None,**kwargs):
        """
        size=number of cells/nodes
        dendrite_size=size of connections for each dendrite
        bigger_dendrite= if the number of input is uneven, will make the last dendrite bigger or smaller
        function= use simple addition or average
        one_permitation=use only one connection scheme for every cells or not
        idx=dimention of the input to use to make segments
        """
        super(dendriter, self).__init__(trainable=trainable, 
                                activity_regularizer=activity_regularizer,
                                **kwargs)
        self.units=units
        try:
            self.function=[tf.unsorted_segment_sum,tf.unsorted_segment_mean][function%2]
            #cannot import unsorted_segment_mean, with tf1.7 on my setup, but worth trying
        except ImportError:
            self.function=tf.unsorted_segment_sum
        self.dendrite_size=float(dendrite_size)
        self.bigger_dendrite=bigger_dendrite
        self.one_perm=one_permutation
        self.idx=idx
        self.use_bias=bias
        self.uniqueW=uniqueW
        self.uniqueB=uniqueB
        self.manyW=many_weights
        self.Weight_initializer=W_init
        if B_init is None:
            self.Bias_initializer=tf.initializers.ones
        else:
            self.Bias_initializer=B_init
        self.Weight_regularizer=W_reg
        self.Weight_constraint=W_constrain
        self.Bias_regularizer=B_reg
        self.Bias_constraint=B_constrain
        
    def segmenter(self,):
        """
        must work on the node (units) and not the data itself?
        makes list with permuted index
        connections= size of input sequence
        permutations= number of lists
        connection_size=size of tuples in the list
        """
        connections_list=[]
        for i in range(self.units):
            con_pre_tup=np.random.permutation(self.connections).tolist()
            if con_pre_tup in connections_list:
                print('if ok')
            while con_pre_tup in connections_list:
                if len(connections_list)>(self.connections**2):
                    break
                else:
                    con_pre_tup=np.random.permutation(self.connections).tolist()
            connections_list.append(con_pre_tup)
        tuples=[]
        
        bigger=False if self.connections%self.dendrite_size==0 else self.bigger_dendrite
        groups = (self.connections//self.dendrite_size)
        if bigger:
            groups-=1
        for perm in connections_list:
            temp=[perm[self.dendrite_size*i:self.dendrite_size*(i+1)] for i in range(groups)]
            temp.append(list(perm[self.dendrite_size*groups:]))
            tuples.append(temp)
        num_id=len(tuples[0])
        output=np.empty((self.units,self.connections,),dtype=int)
        for iseq,sequence in enumerate(tuples):
            for value,indexes in enumerate(sequence):
                for index in indexes:
                    output[iseq,index]=value
        print(output.shape,'oneshot, output shape')
        return(output,num_id)

    
    
    def build(self,input_shape):
        input_shape=input_shape.shape.as_list()
        self.len_input=len(input_shape)
        self.connections=input_shape[-1]
        self.dendrites=self.make_seg_id(id)#list of dendrites per neuron
        self.divider=[ len(tup) for tup in self.seg_sample]#placeholder
        self.ldiv=len(self.divider)#number of dendrites per neuron
        self.divider=tf.reshape(self.divider,(1,1,1,self.ldiv,1,1))
        self.pre_dendrites=self.connections*self.units#neurons*previous_layer_neurons
        #self.num_dendrites=self.pre_dendrites/self.dendrite_size
        #if self.bigger_dendrite:
        #    self.num_dendrites=math.floor(self.num_dendrites)
        #else:
        #    self.num_dendrites=math.ceil(self.num_dendrites)
        
        #input_shape = tensor_shape.TensorShape(input_shape)
        if self.uniqueW:
            self.kernel=self.add_variable('Weight',shape=[*[1 for _ in range(self.len_input-1)],input_shape[-1], self.units],
                                initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                constraint=self.Weight_constraint,dtype=self.dtype,
                                trainable=True)
                        
        else:
            self.kernel=self.add_variable('Weight',shape=[1, self.units],
                                initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                constraint=self.Weight_constraint,dtype=self.dtype,
                                trainable=True)
        
        self.dendriticW=self.add_variable('dendriticWeight',shape=[self.num_id, self.units],
                                initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                constraint=self.Weight_constraint,dtype=self.dtype,
                                trainable=True)
        if self.use_bias:
            if self.uniqueB:
                self.bias=self.add_variable('bias',shape=[input_shape[-1],self.units,],
                                initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                constraint=self.Bias_constraint,dtype=self.dtype,
                                trainable=True)
            else:
                self.bias=self.add_variable('bias',shape=[self.units,],
                                initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                constraint=self.Bias_constraint, dtype=self.dtype,
                                trainable=True)
            if self.uniqueB:
                self.dendriticB=self.add_variable('dendriticBias',shape=[self.num_id,self.units,],
                                initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                constraint=self.Bias_constraint,dtype=self.dtype,
                                trainable=True)
            else:
                self.dendriticB=self.add_variable('dendriticB',shape=[self.units,],
                                initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                constraint=self.Bias_constraint,dtype=self.dtype,
                                trainable=True)
        self.input_spec = tf.layers.InputSpec(min_ndim=2,max_ndim=3,axes={-1: self.connections})
        self.built=True

    def call(self, inputs):
        return(self.__call__(inputs))

    def compute_output_shape(self, input_shape):
        input_shape = tf.tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)


    def __call__(self,inputs):
        #inputs = tf.ops.convert_to_tensor(inputs, dtype=self.dtype)
        output=tf.transpose(tf.expand_dims(inputs,-1))
        if self.manyW:
            output=tf.multiply(output, self.kernel)
        else:
            output=tf.tensordot(output,self.kernel,(-1,0))
        if self.use_bias:
            output=tf.nn.bias_add(output,self.bias)
        output=self.function(tf.transpose(output), self.dendrites, self.num_id,)
        output=tf.matmul(output,self.dendriticW)
        if self.use_bias:
            output=tf.nn.bias_add(output, self.dendriticB)
        if self.activation is not None:
          return self.activation(output)  
        return(output)
