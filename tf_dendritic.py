
import numpy as np
import tensorflow as tf


class minval_constraint():
    def __init__(self,minval=0.0001):
        self.minvalval=minval
        self.minval=tf.constant(minval,dtype=K.floatx())
    def get_config(self):
        return({"minimal value":self.minvalval})
    def __call__(self,w):
        return(tf.maximum(w, self.minval))
class maxval_constraint():
    def __init__(self,maxval=4.0):
        self.maxvalval=maxval
        self.maxval=tf.constant(maxval,dtype=K.floatx())
    def get_config(self):
        return({"maximal value":self.maxvalval})
    def __call__(self,w):
        return(tf.minimum(w, self.maxval))
class minmax_constraint():
    def __init__(self,minval=0.0001,maxval=0.0001):
        self.maxvalval=maxval
        self.maxval=tf.constant(maxval,dtype=K.floatx())
        self.minvalval=minval
        self.minval=tf.constant(minval,dtype=K.floatx())
    def get_config(self):
        return({"maximal value":self.maxvalval,"minimal value":self.minvalval})
    def __call__(self,w):
        return(tf.maximum(tf.minimum(w, self.maxval),self.minval))

class dendriter(tf.layers.Layer):
    def __init__(self,units,dendrite_size,bigger_dendrite=False,activation=None,function:int=0,one_permutation:bool=False,idx=-2,
                 weight_twice=True,dendrites=None,#sequences
                 bias:bool=True,uniqueW=False,trainable=True,activity_regularizer=None,
                 W_init=tf.glorot_normal_initializer(),B_init=tf.glorot_normal_initializer(),
                 W_reg=None,B_reg=None,
                 W_constrain=minval_constraint(minval=0.0001),B_constrain=None,version=1,**kwargs):
        """
        size=number of cells/nodes
        dendrite_size=size of connections for each dendrite
        bigger_dendrite= if the number of input is uneven, will make the last dendrite bigger or smaller
        function= use simple addition or average
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
                                activity_regularizer=activity_regularizer,
                                **kwargs)
        self.units=units
        try:
            self.function=[tf.unsorted_segment_sum,tf.unsorted_segment_mean][function%2]
            #cannot import unsorted_segment_mean, with tf1.7 on my setup, but worth trying
        except (ImportError,AttributeError):
            self.function=tf.unsorted_segment_sum
        self.dendrite_size=dendrite_size
        self.bigger_dendrite=bigger_dendrite
        self.one_perm=one_permutation
        self.idx=idx
        self.weight_twice=weight_twice
        self.use_bias=bias
        self.uniqueW=uniqueW
        self.dendrites=dendrites
        self.Weight_initializer=W_init
        if B_init is None:
            self.Bias_initializer=tf.initializers.ones
        else:
            self.Bias_initializer=B_init
        self.Weight_regularizer=W_reg
        self.Weight_constraint=W_constrain
        self.Bias_regularizer=B_reg
        self.Bias_constraint=B_constrain
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
        print(self.dendrite_size,'dendrite_size')
        for perm in connections_list:
            temp=[perm[self.dendrite_size*i:self.dendrite_size*(i+1)] for i in range(groups)]
            temp.append(list(perm[self.dendrite_size*groups:]))
            tuples.append(temp)
        self.seql=len(tuples[0])
        if self.version==2:
            self.num_id=self.seql*len(tuples)
        else:
            self.num_id=self.seql
        output=np.empty((self.units,self.connections,),dtype=int)
        for iseq,sequence in enumerate(tuples):
            for value,indexes in enumerate(sequence):
                for index in indexes:
                    if self.version==2:
                        output[iseq,index]=value+iseq*self.seql
                    else:
                        output[iseq,index]=value
        print(output.shape,'segmenter, output shape')
        return(output)
    
    def build(self,input_shape):
        self.input_shapes=input_shape.shape.as_list()
        self.len_input=len(self.input_shapes)
        self.connections=self.input_shapes[-1]
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
        if self.weight_twice:
            if self.uniqueW:
                self.kernel=self.add_variable('Weight',shape=[*[1 for _ in range(self.len_input-1)],self.input_shapes[-1], self.units],
                                    initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                    constraint=self.Weight_constraint,dtype=self.dtype,
                                    trainable=True)
                            
            else:
                self.kernel=self.add_variable('Weight',shape=[1, self.units],
                                    initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                    constraint=self.Weight_constraint,dtype=self.dtype,
                                    trainable=True)

        self.dendriticW=self.add_variable('dendriticWeight',shape=dwshape,
                                initializer=self.Weight_initializer,regularizer=self.Weight_regularizer,
                                constraint=self.Weight_constraint,dtype=self.dtype,
                                trainable=True)
        if self.use_bias:
            if self.weight_twice:
                if self.uniqueW:
                    self.bias=self.add_variable('bias',shape=[self.input_shapes[-1],self.units,],
                                    initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                    constraint=self.Bias_constraint,dtype=self.dtype,
                                    trainable=True)
                else:
                    self.bias=self.add_variable('bias',shape=[self.units,],
                                    initializer=self.Bias_initializer,regularizer=self.Bias_regularizer,
                                    constraint=self.Bias_constraint, dtype=self.dtype,
                                    trainable=True)
            if self.uniqueW:
                self.dendriticB=self.add_variable('dendriticBias',shape=[self.seql,self.units,],
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
        if not(inputs.dtype==self.dendriticW.dtype):
            print("casting")
            inputs=tf.cast(inputs,dtype=self.dendriticW.dtype)
        print('input shape',inputs.shape)
        if self.weight_twice:
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
            output=tf.transpose(inputs)

        if self.version==4:#https://stackoverflow.com/questions/37441140/how-to-use-tf-while-loop-in-tensorflow
            #loopv2
            #output=tf.unstack(output)
            ix=tf.constant(0,dtype=tf.int32)
            holder=tf.TensorArray(output.dtype,size=self.units)
            incr=tf.constant(1,dtype=tf.int32)
            
            def test(i, *args):
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

            i, output = tf.while_loop(test, iteration,[ix,holder])
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
        #print('GOOD OUTPUT SHAPE') if output.shape==(*self.input_shapes[:-1],self.units) else print("BAD OUTPUT SHAPE")
        return(output)
