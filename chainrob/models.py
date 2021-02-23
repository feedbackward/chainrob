
# models.py

import numpy as np
import chainer as ch
import config


## FunctionNode objects. ##

class LinearFunction_Robust(ch.function_node.FunctionNode):
    '''
    Usual linear transformation, defined on Variable
    objects and returning Variable objects. The only
    difference from the default linear layers is the
    use of a robustification sub-routine when passed
    a robustifier.

    More concretely, if robustifier is not None, then
    in a per-dimension fashion, we robustly aggregate
    over the mini-batch elements.
    '''

    def __init__(self, robustifier, nfactor):
        super(LinearFunction_Robust, self).__init__()
        self.robustifier = robustifier
        self.nfactor = nfactor

        
    def forward(self, inputs):
        '''
        Forward pass, common for CPU/GPU.
        Works indentically to Chainer's off-the-shelf
        linear layer.
        '''
        
        # Unpack the tuple of inputs.
        if len(inputs) == 3:
            x, W, b = inputs
        else:
            (x, W), b = inputs, None
        
        y = x.dot(W.T)

        # Add a bias term, if relevant.
        if b is not None:
            y += b
            self.retain_inputs((0,1,2))
        else:
            self.retain_inputs((0,1))
        
        # Return as a tuple, always.
        return (y,)


    def backward(self, indices, grad_outputs):
        '''
        Backward pass, common for both CPU/GPU.

        This is the interesting part, where in
        the event of a mini-batch greater than
        size 2, we carry out a robustification
        rather than just summing over data points.
        '''
        
        # Pick up the retained inputs.
        if len(self.get_retained_inputs()) > 2:
            x, W, b = self.get_retained_inputs()
        else:
            x, W = self.get_retained_inputs()
        
        # Get gradient of objective WRT unit *outputs*.
        gy = grad_outputs[0]

        # This gy has shape (n,k), where k is the
        # number of outputs of this FunctionNode.
        n = gy.shape[0]
        k, d = W.shape
        
        # Robustification is only relevant for W and b.
        out = []
        
        # Now for the robustification routines as needed.
        if 0 in indices:
            gx = gy @ W
            out.append(ch.functions.cast(gx, x.dtype))
            
        if 1 in indices:
            
            if n < 3 or self.robustifier is None:
                gW = gy.T @ x
                
            else:
                
                # If batch has enough samples, can aggregate robustly.
                gW = np.zeros(W.shape, dtype=W.dtype) # start as ndarray.

                # Loop over output dimension.
                for i in range(k):
                    
                    gradsW = x.array * np.take(gy.array, [i], 1)
                    if self.nfactor:
                        gradsW *= n

                    # Do without loop over input dimension.
                    gW[i,:] = self.robustifier(x=gradsW)
                    
                # Finally, convert this gW to Variable form.
                gW = ch.Variable(gW)
            
            out.append(ch.functions.cast(gW, W.dtype))
        
        if 2 in indices:
            
            if n < 3 or self.robustifier is None:
                gb = ch.functions.sum(gy, axis=0)
                
            else:
                # If batch has enough samples, can aggregate robustly.
                gb = np.zeros((k,), dtype=W.dtype) # start as ndarray.
                # note: use same type as W.

                # Do without loop over outputs here.
                if self.nfactor:
                    gb = self.robustifier(x=gy.array*n).flatten()
                else:
                    gb = self.robustifier(x=gy.array).flatten()
                
                gb = ch.Variable(gb)
                    
            out.append(ch.functions.cast(gb, W.dtype))

            
        return tuple(out)
    

def linear_robust(x, W, b, robustifier, nfactor):
    '''
    A thin wrapper for LinearFunction_Robust.
    '''
    if b is None:
        args = (x,W)
    else:
        args = (x,W,b)

    # Output is tuple of length one; extract then return.
    return LinearFunction_Robust(robustifier=robustifier,
                                 nfactor=nfactor).apply(args)[0]


## Link objects. ##

class Linear_Robust(ch.Link):
    '''
    A link based upon our linear transformation function,
    with robustification, implemented in the FunctionNode
    sub-class called LinearFunction_Robust.
    '''

    def __init__(self, in_size, out_size, robustifier, nfactor,
                 nobias=False, init_W=None, init_b=None, init_delta=None):

        super(Linear_Robust, self).__init__()
        
        self.robustifier = robustifier
        self.nfactor = nfactor
        
        # Here we initialize and register the "parameters" of
        # interest, namely those to be set via learning algos.
        
        with self.init_scope():
            
            # If provided a Variable object, copy its contents.
            # Else, use a built-in initializer.
            if init_W is not None:
                self.W = ch.Parameter()
                self.W.initialize(shape=init_W.shape)
                self.W.copydata(init_W)
            else:
                W_initializer = ch.initializers.Uniform(scale=init_delta,
                                                        dtype=np.float32)
                self.W = ch.Parameter(initializer=W_initializer,
                                      shape=(out_size, in_size))

            # Basically the same deal for the bias term.
            if nobias:
                self.b = None
            else:
                if init_b is not None:
                    self.b = ch.Parameter()
                    self.b.initialize(shape=init_b.shape)
                    self.b.copydata(init_b)
                else:
                    b_initializer = ch.initializers.Uniform(scale=init_delta,
                                                            dtype=np.float32)
                    self.b = ch.Parameter(initializer=b_initializer,
                                          shape=(out_size,))


    def __call__(self, x):
        '''
        Makes the Link object itself callable.
        '''
        return linear_robust(x, self.W, self.b,
                             robustifier=self.robustifier,
                             nfactor=self.nfactor)


## Chain objects. ##

class Chain_Class_H2_ReLU_Robust(ch.Chain):
    '''
    A simple architecture for classification.
    Feedforward structure.
    Hidden layers: 2
    Hidden layer output activations: Rectified linear unit.
    Final outputs are just "log probabilities", i.e.,
    non-normalized scores; the number of final outputs
    is assumed to be the number of classes.
    '''

    def __init__(self, out_l0, out_l1, out_l2,
                 out_l3, robustifiers, nfactors,
                 nobias=False, init_delta=config.INIT_UNIF_WIDTH):

        super(Chain_Class_H2_ReLU_Robust, self).__init__()

        with self.init_scope():

            self.l1 = Linear_Robust(in_size=out_l0,
                                    out_size=out_l1,
                                    robustifier=robustifiers[0],
                                    nfactor=nfactors[0],
                                    init_W=None,
                                    init_b=None,
                                    init_delta=init_delta,
                                    nobias=nobias)
            self.l2 = Linear_Robust(in_size=out_l1,
                                    out_size=out_l2,
                                    robustifier=robustifiers[1],
                                    nfactor=nfactors[1],
                                    init_W=None,
                                    init_b=None,
                                    init_delta=init_delta,
                                    nobias=nobias)
            self.l3 = Linear_Robust(in_size=out_l2,
                                    out_size=out_l3,
                                    robustifier=robustifiers[2],
                                    nfactor=nfactors[2],
                                    init_W=None,
                                    init_b=None,
                                    init_delta=init_delta,
                                    nobias=nobias) # robust Link.


    def __call__(self, x):
        # Layer 1: Rectified linear unit.
        # Layer 2: Rectified linear unit.
        # Layer 3: final outputs are un-normalized log probabilities.
        return self.l3(
            ch.functions.relu(
                self.l2(ch.functions.relu(self.l1(x)))
            )
        )



class Chain_FFWD_ReLU(ch.Chain):
    '''
    Feed-forward neural network with
    all ReLU activation functions, and
    the flexibility to be robustified
    at any layer.
    '''
    
    def __init__(self, dims, robustifiers, nfactors,
                 nobias=False, init_delta=config.INIT_UNIF_WIDTH):

        super(Chain_FFWD_ReLU, self).__init__()

        # Number of output-producing links.
        self.num_links = len(dims)-1

        if self.num_links != len(robustifiers):
            raise ValueError(
                "The number of robustifiers must match the number of links."
            )
        
        # Link names.
        self.link_names = [ "l"+str(i) for i in range(self.num_links) ]
        
        # Define all the child links.
        with self.init_scope():
            
            for i in range(self.num_links):
                self.add_link(name=self.link_names[i],
                              link=Linear_Robust(in_size=dims[i],
                                                 out_size=dims[i+1],
                                                 robustifier=robustifiers[i],
                                                 nfactor=nfactors[i],
                                                 init_W=None,
                                                 init_b=None,
                                                 init_delta=init_delta,
                                                 nobias=nobias))
        
    def __call__(self, x):
        
        if self.num_links == 1:
            return self.l0(x)
        
        else:
            out = x
            for link_name in self.link_names:
                link_fn = self.__getitem__(name=link_name)
                out = link_fn(out)
                if link_name != self.link_names[-1]:
                    out = ch.functions.relu(out)
                # no activation at last layer.
            return out

        