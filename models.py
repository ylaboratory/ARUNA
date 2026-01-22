import torch
import torch.nn as nn


class DAE(nn.Module):

    """
    One difference to the original rigid DAE:
        Last encoder layer now has a nonlinearity added.
    """
    
    def __init__(self, input_dim, 
                 next_dim, bn_dim,
                 compress = 2, 
                 nonlinearity = "relu",
                 last_act = "sigmoid"):
        
        """
        compress: a reduction factor wthat determines how layer sizes change.
                   affects the total number of layers in the model.
        """
        
        super().__init__()

        layer_dims = [input_dim, next_dim] # add first layer
        if compress == 1:
            raise ValueError("Reduction factor cannot be 1!")
        
        if bn_dim > next_dim:
            raise ValueError("Bottleneck dim cannot be greater than first layer size!")
        
        # generate Linear layer sizes
        while next_dim > bn_dim:
            next_dim = next_dim/compress
            if next_dim > bn_dim:
                layer_dims.append(int(next_dim))
        layer_dims.append(bn_dim) # add final layer
        
        self.num_dims = len(layer_dims)
        self.num_layers = (self.num_dims-1) * 2

        if nonlinearity == "relu":
            act_fn = nn.ReLU(inplace = True)
        elif nonlinearity == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError("Supplied nonlinearity not supported!")

        if last_act == "sigmoid":
            last_act_fn = nn.Sigmoid()
        else:
            raise ValueError("Supplied last layer activation not supported!")
        
        encoder_layers = nn.ModuleList()
        decoder_layers = nn.ModuleList()
        for i in range(self.num_dims-1):
            encoder_layers.append(nn.Linear(layer_dims[i],
                                            layer_dims[i+1]))
            encoder_layers.append(act_fn)

            decoder_layers.append(nn.Linear(layer_dims[self.num_dims-(i+1)],
                                            layer_dims[self.num_dims-(i+2)]))
            if i == self.num_dims-2:
                decoder_layers.append(last_act_fn)
            else:
                decoder_layers.append(act_fn)


        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, sparse_seq):
        latent_repr = self.encoder(sparse_seq)
        recons_seq = self.decoder(latent_repr)
        return(latent_repr, recons_seq)



class DCAE_2CH(nn.Module): # A Fully pyramidal naive architecture

    def __init__(self, num_layers, 
                 kernel_size, stride = 2, init_outchannel = 16,
                 channel_factor = 2, nonlinearity = "relu", 
                 last_act = "sigmoid", batch_norm = False):
        
        """
        Does not support mixed kernel sizes.
        Does not support custom changes in layer-wise channel numbers.
        Channel factor is always 2.
        Dropouts are not recommended with Convolutional layers.
        Shape mismatch from the decoder is due to the input shape and its divisibility by 2^(n_EncoderLayers)
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size//2
        self.init_inchannels = 2 # nature of input
        self.init_outchannel = init_outchannel # define-able param (affects all subsequent channel nos. by channel_factor)
        self.channel_factor = channel_factor
        self.batch_norm = batch_norm

        if nonlinearity == "relu":
            self.act_fn = nn.ReLU(inplace = True)
        elif nonlinearity == "leakyrelu":
            self.act_fn = nn.LeakyReLU(inplace = True)
        elif nonlinearity == "gelu":
            self.act_fn = nn.GELU()
        
        # self.nonlinearity = nonlinearity
        if last_act == "sigmoid":
            self.last_act_fn = nn.Sigmoid()

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        
        encoder_channels = [self.init_inchannels] # to help with decoder definition
        for i in range(num_layers):
            if i == 0:
                self.encoder.append(self._make_encoder_layer(self.init_inchannels, 
                                                             self.init_outchannel))
                last_outchannel = self.init_outchannel
            else:
                self.encoder.append(self._make_encoder_layer(last_outchannel, 
                                                             last_outchannel * self.channel_factor))
                last_outchannel = last_outchannel * self.channel_factor
            encoder_channels.append(last_outchannel)

        decoder_channels = encoder_channels[::-1] # channels/filter maps in reverse order
        for i, in_channels in enumerate(decoder_channels[:-1]): # all but last layer
            out_channels = decoder_channels[i+1]
            if out_channels == self.init_inchannels:
                final_layer = True
            else:
                final_layer = False
            self.decoder.append(self._make_decoder_layer(in_channels, 
                                                         out_channels, 
                                                         final_layer))


    def _make_encoder_layer(self, in_channels, out_channels):

        if self.batch_norm:
            model_block = nn.Sequential(nn.Conv1d(in_channels, 
                                                    out_channels, 
                                                    kernel_size=self.kernel_size, 
                                                    stride = self.stride, 
                                                    padding = self.padding), 
                                        nn.BatchNorm1d(out_channels), 
                                        self.act_fn)

        else:
            model_block = nn.Sequential(nn.Conv1d(in_channels, 
                                                    out_channels, 
                                                    kernel_size=self.kernel_size, 
                                                    stride = self.stride, 
                                                    padding = self.padding), 
                                        self.act_fn)
        return (model_block)


    def _make_decoder_layer(self, in_channels, out_channels, final_layer=False):

        model_block = [nn.ConvTranspose1d(in_channels, 
                                        out_channels, 
                                        kernel_size=self.kernel_size, 
                                        stride = self.stride, 
                                        padding=self.padding, 
                                        output_padding = 1)] # output_padding always 1 to help infer output shape as per pytorch docs
        if not final_layer:
            if self.batch_norm:
                model_block.extend([
                    nn.BatchNorm1d(out_channels),
                    self.act_fn])
            else:
                model_block.extend([self.act_fn])

        else:
            model_block.append(nn.Sigmoid()) # NOTE: no Batch Norm in final layer
        return (nn.Sequential(*model_block))


    def forward(self, x):
        
        # Encode
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
        # Decode
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        #return (encoder_outputs[-1], x) # latent_repr, recons_seq
        return (encoder_outputs, x) # all latent representations, reconstructed sequence
    


# New, improved implementation
class DCAE_MPATCH(nn.Module): # Fully or partially Pyramidal; Symmetric

    """
    CHANGELOG (Jul '25): Now handled in model class instead of elsewhere;
        - Added "posn_embed", "stem_bnorm", "stem_activation" and "enc_last_activation" to config YAML.
            - Removed "use_pe", "use_stem", "pe_combine".
        - Constructor changes to be simpler, getting config["model"] from YAML from main().
        - Determination and init of initial input_channels (based on stem, PE aug type etc.)
        - Position embedding concat/sum now only takes place in forward() instead of on a case-by-case basis in DataLoader.
        - Model with and without stem - all from the same class.
        - Stem changed to follow similar structure to encoder and decoder helper modules.
        - Stem now supports BatchNorm and optional activation function.
        - Output from forward now consistent shape: (B,1,input_size) where last dim is #CpGs.

         Parameters
        ---------
            input_size : int, default=None
                        num_cpgs chosen as hyperparameter for MPatches
            input_channels : int, default=None
                            Usually 1 (when supplied only betas or added posn_vec) or 2 (with concat posn_vec).
            block_list : list of strings, default=None
                        Types of block which compose the DCAE architecture.
                        Examples include "cnn_same", "cnn_halve" etc
                        cnn_same: stride = 1, preserves input dimensionality
                        cnn_halve: stride = 2, halves input dimensionality
                        CNN Padding is calculated accordingly.
                        TODO: Add other block types like MLP, Attention etc.
            kernel_list : list, default=None
                        kernel/filter/receptive field sizes for each CNN layer.
            channel_list : list, default=None
                        num output channels for each layer.
            nonlinearity : string, default="relu"
                        One of "relu", "leakyrelu" or "gelu" applied after each layer.
                        TODO: Mixed activations not supported.
            last_act : string, default="sigmoid"
                    Activation function applied after the last (output) layer.
            batch_norm: bool, default=False
                        Whether to apply torch.BatchNorm1d after each (except the last) block.
                        TODO: Custom batch norm applications over blocks not supported.
    """
    
    def __init__(self, input_size, config):
        
        super().__init__()

        block_list = config["layers"]
        kernel_list = config["kernel_sizes"]
        channel_list = config["out_channels"]
        nonlinearity = config["activation"]
        last_act = config["last_activation"]
        stem_act = config["stem_activation"]
        assert len(block_list) == len(kernel_list) == len(channel_list),\
        "Please check your inputs that all info for each layer is provided!"

        # main attributes
        self.input_size = input_size
        self.stem_dict = config["stem_dict"] # can be None
        self.posn_embed = config["posn_embed"]
        
        if (self.stem_dict) or (not self.posn_embed): # when stem(+type2) or no PE
            self.init_inchannels = 1
        elif (self.posn_embed.split("_")[0] == "type1") and (self.posn_embed.split("_")[1] == "sum"):
            self.init_inchannels = 1
        elif (self.posn_embed.split("_")[0] == "type1") and (self.posn_embed.split("_")[1] == "concat"):
            self.init_inchannels = 2

        self.batch_norm = config["use_bnorm"]
        self.stem_batch_norm = config["stem_bnorm"]
        self.apply_encLastAct = config["enc_last_activation"]
        self.last_outchannels = self.init_inchannels # use copy or something else?
        
        if nonlinearity == "relu":
            self.act_fn = nn.ReLU(inplace = False)
        elif nonlinearity == "leakyrelu":
            self.act_fn = nn.LeakyReLU(inplace = False)
        elif nonlinearity == "gelu":
            self.act_fn = nn.GELU()

        if last_act == "sigmoid":
            self.last_act_fn = nn.Sigmoid()
        elif last_act == "identity":
            self.last_act_fn = nn.Identity() # for M-values

        if stem_act == "relu":
            self.stem_actfn = nn.ReLU(inplace = False)
        elif stem_act == "leakyrelu":
            self.stem_actfn = nn.LeakyReLU(inplace = False)
        else:
            self.stem_actfn = None

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        if self.stem_dict:
            self.stem_layers = nn.ModuleList([])
            # Populate the stem layers
            for stem_block in list(self.stem_dict.keys()):
                # Extract block type from key (e.g., "cnn_1" -> "cnn")
                block_type = stem_block.split("_")[0]
                l_inCh = self.last_outchannels # initial value is same as init_inchannels
                l_outCh = self.stem_dict[stem_block]["out_channels"]
                l_kernelSize = self.stem_dict[stem_block]["kernel_size"]
                self.stem_layers.append(self._make_stem_layer(block_type, l_inCh, 
                                                              l_outCh, l_kernelSize))
                self.last_outchannels = l_outCh
            
            if self.posn_embed.split("_")[1] == "concat":
                self.last_outchannels = 2*l_outCh
            
            self.embed_dim = l_outCh # last outchannels is the embedding dimension


        # Populate Encoder
        for i, btype in enumerate(block_list):
            l_kernelSize = kernel_list[i]
            l_outCh = channel_list[i]
            if i == 0:
                l_inCh = self.last_outchannels
                is_finalLayer = False
            elif i < len(block_list)-1:
                l_inCh = channel_list[i-1]
                is_finalLayer = False
            else:
                l_inCh = channel_list[i-1]
                is_finalLayer = True
            self.encoder.append(self._make_encoder_layer(btype, l_inCh,
                                                         l_outCh, l_kernelSize,
                                                         is_finalLayer))
        

        # Populate Decoder (reverses the channel dimensions from the encoder)
        for i, btype in reversed(list(enumerate(block_list))):
            l_kernelSize = kernel_list[i]
            l_inCh = channel_list[i]
            if i == 0:
                # l_outCh = self.init_inchannels
                l_outCh = 1 # this only comes into play for type1_concat, else it's 1 anyway
                is_finalLayer = True
            else:
                l_outCh = channel_list[i-1]
                is_finalLayer = False
            self.decoder.append(self._make_decoder_layer(btype, l_inCh,
                                                         l_outCh, l_kernelSize,
                                                         is_finalLayer))

    def _make_stem_layer(self, block_type, in_ch, out_ch, ks):
        # Same stride/padding logic as encoder but always preserve dimensions
        if block_type == "cnn":
            s = 1
            pad = (ks-1)//2  # Always preserve input dimensionality
            
            model_block = [nn.Conv1d(in_channels=in_ch, 
                                    out_channels=out_ch, 
                                    kernel_size=ks, 
                                    stride=s, 
                                    padding=pad,
                                    bias=True, 
                                    padding_mode="zeros")]
        
        # Add batch norm and activation like encoder
        if self.stem_batch_norm:
            model_block.append(nn.BatchNorm1d(out_ch))
        if self.stem_actfn:
            model_block.append(self.stem_actfn)
           
        return (nn.Sequential(*model_block))


    def _make_encoder_layer(self, block_type, in_ch, out_ch, ks,
                            final_layer = False):

        # stride and padding calculation is placed here (and repeated in 2 private methods)
        # this is to ensure flexibility for other layer types and not clutter method calls
        
        if block_type.split("_")[0] == "cnn":
            if block_type.split("_")[1] == "same":
                s = 1
                pad = (ks-1)//2
            elif block_type.split("_")[1] == "halve":
                s = 2
                pad = (ks//2)-1

            model_block = [nn.Conv1d(in_channels = in_ch, 
                                     out_channels = out_ch, 
                                     kernel_size = ks, 
                                     stride = s, 
                                     padding = pad,
                                     bias = True, 
                                     padding_mode = "zeros")]

        if not final_layer:
            if self.batch_norm:
                model_block.extend([nn.BatchNorm1d(out_ch), 
                                    self.act_fn])
            else:
                model_block.extend([self.act_fn])
        else:
            if self.apply_encLastAct: # if False, last encoder layer is convolution output
                model_block.append(self.last_act_fn) # NOTE: no Batch Norm in final layer

        return (nn.Sequential(*model_block))


    def _make_decoder_layer(self, block_type, in_ch, out_ch, ks, final_layer=False):

        # stride and padding calculation is relegated here (and repeated in 2 private methods)
        # this is to ensure flexibility for other layer types and not clutter method calls
        
        if block_type.split("_")[0] == "cnn":
            if block_type.split("_")[1] == "same":
                s = 1
                pad = (ks-1)//2
            elif block_type.split("_")[1] == "halve":
                s = 2
                pad = (ks//2)-1

            model_block = [nn.ConvTranspose1d(in_channels = in_ch, 
                                            out_channels = out_ch, 
                                            kernel_size = ks, 
                                            stride = s, 
                                            padding = pad, 
                                            output_padding = 0, 
                                            bias = True, 
                                            padding_mode = "zeros")]
        if not final_layer:
            if self.batch_norm:
                model_block.extend([nn.BatchNorm1d(out_ch), 
                                    self.act_fn])
            else:
                model_block.extend([self.act_fn])

        else:
            model_block.append(self.last_act_fn) # NOTE: no Batch Norm in final layer
        
        return (nn.Sequential(*model_block))


    def forward(self, x, batch_pe = None):

        x = torch.unsqueeze(x,1) # (B,L) --> (B,1,L)

        if self.posn_embed:
            assert batch_pe is not None, "batch_pe must be provided if posn_embed is enabled"
            
            if self.stem_dict: # type 2 PE
                for stem_layer in self.stem_layers:
                    x = stem_layer(x)
            else: # type 1 PE
                batch_pe = torch.unsqueeze(batch_pe, 1) # (B,L) --> (B,1,L)
            
            if self.posn_embed.split("_")[1] == "sum":
                x = x + batch_pe 
            else:
                x = torch.concat((x, batch_pe), axis = 1)

        # Encode
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
        
        # Decode
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        
        # return (encoder_outputs, x)
        return(x)
        


class DCAE_MSLICE(nn.Module):

    """ 
    IDEAS: 
    (i) Try asymmetric kernel
    (ii) Try 1D CNN after collapsing along Sample Dim

    TODO: Currently only supports symmetric kernels.
    """

    def __init__(self, config):
        
        super().__init__()

        channel_list = config["out_channels"]
        stride_list = config["strides"]
        kernel_list = config["kernel_sizes"]
        nonlinearity = config["activation"]
        last_act = config["last_activation"]
        stem_act = config["stem_activation"]
        assert len(channel_list) == len(stride_list) == len(kernel_list),\
        "Please check your inputs that all info for each layer is provided!"

        self.stem_dict = config["stem_dict"] # can be None
        self.posn_embed = config["posn_embed"]
        self.numLayers = len(channel_list)

        if (self.stem_dict) or (not self.posn_embed): # when stem(+type2) or no PE
            self.init_inchannels = 1
        elif (self.posn_embed.split("_")[0] == "type1") and (self.posn_embed.split("_")[1] == "sum"):
            self.init_inchannels = 1
        elif (self.posn_embed.split("_")[0] == "type1") and (self.posn_embed.split("_")[1] == "concat"):
            self.init_inchannels = 2
        
        self.batch_norm = config["use_bnorm"]
        self.stem_batch_norm = config["stem_bnorm"]
        self.apply_encLastAct = config["enc_last_activation"]
        self.last_outchannels = self.init_inchannels # use copy or something else?

        if nonlinearity == "relu":
            self.act_fn = nn.ReLU(inplace = False)
        elif nonlinearity == "leakyrelu":
            self.act_fn = nn.LeakyReLU(inplace = False)
        elif nonlinearity == "gelu":
            self.act_fn = nn.GELU()

        if last_act == "sigmoid":
            self.last_act_fn = nn.Sigmoid()
        elif last_act == "identity":
            self.last_act_fn = nn.Identity() # for M-values

        if stem_act == "relu":
            self.stem_actfn = nn.ReLU(inplace = False)
        elif stem_act == "leakyrelu":
            self.stem_actfn = nn.LeakyReLU(inplace = False)
        else:
            self.stem_actfn = None

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        # MAKE STEM
        if self.stem_dict:
            self.stem_layers = nn.ModuleList([])
            for stem_block in list(self.stem_dict.keys()):
                # Extract block type from key (e.g., "cnn_1" -> "cnn")
                block_type = stem_block.split("_")[0]
                inCh = self.last_outchannels # initial value is init_inchannels
                outCh = self.stem_dict[stem_block]["out_channels"]
                ks = self.stem_dict[stem_block]["kernel_size"]
                s = self.stem_dict[stem_block]["stride"]

                self.stem_layers.append(self._make_stem_layer(block_type, inCh, 
                                                              outCh, ks, s))
                self.last_outchannels = outCh
            
            self.embed_dim = outCh # pre-PE dim is the embedding dim
            # Type 2 PE is applied after Stem-embedding
            if self.posn_embed.split("_")[1] == "concat":
                self.last_outchannels = 2*outCh
            
        # MAKE ENCODER
        for i in range(self.numLayers):
            if i == 0:
                inCh = self.last_outchannels
                is_finalLayer = False
            elif i < (self.numLayers-1):
                inCh = channel_list[i-1]
                is_finalLayer = False
            else:
                inCh = channel_list[i-1]
                is_finalLayer = True
            outCh = channel_list[i]
            ks = kernel_list[i]
            s = stride_list[i]

            self.encoder.append(self._make_encoder_layer(inCh, outCh, ks, 
                                                         s, is_finalLayer))
        
        # MAKE DECODER
        for i in reversed(list(range(self.numLayers))):
            if i == 0:
                outCh = 1
                is_finalLayer = True
            else:
                outCh = channel_list[i-1]
                is_finalLayer = False
            inCh = channel_list[i]
            ks = kernel_list[i]
            s = stride_list[i]

            self.decoder.append(self._make_decoder_layer(inCh, outCh, ks, 
                                                         s, is_finalLayer))


    def _make_stem_layer(self, block_type, in_ch, out_ch, ks, s):

        if block_type == "cnn":
            pad = (ks-1)//2
            model_block = [nn.Conv1d(in_channels=in_ch, 
                                    out_channels=out_ch, 
                                    kernel_size=ks, 
                                    stride=s, 
                                    padding=pad,
                                    bias=True, 
                                    padding_mode="zeros")]
        
        if self.stem_batch_norm:
            model_block.append(nn.BatchNorm1d(out_ch))
        if self.stem_actfn:
            model_block.append(self.stem_actfn)
           
        return (nn.Sequential(*model_block))
    
    def _make_encoder_layer(self, inCh, outCh, ks, s, is_finalLayer):
            
            k_h, k_w = self._normalize_ks(ks)
            s_h, s_w = self._normalize_stride(s)
            
            # pad = (ks-1)//2
            pad = ((k_h - 1) // 2, (k_w - 1) // 2)
            
            model_block = [nn.Conv2d(in_channels = inCh,
                                     out_channels = outCh,
                                     kernel_size = (k_h, k_w),
                                     stride = (s_h, s_w),
                                     padding = pad,
                                     bias = True,
                                     padding_mode = "zeros")]
            
            if not is_finalLayer:
                if self.batch_norm:
                    model_block.extend([nn.BatchNorm2d(outCh), 
                                        self.act_fn])
                else:
                    model_block.extend([self.act_fn])
            else:
                if self.apply_encLastAct: # if False, last encoder layer is convolution output
                    model_block.append(self.last_act_fn) # NOTE: no Batch Norm in final layer

            return(nn.Sequential(*model_block))

    def _make_decoder_layer(self, inCh, outCh, ks, s, is_finalLayer):
        
        k_h, k_w = self._normalize_ks(ks)
        s_h, s_w = self._normalize_stride(s)

        pad = ((k_h - 1) // 2, (k_w - 1) // 2)
        out_pad = (s_h - 1, s_w - 1)

        # pad = (ks-1)//2
        # out_pad = tuple([i-1 for i in s]) # for asymmetric strides in encoder
        model_block = [nn.ConvTranspose2d(in_channels = inCh,
                                          out_channels = outCh,
                                          kernel_size = (k_h, k_w), 
                                          stride = (s_h, s_w),
                                          padding = pad,
                                          output_padding = out_pad,
                                          bias = True,
                                          padding_mode = "zeros")]

        if not is_finalLayer:
            if self.batch_norm:
                model_block.extend([nn.BatchNorm2d(outCh), 
                                    self.act_fn])
            else:
                model_block.extend([self.act_fn])
        else:
            model_block.append(self.last_act_fn) 
        
        return(nn.Sequential(*model_block))
        
    def _normalize_ks(self, ks):
        # following changes meant to add support for asymmetric kernels
        # ks can be int or (h,w)
        if isinstance(ks, int):
            return (ks, ks)
        elif isinstance(ks, (list, tuple)) and len(ks) == 2:
            return (int(ks[0]), int(ks[1]))
        else:
            raise ValueError(f"Invalid kernel_size: {ks}")

    def _normalize_stride(self, s):
        # following changes meant to add support for asymmetric strides
        # s can be int or (sh,sw)
        if isinstance(s, int):
            return (s, s)
        elif isinstance(s, (list, tuple)) and len(s) == 2:
            return (int(s[0]), int(s[1]))
        else:
            raise ValueError(f"Invalid stride: {s}")

    def forward(self, x, batch_pe = None):
        
        x = torch.unsqueeze(x,1) # (B,sps,L)-->(B,1,sps,L)        

        if self.posn_embed:
            assert batch_pe is not None, "batch_pe must be provided if posn_embed is enabled"
            
            if self.stem_dict: # type 2 PE
                for stem_layer in self.stem_layers:
                    x = stem_layer(x)
            else: # type 1 PE
                 batch_pe = torch.unsqueeze(batch_pe, 1) # (B,sps,L) --> (B,1,sps,L)

            if self.posn_embed.split("_")[1] == "sum":
                    x = x + batch_pe 
            else:
                x = torch.concat((x,batch_pe), axis = 1) 

        # Encode
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
        encoder_embed = x
        
        # Decode
        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        return(x, encoder_embed)