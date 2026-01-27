import torch
import torch.nn as nn


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
        
        # Decode
        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        return x