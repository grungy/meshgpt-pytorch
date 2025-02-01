class QuantizerMixin:
    def __init__(
        self,
        dim_codebook,
        num_quantizers,
        codebook_size,
        use_residual_lfq,
        rq_kwargs,
        rvq_kwargs,
        rlfq_kwargs,
        rvq_stochastic_sample_codes,
        checkpoint_quantizer
    ):
        self.dim_codebook = dim_codebook
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.use_residual_lfq = use_residual_lfq
        self.rq_kwargs = rq_kwargs
        self.rvq_kwargs = rvq_kwargs
        self.rlfq_kwargs = rlfq_kwargs
        self.rvq_stochastic_sample_codes = rvq_stochastic_sample_codes
        self.checkpoint_quantizer = checkpoint_quantizer

        # Initialize quantizers here
        # Example: self.quantizer = SomeQuantizerClass(...) 