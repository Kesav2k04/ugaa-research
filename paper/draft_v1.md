## Abstract

Visual language models (VLMs) hallucinate objects that are not present in images. Existing mitigations require retraining or fine‑tuning, which is costly and brittle. We propose **UGAA**, a training‑free inference‑time intervention that applies a constant *NO‑bias* to the LLM decoder’s yes/no output logits. On the POPE benchmark, UGAA achieves **F1 = 0.8211** (+0.017 over the baseline). On the What’s Up spatial‑reasoning benchmark, log‑likelihood scoring reaches **89 % accuracy**. Ablation reveals that attention‑based certainty gating provides no benefit over a fixed bias, suggesting the decoder logit gap already encodes visual grounding.

## 4. Ablation Finding

All attention‑based certainty variants (entropy, magnitude, CLS, object‑word) produce identical or worse results compared to a uniform constant bias. We conclude that the decoder’s yes/no logit gap is already discriminative — additional attention probing adds noise rather than signal on the POPE benchmark. This insight redirects future work toward leveraging the decoder’s intrinsic uncertainty signal rather than complex attention mechanisms.
