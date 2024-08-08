## Environment Setup

Follow the instructions in [UPRISE](https://github.com/microsoft/LMOps/tree/main/uprise) to clone the repo and download pre-trained retriever and pre-constructed demonstration pool. 

Once the prompt pool is encoded, you can run inference with LLaMa3 by using the following script:

```bash
export TASK={} # task name for evaluation, should be the same as the name in the task.py file
export LLM="meta-llama/Meta-Llama-3-8B" # LLM for inference

bash inference_hf.sh
```

Note that currently, the `inference_hf.sh` supports vanill zero-shot baseline. If you don't want this, just simply comment them out. To experiment with different k demonstration examples, kindly change `num_prompts`.
