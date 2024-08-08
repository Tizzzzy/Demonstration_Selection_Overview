## Prerequisites

Follow the instructions in [LLM Retriever](https://github.com/microsoft/LMOps/tree/main/llm_retriever) to clone the repo and download the data.

The code is tested on python 3.9.19

### Install Dependencies

```shell
pip install -r requirements.txt
```

## Evaluate the Performance

```shell
OUTPUT_DIR=outputs/llm-retriever-base/ bash scripts/eval_retriever.sh intfloat/llm-retriever-base
```

Note, right now the `scripts/eval_retriever.sh` is testing all datasets. You can change that by changing the `EVAL_TASKS`. Also, to experiment with different k examples, you can change `N_SHOTS`. 
