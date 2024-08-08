## Preparation

Follow the instructions in [rethinking-demonstrations](https://github.com/Alrope123/rethinking-demonstrations/tree/main) to clone the repo and setup MetaICL codebase.

The code is tested with python 3.8.

Install the data dependencies and download the data.
```bash
conda conda create --name metaicl-data python=3.8
conda activate metaicl-data
pip install datasets==1.4.0 wget
cd preprocess
```

Currently, the `_build_gym.py`, is going to preprocess all the datasets. However, if we only want to preprocess 5 datasets to save time, please use our `_build_gym.py`.

Then run the preprocess code with varying `k`.
```bash
python _build_gym.py --build --n_proc=40 --do_test --test_k {4|8|10|20}
```

After preprocesisng is done, come back to the main directory.
```bash
cd ../
conda deactivate
```

Now, install the model dependencies to run the model. Please note that the Transformer version is not compatible to the datasets library used to download the data, so make sure to use a different environment.
```
conda conda create --name metaicl python=3.8
conda activate metaicl
pip install torch==1.9.0
pip install git+https://github.com/huggingface/transformers.git@c37573806ab3526dd805c49cbe2489ad4d68a9d7
```

Before you run the model, we should add LLaMa3 to the code. To do that, please change `model.py` from path `rethinking-demonstrations/metaicl` to our `model.py`. Also change `test.py` into our `test.py`.

After the changes, you can run the evaluation with this:
``` bash
# Direct LLaMa3
python test.py --dataset {dataset} --gpt2 llama3 --method direct --out_dir out/direct-llama3 --do_zeroshot --use_demonstrations --k {4|8|10|20} --seed 100,13,21,42,87 --test_batch_size 4

# Channel LLaMa3
python test.py --dataset {dataset} --gpt2 llama3 --method channel --out_dir out/channel-llama3 --do_zeroshot --use_demonstrations --k {4|8|10|20} --seed 100,13,21,42,87 --test_batch_size 4
```
Note that the code provide both `direct` and `channel` approach to evaluate the performance.

