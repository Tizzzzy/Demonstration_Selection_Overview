## Preparation

Follow the instructions in [concept-based-demonstration-selection](https://github.com/WANGXinyiLinda/concept-based-demonstration-selection) to clone the repo.

Go to `config` foler and change `tune.json` to:
``` json
{"train": ["glue-mrpc", "glue-qnli", "glue-sst2", "commonsense_qa", "swag"]}
```

The code is tested with python 3.8.

Install the data dependencies and download the data.
```bash
conda create --name metaicl-data python=3.8
conda activate metaicl-data
pip install datasets==1.4.0 wget
cd preprocess
python run.py
```

After preprocesisng is done, come back to the main directory.
```bash
cd ../
conda deactivate
```

Now, install the model dependencies to run the model. Please note that the Transformer version is not compatible to the datasets library used to download the data, so make sure to use a different environment.
```
conda create --name metaicl python=3.8
conda activate metaicl
pip install torch==1.9.0
pip install git+https://github.com/huggingface/transformers.git@c37573806ab3526dd805c49cbe2489ad4d68a9d7
pip install sentence-transformers
```

## Latent concept learning

Follow the instructions in [concept-based-demonstration-selection](https://github.com/WANGXinyiLinda/concept-based-demonstration-selection/tree/main?tab=readme-ov-file#latent-concept-learning) to obtain the "concept" tokens.

## Demonstration selection

Follow the instructions in [concept-based-demonstration-selection](https://github.com/WANGXinyiLinda/concept-based-demonstration-selection/tree/main?tab=readme-ov-file#demonstration-selection) to obtain performance on test set. After this step, you will get a `concept-likelihood` folder. Then you can use this folder to obtain performance on LLaMa3.

## Test on LLaMa3

In `test.py`, the first change need to do is add `"meta-llama/Meta-Llama-3-8B"` in the choices of `parser.add_argument("--gpt")`

Then add these code in the begging of the `main` function:
``` python
if args.gpt.startswith("meta"):
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    llama_model.cuda()
    llama_model.eval()
    add_newlines = False
    metaicl_model = llama_model
```

Then add these code in the begging of the `run` function:
``` python
if args.gpt.startswith("meta"):
    # Tokenize the inputs for LLaMA3
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    # Prepare data
    dev_inputs = []
    for dp in dev_data:
        prompt = ""
        if args.use_demonstrations and train_data:
            # Concatenate demonstrations with the test input
            for demo in train_data:
                prompt += f"Input: {demo['input']}\nOutput: {demo['output']}\n\n"
        prompt += f"Input: {dp['input']}\nOutput:"

        tokenized_prompt = llama_tokenizer(prompt, return_tensors="pt").to("cuda")
        dev_inputs.append(tokenized_prompt)

    print("Testing data samples with demonstrations:")
    for i, (dp, tokenized_input) in enumerate(zip(dev_data[:5], dev_inputs[:5])):  # Print first 5 samples for brevity
        print(f"Input {i}: {dp['input']}")
        print(f"Ground truth {i}: {dp['output']}")
        print(f"Tokenized input {i}: {llama_tokenizer.decode(tokenized_input.input_ids[0])}")
    
    all_nlls = []
    predictions = []
    for inputs in dev_inputs:
        with torch.no_grad():
            outputs = metaicl_model.generate(input_ids=inputs.input_ids, 
                                             attention_mask=inputs.attention_mask,
                                             max_new_tokens=20,
                                             eos_token_id=[llama_tokenizer.eos_token_id],
                                             pad_token_id=llama_tokenizer.eos_token_id)
            response = outputs[0][inputs.input_ids.shape[-1]:]
            predictions.append(llama_tokenizer.decode(response, skip_special_tokens=True))
    
    print("Generated responses:")
    for i, pred in enumerate(predictions[:5]):  # Print first 5 predictions for brevity
        print(f"Prediction {i}: {pred}")

    # Evaluate predictions
    groundtruths = [dp["output"] for dp in dev_data]
    f1, acc = metaicl_data.evaluate(predictions, groundtruths, is_classification, return_all)
    
    return f1, acc, predictions, groundtruths, all_nlls, groundtruths
```

Finally, run `prior.sh` again, but this time set `MODEL` to `meta-llama/Meta-Llama-3-8B`
