## Preparation:

Follow the instruction in [OpenICL](https://github.com/Shark-NLP/OpenICL?tab=readme-ov-file) to install OpenICL by either **Using Pip**:
```
pip install openicl
```
Or **Installation for local development:**
```
git clone https://github.com/Shark-NLP/OpenICL
cd OpenICL
pip install -e .
```

Then simply run `openicl.py`. Note the current code is an example of using `TopK` method to perform demonstration selection on `sst2` dataset. Then evaluate on LLaMa3 model. You can also experiment with different `k` by changing the parameter `ice_num`.

If you want to run `mrpc` dataset, please change `data` and `template` into:
``` python
dataset = load_dataset('nyu-mll/glue', name='mrpc')
data = DatasetReader(dataset, input_columns=['sentence1', 'sentence2'], output_column='label')

template = PromptTemplate(template={
                                        0: '</E>not equivalent: </sentence1> </sentence2>',
                                        1: '</E>equivalent: </sentence1> </sentence2>' 
                                   },
                          column_token_map={'sentence1' : '</sentence1>', 'sentence2' : '</sentence2>'},
                          ice_token='</E>'
           )
```

If you want to run `qnli` dataset, please change `data` and `template` into:
``` python
dataset = load_dataset('nyu-mll/glue', name='qnli')
data = DatasetReader(dataset, input_columns=['question', 'sentence'], output_column='label')

template = PromptTemplate(template={
                                        0: '</E>Entailment: </question> </sentence>',
                                        1: '</E>Not Entailment: </question> </sentence>' 
                                   },
                          column_token_map={'question' : '</question>', 'sentence' : '</sentence>'},
                          ice_token='</E>'
           )
```

If you want to run `commonsense_qa` dataset, please change `data` and `template` into:
``` python
dataset = load_dataset('tau/commonsense_qa')
data = DatasetReader(dataset, input_columns=['question', 'choices'], output_column='answerKey')

start_time = time.time()

template = PromptTemplate(template={
                                        'A': '</E>A: </question> </choices>',
                                        'B': '</E>B: </question> </choices>',
                                        'C': '</E>C: </question> </choices>',
                                        'D': '</E>D: </question> </choices>',
                                        'E': '</E>E: </question> </choices>'
                                   },
                         column_token_map={'question': '</question>', 'choices': '</choices>'},
                         ice_token='</E>',
          )
```

If you want to run `swag` dataset, please change `data` and `template` into:
``` python

dataset = load_dataset('allenai/swag')
data = DatasetReader(dataset, input_columns=['sent1', 'sent2', 'ending0', 'ending1', 'ending2', 'ending3'], output_column='label')

template = PromptTemplate(template={
                                        0: '</E>Context: </sent1> </sent2> Choices: </ending0>, </ending1>, </ending2>, </ending3> Answer: 0',
                                        1: '</E>Context: </sent1> </sent2> Choices: </ending0>, </ending1>, </ending2>, </ending3> Answer: 1',
                                        2: '</E>Context: </sent1> </sent2> Choices: </ending0>, </ending1>, </ending2>, </ending3> Answer: 2',
                                        3: '</E>Context: </sent1> </sent2> Choices: </ending0>, </ending1>, </ending2>, </ending3> Answer: 3',
                                   },
                         column_token_map={'sent1': '</sent1>', 'sent2': '</sent2>', 'ending0': '</ending0>', 'ending1': '</ending1>', 'ending2': '</ending2>', 'ending3': '</ending3>'},
                         ice_token='</E>',
          )
```
