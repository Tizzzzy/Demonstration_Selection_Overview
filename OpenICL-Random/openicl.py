from huggingface_hub import login
login("hf_")

from datasets import load_dataset
from openicl import DatasetReader, PromptTemplate, TopkRetriever, PPLInferencer, RandomRetriever
from openicl import AccEvaluator

dataset = load_dataset('gpt3mix/sst2')
data = DatasetReader(dataset, input_columns=['text'], output_column='label')

template = PromptTemplate(template={
                                        0: 'Positive Movie Review: </text>',
                                        1: 'Negative Movie Review: </text>' 
                                    },
                          column_token_map={'text' : '</text>'} 
           )

# TopK Retriever
retriever = RandomRetriever(data, ice_num=10)

# Define a Inferencer
inferencer = PPLInferencer(model_name='meta-llama/Meta-Llama-3-8B')

# Inference
predictions = inferencer.inference(retriever, ice_template=template, output_json_filename='sst2')
print(predictions)
print(data.references)
score = AccEvaluator().score(predictions=predictions, references=data.references)
print(score)