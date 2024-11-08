# Awesome Demonstration Selection Algorithms

## Our works

🔥🔥🔥 **Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning**

**[Paper](https://arxiv.org/pdf/2410.23099)** | **[Project Page [This Page]](https://github.com/Tizzzzy/Demonstration_Selection_Overview)**

The first comparative analysis for Demonstration Selection Algorithms. :sparkles: </div>  

## News

- [11/2024] Our paper has been accepted by AAAI-25 Student Abstract!

---

This includes an original implementation of "[Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning](https://arxiv.org/pdf/2410.23099)" by [Dong Shu](https://scholar.google.com/citations?user=KfIlTroAAAAJ&hl=en) and [Mengnan Du](https://mengnandu.com/).

This code provides:
- Links to the paper and GitHub repository related to the **demonstration selection** topic.
- Codes for evaluating the demonstration selection algorithm used in the experiments.

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper:
```
@article{shu2024comparative,
  title={Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning},
  author={Shu, Dong and Du, Mengnan},
  journal={arXiv preprint arXiv:2410.23099},
  year={2024}
}
```

Demonstration selection algorithms play a crucial role in enhancing the performance of Large Language Models (LLMs) on various tasks. These algorithms assist users in selecting the best k input-label pairs (demonstration examples) based on a given test input, enabling LLMs to in-context learn the relationship between the provided examples and the test inputs. Despite all the proposed demonstration selection algorithms, their efficiency and effectiveness remain unclear. This lack of clarity make it difficult to apply these algorithms in real-world scenarios and poses challenges for future research aimed at developing improved methods. This paper revisits seven proposed algorithms, evaluating them on five datasets from both efficiency and effectiveness perspectives.

## Content

1. [Existing Paper Overview](#existing-paper-overview)
2. [Reproducing Main Experiments](#reproducing-main-experiments) (Section `Experiments` of the paper)
   * [CBDS](#cbds)
   * [RD-direct](#rd-direct)
   * [RD-channel](#rd-channel)
   * [LLM Retriever](#llm-retriever)
   * [UPRISE](#uprise)
   * [OpenICL TopK](#openicl-topk)
   * [OpenICL Random](#openicl-random)

## Existing Paper Overview
The table below tracks all existing demonstration selection algorithms. Please leave issues for any unlisted papers or code.

| Paper              | Github | Maintain | Year | Approach |
| :---------------- | :------: | :----: | :----: | :---- |
| [Active Example Selection for In-Context Learning](https://arxiv.org/pdf/2211.04486) |   [True](https://github.com/ChicagoHAI/active-example-selection?tab=readme-ov-file)   | Archived | 2022 | Uses an active learning framework to iteratively select the most informative examples that improve in-context learning performance.|
| [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/pdf/2202.12837) |  [True](https://github.com/Alrope123/rethinking-demonstrations?tab=readme-ov-file)   | Yes | 2022 | Investigates various factors that influence the effectiveness of demonstrations, including similarity to the input and diversity among the demonstrations. |
| [Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator](https://arxiv.org/pdf/2206.08082) |  [True](https://github.com/heyjoonkim/sg_icl) | No | 2022 | Uses auto-regressive language models to generate demonstrations on the fly, reducing dependency on pre-existing datasets.|
| [Can language models learn from explanations in context?](https://arxiv.org/pdf/2204.02329) |  False | -- | 2022 | Investigates whether adding explanations to demonstrations can improve in-context learning performance and understanding.|
| ---------------- | ------ | ---- | ---- | ---- |
| [Compositional Exemplars for In-context Learning](https://arxiv.org/pdf/2302.05698) |  [True](https://github.com/HKUNLP/icl-ceil?tab=readme-ov-file) | No | 2023 | Focuses on selecting compositional examples that can generalize well to new tasks by leveraging the compositional structure of language.|
| [Unified Demonstration Retriever for In-Context Learning](https://arxiv.org/pdf/2305.04320) |  [True](https://github.com/KaiLv69/UDR?tab=readme-ov-file) | Yes | 2023 | Proposes a unified retriever system that selects demonstrations from multiple sources based on their relevance and effectiveness for the target task.|
| [OpenICL: An Open-Source Framework for In-context Learning](https://arxiv.org/pdf/2303.02913) |  [True](https://github.com/Shark-NLP/OpenICL?tab=readme-ov-file) | No | 2023 | OpenICL provides an easy interface for in-context learning, with many state-of-the-art retrieval and inference methods built in to facilitate systematic comparison of LMs and fast research prototyping. |
| ---------------- | ------ | ---- | ---- | ---- |
| [DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task](https://arxiv.org/pdf/2406.16332) |  [True](https://github.com/8421bcd/demorank) | Yes | 2024 | Proposes a ranking-based method to select demonstrations that are most likely to improve model performance on ranking tasks.|
| [Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning](https://arxiv.org/pdf/2301.11916) |  [True](https://github.com/WANGXinyiLinda/concept-based-demonstration-selection)   | No | 2024 | Proposes a method to find good demonstrations by modeling them as latent variables and optimizing their selection using probabilistic methods.|
| [Learning to Retrieve In-Context Examples for Large Language Models](https://arxiv.org/pdf/2307.07164) |  [True](https://github.com/microsoft/LMOps/tree/main/llm_retriever) | Yes | 2024 | Uses retrieval models to select examples from a large corpus that are most similar to the test input, improving in-context learning performance.|
| [Enhancing In-Context Learning via Implicit Demonstration Augmentation](https://arxiv.org/pdf/2407.00100) |  False | -- | 2024 | Enhances in-context learning by augmenting the training data with implicitly generated demonstrations that are contextually relevant.|
| [In-Context Learning with Iterative Demonstration Selection](https://arxiv.org/pdf/2310.09881) |  False | -- | 2024 | Iteratively selects and refines demonstrations based on their impact on model performance, aiming for optimal demonstration sets.|
| [IN-CONTEXT LEARNING DEMONSTRATION SELECTION VIA INFLUENCE ANALYSIS](https://arxiv.org/pdf/2402.11750) |  False | -- | 2024 | Uses influence functions to analyze the impact of different demonstrations on model predictions and selects those with the highest positive influence.|
| [Unraveling the Mechanics of Learning-Based Demonstration Selection for In-Context Learning](https://arxiv.org/pdf/2406.11890) |  False | -- | 2024 | Examines the underlying mechanics of various learning-based demonstration selection strategies and their impact on in-context learning.|
| [COMPARABLE DEMONSTRATIONS ARE IMPORTANT IN IN-CONTEXT LEARNING: A NOVEL PERSPECTIVE ON DEMONSTRATION SELECTION](https://arxiv.org/pdf/2312.07476) |  False | -- | 2024 | Emphasizes the importance of selecting comparable demonstrations that are similar in difficulty and structure to the target task.|
| [The Impact of Demonstrations on Multilingual In-Context Learning: A Multidimensional Analysis](https://arxiv.org/pdf/2402.12976) |  [True](https://github.com/uds-lsv/multilingual-icl-analysis?tab=readme-ov-file) | Unsure | 2024 | Conducts a multidimensional analysis of how different types of demonstrations affect multilingual in-context learning performance.|
| [MDR: Model-Specific Demonstration Retrieval at Inference Time for In-Context Learning](https://aclanthology.org/2024.naacl-long.235.pdf) |  [True](https://github.com/kiming-ng/MDR) | Yes | 2024 | Implements a model-specific retrieval system that selects demonstrations at inference time based on model predictions and task requirements.|
| [Revisiting Demonstration Selection Strategies in In-Context Learning](https://arxiv.org/pdf/2401.12087v2) |  [True](https://github.com/romainpkq/revisit_demon_selection_in_icl) | Yes | 2024 | Revisits and compares various demonstration selection strategies, providing insights into their effectiveness and limitations.|
| [In-context Learning with Retrieved Demonstrations for Language Models: A Survey](https://arxiv.org/pdf/2401.11624) |  False | -- | 2024 | Surveys different methods of retrieving demonstrations for in-context learning and compares their effectiveness across various tasks.|


## Reproducing Main Experiments

This is for reproducing experiments in Section `Experiments` of the paper.
Evaluation datasets are:
* Classification (3 datasets): `glue-mrpc`, `glue-qnli`,`glue-sst2`,
* Multi-choice (2 datasets): `commonsense_qa`, `swag`

1. Requesting model access from META
    * visit this [link](https://ai.meta.com/blog/meta-llama-3/) and request the access to the LLaMa3-8B model. 
2. Requesting model access from Hugging Face
    * Once request is approved, use the same email adrress to get the access of the model from HF [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
3. Authorising HF token
    * Once HF request to access the model has been approved, create huggingface token [here](https://huggingface.co/settings/tokens). Run below code and enter your token. It will authenticate your HF account
        ```python
        >>> huggingface-cli login
        
        or
        
        >>> from huggingface_hub import login
        >>> login(YOUR_HF_TOKEN)
        ```

    * Once you successfully login, follow the below algorithms.

Below are all the algorithms we tested in our paper. To run those algorithms, please redirect to the algorithm folder by clicking the link.
#### [CBDS](https://github.com/Tizzzzy/Demonstration_Selection_Overview/tree/main/CBDS)
* Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning
#### [RD-direct](https://github.com/Tizzzzy/Demonstration_Selection_Overview/tree/main/RD-direct)
* Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
#### [RD-channel](https://github.com/Tizzzzy/Demonstration_Selection_Overview/tree/main/RD-channel)
* Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
#### [LLM Retriever](https://github.com/Tizzzzy/Demonstration_Selection_Overview/tree/main/LLM-Retriever)
* Learning to Retrieve In-Context Examples for Large Language Models
#### [UPRISE](https://github.com/Tizzzzy/Demonstration_Selection_Overview/tree/main/UPRISE)
* UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation
#### [OpenICL TopK](https://github.com/Tizzzzy/Demonstration_Selection_Overview/tree/main/OpenICL-TopK)
* OpenICL: An Open-Source Framework for In-context Learning
#### [OpenICL Random](https://github.com/Tizzzzy/Demonstration_Selection_Overview/tree/main/OpenICL-Random)
* OpenICL: An Open-Source Framework for In-context Learning

### 📞 Contact

If you have any question or suggestion related to this project, feel free to open an issue or pull request.

### ✨ Citation

If you find this repository useful, please consider giving a star ⭐ and citation

```
@article{shu2024comparative,
  title={Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning},
  author={Shu, Dong and Du, Mengnan},
  journal={arXiv preprint arXiv:2410.23099},
  year={2024}
}
```
