# Hidden Engrams: Long Term Memory for Transformer Model Inference

State-of-the art transformer models like GPT3 can generate realistic text, but the window of text the transformer is able to look at is still relatively small.
Hidden Engrams aims to remedy this problem by introducing an approximation of long term memory using the transformer's hidden states. These values can then be used to quickly sort all past "memories" by relevance to the current input. Once sorted, an optimized prompt can be built including only the most relevant information

## Usage
First, ensure the transformer model you want to use is configured properly in transformer.py. Engrams are incompatible across different models.
To encode your own datasets, modify "encode.py" as needed to load your data
example.py provides a simple example use case for this: chat bots. Previous messages are encoded and stored, then used to build future prompts


This is a very early proof-of-concept. More to come soon!


```
@misc{hiddenengrams,
  author = {Luke Fay aka AeroScripts},
  title = {Hidden Engrams: Long Term Memory for Transformer Model Inference},
  howpublished = {\url{https://github.com/AeroScripts/HiddenEngrams}},
  year = 2021,
  month = June
}
```
