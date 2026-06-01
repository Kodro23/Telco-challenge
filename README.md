<!-- README TOP -->
<a name="readme-top"></a>

# The AI Telco Troubleshooting Challenge 

<!-- Project presentation -->
## 👨‍🏫1. Project goal
[The AI Telco Troubleshooting Challenge by ITU](https://zindi.africa/competitions/the-ai-telco-troubleshooting-challenge) aims to use telelogs, the automatically generated fault and event logs produced by network equipment, to predict root case of network failures. 

### 2. Data
Training data is a table including:
- ID of telelog (2400 in total) 
- Content of telelog with information on location, mobility, radio signal quality, performane of device
- Root cause among 8 possibilities:
C1: The serving cell's downtilt angle is too large, causing weak coverage at the far end.
C2: The serving cell's coverage distance exceeds 1km, resulting in over-shooting.
C3: A neighboring cell provides higher throughput.
C4: Non-colocated co-frequency neighboring cells cause severe overlapping coverage.
C5: Frequent handovers degrade performance.
C6: Neighbor cell and serving cell have the same PCI mod 30, leading to interference.
C7: Test vehicle speed exceeds 40km/h, impacting user throughput.
C8: Average scheduled RBs are below 160, affecting throughput.

### 3. Method
- Process data by reformating long telelog text content columns with characteristics of device.
- 



### 📜2. Poem generation 
We use OpenAI's GPT-2 as base, which is a trained large-scale unsupervised language model, which generates coherent paragraphs of text ([OpenAI's GPT-2 ](https://openai.com/index/better-language-models/)). We use GPT-2 as it is fully open source with no API costs and have low hardware requirements, even though recent versions (GPT-3, GPT-4) are more efficient.
The model is fine-tuned on poems's dataset. For the generation, the label of the input image is used as the theme of the poem, whether it's a classical english poem or an haiku.

### ⚠️3. Disclaimer
The poems are a little wonky.

### 🧰4. Built with
* python 3.12.9

### 📈5. Improvements
Points of improvement:
* Enrich the datasets with more poems, with better filtering of texts on their quality
* Readjust the parameters for the fine-tuning of gpt2
* Readjust parameters for the poem generator
* More esthetic use friendly API
......