<!-- README TOP -->
<a name="readme-top"></a>

# The AI Telco Troubleshooting Challenge 

<!-- Project presentation -->
## 👨‍🏫1. Project presentation
[The AI Telco Troubleshooting Challenge by ITU](https://zindi.africa/competitions/the-ai-telco-troubleshooting-challenge) aims to use telelogs, the automatically generated fault and event logs produced by network equipment, to predict root case of network failures. 

### 1. Data
Training data is a table including:
- ID of telelog (2400 logs in total) 
- Content of telelog with information on location, mobility, radio signal quality, performane of device
- Root cause among 8 possibilities:\
    C1: The serving cell's downtilt angle is too large, causing weak coverage at the far end.\
    C2: The serving cell's coverage distance exceeds 1km, resulting in over-shooting.\
    C3: A neighboring cell provides higher throughput.\
    C4: Non-colocated co-frequency neighboring cells cause severe overlapping coverage.
    C5: Frequent handovers degrade performance.\
    C6: Neighbor cell and serving cell have the same PCI mod 30, leading to interference.\
    C7: Test vehicle speed exceeds 40km/h, impacting user throughput.\
    C8: Average scheduled RBs are below 160, affecting throughput.\

### 2. Method and results
- Process data by reformating long telelog text content into columns representing characteristics of device. 

Going from raw format

| ID            | Content                                                                                  | root cause     |
|---------------|------------------------------------------------------------------------------------------|----------------|
| ID_1P7PJMPV0R | Analyze the 5G wireless network drive-test user plane data and engineering parameters... | C2             |
| ID_8B1D1TUTFA | Analyze the 5G wireless network drive-test user plane data and engineering parameters... | C1             |
| ID_IGGXMA9GZH | Analyze the 5G wireless network drive-test user plane data and engineering parameters... | C2             |
| ID_D6C9N2X295 | Analyze the 5G wireless network drive-test user plane data and engineering parameters... | C2             |
| ID_8JC15PNP3Q | Analyze the 5G wireless network drive-test user plane data and engineering parameters... | C5             |

to 
| Timestamp           | Longitude  | Latitude  | GPS Speed (km/h) | PCI | RSRP (dBm) | SINR (dB) | DL Throughput (Mbps) | Top1 PCI | Top2 PCI | Mechanical Downtilt | Digital Tilt | Beam Scenario | Height | TxRx Mode | Max Tx Power | Antenna Model | ID |
|--------------------|------------|-----------|------------------|-----|------------|-----------|----------------------|----------|----------|---------------------|--------------|--------------|--------|----------|--------------|---------------|----------------|
| 2025-05-07 15:23:52 | 128.188169 | 32.579273 | 1                | 712 | -77.00     | 15.93     | 1351.25              | 258      | 71       | 6.0                 | 4.0          | DEFAULT      | 5.0    | 64T64R   | 34.9         | NR AAU 2      | ID_1P7PJMPV0R |
| 2025-05-07 15:23:53 | 128.188140 | 32.579223 | 2                | 71  | -80.97     | 6.60      | 366.57               | 258      | 712      | 3.0                 | 10.0         | DEFAULT      | 29.7   | 32T32R   | 34.9         | NR AAU 1      | ID_1P7PJMPV0R |
| 2025-05-07 15:23:54 | 128.188117 | 32.579174 | 16               | 71  | -85.50     | 1.81      | 334.00               | 258      | 712      | 3.0                 | 10.0         | DEFAULT      | 29.7   | 32T32R   | 34.9         | NR AAU 1      | ID_1P7PJMPV0R |
| 2025-05-07 15:23:55 | 128.188103 | 32.579113 | 14               | 71  | -88.21     | 5.40      | 431.94               | 712      | 258      | 3.0                 | 10.0         | DEFAULT      | 29.7   | 32T32R   | 34.9         | NR AAU 1      | ID_1P7PJMPV0R |
| 2025-05-07 15:23:56 | 128.188088 | 32.579075 | 19               | 71  | -78.45     | 13.59     | 566.34               | 712      | 258      | 3.0                 | 10.0         | DEFAULT      | 29.7   | 32T32R   | 34.9         | NR AAU 1      | ID_1P7PJMPV0R |

- Since each telelog content contains repeated measures by timestamp, we use a Long Short-Term Memory (LSTM) model to account for the time series structure. We obtain an accuray of 80% and F1-score of 83%.

### 🧰3. Built with
Python 3.13.13

### 📈4. Improvements
Points of improvement:
- Improve predictive performances (feature engineering,explore other deep learning models,...)
- More esthetic and user-friendly API
- Use a large language model (LLM): the original challenge has been thought to use the logs to fine-tune specialised LLMs capable of performing root-cause analysis. The advantage of LLMs are they are less dependent on a dataframe structure, but they are more computationally expensive. Our original plan was to compare both methods (deep learning and LLMs).
......

<!-- User's guide -->
## 📄II. User's guide
### 🧬1. How to use
- Clone the repository in python environment and go to dir "/Telco-challenge"
- In terminal:
    - Create vitual envrionment
    ```
    python -m venv name_of_environment
    source  name_of_environment/bin/activate
    pip install -r requirements.txt

    ```
    - To run the API use the command :
    ```
    uvicorn app.api:app --reload --host "0.0.0.0" --port 8000
    ```

The documentation of the API is accessible via the requests "/docs". 
- Select "POST/predict",
- Click "Try out" to make predictions,
- Paste in in appropriate format (see request.json for template), 
- Execute. 
- Predictions will be output in section "Response".