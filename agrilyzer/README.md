# Agrilizer: The Digital Fertilizer of Data and Insight

**Agrilizer** is a modular agrometeorological intelligence system. The library provides a Python client for retrieving and visualizing NASA POWER daily weather data together with agrimetric parameters. The agrimetric parameters can be used in machine learning and data analysis alone, or be combined with weather data for a more comprehensive understanding. PyTorch Frame powers the multimodal models for crop yield prediction. It also features a configurable visualization with built-in themes, allowing clear visualization of weather and agrimetric data.


The data from the POWER API can be used for various agricultural purposes, such as:

* Crop Production and Land Use: Understanding land surface reflectance, land temperature, and vegetation greenness.
* Weather Impact on Crops: Tracking rainfall, droughts, and severe weather events to assess growing conditions and threats.
* Water Management: Calculating evaporation patterns and water runoff.

## Features

* 📡 Fetch daily weather data directly from the NASA POWER API
* 📊 Plot and overlay time series of weather and agrimetric data
* 🔍 Filter by date ranges and custom value thresholds
* 🧠 Built-in parameter descriptions for interpretability
* 🗺️ Interactive location mapping
* 🎨 Configurable visualization themes

## Getting Started

### Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/agrilizer.git
cd agrilizer
pip install -r requirements.txt
```

### Configuration

Modify `config.json` to adjust:

* Default weather parameters
* Colormaps and plot style
* NASA POWER base API URL

### Usage

```python
from agrilizer.analyzers import Agrilizer
from agrilizer.visualization import Visualizer
from agrilizer.config import cfg

client = Agrilizer("MySite", lat=-10.5, lon=-55.3, start="2023-01-01", end="2023-12-31")
df = client.get(long_names=True)

viz = Visualizer("MySite", cfg["POWER_PARAM_DESCRIPTIONS"], client.get)
viz.plot(cols=["T2M", "PRECTOTCORR"], ma=7, dual=True)
```

## Example Output

* 📈 Time-series plots with moving averages
* 🌦 Overlay plots for crop and weather data
* 🔁 Monthly aggregation with correlation analysis

## File Structure

```text
agrilizer/
│
├── config.py         # Loads and manages configuration
├── config.json       # Contains weather param definitions and plot settings
├── core.py           # Main client for NASA POWER data
├── dataviz.py        # Visualization utilities
```

## Requirements

* Python ≥ 3.8

* `pandas`, `requests`, `matplotlib`, `folium`, `numpy`

See `requirements.txt` for a full list.

## Author

Cleverson Matiolli — [github.com/matiollipt](https://github.com/matiollipt)

---


## Data Source

This library uses data from the [NASA POWER Project](https://power.larc.nasa.gov/), which is publicly available. NASA does not endorse this product.

All code in this repository is © Cleverson Matiolli, 2025, and is licensed under the MIT License.

## License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2025 Cleverson Matiolli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      
copies of the Software, and to permit persons to whom the Software is          
furnished to do so, subject to the following conditions:                       

The above copyright notice and this permission notice shall be included in     
all copies or substantial portions of the Software.                            

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN      
THE SOFTWARE.
```
