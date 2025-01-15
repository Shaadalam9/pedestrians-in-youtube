# youtube-pedestrian

## Overview
Welcome to the YOLOv8 YouTube Video Analysis project! This repository contains code that extracts YouTube videos based on a mapping.csv file and performs object detection using YOLOv8. The primary objective of this work is to evaluate pedestrian behavior in a cross-country or cross-cultural context using freely available YouTube videos.

This study presents a comprehensive cross-cultural evaluation of pedestrian behavior during road crossings, examining variations between developed and developing states worldwide. As urban landscapes evolve and autonomous vehicles (AVs) become integral to future transportation, understanding pedestrian behavior becomes paramount for ensuring safe interactions between humans and AVs. Through an extensive review of global pedestrian studies, we analyze key factors influencing crossing behavior, such as cultural norms, socioeconomic factors, infrastructure development, and regulatory frameworks. Our findings reveal distinct patterns in pedestrian conduct across different regions. Developed states generally exhibit more structured and rule-oriented crossing behaviors, influenced by established traffic regulations and advanced infrastructure. In contrast, developing states often witness a higher degree of informal and adaptive behavior due to limited infrastructure and diverse cultural influences. These insights underscore the necessity for AVs to adapt to diverse pedestrian behaviors on a global scale, emphasizing the importance of incorporating cultural nuances into AV programming and decision-making algorithms. As the integration of AVs into urban environments accelerates, this research contributes valuable insights for enhancing the safety and efficiency of autonomous transportation systems. By recognizing and accommodating diverse pedestrian behaviors, AVs can navigate complex and dynamic urban settings, ensuring a harmonious coexistence with human road users across the globe.

## Usage of the code
The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code 😍😄 For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Getting Started
Tested with Python 3.9.19. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows:

**Step 1:**

Clone the repository
```command line
git clone https://github.com/Shaadalam9/pedestrians-in-youtube.git
```

**Step 2:**

Create a new virtual environment
```command line
python -m venv venv
```

**Step 3:**

Activate the virtual environment
```command line
source venv/bin/activate
```

On Windows use
```command line
venv\Scripts\activate
```

**Step 4:**

Install dependencies
```command line
pip install -r requirements.txt
```

**Step 5:**

Ensure you have the required datasets in the data/ directory, including the mapping.csv file.

**Step 6:**

Run the code:
```command line
python3 analysis.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Example

https://github.com/Shaadalam9/youtube-pedestrian/assets/88769183/5303f4a5-52a2-4230-bd05-89a53927a5be


## Results

Countries where the study has been conducted
[![Countries where the study has been conducted](figures/world_map.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/world_map.html)

Time to start crossing(sorted by countries)
[![Time to start crossing(sorted by coutries](figures/time_to_start_cross.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/time_to_start_cross.html)

Speed of crossing(sorted by countries)
[![Speed of crossing(sorted by coutries](figures/speed_of_crossing.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/speed_of_crossing.html)

Time to start crossing(sorted by average of day and night)
[![Time to start crossing(sorted by average of day and night)](figures/time_to_start_cross_by_avg.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/time_to_start_cross_by_avg.html)

Speed of crossing(sorted by average of day and night)
[![Speed of crossing(sorted by average of day and night)](figures/speed_of_crossing_by_avg.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/speed_of_crossing_by_avg.html)

Merged figure of the speed of crossing and time to start crossing(sorted by countries)
[![Merged figure of speed of crossing ad time to start crossing(sorted by coutries)](figures/consolidated.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/consolidated.html)

Correlation Matrix
[![Correlation Matrix based on average speed and time to start cross](figures/correlation_matrix_heatmap_averaged.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_averaged.html)

Correlation Matrix at day-time
[![Correlation Matrix at day-time](figures/correlation_matrix_heatmap_in_daylight.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_in_daylight.html)

Correlation Matrix at night-time
[![Correlation Matrix at night-time](figures/correlation_matrix_heatmap_in_night.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_in_night.html)

Correlation Matrix of Africa
[![Correlation Matrix of Africa](figures/correlation_matrix_heatmap_Africa.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_Africa.html)

Correlation Matrix of Asia
[![Correlation Matrix of Asia](figures/correlation_matrix_heatmap_Asia.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_Asia.html)

Correlation Matrix of Australia
[![Correlation Matrix of Australia](figures/correlation_matrix_heatmap_Oceania.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_Oceania.html)

Correlation Matrix of Europe
[![Correlation Matrix of Europe](figures/correlation_matrix_heatmap_Europe.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_Europe.html)

Correlation Matrix of North America
[![Correlation Matrix of North America](figures/correlation_matrix_heatmap_North_America.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_North_America.html)

Correlation Matrix of South America
[![Correlation Matrix of South America](figures/correlation_matrix_heatmap_South_America.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/correlation_matrix_heatmap_South_America.html)

## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com
