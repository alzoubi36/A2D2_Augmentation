## Bachelor Thesis: Generating Synthetic Combined Camera and LiDAR Data for 3D Object Detection

This repository contains the research work done in the bachelor's thesis of Mohammad Al Zoubi with the title "Generating Synthetic Combined Camera and LiDAR Data for 3D Object Detection". The thesis was supervised by Prof. Dr.-Ing. Christian Hühne and advised by Marco Brysch, Dr. Cay-Christian Kalmbach, and Christian Freystein at the Institute of Mechanics and Adaptronics at TU Braunschweig.

The full thesis can be found [here](https://drive.google.com/file/d/1PYzMn8CMUyDeBIYRjsI1SSUXuRhjm5DA/view?usp=sharing).

## Research Objective

This research aims to develop a novel algorithm for generating synthetic data from combined camera and LiDAR data for 3D object detection. The A2D2 dataset for autonomous driving released by Audi was used as the basis for this contribution. The thesis investigates the feasibility of generating realistic synthetic data by placing foreground objects from one frame into other frames while ensuring no collisions with other objects.

## Repository Structure

This repository is organized into the following subdirectories:

* **evaluation**: Contains scripts and code for evaluating the generated synthetic data.
* **generation**: Includes scripts and code for generating synthetic data from the A2D2 dataset.
* **utils**: Provides utility functions and scripts used across the project.
* **data**: Contains the processed A2D2 dataset used for the research.
* **reports**: Includes the thesis document and other relevant reports.

## Getting Started

1. Clone the repository:

```
git clone https://github.com/mohammad-al-zoubi/a2d2-augmentation.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the following:
```
a2d2-augmentation/synthetic_data_generator.py
```

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Disclaimer

Please note that this repository contains the research work of Mohammad Al Zoubi and is not an official release of the A2D2 dataset or any other related projects. The code and data provided here are for educational and research purposes only.
