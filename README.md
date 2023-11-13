# Image Comparison and Analysis Tool

## Description
This Flask application is designed to compare and analyze product images from different online shops. It uses image processing techniques to compare images, generate heatmaps for differences, and evaluate similarity metrics such as Structural Similarity Index (SSIM).

## Installation
To set up the Image Comparison and Analysis Tool, follow these steps:

1. Clone the repository:
git clone https://github.com/senadgfe/dh-image-comparison.git

2. Install the required Python packages:
pip install -r requirements.txt


## Usage
Run the application:
python3 app.py


Access the application through a web browser at `http://localhost:5000`.

## Features
- **Image Comparison**: Compares images of products from different online shops.
- **Heatmap Generation**: Generates heatmaps to visualize differences between images.
- **Similarity Metrics**: Uses SSIM and other metrics to quantify image similarity.
- **Web Scraping**: Extracts product images from specified URLs using BeautifulSoup.
- **Dynamic Thresholding**: Allows users to set a threshold for image similarity.

 