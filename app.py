from flask import Flask, render_template, request, send_from_directory
from PIL import Image, ImageOps
import os
import glob
import re
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import imagehash
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage import img_as_ubyte
import cv2
import hashlib
from skimage import measure, morphology, segmentation

app = Flask(__name__)

# Add the extract_foreground function


def extract_foreground(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    mask = np.zeros_like(thresh, dtype=np.uint8)
    cv2.drawContours(mask, cnt[0], -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width -
               pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill="white")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/', methods=['GET', 'POST'])
def index():
    print("Index route called")

    similarity_threshold = 97  # default value
    if request.method == 'POST':
        # get the threshold from the form
        similarity_threshold = int(request.form.get('similarity'))

    selected_PZN = 6560987
    if request.method == 'POST':
        selected_PZN = request.form.get('PZN')

    selected_shop = 'shop-apotheke'  # default value
    if request.method == 'POST':
        selected_shop = request.form.get('shop')

    grouped_results = {}
    if request.method == 'POST':
        PZNs = request.form.get('PZN')
        PZNs = [pzn.strip() for pzn in PZNs.split(',')]

        for PZN in PZNs:
            print(f"Processing PZN: {PZN}")

            try:
                reference_images = []
                image_dir = os.path.join(
                    app.root_path, 'static', 'images', PZN)
                image_files = glob.glob(os.path.join(image_dir, '*'))

                for image_file in image_files:
                    if os.path.isfile(image_file):
                        reference_image = Image.open(image_file)
                        reference_image = resize_with_padding(
                            reference_image, (256, 256))
                        reference_image = reference_image.convert("RGB")
                        reference_image_extracted = extract_foreground(
                            np.array(reference_image))
                        reference_image_filename = os.path.basename(image_file)
                        reference_images.append(
                            (reference_image, reference_image_extracted, reference_image_filename))

                URL = None
                if selected_shop == 'mycare.de':
                    URL = f"https://www.mycare.de/suche/dewd?q={PZN}"
                elif selected_shop == 'medpex.de':
                    URL = f"https://www.medpex.de/search.do?q={PZN}"
                else:
                    URL = f"https://www.shop-apotheke.com/arzneimittel/{PZN}/"

                getURL = requests.get(
                    URL, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(getURL.text, 'html.parser')

                images = soup.find_all('img')
                if selected_shop == 'medpex.de':
                    images = soup.find_all('img', class_='product')

                print(f"Found {len(images)} images")  # Debug statement

                analyzed_urls = set()

                PZN_results = []  # Initialize PZN_results list
                for image in images:
                    srcset = image.get('srcset')
                    if srcset:
                        urls = re.findall(r'(https?://\S+)', srcset)
                    else:
                        urls = [image.get('src')]

                    for url in urls:
                        # Skip if PZN is not in the URL and the shop is not medpex.de
                        if selected_shop != 'medpex.de' and PZN not in url:
                            continue

                        original_url = re.sub(
                            r'/images/\d+x\d+/', '/images/', url)
                        if original_url not in analyzed_urls:
                            response = requests.get(original_url)
                            try:
                                original_image = Image.open(
                                    BytesIO(response.content))
                                original_image = resize_with_padding(
                                    original_image, (256, 256))
                                original_image_extracted = extract_foreground(
                                    np.array(original_image))
                            except IOError:
                                print(
                                    f"Cannot open {original_url} as an image.")
                                continue

                            for i, (reference_image, reference_image_extracted, filename) in enumerate(reference_images):
                                hash_diff = imagehash.phash(Image.fromarray(reference_image_extracted)).__sub__(
                                    imagehash.phash(Image.fromarray(original_image_extracted)))

                                hash_similarity_percentage = 100 * \
                                    (1 - (hash_diff / 64))

                                # Convert images to grayscale for SSIM
                                grayA = Image.fromarray(
                                    reference_image_extracted).convert("L")
                                grayB = Image.fromarray(
                                    original_image_extracted).convert("L")

                                # Compute SSIM between two images
                                ssim_value = ssim(
                                    np.array(grayA), np.array(grayB))

                                # Inside the loop where you calculate similarity metrics:
                                heatmap_diff = cv2.absdiff(
                                    np.array(reference_image), np.array(original_image))
                                heatmap_diff_norm = heatmap_diff.astype(
                                    np.float32) / 255.0  # Normalize to [0, 1]
                                heatmap_similarity = 1 - \
                                    np.mean(heatmap_diff_norm)
                                heatmap_similarity_percentage = heatmap_similarity * 100.0

                                # Decide if the image is a match
                                status = "MATCH" if heatmap_similarity_percentage >= 99 else "NO MATCH"

                                # Calculate heatmap of differences
                                heatmap = cv2.absdiff(
                                    np.array(reference_image.convert("L")), np.array(original_image.convert("L")))
                                heatmap = img_as_ubyte(heatmap)
                                heatmap = cv2.applyColorMap(
                                    heatmap, cv2.COLORMAP_JET)

                                # Generate unique filename for heatmap
                                hash_object = hashlib.md5()
                                hash_object.update(filename.encode('utf-8'))
                                hash_object.update(
                                    original_url.encode('utf-8'))
                                heatmap_filename = f"{hash_object.hexdigest()}.png"

                                # Save heatmap image
                                heatmap_path = os.path.join(
                                    app.root_path, 'static', 'heatmaps', PZN, heatmap_filename)
                                os.makedirs(os.path.dirname(
                                    heatmap_path), exist_ok=True)
                                cv2.imwrite(heatmap_path, heatmap)

                                match_result = {
                                    "Reference_Image": filename,
                                    "Matched_Image": original_url,
                                    "Status": status,
                                    "Similarity": f"{hash_similarity_percentage:.2f}%",
                                    "SSIM_Similarity": f"{ssim_value * 100:.2f}%",
                                    "Heatmap_Path": os.path.join('heatmaps', PZN, heatmap_filename),
                                    "Heatmap_Similarity": heatmap_similarity_percentage
                                }
                                PZN_results.append(match_result)

                            analyzed_urls.add(original_url)

                grouped_results[PZN] = {
                    'reference_images': [filename for _, _, filename in reference_images],
                    'matches': PZN_results,
                    'product_url': URL
                }

            except Exception as e:
                print(f"Error processing PZN: {PZN}, error: {e}")

    return render_template('index.html', results=grouped_results, similarity_threshold=similarity_threshold, selected_PZN=selected_PZN, selected_shop=selected_shop)


if __name__ == "__main__":
    app.run(debug=True)
