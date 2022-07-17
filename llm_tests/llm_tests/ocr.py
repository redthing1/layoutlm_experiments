import os
import time
import requests

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials



class MicrosoftReadOCR:
    def __init__(self):
        endpoint = os.environ["ACCOUNT_ENDPOINT"]
        key = os.environ["ACCOUNT_KEY"]
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    def simplify_bbox(self, raw_bbox):
        return [raw_bbox[0], raw_bbox[1], raw_bbox[4], raw_bbox[5]]

    def analyze_file(self, in_file):
        # Open the image
        read_image = open(in_file, "rb")

        # Call API with image and raw response (allows you to get the operation location)
        read_response = self.client.read_in_stream(read_image, raw=True)
        # Get the operation location (URL with ID as last appendage)
        read_operation_location = read_response.headers["Operation-Location"]
        # Take the ID off and use to get results
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for the retrieval of the results
        print("waiting for MS Read...")
        while True:
            read_result = self.client.get_read_result(operation_id)
            if read_result.status.lower() not in ["notstarted", "running"]:
                break
            time.sleep(1)

        # Print results, line by line
        results = []
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    # x1, y1, x2, y2 = self.simplify_bbox(line.bounding_box)
                    # simple_box = [x1, y1, x2, y2]
                    
                    # # print(line.text)
                    # # print(simple_box)
                    # # print('ms read result:', line.text, simple_box)
                    # simple_words = [word.text for word in line.words]
                    # results.append((simple_words, simple_box))
                    # add words and bounding boxes
                    for word in line.words:
                        results.append([word.text, self.simplify_bbox(word.bounding_box)])
        # print('ms read results:', results)

        return results


class TrOCR:
    def __init__(self, model):
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
