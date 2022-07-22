import typer
import os
import sys
import tempfile
import json
from typing import Optional
from types import SimpleNamespace
import importlib.util

import torch
import pandas as pd
from PIL import Image
import editdistance
import cv2

from transformers import (
    AutoProcessor,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    LayoutLMv3FeatureExtractor,
)
from datasets import load_dataset, load_from_disk

COMBINE_ROW_DIFF = -1  # 8
COMBINE_COL_DIFF = -1  # 40


def cli(
    ocr_data_path: str,
    segified_data_path: str,
    row_diff: int,
    col_diff: int,
    tiny_subset: bool = False,
    root_dir: str = ".",
    debug: bool = False,
):
    global COMBINE_ROW_DIFF, COMBINE_COL_DIFF

    COMBINE_ROW_DIFF = row_diff
    COMBINE_COL_DIFF = col_diff

    ocr_dataset = load_from_disk(ocr_data_path)

    if tiny_subset:
        ocr_dataset = ocr_dataset.select(range(min(20, len(ocr_dataset))))

    print("ocr data:", ocr_dataset)
    print("ocr features:", ocr_dataset.features)

    print('running segify')
    segified_dataset = ocr_dataset.map(segify_boxes)
    # segified_dataset = ocr_dataset.map(lambda x: x)
    print('segify done')

    if debug:
        # show visualizations
        for item in segified_dataset:
            # get the image
            image = item["image"]
            # get the boxes
            boxes = item["boxes"]
            # get the words
            words = item["words"]

            old_boxes = item["old_boxes"]

            # draw the image and boxes
            # load the image from file
            image_file = f"{root_dir}/{image}"
            print("loading image:", image_file)
            raw_img = cv2.imread(image_file)
            h, w, ch = raw_img.shape

            print("marking up image")
            print("first word and box:", words[0], boxes[0])
            for i, (word, box, oldbox) in enumerate(zip(words, boxes, old_boxes)):
                box_col = (0, 0, 255)
                # draw box
                box_x1 = int(round((box[0] / 1000) * w))
                box_y1 = int(round((box[1] / 1000) * h))
                box_x2 = int(round((box[2] / 1000) * w))
                box_y2 = int(round((box[3] / 1000) * h))

                # print('box:', box, 'coords:', box_x1, box_y1, box_x2, box_y2)

                im = cv2.rectangle(
                    raw_img, (box_x1, box_y1), (box_x2, box_y2), box_col, 3
                )
                # draw word
                im = cv2.putText(
                    im,
                    word,
                    (box_x1, box_y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_col,
                    1,
                )

                # ob_x1 = int(round((oldbox[0] / 1000) * w))
                # ob_y1 = int(round((oldbox[1] / 1000) * h))
                # ob_x2 = int(round((oldbox[2] / 1000) * w))
                # ob_y2 = int(round((oldbox[3] / 1000) * h))
                # im = cv2.rectangle(
                #     im, (ob_x1, ob_y1), (ob_x2, ob_y2), (0, 255, 0), 1
                # )

            print("showing image")
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", im)

            k = cv2.waitKey(0)
            quit_now = False
            while k != ord(" "):
                if k == ord("q"):
                    quit_now = True
                    break
                k = cv2.waitKey(0)

            if quit_now:
                break

    print("segified data:", segified_dataset)
    segified_dataset = segified_dataset.remove_columns("old_boxes")

    print('saving segified data')
    segified_dataset.save_to_disk(segified_data_path)

    print('done')


def segify_boxes(row):
    global COMBINE_ROW_DIFF, COMBINE_COL_DIFF

    words = row["words"]
    boxes = row["boxes"]

    print("words:", words)
    print("boxes:", boxes)

    seg_boxes = []

    # segmentify all these boxes
    last_row_box = None
    boxes_in_last_row = []

    for i, (word, box) in enumerate(zip(words, boxes)):
        # print('i', i, 'word', word, 'box', box)

        # box format is x, y, x, y

        print(f"checking: {word} {box}")

        new_box = box  # untouched

        if last_row_box is None:
            last_row_box = box
            seg_boxes.append(box)
            print(" ate box (first):", new_box)
            continue

        # there is a last row box
        # check if we can combine this box with it
        # check if the X positions are close enough
        # box position numbers are normalized around 1000

        curr_box_ul_y = box[1]
        curr_box_ul_x = box[0]

        lastrow_ul_y = last_row_box[1]
        lastrow_ul_x = last_row_box[0]

        lastrow_ur_y = lastrow_ul_y
        lastrow_ur_x = last_row_box[2]

        lastrow_dr_y = last_row_box[3]
        lastrow_dr_x = lastrow_ur_x

        # diff in y from curr box UL to lastrow UR
        y_gap_diff = curr_box_ul_y - lastrow_ur_y
        x_gap_diff = curr_box_ul_x - lastrow_ur_x

        # print(' gap diffs:', y_gap_diff, x_gap_diff)

        if abs(y_gap_diff) < COMBINE_ROW_DIFF and abs(x_gap_diff) < COMBINE_COL_DIFF:
            # combine them, aka add this to that box's segment

            x_diff = box[0] - last_row_box[0]
            y_diff = box[1] - last_row_box[1]

            print(" combining", box, "with", last_row_box, "diffs:", x_diff, y_diff)

            # x diff should ALWAYS be small
            # assert x_diff > -COMBINE_ROW_DIFF, f'x_diff: {x_diff} should always be small'

            # adjust the box to account for the combined box

            # x1
            last_row_box[0] = min(box[0], last_row_box[0])
            # y1
            last_row_box[1] = min(box[1], last_row_box[1])
            # x2
            last_row_box[2] = max(box[2], last_row_box[2])
            # y2
            last_row_box[3] = max(box[3], last_row_box[3])

            # copy the new box
            new_box = last_row_box.copy()
            boxes_in_last_row.append(i)

            seg_boxes.append(new_box)
            print(" ate box (merge):", new_box)
        else:
            # failed to combine
            # flush old boxes in last row
            for old_box_i in boxes_in_last_row:
                # in case the box changed, set the box again
                seg_boxes[old_box_i] = last_row_box.copy()

            # set new last row box
            last_row_box = box
            boxes_in_last_row = []

            seg_boxes.append(box)
            print(" ate box (alone):", new_box)

    # sanity checking
    assert len(seg_boxes) == len(words), f"boxes and words mismatch: len(seg_boxes) != len(words)"

    # save new boxes
    row["boxes"] = seg_boxes
    row["old_boxes"] = boxes

    return row


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
