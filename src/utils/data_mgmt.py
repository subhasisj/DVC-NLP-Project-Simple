import logging
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET
import re

def process_posts(f_in, f_train, f_test, target_tag, split_ratio):
    for i, line in enumerate(tqdm(f_in,desc='Reading Posts')):
        try:
            line = line.strip()
            if line.startswith('<row'):
                attrib = ET.fromstring(line).attrib
                pid = attrib.get("Id", "")
                label = 1 if target_tag in attrib.get("Tags", "") else 0
                title = re.sub(r"\s+", " ", attrib.get("Title","")).strip()
                body = re.sub(r"\s+", " ", attrib.get("Body","")).strip()
                text = title + " " + body

                if random.random() > split_ratio:
                    f_train.write(f"{pid}\t{text}\t{label}\n")
                else:
                    f_test.write(f"{pid}\t{text}\t{label}\n")

        except Exception as e:
            msg = f"Skipping line: {i} Exception: {e}\n"
            logging.exception(msg)