#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TODO"""
from ccai.app import create_app
from io import BytesIO

client = create_app().test_client()

with open("/home/corneau/themes/firewatch/palette.png", "rb") as image:
    img = BytesIO(image.read())

data = {'file': (img, 'filename.png'), }

print(data)
res = client.post('/upload_file', data=data)
