#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TODO"""
from ccai.app import create_app
from io import BytesIO
from zipfile import ZipFile

client = create_app().test_client()

with open("/home/corneau/themes/firewatch/palette.png", "rb") as image:
    img = BytesIO(image.read())

data = {'file': (img, 'filename.png'), }

print(data)
res = client.post('/upload_file', data=data)

with ZipFile('images.zip', 'w') as _zip:
    _zip.write('/home/corneau/themes/firewatch/firewatch.png', 'firewatch.png')

with open('./images.zip', 'rb') as _zip:
    zp = BytesIO(_zip.read())

data = {'file': (zp, 'images.zip'), }

print(data)
res = client.post('/upload_file', data=data)
