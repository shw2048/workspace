# -*- coding:utf-8 -*-
'''
生成随机验证码（数字）
'''
import random
import string
import sys
import math
from PIL import Image,ImageDraw,ImageFilter,ImageFont

class Captcha(object):
    def __init__(self, size=(100, 40), fontSize=30):
        self.font = ImageFont.truetype(r'C:\Windows\Fonts\Arial.ttf', fontSize)
        self.size = size
        self.image = Image.new('RGBA', self.size, (255,) * 4)
        self.texts = self.randNum(5)

    def rotate(self):
        rot = self.image.rotate(random.randint(-10, 10), expand=0)
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        self.image = Image.composite(rot, fff, rot)

    def randColor(self):
        self.fontColor = (random.randint(0, 250), random.randint(0, 250), random.randint(0, 250))

    def randNum(self, bits):
        return ''.join(str(random.randint(0, 9)) for i in range(bits))

    def write(self, text, x):
        draw = ImageDraw.Draw(self.image)
        draw.text((x, 4), text, fill=self.fontColor, font=self.font)

    def writeNum(self):
        x = 10
        xplus = 15
        for text in self.texts:
            self.randColor()
            self.write(text, x)
            self.rotate()
            x += xplus
        return self.texts

    def save(self):
        self.image.save('captcha.bmp')


img = Captcha()
num = img.writeNum()
img.image.show()
img.save()
