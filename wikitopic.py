#!/usr/bin/env python3

from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import cv2
import json
import luigi
import luigi.contrib.external_program
import numpy as np
import re
import requests


url = 'https://en.wikipedia.org/w/api.php'
cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6})|parser|MW|template|output;')


def remove_html(html):
    cleantext = re.sub(cleanr, '', html)
    return cleantext


class DownloadText(luigi.Task):
    topic = luigi.Parameter()

    def output(self):
        filename = '{}.html'.format(self.topic)
        return luigi.LocalTarget(filename)

    def run(self):
        params = {
            'action': 'parse',
            'prop':   'text',
            'format': 'json',
            'page':   self.topic,
        }
        response = requests.get(url, params=params, verify=False)

        j = json.loads(response.text)
        text = j['parse']['text']['*']
        with self.output().open('w') as f:
            cleaned = remove_html(text)
            f.write(cleaned)


class GetImageUrl(luigi.Task):
    topic = luigi.Parameter()

    def output(self):
        filename = '{}-image-url.txt'.format(self.topic)
        return luigi.LocalTarget(filename)

    def run(self):
        params = {
            'action': 'query',
            'prop':   'pageimages',
            'format': 'json',
            'piprop': 'original',
            'titles': self.topic,
        }
        response = requests.get(url, params=params, verify=False)

        j = json.loads(response.text)
        img_info = next(iter(j['query']['pages'].values()))
        img_url = img_info['original']['source']
        with self.output().open('w') as f:
            f.write(img_url)


class DownloadImage(luigi.contrib.external_program.ExternalProgramTask):
    topic = luigi.Parameter()

    @property
    def img_url(self):
        with self.input().open() as f:
            return f.read()

    @property
    def filename(self):
        return self.img_url.split('/')[-1]

    def complete(self):
        """
        We need a custom completeness property because we don't know our target
        name before the required tasks have been run.
        """
        if not self.input().exists():
            return False
        return self.output().exists()

    def output(self):
        return luigi.LocalTarget(self.filename)

    def requires(self):
        return GetImageUrl(self.topic)

    def program_args(self):
        with self.input().open() as f:
            img_url = f.read()
        return ['curl', '-o', self.filename, img_url]


class WikiWordCloud(luigi.Task):
    topic = luigi.Parameter()
    max_resolution = luigi.IntParameter(default=512)

    def output(self):
        filename = '{}-wordcloud.png'
        return luigi.LocalTarget(filename)

    def requires(self):
        return {
            'text': DownloadText(self.topic),
            'image': DownloadImage(self.topic),
        }

    def run(self):
        inputs = self.input()
        with inputs['text'].open() as f:
            text = f.read()

        raw_image = np.array(Image.open(inputs['image'].fn))
        scale = min(1.0, self.max_resolution / max(raw_image.shape))
        shape = (int(raw_image.shape[1] * scale),
                 int(raw_image.shape[0] * scale))
        image = cv2.resize(raw_image,
                           dsize=shape,
                           interpolation=cv2.INTER_CUBIC)
        # image = raw_image

        image_colors = ImageColorGenerator(image)

        wordcloud_generator = WordCloud(mask=image,
                                        max_words=4096,
                                        mode='RGB',
                                        repeat=True,)
        wordcloud = wordcloud_generator.generate(text)
        colored = wordcloud.recolor(color_func=image_colors)

        import matplotlib.pyplot as plt

        # show
        plt.style.use('dark_background')

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(colored, interpolation="bilinear")
        axes[1].imshow(image, cmap=plt.cm.gray, interpolation="bilinear")
        for ax in axes:
            ax.set_axis_off()
        plt.show()
