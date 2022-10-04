import os

import fitz

import torch
from transformers import pipeline

import cv2
import pytesseract
from pytesseract import Output


class DocVQAModel():
    def __init__(self,model_name_path="impira/layoutlm-document-qa"):
        """DocVQA model class

        Args:
            model_name_path (str): Path to the pre-trained model.
        """
        self._model = pipeline(
            "document-question-answering",
            model=model_name_path,
            )
        self.max_seq_len = self._model.tokenizer.init_kwargs['model_max_length']

    @property
    def model(self):
        """ Propery of DocVQA model.
        Returns:
            Encoder-only model object (based on model name)
        """
        return self._model

    def normalize_box(self, box, width, height):
        """
            Normalize bounding boxes in the range of 0 to 1000
        """
        return [
            max(0, min(1000, int(1000 * (box[0] / width)))),
            max(0, min(1000, int(1000 * (box[1] / height)))),
            max(0, min(1000, int(1000 * (box[2] / width)))),
            max(0, min(1000, int(1000 * (box[3] / height)))),
        ]

    def preprocess_pdf(self, pdf, tolerance=2, page_no=1):
        """
            This function uses pymupdf to extract text from the input pdf document.
        Args: 
            Pdf (str): Pdf file with a text layer 
            page (int): Page number for extraction, default 1.

        Returns:
            words (list): List of words in the page
            bboxes (list): List of bounding boxes for each word in the page
        """
        document = fitz.open(pdf)
        if page_no > len(document):
            raise AssertionError("Input document has only %s pages. Please use page number within this range." % len(document))

        page = document[page_no-1 if page_no>0 else 0]
        width, height = page.rect.width, page.rect.height
        text_words = page.get_text_words()
    
        if document.is_form_pdf is False:
            # The words should be ordered by block number
            sorted_words = sorted(text_words, key=lambda x: x[5])
        else:
            # The words should be ordered by y1 and x0
            sorted_words = sorted(text_words, key=lambda x: x[3])
    
        blocks = []
        for wid, word in enumerate(sorted_words):
            if document.is_form_pdf == False:
                blocks.append([word])
            else:
                if wid == 0:
                    blocks.append([word])
                else:
                    diff = abs(blocks[-1][-1][3] - word[3])
                    if diff < tolerance:
                        blocks[-1].append(word)
                    else:
                        blocks[-1] = sorted(blocks[-1], key=lambda x: x[0])
                        blocks.append([word])
    
        bboxes, words = [], []
        for block in blocks:
            for word_info in block:
                x0, y0, x1, y1, word, block_no, line_no, word_no = word_info
                word = [ch for ch in word if ch != "_"]
                word = "".join(word)
                if not word: continue
                box = [int(b) if int(b)>0 else 0 for b in [x0, y0, x1, y1]]
                normalized_box = self.normalize_box(box, width, height)
                bboxes.append(normalized_box)
                words.append(word)
        return words, bboxes

    def preprocess_image(self, image):
        """
            This function uses tesseract OCR to extract text from the input image.
        Args: 
            Image (str): Image file 

        Returns:
            words (list): List of words in the image
            bboxes (list): List of bounding boxes for each word in the image
        """
        img = cv2.imread(image)
        height, width, channels = img.shape
        page = pytesseract.image_to_data(img, output_type=Output.DICT)     

        tokens = page['text']
        left = page['left']
        top = page['top']
        pwidth = page['width']
        pheight = page['height']

        words, bboxes = [], []
            
        for (word,x,y,w,h) in zip(tokens,left,top,pwidth,pheight):
            #ignore empty strings
            if not word.strip(): continue
            words.append(word.strip())
            box = [x, y, x+w, y+h]
            normalized_box = self.normalize_box(box, width, height)
            bboxes.append(normalized_box)
        return words, bboxes

    def predict(self, images_queries_list, page=1):
        """This function takes a table dictionary and a list of queries as input and returns the answer to the queries using the DocVQA model.

        Args:
            images_queries_list (List): List of queries per image

        Returns:
            Dict: Returns a dictionary of query and the predicted answer.
        """
        
        query_answer_dicts = []
        for image, queries in images_queries_list:
            _, extention = os.path.splitext(image)
            if extention.lower() not in ['.png', '.pdf', '.jpeg', '.jpg', '.ps', '.jp2']:
                raise AssertionError('File format of type %s not supported' % (extention))

            if extention in [".pdf", ".ps"]:
                words, bboxes = self.preprocess_pdf(image, page_no=page)
            else:
                words, bboxes = self.preprocess_image(image)

            if not words:
                raise AssertionError("No textlayer found from the input document")
            
            query_answer_dict = {}
            for query in queries:
                inputs = {'image': None, 'question':query, 'word_boxes': list(zip(words, bboxes))}
          
                encoding = self._model.preprocess(inputs)
                for k,v in encoding.items():
                    if isinstance(v, torch.Tensor):
                        v = v[:, :self.max_seq_len]
                        encoding[k] = v
                    else:
                        if k == 'words':
                            encoding[k] = v[:self.max_seq_len]        
                        else:
                            encoding[k] = [v[0][:self.max_seq_len]]
                             
                outputs = self._model._forward(encoding)
                outputs.start_logits = outputs.start_logits.detach()
                outputs.end_logits = outputs.end_logits.detach()
                answer_dict = self._model.postprocess(outputs)
                answer = answer_dict.get('answer', '')
                query_answer_dict[query] = answer
            query_answer_dicts.append(query_answer_dict)
                
        return query_answer_dicts
