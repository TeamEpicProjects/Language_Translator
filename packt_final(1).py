''' This translate.py script file helps in making a basic UI for the Story Translation. '''
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
import streamlit as st                                           # to make the user interface of the app
import os                                                        # github repository cloning
import re                                                        # for filteration of query
import time                                                      # for loading animations
os.environ["GIT_PYTHON_REFRESH"] = "quiet"                           
import base64                                                    # text file encoding
#from indicTrans.inference.engine import Model                   # importing translation model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM    # Helsinki huggingface model
import nltk
from tensorflow import keras                                     # to load our custom-made model
import pickle                                                    # to load custom tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
def get_words(n, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == n:
      return word
  return None
def predict_sequence(model, tokenizer, source,tokenizer2):
  source=pd.Series(source)
  seq = tokenizer2.texts_to_sequences(source)
  seq = pad_sequences(seq, maxlen = 30, padding = 'post')
  prediction = model.predict(seq, verbose = 0)[0]
  integers = [np.argmax(vector) for vector in prediction]
  target = list()
  for i in integers:
    word = get_words(i, tokenizer)
    if word is not None:
      target.append(word)
    # target.append(word)
  res = []
  for i in target:
      if i not in res:
          res.append(i)
  # res = list(set(target))
  if len(res)!=0:
    en_story=' '.join(res)
  else:
    en_story=' sorry model is not able to translate : '+source
  return en_story.strip()

def hel_translate(hi_story, tokenizer, model):
    en_story = []
    sent_story = nltk.sent_tokenize(hi_story)
    for sent in sent_story:
        input_ids = tokenizer.encode(sent, return_tensors="pt", padding=True)
        outputs = model.generate(input_ids)
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        en_story.append(decoded_text)
    en_story = ' '.join(en_story)
    return en_story.strip()


def sale_translate(hi_story, tokenizer, model):
    en_story = []
    sent_story = nltk.sent_tokenize(hi_story)
    for sent in sent_story:
        input_ids = tokenizer.encode(sent, return_tensors="pt", padding = True)
        outputs = model.generate(input_ids, num_beams= None, early_stopping = True)
        decoded_text = tokenizer.decode(outputs[0]).replace('<pad>', "").strip().lower()
        en_story.append(decoded_text)
    en_story = ' '.join(en_story)
    return en_story.strip()

def text_downloader(raw_text):
	b64 = base64.b64encode(raw_text.encode()).decode()
	st.markdown("#### Download Translated English Story File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="Translated_Story.txt">Download File!!</a>'
	st.markdown(href,unsafe_allow_html=True)
    
    
def main():
    
    #indic2en_model = Model(expdir='../indic-en')
    
    # For custom made lstm model
    lstm_tokenizer = pickle.load(open('/content/eng_tokenizer.pkl','rb'))
    lstm_tokenizer2 = pickle.load(open('/content/hi_tokenizer.pkl','rb'))
    lstm_model = keras.models.load_model('/content/My_Translation_HI_EN_Model.h5')
    
    # For Helsinki Model
    hel_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hel_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-hi-en")   
    
    # For Salesken Model
    sale_tokenizer = AutoTokenizer.from_pretrained("salesken/translation-hi-en")
    sale_model = AutoModelForSeq2SeqLM.from_pretrained("salesken/translation-hi-en")
    
    
    # app title
    st.header('Language Translator')
    
    html_temp = '''
    <div style="background-color:teal; padding:20px; border-radius: 25px;">
    <h2 style="color:white; text-align:center; font-size: 30px;"><b>Hindi Story to English Translation Model</b></h2>
    </div><br><br>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # text input
    hi_story = st.text_area('Please type hindi story:')
    
    #model choice
    model = st.sidebar.radio("Choose what type of model to be used",("Customed Made using LSTM", "Huggingface's Helsinki Model","Huggingface's Salesken Model"))
    # if model == 'IndicTRANS':
    #     lang = st.sidebar.radio("Story Language (only for indicTRANS)",("Hindi","Punjabi","Gujarati", "Tamil","Bengali", "Nepali", "Assamese", "Marathi"))
    #     lang_dict = {'Hindi' : 'hi',"Punjabi" : 'pa',"Gujarati" : 'gu', "Tamil" : 'ta',"Bengali" : 'bn', "Nepali" : 'np', "Assamese" : 'as', "Marathi" : 'mr'}
    #     lang = lang_dict[lang]
        
        
    # predicting ticket_type
    if st.button('Translate'):
        
        # necessary requirements
        
        # checker for empty text
        if hi_story.strip()=='':
            st.warning('No information has been written! Kindly write your story again.')
            st.stop()
            
        # checker for punctuation only in the text
        if str(re.sub(r"[,./;'\|!@#$%^&*-_=+`~']+", ' ', hi_story)).strip()=='':
            st.warning('You have written punctuation only. Kindly write a proper story again.')
            st.stop()
            
        # checker for numbers only in the text
        if str(re.sub(r"[0-9]+", ' ', hi_story)).strip()=='':
            st.warning('You have written numbers only. Kindly write a proper story again.')
            st.stop()
        if str(re.sub(r"[a-zA-Z0-9]", ' ', hi_story)).strip()=='':
            st.warning('You have written in other language. Kindly write a proper hindi story again.')
            st.stop()
        # ^[a-zA-Z0-9]*$
        # text should have atleast 5 words in it.
        if len(hi_story.split(' ')) < 5:
            st.warning('Story information provided is too low. Kindly write atleast five words in the story.')
            st.stop()
            
        with st.spinner(text = 'Translating...'):
            time.sleep(3)
        
        
        cols = st.columns(2)
        
        with cols[0]:
            st.info('Original Hindi Story')
            st.text_area(label = ' ' ,value = hi_story, height = 500)
        
        with cols[1]:
            st.info('Translated English Story')
            
            if model == 'IndicTRANS':
                #en_story = indic2en_model.translate_paragraph(hi_story, lang, 'en')
                en_story = 'Indic'
            elif model == "Customed Made using LSTM":
                en_story = predict_sequence(lstm_model,lstm_tokenizer,hi_story,lstm_tokenizer2)
                # en_story = LSTM_translate(hi_story, lstm_tokenizer, lstm_model)
            elif model == "Huggingface's Salesken Model":
                en_story = sale_translate(hi_story, sale_tokenizer, sale_model)                
            else:
                en_story = hel_translate(hi_story, hel_tokenizer, hel_model)
            
            
            st.text_area(label = ' ', value = en_story, height = 200)
            
            # for downloading file
            text_downloader(en_story)
        
        
if __name__=='__main__':
    main()