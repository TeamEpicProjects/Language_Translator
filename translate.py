import streamlit as st                       # to make the user interface of the app
import os                                    # github repository cloning
import re                                    # for filteration of query
import time                                  # for animations
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
#import git                                  # to clone repository
#from git import Repo
#import zipfile                              # extraction purpose
import base64                                # text file encoding
#from indicTrans.inference.engine import Model            # importing translation model

def text_downloader(raw_text):
	b64 = base64.b64encode(raw_text.encode()).decode()
	st.markdown("#### Download Translated English Story File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="Translated_Story.txt">Download File!!</a>'
	st.markdown(href,unsafe_allow_html=True)
    
    
def main():
    
    #!wget clone "https://github.com/AI4Bharat/indicTrans.git"
    #indic2en_model = Model(expdir='../indic-en')
        
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
        # translation of the story
        #en_story = indic2en_model.translate_paragraph(hi_story, 'hi', 'en')
        with cols[1]:
            st.info('Original English Story')
            en_story = 'HELLLOOZZZZZZz'
            st.text_area(label = ' ', value = en_story, height = 500)
            
            # for downloading file
            text_downloader(en_story)
        
        
if __name__=='__main__':
    main()