import streamlit as st
from gtts import gTTS
import os
import torch
st.set_option('deprecation.showfileUploaderEncoding', False)

html_temp = """
    <div style="background-color:black ;padding:10px">
    <h1 style="color:white;text-align:center;">I'M READY TO HELP</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
html_temp69 = """
    <div style="background-color:white ;padding:10px">
    <h3 style="color:black;text-align:center;">PLEASE HAVE A LOOK AT THE SIDEBAR TO MAKE THE BEST USE OF this WEBAPP</h3>
    </div>
    """
st.markdown(html_temp69, unsafe_allow_html=True)
st.sidebar.header("Ypu don't need to read everything, cause it does it for you and is very exact while providing answers")
st.sidebar.markdown('<b>Best used while answering interviews</b>', unsafe_allow_html=True)
st.sidebar.markdown('<b>Created by:Nimisha Bhide</b>', unsafe_allow_html=True)
st.sidebar.markdown('<b>Email id:nbhide.nb@gmail.com</b>', unsafe_allow_html=True)
try:
import torch
myText=st.text_input("PLEASE ENTER THE TEXT HERE")
question=st.text.input("PLEASE ENTER THE QUESTION HERE")
if st.button('ANSWER'):
  from transformers import BertForQuestionAnswering
  model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
  from transformers import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
  input_ids = tokenizer.encode(question, myText)
  tokens = tokenizer.convert_ids_to_tokens(input_ids)
  sep_index = input_ids.index(tokenizer.sep_token_id)
  num_seg_a = sep_index + 1
  num_seg_b = len(input_ids) - num_seg_a
  segment_ids = [0]*num_seg_a + [1]*num_seg_b
  assert len(segment_ids) == len(input_ids)
  start_scores, end_scores = model(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]))
  answer_start = torch.argmax(start_scores)
  answer_end = torch.argmax(end_scores)
  tokens = tokenizer.convert_ids_to_tokens(input_ids)
  answer = tokens[answer_start]
  answer = tokens[answer_start]
  for i in range(answer_start + 1, answer_end + 1):
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    else:
        answer += ' ' + tokens[i]

  print('Answer: "' + answer + '"')
  language="en"
  output=gTTS(text=answer,lang=language,slow=False)
  output.save("voice.ogg")
  audio_file = open('voice.ogg', 'rb')
  audio_bytes = audio_file.read()
  st.audio(audio_bytes, format='audio/ogg')
except AssertionError:
    st.error("Please enter text that you want me to answer")
