import streamlit as st
import joblib
vect1=joblib.load('vect.joblib')
mod=joblib.load('cl.ml')

def home(input_data):
    ls=[]
    ls.clear()
    ls.append(input_data)
    p=vect1.transform(ls)
    pred=mod.predict(p)
    
    if pred==1:
        return 'Spam'
    else:
        return 'Ham'
    
    
def main():
    st.title('Spam Detection Web App')
    st.text('This program classifies text as either \'spam\' or \'ham\'')
    
    #input
    text=st.text_input('Input some text in space below:')
    
    #prediction
    clfn=''
    
    #button
    if st.button('Classify Text'):
        clfn=home(text)
    st.success(clfn)
    
if __name__=='__main__':
    main()
          
    
    
    
    
    



