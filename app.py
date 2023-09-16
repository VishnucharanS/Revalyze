import streamlit as st
-m pip install
--pip install selenium
from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException, ElementClickInterceptedException
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax
import base64
from io import BytesIO
import re
import tensorflow as tf
import torch


#collecting reviews
def data(pl):
    path_to_chromedriver = r"C:\Path\To\chromedriver.exe"
    chrome_binary_path = r"C:\Drivers\chrome-win64\chrome.exe"
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = chrome_binary_path
    chrome_options.add_argument('executable_path=' + path_to_chromedriver)
    #opening browser and link
    browser = webdriver.Chrome(options=chrome_options)
    browser.get(pl)
    browser.maximize_window()
    #finding all reviews
    mydiv = browser.find_element(By.CLASS_NAME, '_3UAT2v._16PBlm')
    parent = mydiv.find_element(By.XPATH, '..')
    reviewlink = parent.get_attribute('href')
    browser.get(reviewlink)#opened all reviews
    
    #scraping, sorting fake and small reviews into a df
    reviews = []
    i=0
    while True:
        try:
            if i == 0:
                review = browser.find_elements(By.CLASS_NAME,'t-ZTKy')
                for r in review:
                    rt = r.text
                    if len(rt.split()) > 2:
                        rt= re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', rt)
                        reviews.append(rt)
                    nextreviewpage = browser.find_element(By.CLASS_NAME,"_1LKTO3")
                nextreviewpage = nextreviewpage.get_attribute('href')
                browser.get(nextreviewpage)
                i+=1
            
            if browser.find_element(By.CLASS_NAME,"_1LKTO3"):
                i+=1
                review = browser.find_elements(By.CLASS_NAME,'t-ZTKy')
                for r in review:
                    rt = r.text
                    if len(rt.split()) > 2:
                        rt= re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', rt)
                        reviews.append(rt)
                nextreviewpages=[]
                nextreviewpage = browser.find_elements(By.CLASS_NAME,"_1LKTO3")
                for review in nextreviewpage:
                    review = review.get_attribute('href')
                    nextreviewpages.append(review)
                if len(nextreviewpages)>1:
                    browser.get(nextreviewpages[1])
                else:
                    review = browser.find_elements(By.CLASS_NAME,'t-ZTKy')
                    for r in review:
                        rt = r.text
                        if len(rt.split()) > 2:
                            rt= re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', rt)
                            reviews.append(rt)
                    print(len(reviews), "Reviews extracted")
                    break
        except (StaleElementReferenceException, ElementClickInterceptedException, TimeoutException, NoSuchElementException):
            break
    reviewsdf = pd.DataFrame(reviews, columns=["Reviews"])
    return reviewsdf

def polarity_scores_roberta(df):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

    encoded_text = tokenizer(df, return_tensors='tf')
    output = model(**encoded_text)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

def robertaslow(df):
    df = df.reset_index().rename(columns={'index':'ID'})
    results = []
    res = {}
    for i in range(len(df)):
        res[df['ID'][i]]= polarity_scores_roberta(df['Reviews'][i])
    results = pd.DataFrame(res).T
    results = results.reset_index().rename(columns={'index':'ID'})
    results = results.merge(df, how='left')
    return results

def slowbutton(pl):
    strl = str(pl)
    df = data(strl)
    resultsf = robertaslow(df)
    total_pos = resultsf['roberta_pos'].sum()
    total_neg = resultsf['roberta_neg'].sum()
    total_neu = resultsf['roberta_neu'].sum()

    if total_pos > total_neg and total_pos > total_neu:
        overall_sentiment = 'Positive'
    elif total_neg > total_pos and total_neg > total_neu:
        overall_sentiment = 'Negative'
    else:
        overall_sentiment = 'Neutral'

    #print(f"Overall sentiment for the product: {overall_sentiment}")
    
    sentiments = ['Positive', 'Neutral', 'Negative']
    scores = [total_pos, total_neu, total_neg]
    
    plt.bar(sentiments, scores, color=['green', 'gray', 'red'])
    plt.xlabel('Sentiment')
    plt.ylabel('Aggregated Score')
    plt.title('Overall Sentiment Analysis for the Product')
   
    fig, ax = plt.subplots()
    ax.bar(sentiments, scores, color=['green', 'gray', 'red'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Aggregated Score')
    ax.set_title('Overall Sentiment Analysis for the Product')
    

    # Save the plot to a BytesIO buffer
    #buffer = BytesIO()
    #plt.savefig(buffer, format='png')
    #buffer.seek(0)
    
    # Convert the plot to a base64-encoded image
    #plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    #st.image(plot_base64)'''
    
    return overall_sentiment, fig

def sentifast(df):
    senti = pipeline("sentiment-analysis")
    df = df.reset_index().rename(columns={'index':'ID'})
    ress = {}
    results = []
    for i in range(len(df)):
        ress[df['ID'][i]]= senti(df['Reviews'][i])
    flat_data = [item[0] for item in ress.values()]
    results= pd.DataFrame(flat_data)
    results = pd.DataFrame(results)
    results = results.reset_index().rename(columns={'index':'ID'})
    results = results.merge(df, how='right')
    return results

def fastbutton(pl):
    strl = str(pl)
    df = data(strl)
    resultsf = sentifast(df)
    total_pos = 0
    total_neg = 0
    for i in range(len(resultsf)):
        if resultsf['label'][i] == "POSITIVE":
            total_pos += resultsf['score'][i]
        else:
            total_neg += resultsf['score'][i]
    if total_neg > total_pos:
        overall_sentiment = 'Negative'
    else:
        overall_sentiment = 'Positive'
    #print(f"Overall sentiment for the product: {overall_sentiment}")
    sentiments = ['Positive', 'Negative']
    scores = [total_pos, total_neg]
    fig, ax = plt.subplots()
    ax.bar(sentiments, scores, color=['green', 'red'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Aggregated Score')
    ax.set_title('Overall Sentiment Analysis for the Product')
    

    # Save the plot to a BytesIO buffer
    #buffer = BytesIO()
    #plt.savefig(buffer, format='png')
    #buffer.seek(0)
    
    # Convert the plot to a base64-encoded image
    #plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    #st.image(plot_base64)'''

    return overall_sentiment, fig

def main():
    st.title("Revalyze")
    st.markdown('<link rel="stylesheet" type="text/css" href="style.css">', unsafe_allow_html=True)

    # Input for the link
    link = st.text_input('Paste the Link:')

    # Buttons to choose between slow and fast processing
    if st.button("Fast (Sentiment Analysis Pipeline from Transformers)"):
        result_sentiment, result_plot = fastbutton(link)
        st.write(f"Overall sentiment for the product: {result_sentiment}")
        st.pyplot(result_plot)
        
    if st.button("Slow (RoBERTa Model which also shows Neutral Sentiment)"):
        result_sentiment, result_plot = slowbutton(link)
        st.write(f"Overall sentiment for the product: {result_sentiment}")
        st.pyplot(result_plot)
    
    

if __name__ == "__main__":
    main()
