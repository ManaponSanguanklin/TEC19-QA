from langdetect import detect
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
from langdetect import detect
from transformers import AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Plus
from rank_bm25 import BM25L

#setup model
#QA model
tokenizer = AutoTokenizer.from_pretrained("ManaponS/TEC19-QA_QA_Model")
model = AutoModelForQuestionAnswering.from_pretrained("ManaponS/TEC19-QA_QA_Model")
question_answerer = pipeline("question-answering", model=model,tokenizer=tokenizer)
#Thai to English model
# tokenizerTH2EN = AutoTokenizer.from_pretrained('./TL_Model/TH2EN')
# modelTH2EN = AutoModelForSeq2SeqLM.from_pretrained('./TL_Model/TH2EN')
tokenizerTH2EN = AutoTokenizer.from_pretrained('ManaponS/TEC19-QA_TH2EN_MT')
modelTH2EN = AutoModelForSeq2SeqLM.from_pretrained('ManaponS/TEC19-QA_TH2EN_MT')
#English to Thai model
# tokenizerEN2TH = AutoTokenizer.from_pretrained('./TL_Model/EN2TH')
# modelEN2TH = AutoModelForSeq2SeqLM.from_pretrained('./TL_Model/EN2TH')
tokenizerEN2TH = AutoTokenizer.from_pretrained('ManaponS/TEC19-QA_EN2TH_MT')
modelEN2TH = AutoModelForSeq2SeqLM.from_pretrained('ManaponS/TEC19-QA_EN2TH_MT')


def TranslateThaiToEnglish(query):
    #translate from TH question to ENG
    inputsQ = tokenizerTH2EN(query, return_tensors="pt").input_ids
    outputsQ = modelTH2EN.generate(inputsQ, max_new_tokens=10000, do_sample=True, top_k=50, top_p=0.95)
    query = tokenizerTH2EN.decode(outputsQ[0], skip_special_tokens=True)
    print("แปลไทย -> อังกฤษ: "+query)
    return query

def TranslateEnglishToThai(query):
    #Translate ENG Answer to TH
    inputs = tokenizerEN2TH(query, return_tensors="pt").input_ids
    outputs = modelEN2TH.generate(inputs, max_new_tokens=10000, do_sample=True, top_k=50, top_p=0.95)
    query = tokenizerEN2TH.decode(outputs[0], skip_special_tokens=True)
    print("แปลอังกฤษ -> ไทย: "+query)

    return query

def QuestionAnswering(Query,Context,topN):
    for i in range(topN):
        context = Context[i]
        print(context)
        print('--------------------------------------------------------------------------------------')
        Question = Query
        ans = question_answerer(question=Question, context=context)
        print(ans)
        print('--------------------------------------------------------------------------------------')
        if i == 0:
            ans_df = pd.DataFrame([ans])
        else:
            ans_df = pd.concat([ans_df, pd.DataFrame([ans])], ignore_index=True)
        print(ans_df)
    return ans_df

def Get_Ans(query,context,topN):
    lang = detect(query)
    if lang == 'th':
        print('Thai')
        print("คำถาม(ไทย): "+query)
        query = TranslateThaiToEnglish(query)
        Ans = QuestionAnswering(query,context,topN)
    elif lang == 'en':
        print('English')
        print("Question: "+query)
        Ans = QuestionAnswering(query,context,topN)
    else:
        lang = 'else'
        print("Error: Please input Thai or English Question")
        Ans='Please input Thai or English Question'
        
    return Ans,lang

def Select_RankingFunction(Bm25fn,tokenized_corpus):
    if Bm25fn == 'BM25':
        print('BM25')
        Bm25 = BM25Okapi(tokenized_corpus)
    if Bm25fn == 'BM25L':
        print('BM25L')
        Bm25 = BM25L(tokenized_corpus)
    if Bm25fn == 'BM25P':
        print('BM25P')
        Bm25 = BM25Plus(tokenized_corpus)
    return Bm25

def EngAnsToThai(ans_list):
    tl_df = []
    for i in ans_list:
        tl_df.append(TranslateEnglishToThai(i)) 
    return tl_df